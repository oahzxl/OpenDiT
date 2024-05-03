import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizers warning

import warnings

warnings.filterwarnings("ignore")

import argparse

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from opendit.core.parallel_mgr import get_parallel_manager, set_parallel_manager
from opendit.datasets.dataloader import prepare_dataloader
from opendit.embed.t5_text_emb import T5Encoder
from opendit.models.opensora.datasets import DatasetFromCSV, get_transforms_video
from opendit.models.opensora.scheduler import IDDPM
from opendit.models.opensora.stdit import STDiT_XL_2
from opendit.utils.profile import PerformanceEvaluator
from opendit.utils.utils import str_to_dtype
from opendit.vae.wrapper import VideoAutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({}, seed=args.global_seed)
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = str_to_dtype(args.mixed_precision)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Process Group
    # ==============================
    sp_size = args.sequence_parallel_size
    dp_size = dist.get_world_size() // sp_size
    set_parallel_manager(dp_size, sp_size, dp_axis=0, sp_axis=1, method=args.sp)

    # ======================================================
    # Initialize Model, Objective, Optimizer
    # ======================================================
    # Create VAE encoder
    vae = VideoAutoencoderKL(args.vae_pretrained_path, split=4).to(device, dtype)

    # Configure input size
    input_size = (args.num_frames, args.image_size[0], args.image_size[1])
    latent_size = vae.get_latent_size(input_size)
    text_encoder = T5Encoder(
        args.text_pretrained_path, args.text_max_length, shardformer=args.text_speedup, device=device
    )

    # Shared model config for two models
    model_config = {
        "time_scale": args.model_time_scale,
        "space_scale": args.model_space_scale,
        "input_size": latent_size,
        "in_channels": vae.out_channels,
        "caption_channels": text_encoder.output_dim,
        "model_max_length": text_encoder.model_max_length,
        "enable_layernorm_kernel": args.enable_layernorm_kernel,
    }

    # Create DiT model
    model = STDiT_XL_2(
        enable_flashattn=args.enable_flashattn,
        dtype=dtype,
        **model_config,
    ).to(device, dtype)

    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create diffusion
    # default: 1000 steps, linear noise schedule
    scheduler = IDDPM(timestep_respacing="")

    # Setup optimizer
    # We used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, weight_decay=0, adamw_mode=True
    )
    # You can use a lr scheduler if you want
    # Recommend if you continue training from a model
    lr_scheduler = None

    model.train()

    # Setup data:
    dataset = DatasetFromCSV(
        args.data_path,
        transform=get_transforms_video((args.image_size[0], args.image_size[1])),
        num_frames=args.num_frames,
        frame_interval=args.frame_interval,
    )
    dataloader = prepare_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        pg_manager=get_parallel_manager(),
    )

    # Boost model for distributed training
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )
    torch.set_default_dtype(torch.float)

    perf = PerformanceEvaluator(ignore_steps=args.warmup, dp_world_size=dp_size)

    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    # VAE encode
    with torch.no_grad():
        x0 = batch["video"].to(device)
        assert x0.shape == (args.batch_size, 3, args.num_frames, args.image_size[0], args.image_size[1]), x0.shape
        y0 = batch["text"]
        # Map input images to latent space + normalize latents:
        x0 = vae.encode(x0.to(dtype))
        # Prepare text inputs
        model_args0 = text_encoder.encode(y0)

    dataloader_iter = iter(dataloader)
    for step in range(args.warmup + args.runtime):
        perf.on_step_start(step)
        x = x0.clone().detach()
        model_args = {k: v.clone().detach() for k, v in model_args0.items()}

        # Diffusion
        t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
        loss_dict = scheduler.training_losses(model, x, t, model_args)

        # Backward & update
        loss = loss_dict["loss"].mean()
        booster.backward(loss=loss, optimizer=optimizer)
        optimizer.step()
        optimizer.zero_grad()
        perf.on_step_end(x)

    perf_result = perf.on_fit_end()
    final_output = f"Config:\n{args}\n\nPerformance: {perf_result}"
    if coordinator.is_master():
        print(final_output)
        with open(
            f"log/batch{args.batch_size}_f{args.num_frames}_h{args.image_size[0]}_w{args.image_size[1]}_sp{args.sequence_parallel_size}_{args.sp}_perf.log",
            "w",
        ) as f:
            f.write(final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--outputs", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Path to the dataset")

    # profile
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runtime", type=int, default=20)

    # sample
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--frame_interval", type=int, default=3)

    # model
    parser.add_argument("--model_space_scale", type=float, default=1.0)
    parser.add_argument("--model_time_scale", type=float, default=1.0)

    # vae
    parser.add_argument("--vae_pretrained_path", type=str, default="stabilityai/sd-vae-ft-ema")

    # text encoer
    parser.add_argument("--text_pretrained_path", type=str, default="t5-v1_1-xxl")
    parser.add_argument("--text_max_length", type=int, default=120)
    parser.add_argument("--text_speedup", action="store_true")

    # kernel
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")

    # parallel
    parser.add_argument("--plugin", type=str, default="zero2")
    parser.add_argument("--sp", type=str, default="dsp")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size, enable if > 1")

    args = parser.parse_args()
    main(args)
