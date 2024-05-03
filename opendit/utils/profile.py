from time import time
from typing import Optional

import torch
import torch.distributed as dist
from colossalai.accelerator import get_accelerator
from colossalai.cluster import DistCoordinator
from torch import Tensor


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


@torch.no_grad()
def all_reduce_mean(x: float, world_size: int) -> float:
    if world_size == 1:
        return x
    tensor = torch.tensor([x], device=get_accelerator().get_current_device())
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration += time() - self.start_time
        self.start_time = None

    def reset(self) -> None:
        self.duration = 0.0


class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        ignore_steps: int = 0,
        dp_world_size: Optional[int] = None,
    ) -> None:
        self.ignore_steps = ignore_steps

        self.coordinator = DistCoordinator()
        self.dp_world_size = dp_world_size or self.coordinator.world_size
        self.disable: bool = False
        self.timer = Timer()
        self.num_samples: int = 0

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        if self.disable:
            return
        get_accelerator().synchronize()
        self.timer.start()

    def on_step_end(self, input_ids: Tensor, **kwargs) -> None:
        if self.disable:
            return
        get_accelerator().synchronize()
        self.timer.end()

        batch_size = input_ids.shape[0]

        self.num_samples += batch_size

    def on_fit_end(self) -> None:
        avg_duration = all_reduce_mean(self.timer.duration, self.coordinator.world_size)
        avg_throughput = self.num_samples * self.dp_world_size / (avg_duration + 1e-12)
        self.coordinator.print_on_master(
            f"\nnum_samples: {self.num_samples}, dp_world_size: {self.dp_world_size}, avg_duration: {avg_duration}, "
            f"avg_throughput: {avg_throughput} samples/sec\n"
        )
