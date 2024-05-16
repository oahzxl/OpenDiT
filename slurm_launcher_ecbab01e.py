import argparse
import os

from accelerate.commands.launch import launch_command, launch_command_parser

# ecbab01e
MACHINE_IP_MAPPING = {i: f"10.49.5.{i}" for i in range(1, 129)}
MACHINE_TOPOLOGY = list(range(1, 129))
CLUSTER_PREFIX = "ecbab01e-"


def parse_node_list(node_list):
    node_list = node_list[len(CLUSTER_PREFIX) :]
    node_list = node_list.strip("[]").split(",")
    expanded_node_list = []
    for node_item in node_list:
        if "-" in node_item:
            start, end = node_item.split("-")
            for idx in range(int(start), int(end) + 1):
                expanded_node_list.append(idx)
        else:
            expanded_node_list.append(int(node_item))
    return expanded_node_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="train.py")
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16", "fp8"], default="no")
    parser.add_argument("--main_process_port", type=int, default=34049)
    args, unparsed = parser.parse_known_args()

    # Should only launch once per machine
    assert int(os.environ["SLURM_NTASKS_PER_NODE"]) == 1
    num_machines = int(os.environ["SLURM_NNODES"])

    # Get the main machine with rank 0
    node_list = parse_node_list(os.environ["SLURM_JOB_NODELIST"])
    main_machine = None
    for idx in MACHINE_TOPOLOGY:
        if idx in node_list:
            main_machine = idx
            break
    assert main_machine is not None

    # Double check on main machine
    if int(os.environ["SLURM_NODEID"]) == 0:
        assert os.environ["SLURMD_NODENAME"] == CLUSTER_PREFIX + f"{main_machine:03d}"

    num_processes = num_machines * int(os.environ["SLURM_GPUS_ON_NODE"])
    accelerate_args = []
    if num_processes > 1:
        accelerate_args.append("--multi_gpu")
    accelerate_args += [
        "--same_network",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        str(num_machines),
        "--machine_rank",
        os.environ["SLURM_NODEID"],
        "--num_cpu_threads_per_process",
        str(args.num_cpu_threads_per_process),
        "--main_process_ip",
        MACHINE_IP_MAPPING[main_machine],
        "--main_process_port",
        str(args.main_process_port),
        "--mixed_precision",
        args.mixed_precision,
        "--dynamo_backend",
        "no",
        args.script,
        *unparsed,
    ]
    accelerate_parser = launch_command_parser()
    accelerate_args = accelerate_parser.parse_args(accelerate_args)

    launch_command(accelerate_args)
