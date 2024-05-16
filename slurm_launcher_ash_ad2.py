import argparse
import os

from accelerate.commands.launch import launch_command, launch_command_parser

# ash-ad2
MACHINE_IP_MAPPING = {
    1: "10.18.74.66",
    2: "10.18.74.128",
    3: "10.18.74.174",
    4: "10.18.74.232",
    5: "10.18.74.63",
    6: "10.18.74.110",
    7: "10.18.74.55",
    8: "10.18.74.135",
    9: "10.18.74.177",
    10: "10.18.74.231",
    11: "10.18.74.121",
    12: "10.18.74.18",
    13: "10.18.74.124",
    14: "10.18.74.158",
    15: "10.18.74.40",
    16: "10.18.74.8",
    17: "10.18.74.68",
    18: "10.18.74.241",
    19: "10.18.74.157",
    20: "10.18.74.50",
    21: "10.18.74.10",
    22: "10.18.74.123",
    23: "10.18.74.13",
    24: "10.18.74.206",
    25: "10.18.74.142",
    26: "10.18.74.165",
    27: "10.18.74.78",
    28: "10.18.74.235",
    29: "10.18.74.114",
    30: "10.18.74.58",
    31: "10.18.74.92",
    32: "10.18.74.213",
    33: "10.18.74.205",
    34: "10.18.74.189",
    35: "10.18.74.172",
    36: "10.18.74.127",
    37: "10.18.74.184",
    38: "10.18.74.19",
    39: "10.18.74.198",
    40: "10.18.74.202",
    41: "10.18.74.6",
    42: "10.18.74.132",
    43: "10.18.74.115",
    44: "10.18.74.185",
    45: "10.18.74.102",
    46: "10.18.74.51",
    47: "10.18.74.170",
    48: "10.18.74.74",
    49: "10.18.74.221",
    50: "10.18.74.71",
    51: "10.18.74.237",
    52: "10.18.74.129",
    53: "10.18.74.249",
    54: "10.18.74.187",
    55: "10.18.74.150",
    56: "10.18.74.3",
    57: "10.18.74.46",
    58: "10.18.74.90",
    59: "10.18.74.64",
    60: "10.18.74.7",
    61: "10.18.74.228",
    62: "10.18.74.236",
    63: "10.18.74.53",
    64: "10.18.74.240",
    65: "10.18.74.244",
    66: "10.18.74.11",
    67: "10.18.74.145",
    68: "10.18.74.96",
    69: "10.18.74.22",
    70: "10.18.74.103",
    71: "10.18.74.87",
    72: "10.18.74.75",
    73: "10.18.74.48",
    74: "10.18.74.94",
    75: "10.18.74.109",
    76: "10.18.74.160",
    77: "10.18.74.54",
    78: "10.18.74.15",
    79: "10.18.74.42",
    80: "10.18.74.175",
    81: "10.18.74.136",
    82: "10.18.74.39",
    83: "10.18.74.140",
    84: "10.18.74.197",
    85: "10.18.74.227",
    86: "10.18.74.65",
    87: "10.18.74.49",
    88: "10.18.74.62",
    89: "10.18.74.217",
    90: "10.18.74.239",
    91: "10.18.74.216",
    92: "10.18.74.14",
    93: "10.18.74.230",
    94: "10.18.74.93",
    95: "10.18.74.162",
    96: "10.18.74.126",
    97: "10.18.74.214",
    98: "10.18.74.84",
    99: "10.18.74.16",
    100: "10.18.74.30",
    101: "10.18.74.77",
    102: "10.18.74.73",
    103: "10.18.74.72",
    104: "10.18.74.134",
    105: "10.18.74.139",
    106: "10.18.74.156",
    107: "10.18.74.201",
    108: "10.18.74.195",
    109: "10.18.74.117",
    110: "10.18.74.125",
    111: "10.18.74.138",
    112: "10.18.74.141",
    113: "10.18.74.104",
    114: "10.18.74.178",
    115: "10.18.74.211",
    116: "10.18.74.252",
    117: "10.18.74.79",
    118: "10.18.74.95",
    119: "10.18.74.183",
    120: "10.18.74.215",
    121: "10.18.74.24",
    122: "10.18.74.27",
}
MACHINE_TOPOLOGY = list(range(1, 129))
CLUSTER_PREFIX = "pika-h100-ash-ad2-"


def parse_node_list(node_list):
    node_list = node_list[len(CLUSTER_PREFIX) :]
    node_list = node_list.strip("[]").split(",")
    expanded_node_list = []
    for node_item in node_list:
        if "-" in node_item:
            start, end = node_item.rsplit("-", 1)
            for idx in range(int(start), int(end) + 1):
                expanded_node_list.append(idx)
        else:
            expanded_node_list.append(int(node_item))
    return expanded_node_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="train.py")
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=14)
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
