import argparse
import os

from accelerate.commands.launch import launch_command, launch_command_parser

# denvrbm
MACHINE_IP_MAPPING = {
    1001: "10.223.255.29",
    1002: "10.223.255.23",
    1003: "10.223.255.35",
    1004: "10.223.255.149",
    1005: "10.223.255.39",
    1006: "10.223.255.42",
    1007: "10.223.255.46",
    1008: "10.223.255.51",
    1009: "10.223.255.115",
    1010: "10.223.255.91",
    1011: "10.223.255.27",
    1012: "10.223.255.34",
    1013: "10.223.255.130",
    1014: "10.223.255.38",
    1015: "10.223.255.43",
    1016: "10.223.255.138",
    1017: "10.223.255.110",
    1018: "10.223.255.140",
    1019: "10.223.255.98",
    # 1020: '',
    1021: "10.223.255.154",
    1022: "10.223.255.60",
    1023: "10.223.255.122",
    1024: "10.223.255.124",
    1025: "10.223.255.136",
    1026: "10.223.255.123",
    1027: "10.223.255.146",
    1028: "10.223.255.134",
    1029: "10.223.255.135",
    1030: "10.223.255.118",
    1031: "10.223.255.22",
    1032: "10.223.255.129",
    1033: "10.223.255.142",
    # 1034: '',
    # 1035: '',
    1036: "10.223.255.128",
    1037: "10.223.255.141",
    1038: "10.223.255.131",
    1039: "10.223.255.152",
    1040: "10.223.255.137",
    1041: "10.223.255.48",
    1042: "10.223.255.53",
    1043: "10.223.255.139",
    1044: "10.223.255.101",
    1045: "10.223.255.24",
    # 1046: '',
    1047: "10.223.255.117",
    1048: "10.223.255.93",
    1049: "10.223.255.71",
    1050: "10.223.255.159",
    1051: "10.223.255.86",
    1052: "10.223.255.76",
    1053: "10.223.255.58",
    # 1054: '',
    # 1055: '',
    1056: "10.223.255.25",
    1057: "10.223.255.85",
    1058: "10.223.255.157",
    1059: "10.223.255.90",
    # 1060: '',
    1061: "10.223.255.78",
    1062: "10.223.255.95",
    1063: "10.223.255.187",
    1064: "10.223.255.64",
    1065: "10.223.255.72",
    1066: "10.223.255.125",
    1067: "10.223.255.106",
    1068: "10.223.255.65",
    # 1069: '',
    1070: "10.223.255.153",
    # 1071: '',
    # 1072: '',
    1073: "10.223.255.126",
    1078: "10.223.255.155",
}
MACHINE_TOPOLOGY = (
    list(range(1001, 1020))
    + list(range(1021, 1034))
    + list(range(1036, 1046))
    + list(range(1047, 1054))
    + list(range(1056, 1060))
    + list(range(1061, 1069))
    + [1070, 1073, 1078]
)
CLUSTER_PREFIX = "denvrbm-"


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
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=26)
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
        assert os.environ["SLURMD_NODENAME"] == CLUSTER_PREFIX + str(main_machine)

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
