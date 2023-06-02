"""Grid Search for FedETuning"""

import os
import sys
import itertools as it
from loguru import logger
from multiprocessing import Pool
from configs.tuning import fed_best_hyperparameter


def run_process(proc):
    os.system(proc)


run_dirs = sys.argv[1]
fl_algorithm = sys.argv[2]
task_name = sys.argv[3]
tuning_type = sys.argv[4]
port_start = int(sys.argv[5])
device = sys.argv[6]

device_idx_list = [idx for idx in device.split(",")]
n_gpu = len(device_idx_list)
world_size = 3
logger.info(f"world_size is {world_size}")

if task_name == "conll":
    max_seq = 32
    data_file = "fedner"
    dataset_name = "ner"
    metric_name = "conll"
    model_output_mode = "token_classification"
else:
    max_seq = 128
    data_file = "fedglue"
    dataset_name = "glue"
    metric_name = "glue"
    model_output_mode = "seq_classification"

logger.info(f"{task_name}'s max_seq is {max_seq}")

cmds = []
gpu_index = 0
for tuning_type in ['lora', 'prefix', 'adapter', 'bitfit', 'fine-tuning']:
    hyper_parameter = fed_best_hyperparameter[task_name][tuning_type]
    # hyper_parameter["seed"] = [42]
    hyper_parameter["num_train_epochs"] = [1]
    # hyper_parameter["alpha"] = [0.1, 10.0]
    # hyper_parameter["sample"] = [0.2, 0.3, 0.4]
    for parameter in it.product(*list(hyper_parameter.values())):
        specific_parameter_dict = {key: parameter[list(hyper_parameter.keys()).index(key)]
                                for key in list(hyper_parameter.keys())}
        if "lora_r" in specific_parameter_dict:
            specific_parameter_dict["lora_alpha"] = specific_parameter_dict["lora_r"]
        port = port_start + gpu_index
        device_index = gpu_index % n_gpu

        # cmd = f'CUDA_VISIBLE_DEVICES={device_idx_list[device_index]} python3 main.py '
        options = [
            "--model_name_or_path", f"{run_dirs}/pretrain/nlp/roberta-base/",
            "--output_dir", f"{run_dirs}/output/{data_file}",
            "--task_name", f"{task_name}",
            "--fl_algorithm", f"{fl_algorithm}",
            "--raw_dataset_path", f"{run_dirs}/data/{data_file}",
            "--partition_dataset_path", f"{run_dirs}/data/{data_file}",
            "--max_seq_length", f"{max_seq}",
            "--world_size", f"{world_size}",
            "--port", f"{port}",
            "--dataset_name", dataset_name,
            "--metric_name", metric_name,
            "--model_output_mode", model_output_mode,
            "--tuning_type", f"{tuning_type}_roberta-base",
            "--do_grid", "True",
        ]
        for key, value in specific_parameter_dict.items():
            options.extend(["--" + key, str(value)])

        server_options = options + ["--rank", "0"]
        server_cmd = f'CUDA_VISIBLE_DEVICES={device_idx_list[0]} python3 main.py ' + " ".join(server_options)
        one_cmd_list = [server_cmd]
        for i in range(1, world_size):
            client_options = options + ["--rank", str(i)]
            client_cmd = f'CUDA_VISIBLE_DEVICES={device_idx_list[i]} python3 main.py ' + " ".join(client_options)
            # client_cmd = "sleep 2s " + client_cmd
            one_cmd_list.append(client_cmd)
        one_cmd = " & ".join(one_cmd_list)
        one_cmd += " & wait"

        gpu_index += 1
        cmds.append(one_cmd)

run_process("sleep 3s")
logger.warning(f"run {len(cmds)} seed-ablation tasks for roberta_{task_name}_{tuning_type}")

# run_process(cmds[0])  # debug
pool = Pool(processes=1)
pool.map(run_process, cmds)
