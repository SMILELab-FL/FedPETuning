from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pickle
from math import log
from matplotlib.ticker import FormatStrFormatter
import random
import numpy as np

ticklabelpad = mpl.rcParams['xtick.major.pad']


# seed=42
parameter_num = {
    "RTE":
        {
            'FedLR': 0.9,
            'FedPF': 10.3,
            'FedAP': 3.0,
            'FedBF': 0.7,
            'FedFT': 124.7,
        },
    "MNLI":
        {
            'FedLR': 0.9,
            'FedPF': 10.3,
            'FedAP': 3.0,
            'FedBF': 0.7,
            'FedFT': 124.7,
        },
    # "MRPC":
    #     {
    #         'FedLR': 0.9,
    #         'FedPF': 10.3,
    #         'FedAP': 1.201,
    #         'FedBF': 0.7,
    #         'FedFT': 124.7,
    #     },
    # "SST-2":
    #     {
    #         'FedLR': 0.9,
    #         'FedPF': 10.3,
    #         'FedAP': 3.0,
    #         'FedBF': 0.7,
    #         'FedFT': 124.7,
    #     },
    # "QNLI":
    #     {
    #         'FedLR': 0.9,
    #         'FedPF': 10.3,
    #         'FedAP': 1.201,
    #         'FedBF': 0.7,
    #         'FedFT': 124.7,
    #     },
    # "QQP":
    #     {
    #         'FedLR': 0.9,
    #         'FedPF': 10.3,
    #         'FedAP': 3.0,
    #         'FedBF': 0.7,
    #         'FedFT': 124.7,
    #     },
}

log_file = {
    "RTE": 
        {
            "FedLR": ["./main_logs/lora_s42_roberta_rte.eval.log", "./main_logs/lora_s2_roberta_rte.eval.log", "./main_logs/lora_s3_roberta_rte.eval.log", "./main_logs/lora_s4_roberta_rte.eval.log", "./main_logs/lora_s5_roberta_rte.eval.log"],
            "FedPF": ["./main_logs/prefix_s42_roberta_rte.eval.log", "./main_logs/prefix_s2_roberta_rte.eval.log", "./main_logs/prefix_s3_roberta_rte.eval.log", "./main_logs/prefix_s4_roberta_rte.eval.log", "./main_logs/prefix_s5_roberta_rte.eval.log"],
            "FedAP": ["./main_logs/adapter_s42_roberta_rte.eval.log", "./main_logs/adapter_s2_roberta_rte.eval.log", "./main_logs/adapter_s3_roberta_rte.eval.log", "./main_logs/adapter_s4_roberta_rte.eval.log", "./main_logs/adapter_s5_roberta_rte.eval.log"],
            "FedBF": ["./main_logs/bitfit_s42_roberta_rte.eval.log", "./main_logs/bitfit_s2_roberta_rte.eval.log", "./main_logs/bitfit_s3_roberta_rte.eval.log", "./main_logs/bitfit_s4_roberta_rte.eval.log", "./main_logs/bitfit_s5_roberta_rte.eval.log"],
            "FedFT": ["./main_logs/fine-tuning_s42_roberta_rte.eval.log", "./main_logs/fine-tuning_s2_roberta_rte.eval.log", "./main_logs/fine-tuning_s3_roberta_rte.eval.log", "./main_logs/fine-tuning_s4_roberta_rte.eval.log", "./main_logs/fine-tuning_s5_roberta_rte.eval.log"]
        },
    # "MNLI": 
    #     {
    #         "FedLR": ["./main_logs/lora_s42_roberta_mnli.eval.log", "./main_logs/lora_s2_roberta_mnli.eval.log", "./main_logs/lora_s3_roberta_mnli.eval.log", "./main_logs/lora_s4_roberta_mnli.eval.log", "./main_logs/lora_s5_roberta_mnli.eval.log"],
    #         "FedPF": ["./main_logs/prefix_s42_roberta_mnli.eval.log", "./main_logs/prefix_s2_roberta_mnli.eval.log", "./main_logs/prefix_s3_roberta_mnli.eval.log", "./main_logs/prefix_s4_roberta_mnli.eval.log", "./main_logs/prefix_s5_roberta_mnli.eval.log"],
    #         "FedAP": ["./main_logs/adapter_s42_roberta_mnli.eval.log", "./main_logs/adapter_s2_roberta_mnli.eval.log", "./main_logs/adapter_s3_roberta_mnli.eval.log", "./main_logs/adapter_s4_roberta_mnli.eval.log"],
    #         "FedBF": ["./main_logs/bitfit_s42_roberta_mnli.eval.log", "./main_logs/bitfit_s2_roberta_mnli.eval.log", "./main_logs/bitfit_s3_roberta_mnli.eval.log", "./main_logs/bitfit_s4_roberta_mnli.eval.log", "./main_logs/bitfit_s5_roberta_mnli.eval.log"],
    #         "FedFT": ["./main_logs/fine-tuning_s42_roberta_mnli.eval.log", "./main_logs/fine-tuning_s2_roberta_mnli.eval.log", "./main_logs/fine-tuning_s3_roberta_mnli.eval.log", "./main_logs/fine-tuning_s4_roberta_mnli.eval.log"]
    #     },
    "MRPC": 
        {
            "FedLR": ["./main_logs/lora_s42_roberta_mrpc.eval.log", "./main_logs/lora_s2_roberta_mrpc.eval.log", "./main_logs/lora_s3_roberta_mrpc.eval.log", "./main_logs/lora_s4_roberta_mrpc.eval.log", "./main_logs/lora_s5_roberta_mrpc.eval.log"],
            "FedPF": ["./main_logs/prefix_s42_roberta_mrpc.eval.log", "./main_logs/prefix_s2_roberta_mrpc.eval.log", "./main_logs/prefix_s3_roberta_mrpc.eval.log", "./main_logs/prefix_s4_roberta_mrpc.eval.log", "./main_logs/prefix_s5_roberta_mrpc.eval.log"],
            "FedAP": ["./main_logs/adapter_s42_roberta_mrpc.eval.log", "./main_logs/adapter_s2_roberta_mrpc.eval.log", "./main_logs/adapter_s3_roberta_mrpc.eval.log", "./main_logs/adapter_s4_roberta_mrpc.eval.log", "./main_logs/adapter_s5_roberta_mrpc.eval.log"],
            "FedBF": ["./main_logs/bitfit_s42_roberta_mrpc.eval.log", "./main_logs/bitfit_s2_roberta_mrpc.eval.log", "./main_logs/bitfit_s3_roberta_mrpc.eval.log", "./main_logs/bitfit_s4_roberta_mrpc.eval.log", "./main_logs/bitfit_s5_roberta_mrpc.eval.log"],
            "FedFT": ["./main_logs/fine-tuning_s42_roberta_mrpc.eval.log", "./main_logs/fine-tuning_s2_roberta_mrpc.eval.log", "./main_logs/fine-tuning_s3_roberta_mrpc.eval.log", "./main_logs/fine-tuning_s4_roberta_mrpc.eval.log", "./main_logs/fine-tuning_s5_roberta_mrpc.eval.log"]
        },
#     "SST-2": 
#         {
#             "FedLR": "./main_logs/lora_s42_roberta_sst-2.eval.log",
#             "FedPF": "./main_logs/prefix_s42_roberta_sst-2.eval.log",
#             "FedAP": "./main_logs/adapter_s42_roberta_sst-2.eval.log",
#             "FedBF": "./main_logs/bitfit_s42_roberta_sst-2.eval.log",
#             "FedFT": "./main_logs/fine-tuning_s42_roberta_sst-2.eval.log",
#         },
#     "QNLI": 
#         {
#             "FedLR": "./main_logs/lora_s42_roberta_qnli.eval.log",
#             "FedPF": "./main_logs/prefix_s42_roberta_qnli.eval.log",
#             "FedAP": "./main_logs/adapter_s42_roberta_qnli.eval.log",
#             "FedBF": "./main_logs/bitfit_s42_roberta_qnli.eval.log",
#             "FedFT": "./main_logs/fine-tuning_s42_roberta_qnli.eval.log",
#         },
#     "QQP": 
#         {
#             "FedLR": "./main_logs/lora_s42_roberta_qqp.eval.log",
#             "FedPF": "./main_logs/prefix_s42_roberta_qqp.eval.log",
#             "FedAP": "./main_logs/adapter_s42_roberta_qqp.eval.log",
#             "FedBF": "./main_logs/bitfit_s42_roberta_qqp.eval.log",
#             "FedFT": "./main_logs/fine-tuning_s42_roberta_qqp.eval.log",
#         },
}

data_best_finetuning = {
    "RTE": 73.0,
    "MRPC": 90.9,
    "SST-2": 92.1,
    "QNLI": 90.8,
    "QQP": 91.1,
    "MNLI": 86.0,
}

different_marks = {
    'FedLR': '^',
    'FedPF': 'D',
    'FedAP': 'x',
    'FedBF': 'o',
    'FedFT': 's',
}

different_line_types = {
    'FedLR': '-',
    'FedPF': '--',
    'FedAP': '-.',
    'FedBF': ':',
    'FedFT': '-',
}

different_colors = {
    'FedLR': 'darkorange',
    'FedPF': 'purple',
    'FedAP': 'g',
    'FedBF': 'r',
    'FedFT': 'b',
}

different_colors = {
    'FedLR': (9/255.0, 147/255.0, 150/255.0),
    'FedPF': (238/255.0, 155/255.0, 0/255.0),
    'FedAP': (174/255.0, 32/255.0, 18/255.0),
    'FedBF': 'r',
    'FedFT': (0/255.0, 48/255.0, 225/255.0),
}

##########################################################     
def read_acc_from_log(file_path, metric_name):
    with open(file_path, "rb") as this_file:
        log_data = pickle.load(this_file)['logs']
    
    this_log_acc = []
    for log_index, log_sample in enumerate(log_data):
        this_acc = float(log_sample['round_' + str(log_index + 1)][metric_name])
        this_log_acc.append(this_acc)
    
    return this_log_acc

def accumulate_max(in_list):
    out_list = []
    current_max = -99999999
    for this_value in in_list:
        current_max = max(current_max, this_value)
        out_list.append(current_max)
    return out_list
    
def read_log(data_name, tuning_type):
    if data_name == "MRPC":
        metric_name = "f1"
    else:
        metric_name = "acc"
            
    # read file
    log_path = log_file[data_name][tuning_type]
    
    if isinstance(log_path, list):
        log_acc_list = []
        for this_log in log_path:
            log_acc_list.append(read_acc_from_log(this_log, metric_name))
        
        # use accumulate max
        log_acc_list = [accumulate_max(this_l) for this_l in log_acc_list]
        
        this_log_acc = []
        this_log_collection = []
        for this_index, _ in enumerate(log_acc_list[0]):
            this_element = 0.0
            this_collection = []
            for temp_acc_list in log_acc_list:
                this_element += temp_acc_list[this_index]
                this_collection.append(temp_acc_list[this_index])
                
            this_log_acc.append(this_element/len(log_acc_list))
            this_log_collection.append(this_collection)
    else:
        this_log_acc = read_acc_from_log(log_path, metric_name)
        this_log_collection = None
    
    print(f"Read {data_name} {tuning_type} Round {len(this_log_acc)} log.")
    
    # select data
    x_communication = []
    y_acc = []
    y_max, y_min = [], []
    
    current_best_acc = -1.0
    for acc_index, this_acc in enumerate(this_log_acc):
        if this_acc > current_best_acc:
            current_best_acc = this_acc
                
            y_acc.append(current_best_acc)
            x_communication.append(parameter_num[data_name][tuning_type] * (acc_index + 1) * 4 * 10)
            if this_log_collection is not None:
                this_std = np.std(this_log_collection[acc_index], ddof=0)
                y_max.append(y_acc[-1] + this_std)
                y_min.append(y_acc[-1] - this_std)
    
    return x_communication, y_acc, metric_name, (y_max, y_min)

##########################################################
    
font_size = 90

my_nrows = 1
my_ncols = len(log_file.keys()) // my_nrows

fig, axes = plt.subplots(nrows=my_nrows ,ncols=my_ncols, figsize=(80, 30))
fig.subplots_adjust(hspace=0.45)

# iter by task_name
for index, data_name in enumerate(log_file.keys()):
    if my_nrows > 1:
        this_axes = axes[index // my_ncols][index % my_ncols]
    elif my_ncols == 1:
        this_axes = axes
    else:
        this_axes = axes[index]

    # iter by tuning type
    fine_tuning_best = -1.0
    for i_index, tuning_type in enumerate(log_file[data_name].keys()):
        
        # read results from file
        x_communication, y_acc, metric_name, (y_max, y_min) = read_log(data_name, tuning_type)

        if tuning_type == 'FedFT':
            fine_tuning_best = y_acc[-1]
            
        this_axes.plot(x_communication, y_acc, markersize=50, label=tuning_type, linewidth=13, color=different_colors[tuning_type])
        if len(y_max) > 0:
            this_axes.fill_between(x_communication, y_min, y_max, alpha=0.1, color=different_colors[tuning_type])
        
    plt.setp(this_axes.spines.values(), linewidth=6)
    this_axes.spines['top'].set(linewidth=1, color="lightgray")
    this_axes.spines['right'].set(linewidth=1, color="lightgray")
    this_axes.spines['top'].set_visible(False)
    this_axes.spines['right'].set_visible(False)
    
    this_axes.tick_params(width=6, length=15)
    this_axes.tick_params(labelsize=font_size, pad=15)
    
    this_axes.set_xlabel("Communication Budget / MB", fontsize=font_size+10, labelpad=10)
    if metric_name == "f1":
        y_metric = "F1 Score"
    else:
        y_metric = "Accuracy (%)"
    this_axes.set_ylabel(y_metric, fontsize=font_size+10, labelpad=50)
    
    this_axes.set_title(data_name, fontdict={'fontsize': font_size, 'horizontalalignment': 'center'})
    this_axes.legend(fontsize=font_size, framealpha=0.9)
    
    this_axes.grid(axis='y', linestyle="-", color="lightgray", linewidth=0.5)  
    this_axes.axhline(y=data_best_finetuning[data_name]*0.95, c='gray', ls='--', lw=5)  # 垂直于y轴的参考线
    

    this_axes.set_xscale('log')


fig.align_labels()
fig.savefig("comm_exp.png")