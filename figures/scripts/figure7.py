from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl


ticklabelpad = mpl.rcParams['xtick.major.pad']


# seed=42
epoch_results = {
    # "RTE":
    #     {
    #         'FedLR': ([67.4, 69.7, 65.9], [62.1, 68.2, 65.0], [72.6, 72.2, 66.8]), # mean, min, max
    #         'FedPF': ([58.6, 59.6, 57.8], [55.6, 57.4, 52.7], [61.4, 61.7, 63.2]),
    #         'FedAP': ([69.4, 70.3, 69.5], [65.0, 65.0, 66.1], [71.1, 74.4, 72.2]),
    #         'FedBF': ([61.4, 64.2, 64.3], [59.6, 61.4, 61.0], [63.9, 66.8, 67.5]),
    #         'FedFT': ([70.3, 70.4, 72.7], [68.6, 65.3, 69.7], [71.8, 75.5, 76.2]),
    #     },
    # "RTE":
    #     {
    #         'FedLR': ([67.4, 4.2], [69.7, 1.7], [71.4, 2.6]), # mean, min, max
    #         'FedPF': ([58.6, 2.0], [59.6, 1.5], [61.4, 1.9]),
    #         'FedAP': ([69.4, 2.3], [70.3, 3.1], [69.5, 2.2]),
    #         'FedBF': ([61.4, 1.7], [64.2, 2.1], [64.3, 2.5]),
    #         'FedFT': ([70.3, 1.1], [70.4, 3.6], [72.8, 2.4]),
    #     },
    # "MNLI":
    #     {
    #         'FedLR': ([84.9, 0.3], [84.8, 0.2], [85.2, 0.2]),
    #         'FedPF': ([82.2, 0.3], [83.3, 0.3], [83.8, 0.3]),
    #         'FedAP': ([86.0, 0.3], [86.3, 0.1], [86.5, 0.2]),
    #         'FedBF': ([81.7, 0.2], [82.4, 0.2], [82.5, 0.1]),
    #         'FedFT': ([86.4, 0.2], [86.4, 0.2], [86.5, 0.1]),
    #     },
    "MRPC":
        {
            'FedLR': ([84.5, 4.5], [88.6, 3.1], [89.3, 1.2]),
            'FedPF': ([86.8, 1.0], [88.2, 1.2], [88.0, 0.8]),
            'FedAP': ([89.1, 1.2], [89.5, 0.6], [89.7, 0.9]),
            'FedBF': ([84.6, 2.7], [86.7, 1.1], [86.7, 1.1]),
            'FedFT': ([90.8, 0.3], [90.2, 0.8], [90.9, 0.6]),
        },
    # "CoLA":
    #     {
    #         'lora': [59.0, 60.5, 55.4],
    #         'prefix': [57.4, 60.1, 57.5],
    #         'adapter': [60.8, 62.2, 61.4],
    #         'BitFit': [51.7, 46.6, 41.9],
    #         'fine-tuning': [60.6, 58.7, 61.5],
    #     },
    "SST-2":
        {
            'FedLR': ([93.6, 0.5], [93.6, 0.3], [93.9, 0.3]),
            'FedPF': ([93.0, 0.6], [93.0, 0.4], [93.6, 0.1]),
            'FedAP': ([93.3, 0.6], [93.9, 0.3], [94.1, 0.3]),
            'FedBF': ([92.5, 0.7], [92.8, 0.5], [93.0, 0.3]),
            'FedFT': ([93.9, 0.6], [94.4, 0.2], [94.3, 0.3]),
        },
    "QNLI":
        {
            'FedLR': ([90.9, 0.3], [91.1, 1.0], [91.1, 0.9]),
            'FedPF': ([87.6, 0.5], [89.5, 0.4], [90.2, 0.2]),
            'FedAP': ([90.9, 0.4], [91.0, 1.2], [91.8, 0.2]),
            'FedBF': ([87.2, 0.5], [88.3, 1.4], [89.6, 0.4]),
            'FedFT': ([91.0, 0.4], [91.6, 0.2], [91.6, 0.2]),
        },
    "QQP":
        {
            'FedLR': ([87.4, 0.3], [88.2, 0.00], [88.5, 0.1]),
            'FedPF': ([85.7, 0.3], [86.6, 0.3], [86.9, 0.3]),
            'FedAP': ([88.4, 0.2], [89.1, 0.1], [89.3, 0.2]),
            'FedBF': ([84.5, 0.5], [85.0, 0.3], [85.4, 0.3]),
            'FedFT': ([89.5, 0.1], [89.8, 0.1], [89.9, 0.3]),
        },
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
    # 'FedFT': 'b',
}

different_marks = {
    'FedLR': '^',
    'FedPF': 'D',
    'FedAP': 'x',
    'FedBF': 'o',
    'FedFT': 's',
}

font_size = 80

my_nrows = 2
my_ncols = len(epoch_results.keys()) // my_nrows

fig, axes = plt.subplots(nrows=my_nrows ,ncols=my_ncols, figsize=(60, 40))
fig.subplots_adjust(hspace=0.75)

for index, data_name in enumerate(epoch_results.keys()):
    if my_nrows > 1:
        this_axes = axes[index // my_ncols][index % my_ncols]
    elif my_ncols == 1:
        this_axes = axes
    else:
        this_axes = axes[index]

    alphas = ['e=1', 'e=2', 'e=3']
            
    for a_index, tuning_type in enumerate(epoch_results[data_name].keys()):
        width = 0.25
        
        if isinstance(epoch_results[data_name][tuning_type], tuple):
            this_axes.plot(range(1,4), [xxx[0] for xxx in epoch_results[data_name][tuning_type]], marker=different_marks[tuning_type], markersize=50, label=tuning_type, linewidth=10, color=different_colors[tuning_type])
            this_axes.fill_between(range(1,4), [xxx[0] - xxx[1] for xxx in epoch_results[data_name][tuning_type]], [xxx[0] + xxx[1] for xxx in epoch_results[data_name][tuning_type]], alpha=0.1, color=different_colors[tuning_type])
        else:
            this_axes.plot(range(1,4), epoch_results[data_name][tuning_type], marker=different_marks[tuning_type], markersize=50, label=tuning_type, linewidth=10, color=different_colors[tuning_type])
            
    plt.setp(this_axes.spines.values(), linewidth=6)
    this_axes.spines['top'].set(linewidth=1, color="lightgray")
    this_axes.spines['right'].set(linewidth=1, color="lightgray")
    
    this_axes.tick_params(width=6, length=6)
    
    this_axes.tick_params(labelsize=font_size, pad=15)
    
    if index % my_ncols == 0:
        this_axes.set_ylabel("Accuracy (%)", fontsize=font_size, labelpad=30)
    
    this_axes.set_xlabel("Epoch", fontsize=font_size, labelpad=0)
    
    this_axes.legend(fontsize=font_size, framealpha=0.9)
    this_axes.legend(fontsize=60,loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)
    
        
    this_axes.set_title(data_name, fontdict={'fontsize': font_size, 'horizontalalignment': 'center'})
    
    this_axes.grid(linestyle="-", color="lightgray", linewidth=1)  
    
    plt.sca(this_axes)
    plt.xticks([1, 2, 3],  list(['1', '2', '3']), rotation='horizontal')
    
fig.savefig("epoch_exp.png")