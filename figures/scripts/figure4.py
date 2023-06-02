from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl


ticklabelpad = mpl.rcParams['xtick.major.pad']

###################################################
# 5 seed results
###################################################
data_results = {
    "RTE":
        {
            'FedFT': 1.0,
            'FedAP': 103.8,
            'FedLR': 140.1,
            'FedPF': 12.1,
            'FedBF': 190.0,
        },
    "MNLI":
        {
            'FedFT': 1.0,
            'FedAP': 41.9,
            'FedLR': 140.4,
            'FedPF': 12.1,
            'FedBF': 190.0,
        },
    "MRPC":
        {
            'FedFT': 1.0,
            'FedAP': 103.8,
            'FedLR': 140.6,
            'FedPF': 12.1,
            'FedBF': 190.0,
        },
    "SST-2":
        {
            'FedFT': 1.0,
            'FedAP': 42.0,
            'FedLR': 140.5,
            'FedPF': 12.1,
            'FedBF': 190.0,
        },
    "QNLI":
        {
            'FedFT': 1.0,
            'FedAP': 103.8,
            'FedLR': 140.5,
            'FedPF': 12.1,
            'FedBF': 190.0,
        },
    "QQP":
        {
            'FedFT': 1.0,
            'FedAP': 42.0,
            'FedLR': 140.5,
            'FedPF': 12.1,
            'FedBF': 190.0,
        },
}


different_colors = {
    'FedBF': (255/255.0, 67/255.0, 67/255.0),
    'FedPF': (255/255.0, 169/255.0, 13/255.0),
    'FedLR': (121/255.0, 195/255.0, 197/255.0),
    'FedAP': (205/255.0, 39/255.0, 21/255.0),
    'FedFT': (0/255.0, 48/255.0, 225/255.0),
}

data_names = ['MRPC', 'SST-2', 'QNLI', 'QQP']
methods = list(different_colors.keys())

font_size = 80

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(55, 30*2))

fig.subplots_adjust(hspace=0.5)

methods_results_dict = {}

for index, m in enumerate(methods):
    this_data = []
    for d in data_names:
        this_data.append(data_results[d][m])
    
    methods_results_dict[m] = this_data
    

compact_hyper = 0.18

for i in range(2):
    this_axes = axes[i]
    
    for index, m in enumerate(methods):
        width = 0.03
        
        this_x = list(range(len(data_names)))
        for l in range(len(this_x)):
            this_x[l] = compact_hyper*(this_x[l])
            
        this_x = [x + (width)*index for x in this_x]
            
        bars = this_axes.barh(this_x, methods_results_dict[m], fc=different_colors[m], zorder=10, label=m, edgecolor='black', tick_label = data_names, height=width)

        this_axes.bar_label(bars, fontsize=font_size - 10, padding=10)
        
    plt.setp(this_axes.spines.values(), linewidth=6)
    this_axes.spines['top'].set(linewidth=1, color="lightgray")
    this_axes.spines['right'].set(linewidth=1, color="lightgray")

    this_axes.tick_params(width=6, length=6)
        
    this_axes.legend(fontsize=font_size,loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=True, shadow=True, ncol=5)

    this_axes.tick_params(labelsize=font_size)
    this_axes.grid(axis='x', linestyle="-", color="lightgray", linewidth=3.5)  

    xx = list(range(len(this_x)))
    xx = [(compact_hyper*x + 1.5*width) for x in xx]
    plt.sca(this_axes)
    plt.yticks([index  for index in xx],  data_names, rotation='horizontal')
    

fig.savefig("resource_cost.pdf")