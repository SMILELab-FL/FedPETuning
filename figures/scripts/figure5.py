from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl


ticklabelpad = mpl.rcParams['xtick.major.pad']

###################################################
# 5 seed results
###################################################
std_deviations = {
    "RTE":
        {
            'FedFT': [2.7, 1.1, 0.7],
            'FedAP': [3.8, 2.2, 1.0],
            'FedLR': [2.5, 4.2, 1.2],
            'FedPF': [5.4, 2.0, 3.0],
            'FedBF': [2.3, 1.7, 2.2],
        },
    "MNLI":
        {
            'FedFT': [0.6, 0.2, 0.3],
            'FedAP': [1.1, 0.3, 0.2],
            'FedLR': [0.3, 0.3, 0.2],
            'FedPF': [0.8, 0.3, 0.2],
            'FedBF': [0.5, 0.2, 0.2],
        },
    # "MRPC": 
    #     {
    #         'FedFT': [3.8, 0.3, 0.2],
    #         'FedAP': [0.4, 1.2, 0.8],
    #         'FedLR': [0.5, 4.5, 3.0],
    #         'FedPF': [0.0, 1.0, 1.5],
    #         'FedBF': [0.5, 2.7, 1.0],
    #     },
    # "SST-2":
    #     {
    #         'FedFT': [0.1, 0.6, 0.5],
    #         'FedAP': [0.4, 0.6, 0.4],
    #         'FedLR': [0.7, 0.5, 0.1],
    #         'FedPF': [1.7, 0.6, 0.4],
    #         'FedBF': [0.5, 0.7, 0.5],
    #     },
    "QNLI":
        {
            'FedFT': [2.0, 0.4, 0.2],
            'FedAP': [2.3, 0.4, 0.6],
            'FedLR': [1.2, 0.3, 0.3],
            'FedPF': [1.7, 0.5, 0.2],
            'FedBF': [4.7, 0.5, 0.6],
        },
    "QQP":
        {
            'FedFT': [4.6, 0.8, 0.2],
            'FedAP': [1.7, 0.2, 0.1],
            'FedLR': [1.7, 0.3, 0.2],
            'FedPF': [1.2, 0.3, 1.1],
            'FedBF': [0.9, 0.5, 0.3],
        },
    }

alpha_results = {
    "RTE":
        {
            'FedFT': [68.4, 70.3, 71.7],
            'FedAP': [61.5, 69.4, 70.9],
            'FedLR': [65.2, 67.4, 68.5],
            'FedPF': [53.9, 58.6, 59.4],
            'FedBF': [59.6, 61.4, 63.2],
        },
    "MNLI":
        {
            'FedFT': [84.6, 86.4, 87.0],
            'FedAP': [82.7, 86.0, 86.8],
            'FedLR': [82.8, 84.9, 85.6],
            'FedPF': [78.4, 82.2, 83.3],
            'FedBF': [79.7, 81.7, 82.5],
        },
    # "MRPC":
    #     {
    #         'FedFT': [84.8, 90.8, 91.0],
    #         'FedAP': [81.3, 89.1, 89.5],
    #         'FedLR': [82.0, 84.5, 86.2],
    #         'FedPF': [81.2, 86.8, 88.4],
    #         'FedBF': [81.5, 84.6, 84.9],
    #     },
    # "SST-2":
    #     {
    #         'FedFT': [92.4, 93.9, 94.5],
    #         'FedAP': [92.3, 93.3, 94.0],
    #         'FedLR': [92.0, 93.6, 93.7],
    #         'FedPF': [88.5, 93.0, 93.1],
    #         'FedBF': [91.2, 92.5, 93.5],
    #     },
    "QNLI":
        {
            'FedFT': [84.0, 91.0, 92.3],
            'FedAP': [52.5, 90.9, 91.6],
            'FedLR': [78.8, 90.8, 91.7],
            'FedPF': [52.3, 87.6, 89.7],
            'FedBF': [79.6, 87.2, 88.3],
        },
    "QQP":
        {
            'FedFT': [79.6, 89.5, 90.3],
            'FedAP': [64.2, 88.4, 89.3],
            'FedLR': [80.8, 87.4, 88.2],
            'FedPF': [63.8, 85.7, 87.3],
            'FedBF': [81.5, 84.5, 85.3],
        },
    }

font_size = 80

my_nrows = 2
my_ncols = len(alpha_results.keys()) // my_nrows

fig, axes = plt.subplots(nrows=my_nrows ,ncols=my_ncols, figsize=(55, 35))
fig.subplots_adjust(hspace=0.5)

for index, data_name in enumerate(alpha_results.keys()):
    if my_nrows > 1:
        this_axes = axes[index // my_ncols][index % my_ncols]
    else:
        this_axes = axes[index]
    colors = ['goldenrod', 'red', 'dodgerblue']
    alpha = 1.0
    colors = [(121/255.0, 195/255.0, 197/255.0),(255/255.0, 169/255.0, 13/255.0), (255/255.0, 169/255.0, 13/255.0),]
    
    tuning_types = alpha_results[data_name].keys()
    alphas = ['α=0.1', 'α=1.0', 'α=10.0']
    
    alphas_results_dict = {}
    this_axes_min_y = 1000
    for a_index, alp in enumerate(alphas):
        this_alp_data = []
        for t in tuning_types:
            this_alp_data.append(alpha_results[data_name][t][a_index])
        alphas_results_dict[alp] = this_alp_data
        
        this_axes_min_y = min(this_axes_min_y, min(this_alp_data))
        
    deviation_results_dict = {}
    for a_index, alp in enumerate(alphas):
        if std_deviations[data_name] is None:
            this_alp_data = None
        else:
            this_alp_data = []
            for t in tuning_types:
                this_alp_data.append(std_deviations[data_name][t][a_index])
        deviation_results_dict[alp] = this_alp_data
    
    baseline = alphas_results_dict["α=1.0"]
    del alphas_results_dict["α=1.0"]
    
    for a_index, alp in enumerate(alphas_results_dict.keys()):
        width = 0.25
        
        this_x = list(range(len(tuning_types)))
        this_x = [x + (width)*a_index for x in this_x]
        
        ebar_style = {'elinewidth': 4, 'capthick': 4, 'capsize': 16} 
        this_axes.bar(this_x, [100*(alphas_results_dict[alp][i_b] - b)/b for i_b, b in enumerate(baseline)], fc=colors[a_index], width=width, zorder=10, label=alp, edgecolor='black', tick_label = list(tuning_types))

    plt.setp(this_axes.spines.values(), linewidth=6)
    this_axes.spines['top'].set(linewidth=1, color="lightgray")
    this_axes.spines['right'].set(linewidth=1, color="lightgray")
    
    this_axes.tick_params(width=6, length=6)
        
    this_axes.legend(fontsize=70,loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=5)
    
    if index % my_ncols == 0:
        this_axes.set_ylabel("Relative Accuracy (%)", fontsize=font_size, labelpad=30)
    

    this_axes.tick_params(labelsize=font_size)
    this_axes.set_title(data_name, fontdict={'fontsize': font_size, 'horizontalalignment': 'center'})
    
    this_axes.grid(axis='y', linestyle="-", color="lightgray", linewidth=3.5)  
    
    xx = list(range(len(this_x)))
    xx = [(x + 3*width) for x in xx]
    plt.sca(this_axes)
    plt.xticks([index - 0.5 for index in xx],  list(tuning_types), rotation='horizontal')
    

fig.savefig("relative_alpha_exp.png")