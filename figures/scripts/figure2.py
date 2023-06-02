from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl


ticklabelpad = mpl.rcParams['xtick.major.pad']

#### read data, metirc->method->(x, y)
data_dict = {'Rec.': {}, 'Pre.': {}, 'F1': {}}


for key in data_dict.keys():
    # x(bsz), y
    data_dict[key]['FedFT'] = [[], []]
    data_dict[key]['FedPF'] = [[], []]
    data_dict[key]['FedAP'] = [[], []]
    data_dict[key]['FedBF'] = [[], []]
    data_dict[key]['FedLR'] = [[], []]

method_map = {
    "fine": "FedFT",
    "prefix": "FedPF",
    "adapter": "FedAP",
    "bitfit": "FedBF",
    "lora": "FedLR",
}

with open("PET.txt", "r") as this_file:
    for line in this_file:
        line_split = line.split()
        
        # method name, bsz, pre, rec, f1
        items = [l.split(":")[-1] for l in line_split]
        for index in range(len(items)):
            if index < len(items) - 1:
                items[index] = items[index][:-1]
                
        # save data
        data_dict['Rec.'][method_map[items[0]]][0].append(eval(items[1]))
        data_dict['Rec.'][method_map[items[0]]][1].append(eval(items[3]))
        
        data_dict['Pre.'][method_map[items[0]]][0].append(eval(items[1]))
        data_dict['Pre.'][method_map[items[0]]][1].append(eval(items[2]))
        
        data_dict['F1'][method_map[items[0]]][0].append(eval(items[1]))
        data_dict['F1'][method_map[items[0]]][1].append(eval(items[4]))

print(data_dict['F1']['FedFT'])


# graph setting
different_colors = {
    'FedLR': 'darkorange',
    'FedPF': 'purple',
    'FedAP': 'g',
    'FedBF': 'r',
    'FedFT': 'b',
}

different_colors = {
    'FedLR': (69/255.0, 159/255.0, 161/255.0),
    'FedPF': (238/255.0, 155/255.0, 0/255.0),
    'FedAP': (174/255.0, 32/255.0, 18/255.0),
    'FedBF': 'r',
    'FedFT': (21/255.0, 71/255.0, 255/255.0),
    # 'FedFT': 'b',
}

different_marks = {
    'FedLR': '^',
    'FedPF': 'D',
    'FedAP': 'x',
    'FedBF': 'o',
    'FedFT': 's',
}

font_size = 25

my_nrows = 1
my_ncols = len(data_dict.keys()) // my_nrows

fig, axes = plt.subplots(nrows=my_nrows ,ncols=my_ncols, figsize=(20, 6))
fig.subplots_adjust(hspace=0.75)

for index, data_name in enumerate(data_dict.keys()):
    if my_nrows > 1:
        this_axes = axes[index // my_ncols][index % my_ncols]
    elif my_ncols == 1:
        this_axes = axes
    else:
        this_axes = axes[index]

    for a_index, tuning_type in enumerate(data_dict[data_name].keys()):
        width = 0.25
        
        this_x = [str(x) for x in data_dict[data_name][tuning_type][0]]
        this_x.reverse()
        
        this_y = data_dict[data_name][tuning_type][1]
        this_y.reverse()
        
        this_axes.plot(this_x, this_y, marker=different_marks[tuning_type], markersize=10, label=tuning_type, linewidth=2, color=different_colors[tuning_type])
            
  
    plt.setp(this_axes.spines.values(), linewidth=2)
    this_axes.spines['top'].set(linewidth=1, color="lightgray")
    this_axes.spines['right'].set(linewidth=1, color="lightgray")
    
    this_axes.tick_params(width=2, length=2)
    
    this_axes.tick_params(labelsize=font_size - 5, pad=2)
    
    this_axes.set_xlabel("Batch Size", fontsize=font_size, labelpad=2)
    
    this_axes.legend(fontsize=font_size - 5, framealpha=0.9)
    
    if data_name == "Rec.":
        this_title = "Recall"
    elif data_name == "Pre.":
        this_title = "Precision"
    else:
        this_title = "F1"

    this_axes.set_title(this_title, fontdict={'fontsize': font_size, 'horizontalalignment': 'center'})
    
    this_axes.set_ylim(0, 1)
    
    this_axes.grid(linestyle="-", color="lightgray", linewidth=0.5)  
    
fig.savefig("dlg.pdf")