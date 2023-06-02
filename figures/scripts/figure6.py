
import pickle
import itertools as it
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

def pickle_read(path, read_format="rb"):
    with open(path, read_format) as file:
        obj = pickle.load(file)
    return obj

def plot_pairwise_distance(data_names):
    # client_nums = [100, 10]
    client_nums = [100]
    
    alphas = [0.1, 1.0, 10.0]
    
    fig,axes = plt.subplots(len(data_names), len(client_nums)*len(alphas), figsize=(20, 20))
    
    for d_index, data_name in enumerate(data_names):
        data_partition_path = f"../../../../data/fedglue/{data_name}_partition.pkl"
        data_path = f"../../../../data/fedglue/{data_name}_data.pkl"
        
        data = pickle_read(data_path)
        data_p = pickle_read(data_partition_path)
        client_assignment = [example.label for example in data["train"]]
        cluster_num = len(set(client_assignment))
        
        alpha_color_mapping = {0.1: (9/255.0, 147/255.0, 150/255.0, 1.0),
                            1.0: (238/255.0, 155/255.0, 0/255.0, 1.0),
                            10.0: (174/255.0, 32/255.0, 18/255.0, 1.0)}
        fig.subplots_adjust(hspace=0.35)
        
        axe_num = 0
        for ele in it.product(client_nums, alphas):
            client_num, alpha = ele
            p_method = f"clients={client_num}_alpha={alpha}"

            lable_mapping = data_p[p_method]["attribute"]["lable_mapping"]
            client_data_distribution = []
            for client_idx in range(client_num):
                probability_array = np.zeros(cluster_num)

                client_data_list = data_p[p_method]["train"][client_idx]
                client_label_list = [data["train"][example_id].label for example_id in client_data_list]
                single_client_data = np.array([lable_mapping[label] for label in client_label_list])
                unique, counts = np.unique(single_client_data, return_counts=True)

                for key, value in dict(zip(unique, counts)).items():
                    probability_array[key] = value
                client_data_distribution.append(probability_array)

            heat_map_data = np.zeros((client_num, client_num))
            pdf_data = []
            for i in range(client_num):
                for j in range(i+1, client_num):
                    pairwise_dist = distance.jensenshannon(client_data_distribution[i], client_data_distribution[j])
                    pdf_data.append(pairwise_dist)

            col = axe_num

            line_kws = {'linewidth': 3}
            
            sns.set_palette(None)
            if len(data_names) == 1:
                axesSub = sns.histplot(pdf_data, ax=axes[col], kde=True, stat="density", line_kws=line_kws, color=alpha_color_mapping[alpha])
            else:    
                axesSub = sns.histplot(pdf_data, ax=axes[d_index][col], kde=True, stat="density", line_kws=line_kws, color=alpha_color_mapping[alpha])
                
            
            plt.setp(axesSub.spines.values(), linewidth=3)
            axesSub.spines['top'].set(linewidth=1, color="lightgray")
            axesSub.spines['right'].set(linewidth=1, color="lightgray")

            axesSub.tick_params(width=3, length=3)

            font_size = 20
            axesSub.tick_params(labelsize=font_size - 5)
            if col==0:
                axesSub.set_ylabel(ylabel="Density", fontsize=font_size, labelpad=30)
            else:
                axesSub.set_ylabel(ylabel="", fontsize=font_size)
                
            axesSub.set_xlabel(xlabel="Distance", fontsize=font_size)

            axesSub.set_title(f"{str.upper(data_name)} α="+ str(alpha), fontdict={'fontsize': font_size + 5, 'horizontalalignment': 'center'})
    
            axe_num += 1
            
            # label = "client-level pairwise JS distance"
            
        fig.savefig("distribution_exp.pdf")

data_names = ["mrpc", "sst-2", "qnli", "qqp"]

plot_pairwise_distance(data_names)