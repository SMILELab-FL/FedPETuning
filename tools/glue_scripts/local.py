import random
import json
import pickle as pkl
from collections import Counter, defaultdict


def get_dist(data, idx_list):
    labels = []
    for idx in idx_list:
        labels.append(data[idx].label)
    labels_dist = Counter(labels)
    return labels_dist


def check_dist(data_1, data_2, idx_list_1, idx_list_2, flag):
    label_dist_1 = get_dist(data_1, idx_list_1)
    label_dist_2 = get_dist(data_2, idx_list_2)
    for keys in label_dist_2:
        if not flag[keys] and label_dist_2[keys] != label_dist_1[keys]:
            print(keys)
            print(flag)
            print(label_dist_1[keys])
            print(label_dist_2[keys])
            raise


def build_vaild(data, data_partition, method):
    client_num = int(method.split("_")[0].split("=")[1])

    all_valid_labels_dist = get_dist(data["valid"], [i for i in range(len(data["valid"]))])
    all_test_labels_dist = get_dist(data["test"], [i for i in range(len(data["test"]))])
    print(all_valid_labels_dist, all_test_labels_dist)

    all_valid_labels_list = {key: [i for i, exapmle in enumerate(data["valid"]) if exapmle.label == key] for key in
                             all_valid_labels_dist}
    all_test_labels_list = {key: [i for i, exapmle in enumerate(data["test"]) if exapmle.label == key] for key in
                            all_test_labels_dist}

    for i in range(client_num):
        train_idx_list = data_partition[method]["train"][i]
        train_labels_dist = get_dist(data["train"], train_idx_list)

        vaild_idx_list, test_idx_list = [], []
        valid_flag, test_flag = {}, {}
        for key, value in train_labels_dist.items():
            if key in all_valid_labels_dist.keys():
                if all_valid_labels_dist[key] < value:
                    valid_value = all_valid_labels_dist[key]
                    valid_flag[key] = True
                else:
                    valid_value = value
                    valid_flag[key] = False
                vaild_idx_list.extend(random.sample(all_valid_labels_list[key], valid_value))

            if key in all_test_labels_dist.keys():
                if all_test_labels_dist[key] < value:
                    test_value = all_test_labels_dist[key]
                    test_flag[key] = True
                else:
                    test_value = value
                    test_flag[key] = False
                test_idx_list.extend(random.sample(all_test_labels_list[key], test_value))

        check_dist(data["train"], data["valid"], train_idx_list, vaild_idx_list, valid_flag)
        check_dist(data["train"], data["test"], train_idx_list, test_idx_list, test_flag)

        data_partition[method]["valid"][i] = vaild_idx_list
        data_partition[method]["test"][i] = test_idx_list
    return data_partition