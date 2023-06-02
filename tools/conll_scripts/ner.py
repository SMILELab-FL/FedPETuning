import os
import pickle
import argparse
from loguru import logger

from tools.conll_scripts.partition import get_partition_data
from utils import make_sure_dirs


def parser_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", default="conll", type=str,
        help="Task name")
    parser.add_argument("--output_dir", default="/workspace/data/fedner/", type=str,
        help="The output directory to save partition or raw data")
    parser.add_argument("--clients_num", default=100, type=int,
        help="All clients numbers")
    parser.add_argument("--alpha", default=1.0, type=float,
        help="The label skew degree.")
    parser.add_argument("--overwrite", default=0, type=int,
        help="overwrite")

    args = parser.parse_args()
    return args


def get_ner_examples(args):
    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    train_examples = data["train"]
    valid_examples = data["valid"]
    test_examples = data["test"]
    label_list = data["label_list"]
    output_mode = data["output_mode"]
    return train_examples, valid_examples, test_examples, label_list, output_mode


def convert_conll_to_device_pkl(args):

    logger.info("reading examples ...")

    with open(args.output_data_file, "rb") as file:
        data = pickle.load(file)
    train_examples, valid_examples, test_examples = data["train"], data["valid"], data["test"]
    output_mode, label_list = data["output_mode"], data["label_list"]

    logger.info("partition data ...")
    if os.path.isfile(args.output_partition_file):
        logger.info("loading partition data ...")
        with open(args.output_partition_file, "rb") as file:
            partition_data = pickle.load(file)
    else:
        partition_data = {}

    logger.info(f"partition data's keys: {partition_data.keys()}")

    if f"clients={args.clients_num}_alpha={args.alpha}" in partition_data and not args.overwrite:
        logger.info(f"Partition method 'clients={args.clients_num}_alpha={args.alpha}' has existed "
                    f"and overwrite={args.overwrite}, then skip")

    else:
        lable_mapping = {label: idx for idx, label in enumerate(label_list)}
        attribute = {"lable_mapping": lable_mapping, "label_list": label_list,
                     "clients_num": args.clients_num, "alpha": args.alpha,
                     "output_mode": output_mode
                     }

        clients_partition_data = {"train": get_partition_data(
            examples=train_examples, label_list=label_list, num_clients=args.clients_num,
            dir_alpha=args.alpha, partition="dirichlet"
        ), "valid": get_partition_data(
            examples=valid_examples, label_list=label_list, num_clients=args.clients_num,
            dir_alpha=args.alpha, partition="dirichlet"
        ), "test": get_partition_data(
            examples=test_examples, label_list=label_list, num_clients=args.clients_num,
            dir_alpha=args.alpha, partition="dirichlet"
        ), "attribute": attribute}

        logger.info(f"writing clients={args.clients_num}_alpha={args.alpha} ...")
        partition_data[f"clients={args.clients_num}_alpha={args.alpha}"] = clients_partition_data

        with open(args.output_partition_file, "wb") as file:
            pickle.dump(partition_data, file)

    logger.info("end")


if __name__ == "__main__":
    logger.info("start...")
    args = parser_args()
    make_sure_dirs(args.output_dir)
    args.output_data_file = os.path.join(
        args.output_dir, f"{args.task}_data.pkl"
    )
    args.output_partition_file = os.path.join(
        args.output_dir, f"{args.task.lower()}_partition.pkl"
    )

    logger.info(f"clients_num: {args.clients_num}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"output_data_file: {args.output_data_file}")
    logger.info(f"output_partition_file: {args.output_partition_file}")

    convert_conll_to_device_pkl(args)
