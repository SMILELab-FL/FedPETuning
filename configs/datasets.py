import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    raw_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The raw data dir. if None, it builds from globalhost.py"}
    )

    partition_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The partition data dir. if None, it builds from globalhost.py"}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The cached data dir. if None, it builds from globalhost.py"}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):

        if self.task_name is None:
            raise ValueError(f"The task_name must be set, but {self.task_name} found")
        else:
            self.task_name = self.task_name.lower()

        if self.raw_dataset_path is None:
            raise ValueError(f"The raw_dataset_path must be set, but {self.raw_dataset_path} found")
        else:
            if not self.raw_dataset_path.endswith(".pkl"):
                self.raw_dataset_path = os.path.join(
                    self.raw_dataset_path, f"{self.task_name}_data.pkl"
                )

        if self.partition_dataset_path is None:
            raise ValueError(f"The raw_dataset_path must be set, but {self.raw_dataset_path} found")
        else:
            if not self.partition_dataset_path.endswith(".pkl"):
                self.partition_dataset_path = os.path.join(
                    self.partition_dataset_path, f"{self.task_name}_partition.pkl"
                )
