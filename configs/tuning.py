"""Tuning Args"""

all_delta_config = {
    "adapter_roberta-base":
        {
            "delta_type": "adapter",
            "learning_rate": 1e-3,
            "unfrozen_modules": [
                "deltas",
                "layer_norm",
                "final_layer_norm",
                "classifier",
            ],
            "bottleneck_dim": 16,
        },
    'soft_prompt_roberta-base':
        {
            "delta_type": "soft_prompt",
            "learning_rate": 3e-2,
            "soft_token_num": 100,
            "unfrozen_modules": [
                "deltas",
                "classifier",
            ],
        },
    "lora_roberta-base":
        {
            "rte":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_r": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 80,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "qqp":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_r": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 25,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "mrpc":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.001,
                    "lora_alpha": 16,
                    "lora_r": 16,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 30,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas",
                        "layer_norm"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "mnli":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_r": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 30,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 16,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "cola":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0004,
                    "lora_alpha": 8,
                    "lora_r": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 80,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "qnli":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0004,
                    "lora_alpha": 8,
                    "lora_r": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 25,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "sst-2":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_r": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 60,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                }

        },
    "bitfit_roberta-base":
        {
            "delta_type": "bitfit",
            "learning_rate": 3e-4,
            "unfrozen_modules": [
                "classifier",
                "deltas"
            ],
        },
    "prefix_roberta-base":
        {
            "delta_type": "prefix",
            "learning_rate": 1e-3,
            "unfrozen_modules": [
                "deltas",
                "classifier",
            ],
            "prefix_token_num": 16
        }
}


fed_best_hyperparameter = {
    "rte":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-3],
                "prefix_token_num": [8],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [5e-3],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-3],
                "bottleneck_dim": [64]
            }
        },
    "qqp":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [5e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [5e-4],
                "bottleneck_dim": [64]
            }
        },
    "mnli":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-4],
                "prefix_token_num": [16],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [5e-4],
                "bottleneck_dim": [64]
            }
        },
    "mrpc":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-3],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [5e-3],
            },
            "lora": {
                "learning_rate": [5e-3],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-3],
                "bottleneck_dim": [16]
            }
        },
    "cola":
        {
            "fine-tuning": {
                "learning_rate": [1e-4],
            },
            "prefix": {
                "learning_rate": [1e-3],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [5e-3],
            },
            "lora": {
                "learning_rate": [5e-3],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-3],
                "bottleneck_dim": [64]
            }
        },
    "qnli":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [5e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-3],
                "bottleneck_dim": [16]
            }
        },
    "sst-2":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [5e-4],
                "bottleneck_dim": [16]
            }
        },
}


cen_best_hyperparameter = {
    "rte":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [5e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-3],
                "bottleneck_dim": [16]
            }
        },
    "qqp":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [5e-4],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-4],
                "bottleneck_dim": [64]
            }
        },
    "mnli":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-4],
                "bottleneck_dim": [64]
            }
        },
    "mrpc":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-4],
                "prefix_token_num": [16],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [5e-4],
                "lora_r": [16],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [5e-4],
                "bottleneck_dim": [16]
            }
        },
    "cola":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [5e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [1e-3],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-4],
                "bottleneck_dim": [64]
            }
        },

    "qnli":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [1e-4],
                "prefix_token_num": [16],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [5e-4],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-4],
                "bottleneck_dim": [16]
            }
        },
    "sst-2":
        {
            "fine-tuning": {
                "learning_rate": [5e-5],
            },
            "prefix": {
                "learning_rate": [5e-4],
                "prefix_token_num": [64],
            },
            "bitfit": {
                "learning_rate": [1e-3],
            },
            "lora": {
                "learning_rate": [5e-4],
                "lora_r": [8],  # so to lora_alpha
            },
            "adapter": {
                "learning_rate": [1e-4],
                "bottleneck_dim": [64]
            }
        },
}


hyperparameter_grid = {

    # Hyper-parameter Setup: 6 * (1+2+2+1+3) = 54
    # adapter lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and bottleneck_dim: [16,64]
    # lora lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and lora_alpha & lora_r from [8, 16]
    # bitfit lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # prefix lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and prefix_token_num from [8, 16, 64]

    "fine-tuning": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        # "learning_rate": [5e-5],
    },
    "prefix": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "prefix_token_num": [8, 16, 64],
    },
    "bitfit": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    },
    "lora": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "lora_r": [8, 16],  # so to lora_alpha
    },
    "adapter": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "bottleneck_dim": [16, 64]
    }
}


def get_delta_config(delta_name):
    return all_delta_config[delta_name]


def get_delta_key(delta_type):
    delta_keys = {
        "fine-tuning": "",
        "prefix": "prefix_token_num",
        "bitfit": "",
        "lora": "lora_r",
        "adapter": "bottleneck_dim"
    }
    delta_keys_abb = {
        "fine-tuning": "",
        "prefix": "ptn",
        "bitfit": "",
        "lora": "la",
        "adapter": "dim"
    }
    return delta_keys[delta_type], delta_keys_abb[delta_type]
