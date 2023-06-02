from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/configs/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "model_name"}
    )
    model_output_mode: Optional[str] = field(
        default=None,
        metadata={"help": "model_output_type: {seq_classification, seq_tagging, language_model}"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained configs name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    # Personalized Model Config
    permutation_layers: bool = field(
        default=False,
        metadata={"help": "The flag controls model layers."},
    )
    client_model_layers: List[int] = field(
        default_factory=list,
        metadata={"help": "The client model's size, default=[0,1,2]"},
    )
    server_model_layers: List[int] = field(
        default_factory=list,
        metadata={"help": "The client model's size, default=[i for i in range(12)]"},
    )

    # Efficient Model Config
    tuning_type: str = field(
        default=None,
        metadata={"help": "The Efficient Fine-tuning type, support {adapter, prompt, lora, prefix}"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "lora specific parameters"}
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "lora specific parameters"}
    )
    prefix_token_num: int = field(
        default=16,
        metadata={"help": "prefix-tuning specific parameters"}
    )
    bottleneck_dim: int = field(
        default=64,
        metadata={"help": "adapter specific parameters"}
    )

    def __post_init__(self):
        ...
