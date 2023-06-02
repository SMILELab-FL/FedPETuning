"""Data Process Utils"""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None


def tokenize_and_align_labels(seq_tokens, seq_labels, tokenizer,
                              label_to_id, b_to_i_label, max_seq_length,
                              padding=False, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        seq_tokens,
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(seq_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["label"] = labels
    return tokenized_inputs


def conll_convert_examples_to_features(examples, tokenizer, max_length, label_list,
                                       output_mode, label_all_tokens=False, padding=True):
    seq_tokens = [example["tokens"] for example in examples]
    seq_labels = [example["labels"] for example in examples]

    label_to_id = {l: i for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    tokenized_inputs = tokenize_and_align_labels(
        seq_tokens, seq_labels, tokenizer,
        label_to_id, b_to_i_label, max_length,
        padding=padding, label_all_tokens=label_all_tokens)

    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'label']
    features = []
    for i in range(len(tokenized_inputs["input_ids"])):
        temp = {}
        for key in keys:
            if key in tokenized_inputs:
                temp[key] = tokenized_inputs[key][i]
        features.append(InputFeatures(**temp))

    return features

