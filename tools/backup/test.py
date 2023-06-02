"""Test some code snippets"""
import os
import torch
from loguru import logger
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.data_utils import InputFeatures
from utils import pickle_read
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


# class CoLAPromptTemple(object):
#     def __init__(self, data_path):
#         self.data_path = data_path


# dataset = [  # For simplicity, there's only two examples
#     # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
#     InputExample(
#         guid=0,
#         text_a="Albert Einstein was one of the greatest intellects of his time.",
#     ),
#     InputExample(
#         guid=1,
#         text_a="The film was badly made.",
#     ),
# ]

class MyManualTemplate(ManualTemplate):
    def wrap_one_example(self,
                         example: InputExample):
        if self.text is None:
            raise ValueError("template text has not been initialized")

        text = self.incorporate_text_example(example)

        # not_empty_keys = example.keys()
        not_empty_keys = [key for key in example.__dict__.keys() if getattr(example, key) is not None]
        for placeholder_token in self.placeholder_mapping:
            not_empty_keys.remove(self.placeholder_mapping[placeholder_token])  # placeholder has been processed, remove
        if "meta" in not_empty_keys:
            not_empty_keys.remove('meta')  # meta has been processed

        keys, values = ['text'], [text]
        for inputflag_name in self.registered_inputflag_names:
            keys.append(inputflag_name)
            v = None
            if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                v = getattr(self, inputflag_name)
            elif hasattr(self, "get_default_" + inputflag_name):
                v = getattr(self, "get_default_" + inputflag_name)()
                setattr(self, inputflag_name, v)  # cache
            else:
                raise ValueError("""
                Template's inputflag '{}' is registered but not initialize.
                Try using template.{} = [...] to initialize
                or create an method get_default_{}(self) in your template.
                """.format(inputflag_name, inputflag_name, inputflag_name))

            if len(v) != len(text):
                raise ValueError("Template: len({})={} doesn't match len(text)={}." \
                    .format(inputflag_name, len(v), len(text)))
            values.append(v)
        wrapped_parts_to_tokenize = []
        for piece in list(zip(*values)):
            wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))

        wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
        return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]


def build_dataloader(features, mode="train", model_type="roberta", train_batch_size=5,
                     output_mode="classification", tuning_type="prompt"):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    if model_type not in ["distilbert", "roberta"]:
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    else:
        # distilbert and roberta don't have token_type_ids
        all_token_type_ids = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if "prompt" in tuning_type:
        all_loss_ids = torch.tensor([f.loss_ids for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_loss_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=train_batch_size)

    return dataloader


def convert_glue_to_manual_features(samples, prompt_template,
                                    wrapped_tokenizer, label_list):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for sample in samples:
        wrapped_example = prompt_template.wrap_one_example(sample)
        tokenized_example = wrapped_tokenizer.tokenize_one_example(
            wrapped_example, teacher_forcing=False)
        feature = InputFeatures(**tokenized_example, **wrapped_example[1])
        feature.label = label_map[feature.label]
        features.append(feature)
    return features


# model_inputs = {}
# for split in ['train', 'valid', 'test']:
#     model_inputs[split] = []
#     for sample in raw_data[split][0:10]:
#         wrapped_example = promptTemplate.wrap_one_example(sample)
#         tokenized_example = wrapped_tokenizer.tokenize_one_example(
#             wrapped_example, teacher_forcing=False)
#         feature = InputFeatures(**tokenized_example, **wrapped_example[1])
#         model_inputs[split].append(feature)

task_name = "cola"
raw_dataset_path = os.path.join("/workspace/data/fedglue/", f"{task_name}_data.pkl")
raw_data = pickle_read(raw_dataset_path)

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "/workspace/pretrain/nlp/roberta-base/")

promptTemplate = MyManualTemplate(
    # text='{"placeholder":"text_a"} It was {"mask"}',
    text='{"placeholder":"text_a"} This is {"mask"} .',
    tokenizer=tokenizer,
)

label_list = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "0",
    "1"
]
label_words = {
    "0": ["incorrect", ],
    "1": ["correct"],
}
promptVerbalizer = ManualVerbalizer(
    classes=label_list,
    label_words=label_words,
    tokenizer=tokenizer,
)

# wrapped_example = promptTemplate.wrap_one_example(raw_data['train'][0])
# print(wrapped_example)

wrappedTokenizer = WrapperClass(max_seq_length=128, tokenizer=tokenizer, truncate_method="head")

train_features = convert_glue_to_manual_features(
    raw_data["train"],
    prompt_template=promptTemplate,
    wrapped_tokenizer=wrappedTokenizer,
    label_list=label_list
)
train_data_loader = build_dataloader(train_features)

valid_features = convert_glue_to_manual_features(
    raw_data["valid"],
    prompt_template=promptTemplate,
    wrapped_tokenizer=wrappedTokenizer,
    label_list=label_list
)
train_data_loader = build_dataloader(train_features, train_batch_size=32)
valid_data_loader = build_dataloader(
    valid_features, mode="valid",
    train_batch_size=32)
# You can see what a tokenized example looks like by
# tokenized_example = wrapped_tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
# print(tokenized_example)
# print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))

# print(promptVerbalizer.label_words_ids)
# logits = torch.randn(2, len(tokenizer))  # creating a pseudo output from the plm, and
# print(promptVerbalizer.process_logits(logits)) # see what the verbalizer do

# model_inputs = {}
# for split in ['train', 'valid', 'test']:
#     model_inputs[split] = []
#     for sample in raw_data[split][0:10]:
#         wrapped_example = promptTemplate.wrap_one_example(sample)
#         tokenized_example = wrapped_tokenizer.tokenize_one_example(
#             wrapped_example, teacher_forcing=False)
#         feature = InputFeatures(**tokenized_example, **wrapped_example[1])
#         model_inputs[split].append(feature)

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
    freeze_plm=False
)
from utils import get_parameter_number

print(get_parameter_number(promptModel))
#
# data_loader = PromptDataLoader(
#     dataset=raw_data['train'],
#     # tokenizer=tokenizer,
#     tokenizer_wrapper=wrapped_tokenizer,
#     template=promptTemplate,
#     tokenizer_wrapper_class=WrapperClass,
# )


promptModel = promptModel.cuda()

# Now the training is standard
from transformers import AdamW, get_linear_schedule_with_warmup

num_train_epochs = 10
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
t_total = len(train_data_loader) * num_train_epochs
scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=t_total
        )
for epoch in range(num_train_epochs):
    tot_loss = 0
    for step, batch in enumerate(train_data_loader):

        batch = tuple(t.cuda() for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'label': batch[3],
                  'loss_ids': batch[4]
                  }

        logits = promptModel(inputs)
        # label_ids = batch[3]
        label_ids = inputs["label"]
        loss = loss_func(logits, label_ids)
        loss.requires_grad_(True)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch {}, average loss: {}".format(epoch, tot_loss / len(train_data_loader)), flush=True)

allpreds = []
alllabels = []
for step, batch in enumerate(valid_data_loader):
    batch = tuple(t.cuda() for t in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'label': batch[3],
              'loss_ids': batch[4]
              }
    logits = promptModel(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
print(acc)
from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(alllabels, allpreds))

# promptModel.eval()
# with torch.no_grad():
#     for batch in data_loader:
#         logits = promptModel(batch)
#         preds = torch.argmax(logits, dim = -1)
#         print(classes[preds])


# logger.info(promptModel)
