"""BaseClientTrainer for FedETuning"""

from abc import ABC
from typing import List
from thop import profile
from thop import clever_format

import torch
from transformers import get_linear_schedule_with_warmup, AdamW

from utils import registry
from utils import get_parameter_number
from fedlab.utils import MessageCode, SerializationTool
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.manager import ORDINARY_TRAINER, SERIAL_TRAINER


class BaseClientTrainer(ClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset):

        self._model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self._before_training()

    def _before_training(self):
        """before training function"""

        self.type = SERIAL_TRAINER  # represent serial trainer

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        self.client_num = len(config.F.clients_id_list)
        self.device = config.training_config.device
        self.rank = config.federated_config.rank
        self.param_list = []
        self.logger = registry.get("logger")

        self._build_metric()
        self._build_eval()

        # key: client idx, value: valid metric
        self.loc_best_metric = {}
        # key: client idx, value: test metric
        self.loc_test_metric = {}
        # key: client idx, value: serialized params
        self.loc_best_params = {}
        # local patient times
        self.loc_patient_times = 0
        # local early stop
        self.stop_early = False

        self.metric_name = self.metric.metric_name
        self._model.to(self.device)

        if self.federated_config.rank == -1:
            self._calculate_model_computation()

    def _calculate_model_computation(self):

        dummy_idx = list(self.train_dataset.keys())[0]
        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=dummy_idx)
        for step, batch in enumerate(train_loader):
            self._model.train()
            batch = tuple(t.to(self.device) for t in batch)

            macs, params = profile(self._model.backbone, inputs=(batch[0],))
            flops, params = clever_format([macs, params], "%.3f")
            self.logger.debug(f"Model Type: {self.model_config.model_type}, "
                              f"Tuning Type: {self.training_config.tuning_type}, "
                              f"Parameters: {get_parameter_number(self._model.backbone)}, "
                              f"FLOPs: {flops}")
            break

    @property
    def uplink_package(self):
        return self.param_list

    def _train_alone(self, idx: int, model_parameters: torch.Tensor, *args, **kwargs):
        """local training for Client"""

        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        if model_parameters is not None:
            SerializationTool.deserialize_model(self._model, model_parameters)

        # build optimizer,scheduler,loss
        optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
        self._model, optimizer = self._mixed_train_model(self._model, optimizer)
        self._build_loss()

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()
            self._on_epoch(train_loader, optimizer, scheduler)
            self._on_epoch_end(idx)
            if self.federated_config.pson and self.stop_early:
                self.logger.critical(f"local stop early in {epoch}")
                break

    def _get_dataloader(self, dataset, client_id: int):
        """Get :class:`DataLoader` for ``client_id``."""
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def local_process(self, id_list: List, payload: List):
        """local process for Federated Learning"""
        model_parameters = payload[0]
        self.param_list = self.fed_train(model_parameters, id_list)
        return self.param_list

    def fed_train(self, model_parameters: torch.Tensor, id_list: List):
        param_list = []

        for idx in id_list:
            self._train_alone(
                idx=idx,
                model_parameters=model_parameters
            )
            param_list.append(self.model_parameters)

        return param_list

    def cen_train(self, *args):
        self._train_alone(
            idx=-1,
            model_parameters=None,
        )

    # Local Training Functions
    def _build_loss(self):
        self.criterion = registry.get_loss_class(self.training_config.loss_name)(
            config=self.training_config
        )

    def _build_optimizer(self, model, train_dl_len):
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.get_optimized_model_params(model)

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler

    def get_optimized_model_params(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        # Both pieces of code have the same effect
        # optimizer_grouped_parameters = [
        #     {"params": filter(lambda x: x.requires_grad, model.bert.parameters()),
        #      'weight_decay': 0.0},
        # ]

        return optimizer_grouped_parameters

    def _mixed_train_model(self, model, optimizer):
        if self.training_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.training_config.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
        if self.training_config.n_gpu > 1:
            self.logger.warning("We haven't tested our model under multi-gpu. Please be aware!")
            model = torch.nn.DataParallel(model)

        return model, optimizer

    # Local Test Function
    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.is_decreased_valid_metric
        )

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def test_on_client(self, test_dataloader):

        for idx in self.loc_best_params:
            loc_best_params = self.loc_best_params[idx]
            SerializationTool.deserialize_model(self._model, loc_best_params)
            result = self.eval.test_and_eval(
                model=self._model,
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Test, "
                f"Client:{idx}, Test loss:{test_loss:.3f}, "
                f"Test {self.metric_name}:{test_metric:.3f}"
            )
            self.loc_test_metric[idx] = test_metric

    # Local Epoch Function
    def _on_epoch_begin(self):
        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0
        self.total, self.correct = 0, 0

    def _on_epoch(self, train_loader, optimizer, scheduler):
        for step, batch in enumerate(train_loader):
            # if step >= 2:
            #     break
            self._model.train()
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]
                      }
            label = inputs['labels']
            if self.model_config.model_type != 'distilbert' or self.model_config.model_type != 'roberta':
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs['token_type_ids'] = batch[2] \
                    if self.model_config.model_type in ['bert', 'xlnet'] else None
            outputs = self._model(inputs)

            loss, logits = outputs[:2]
            _, predicted = torch.max(logits, 1)

            optimizer.zero_grad()
            if self.training_config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.training_config.gradient_accumulation_steps > 1:
                loss = loss / self.training_config.gradient_accumulation_steps

            if self.training_config.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.tr_loss += loss.item()
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.training_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.training_config.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                self.global_step += 1

            self.total += label.size(0)
            if self.model_config.model_output_mode == "seq_classification":
                self.correct += (predicted == label).sum().item()

    def _on_epoch_end(self, idx):
        """on epoch end"""

        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, Loss:{self.tr_loss/self.global_step:.3f}, "
                         f"Accuracy:{self.correct/self.total:.3f}")

        if not self.federated_config.pson:
            # not need for local test
            return

        valid_data = self._get_dataloader(dataset=self.valid_dataset, client_id=idx)

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        # TODO hard code
        if not self.loc_best_metric.get(idx, None):
            self.loc_best_metric[idx] = float('-inf')
        if self.loc_best_metric[idx] < test_metric:
            self.loc_best_metric[idx] = test_metric
            self.loc_best_params[idx] = SerializationTool.serialize_model(self._model)
            self.loc_patient_times = 0
        else:
            self.loc_patient_times += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} Eval, "
                          f"Client:{idx}, Loss:{test_loss:.3f}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best {self.metric_name}:{self.loc_best_metric[idx]:.3f}")

        if self.loc_patient_times >= self.training_config.patient_times:
            self.stop_early = True


class BaseClientManager(PassiveClientManager, ABC):
    def __init__(self, network, trainer):
        self.logger = registry.get("logger")
        super().__init__(network, trainer, self.logger)

    def main_loop(self):
        """Actions to perform when receiving a new message, including local trainers.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model trainers procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:

                id_list, payload = payload[0].to(
                    torch.int32).tolist(), payload[1:]

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:  # serial
                    self._trainer.local_process(
                        id_list=id_list,
                        payload=payload
                    )

                elif self._trainer.type == ORDINARY_TRAINER:  # ordinary
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload)

                self.synchronize()

            else:
                raise ValueError(f"Invalid MessageCode {message_code}. Please check MessageCode list.")

    def synchronize(self):
        """Synchronize with server"""
        self.logger.info("Uploading information to server.")
        self._network.send(
            content=self._trainer.uplink_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )
