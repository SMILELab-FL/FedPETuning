# import torch
# from abc import ABC
# from utils import registry
#
# Discard LocalTrainer
#
# class BaseLocalTrainer(ABC):
#     def __init__(self):
#
#         config = registry.get("config")
#         self.model_config = config.model_config
#         self.data_config = config.data_config
#         self.training_config = config.training_config
#         self.federated_config = config.federated_config
#
#         self.device = config.training_config.device
#         self.metric = registry.get_metric_class(self.training_config.metric_name)
#
#         self.logger = registry.get("logger")
#
#         self._before_training()
#
#     def train_model(self, *args):
#         raise NotImplementedError
#
#     def eval_model(self, model, valid_dl):
#         raise NotImplementedError
#
#     def infer_model(self, model, test_dl):
#         raise NotImplementedError
#
#     def _build_loss(self):
#         return registry.get_loss_class(self.training_config.loss_name)(
#             config=self.training_config
#         )
#
#     def _build_metric(self):
#         self.metric = registry.get_metric_class(self.training_config.metric_name)(
#             self.data_config.task_name, self.training_config.is_decreased_valid_metric
#         )
#
#     def _build_optimizer(self, model, iteration_in_total):
#         raise NotImplementedError
#
#     def _mixed_train_model(self, model, optimizer):
#         if self.training_config.fp16:
#             try:
#                 from apex import amp
#             except ImportError:
#                 raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#             model, optimizer = amp.initialize(model, optimizer, opt_level=self.training_config.fp16_opt_level)
#
#             # multi-gpu training (should be after apex fp16 initialization)
#         if self.training_config.n_gpu > 1:
#             self.logger.warning("We haven't tested our model under multi-gpu. Please be aware!")
#             model = torch.nn.DataParallel(model)
#
#         return model, optimizer
#
#     def _freeze_model_parameters(self, model):
#         raise NotImplementedError
#
#     def _before_training(self):
#         self.logger.info("build metric ...")
#         self._build_metric()
