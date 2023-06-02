
class Registry:
    mapping = {
        "state": {},
        "loss_name_mapping": {},
        "data_name_mapping": {},
        "model_name_mapping": {},
        "fl_name_mapping": {},
        "metric_name_mapping": {},
        "eval_name_mapping": {}
    }

    @classmethod
    def register(cls, name, obj):
        cls.mapping["state"][name] = obj

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
                "writer" in cls.mapping["state"]
                and value == default
                and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def register_loss(cls, name):
        def wrap(func):
            from utils.loss import BaseLoss

            assert issubclass(
                func, BaseLoss
            ), "All loss must inherit utils.loss.BaseLoss class"
            cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_data(cls, name):
        def wrap(func):
            from data.base_dataloader import BaseDataLoader

            assert issubclass(
                func, BaseDataLoader
            ), "All dataset must inherit data.base_data_loader.BaseDataLoader class"
            cls.mapping["data_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(func):
            from models.base_models import BaseModels

            assert issubclass(
                func, BaseModels
            ), "All model must inherit models.base_models.BaseModels class"
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_fl_algorithm(cls, name):
        def wrap(func):
            from trainers.FedBaseTrainer import BaseTrainer

            assert issubclass(
                func, BaseTrainer
            ), "All federated algorithm must inherit trainers.base_fed_trainer.BaseTrainer class"
            cls.mapping["fl_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_metric(cls, name):
        def wrap(func):
            from utils.metrics import BaseMetric

            assert issubclass(
                func, BaseMetric
            ), "All metric must inherit utils.metrics.BaseMetric class"
            cls.mapping["metric_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_eval(cls, name):
        def wrap(func):
            from utils.evaluations import BaseEval

            assert issubclass(
                func, BaseEval
            ), "All evaluation must inherit utils.evaluations.BaseEval class"
            cls.mapping["eval_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def get_loss_class(cls, name):
        return cls.mapping["loss_name_mapping"].get(name, None)

    @classmethod
    def get_data_class(cls, name):
        return cls.mapping["data_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_fl_class(cls, name):
        return cls.mapping["fl_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def get_eval_class(cls, name):
        return cls.mapping["eval_name_mapping"].get(name, None)

    @classmethod
    def unregister(cls, name):
        return cls.mapping["state"].pop(name, None)

    @classmethod
    def get_keys(cls):
        keys = list(cls.mapping["state"].keys())
        return keys

    @classmethod
    def debug(cls, info):
        from loguru import logger
        logger.debug(info)


registry = Registry()
