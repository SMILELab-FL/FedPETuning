"""main for FedETuning"""

from utils import registry
from utils import build_config
from utils import setup_logger, setup_imports


def main():

    setup_imports()
    setup_logger()

    config = build_config()

    trainer = registry.get_fl_class(config.federated_config.fl_algorithm)()
    trainer.train()


if __name__ == "__main__":
    main()
