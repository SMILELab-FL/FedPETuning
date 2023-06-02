""" pretty logging for FedETuning """

import sys
from loguru import logger
from utils.register import registry


def formatter(record):
    # default format
    time_format = "<green>{time:MM-DD/HH:mm:ss}</>"
    lvl_format = "<lvl><i>{level:^5}</></>"
    rcd_format = "<cyan>{file}:{line:}</>"
    msg_format = "<lvl>{message}</>"

    if record["level"].name in ["WARNING", "CRITICAL"]:
        lvl_format = "<l>" + lvl_format + "</>"

    return "|".join([time_format, lvl_format, rcd_format, msg_format]) + "\n"


def setup_logger():
    logger.remove()

    logger.add(
        sys.stderr, format=formatter,
        colorize=True, enqueue=True
    )

    registry.register("logger", logger)

    return logger
