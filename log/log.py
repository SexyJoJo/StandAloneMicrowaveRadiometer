import logging
import logging.config
import json


class Log:
    def __init__(self, logger):
        with open(r"config/log.json", "r") as f:
            log_config = json.load(f)
        logging.config.dictConfig(log_config)
        self.logger = logging.getLogger(logger)


train_log = Log("train_logger")
