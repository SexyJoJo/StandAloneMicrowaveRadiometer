from parse.parse_utils import *
from train.training import Train
from const.Consts import CONFIG

if __name__ == '__main__':
    Train.training(CONFIG["data_sources"], CONFIG["activation"], CONFIG["elements"], CONFIG["hidden_nodes"],
                   CONFIG["input_nodes"], CONFIG["output_nodes"], CONFIG["max_iter"], CONFIG["normalization"],
                   CONFIG["solver"])
