import pyhocon
import torch
import numpy as np
import random
from transformers import T5Tokenizer, BartTokenizer

def initialize_config(config_name):
    """Intitalise the config parameters

    Args:
        config_name ([string]): [path to config file]

    Returns:
        [dict]: [config in dict format]
    """

    config = pyhocon.ConfigFactory.parse_file(config_name) #("experiments.conf")[config_name]
    return config