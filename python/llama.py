from base_class import Model, translate_prompt
from definition import *
from functions import *
from util import initialize_config

from langchain.llms import LlamaCpp

class Llama(Model):
    def __init__(self):
        self.config = initialize_config("/Users/arthur/Documents/ETH/Master_Thesis/python/experiments.conf")
        self.name = "Llama"
        self.model = LlamaCpp(
            model_path=self.config['model_llama']['name'],
            n_gpu_layers=self.config['model_llama']['n_gpu_layers'],
            max_tokens=self.config['model_llama']['max_tokens'],
            n_ctx=self.config['model_llama']['n_ctx'],
            n_batch=512,
            f16_kv=True
        )
        
    def __call__(self, prompt):#, grammar="/Users/arthur/Documents/ETH/Master_Thesis/model/llama/test.gbnf"): 
        return self.model(translate_prompt(prompt)).strip()
    