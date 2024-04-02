from base_class import Model, translate_prompt
from functions import *
from util import initialize_config

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch

#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FLAN(Model):
    def __init__(self):
        self.config = initialize_config("/Users/arthur/Documents/ETH/Master_Thesis/python/experiments.conf")
        self.name = "FLAN"
        self.model = T5ForConditionalGeneration.from_pretrained(self.config['model_flan']).to(device)
        #self.tokenizer = T5Tokenizer.from_pretrained(self.config['tokenizer_flan'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_flan'])
        

    def __call__(self, prompt):
        inputs = ""
        if type(prompt) == list: inputs = self.tokenizer(translate_prompt(prompt), return_tensors="pt").to(device)
        elif '[INST]' in prompt: inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        else: 
            print("Wrong prompt type")
            return
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
       
            
    #result_list=[]
        #for predicted_ids in outputs:
        #    one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        #return result_list