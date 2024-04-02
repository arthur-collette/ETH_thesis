import wandb
from tqdm import tqdm

from llama import Llama
from gpt import GPT
from flan import FLAN
from functions import *
from dataloader import * 
import argparse
from util import initialize_config

class File:
    def __init__(self, config_name, model_name, input_path, output_path, start):
        self.config = initialize_config("/Users/arthur/Documents/ETH/Master_Thesis/python/"+config_name)['train']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if model_name == "flan": 
            self.model = FLAN()
        elif model_name == "gpt":
            self.model = GPT("GPT")
        elif model_name == "llama":
            self.model = Llama()
        elif model_name == "gpt4":
            self.model = GPT("GPT4")
        self.output_path = f"/Users/arthur/Documents/ETH/Master_Thesis/data/{output_path}.jsonl"
        self.dataset = IDataset(input_path)
        self.start = start
        self.model_challenge = Llama()
        
        
        
    def run(self, n_sample):
        if n_sample > len(self.dataset):
            n_sample = len(self.dataset)
            print("n_sample is too big and has been set to max value :", n_sample)
        with open(self.output_path, 'w') as write_file:
            for i in tqdm(range(self.start, n_sample+self.start)):
                input = self.dataset[i]
                challenge = self.model_challenge.challenge(input)
                input['challenge'] = challenge
                questions = self.model.all_questions(input)
                input['generated_questions'] = dict(zip(self.model.config["question_types"], questions))
                json_object = json.dumps(input)
                write_file.write(json_object) 
                write_file.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File')

    parser.add_argument("--model_name", type=str, default="gpt4")
    parser.add_argument("--input_path", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="train_gpt4_0")
    parser.add_argument("--config_name", type=str, default="experiments.conf")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n_sample", type=int, default=400)

    args = parser.parse_args()

    runner = File(args.config_name, args.model_name, args.input_path, args.output_path, args.start)
    runner.run(args.n_sample)