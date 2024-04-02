import wandb
from torch.utils.data import DataLoader
from transformers import get_scheduler, AdamW
import pandas as pd
import time
import argparse
import numpy as np
from tqdm import tqdm

from llama import Llama
from gpt import GPT
from flan import FLAN
from functions import *
from dataloader import * 
from util import initialize_config
from metrics import bleu_reward_estimation, correct_ques_num_reward_estimation
from evaluate import load

bertscore = load("bertscore")

class Train:
    def __init__(self, config_name, model_name, input_path):
        self.config = initialize_config("/Users/arthur/Documents/ETH/Master_Thesis/python/"+config_name)['train']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if model_name == "flan": 
            self.model = FLAN()
        elif model_name == "gpt":
            self.model = GPT("GPT")
        elif model_name == "llama":
            self.model = Llama()

        #self.train_dataset = FLANDataset(input_path, self.model)
        self.validation_dataset = QDataset(input_path)
        
        
        
    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=int(self.config["BATCH_SIZE"]), shuffle=True, drop_last=True)
        optim = AdamW(self.model.model.parameters(), lr=self.config["LEARNING_RATE"])
        num_training_steps = int(self.config["EPOCHS"]) * len(train_loader)
        lr_scheduler = get_scheduler(
            self.config['LR_SCHEDULER'],
            optimizer=optim,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        optim.zero_grad()
        self.model.model.train()
        print(num_training_steps)
        pbar = tqdm(range(num_training_steps))
        export_path = os.path.join("Users/arthur/Documents/ETH/Master_Thesis/model", self.model.name + str(int(time.time())))

        wandb.init(project="test", entity="arthur-collette", config=self.config)
        wandb.config.model_name = self.model.name
        wandb.config.export_path = export_path
        wandb.run.name = os.environ.get("RUN_NAME", f'{self.model.name}-finetune')

        table = wandb.Table(data=pd.DataFrame({"input": self.train_dataset[0:5]['raw_input'],
                                               "output": self.train_dataset[0:5]['raw_output']}))
        wandb.log({'training_datasample': table})

        for epoch in range(int(self.config["EPOCHS"])):
            for batch in train_loader:
                optim.zero_grad()
                for k, v in batch.items():
                    if k != "raw_output" and k != "raw_input":
                        batch[k] = v.to(self.device)
                train_loss = self.model.model(input_ids=batch["inpt"], attention_mask=batch["att_mask"], labels=batch["lbl"]).loss
                wandb.log({"train_loss": train_loss})
                train_loss.backward()
                optim.step()
                lr_scheduler.step()
                pbar.update(1)

        self.model.model.save_pretrained(export_path)

    def validation(self,stop=-1, challenge = False, check = False):
        predicted_out = []
        reference_out = []
        n=0
        for sample in tqdm(self.validation_dataset):
            if stop != -1:
                if n < stop: 
                    for q_type in sample['generated_questions']:
                        gen_question = self.model.question(sample, q_type, challenge)
                        if check:
                            if self.model.verify_question(gen_question, sample, q_type):
                                predicted_out.append(gen_question)
                                reference_out.append(sample['generated_questions'][q_type])
                                n+=1
                        else:
                            predicted_out.append(gen_question)
                            reference_out.append(sample['generated_questions'][q_type])
                            n+=1
                else: break
            else:
                for q_type in sample['generated_questions']:
                    gen_question = self.model.question(sample, q_type, challenge)
                    if check:
                        if self.model.verify_question(sample, q_type):
                            predicted_out.append(gen_question)
                            reference_out.append(sample['generated_questions'][q_type])
                
                    else:
                        predicted_out.append(gen_question)
                        reference_out.append(sample['generated_questions'][q_type])
                        
                
        valid_bleu_list = bleu_reward_estimation(reference_out, predicted_out)
        valid_question_count_list = correct_ques_num_reward_estimation(reference_out, predicted_out)
        mean_valid_bleu = np.array(valid_bleu_list).mean()

        mean_valid_question_count = np.array(valid_question_count_list).mean()

        bert = bertscore.compute(predictions=predicted_out, references=reference_out, lang="en")

        print(f"Valid bleu: {mean_valid_bleu}") #, valid question count: {mean_valid_question_count}")
        return predicted_out, reference_out, bert

if __name__ == "__main__":
    print("main")
    parser = argparse.ArgumentParser(description='FineTune')

    parser.add_argument("--model_name", type=str, default="flan")
    parser.add_argument("--input_path", type=str, default="gpt_1978")
    parser.add_argument("--output_path", type=str, default="../model/")
    parser.add_argument("--config_name", type=str, default="experiments.conf")

    args = parser.parse_args()

    runner = Train(args.config_name, args.model_name, args.input_path)
    runner.train()