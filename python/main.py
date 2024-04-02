#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch

from definition import *
from functions import *
from llama import Llama
from gpt import GPT
from flan import FLAN
import openai

parser = argparse.ArgumentParser(description='Create file')

parser.add_argument("--model", type=str, default="")
parser.add_argument("--n_sample", type=int, default=10)
parser.add_argument("--output_path", type=str, default="../data/")
parser.add_argument("--input_path", type=str, default="../data/train_socratic.jsonl")

parser.add_argument("--env", type=str, default="mac")
parser.add_argument("--mode", type=str, default="online")

args = parser.parse_args()

if __name__ == '__main__':

    if args.env == "mac": device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == "flan": 
        model = FLAN()
    elif args.model == "gpt":
        model = GPT()
    elif args.model == "llama":
        model = Llama()
    output_path = args.output_path+args.model+"_"+str(args.n_sample)+".jsonl"

    model.all(args.input_path, output_path, args.n_sample)
    