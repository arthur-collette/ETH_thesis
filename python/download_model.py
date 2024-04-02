#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:56:49 2023

@author: arthur
"""
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser(description='download_model')
parser.add_argument("--input_path", type=str, default="google/flan-t5-xl")
#parser.add_argument("--output_path", type=str, default="/cluster/scratch/acollette/flan/model/")
parser.add_argument("--output_path", type=str, default="./model/")
args = parser.parse_args()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")


tokenizer.save_pretrained(args.output_path+args.input_path.split("/")[1])
model.save_pretrained(args.output_path+args.input_path.split("/")[1])