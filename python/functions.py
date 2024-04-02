#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import math
from tqdm import tqdm
from IPython.display import display, Markdown
from collections import Counter

from definition import question_types

def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def get_questions(line):
    """
    Retreive the questions from the training dataset
    """
    questions = []
    l = line['answer'].split("?")
    questions.append(l[0]+"?")
    for i in range(1,len(l)-1):
        temp = l[i].split("\n")
        if len(temp) == 2 : questions.append(temp[1]+"?")
    return questions

def get_random(file_list):
    return json.loads(file_list[random.randint(0,len(file_list))])

def count_class(file_list):
    count = {}
    n = 0
    for sample in tqdm(file_list):
        line = json.loads(sample)
        questions = line['generated_questions']
        for q_type in questions.keys():
            if q_type in count.keys(): count[q_type]+=1
            else: count[q_type] = 1
            n+=1
    for key in count.keys():
        count[key] /= n
    return count

def questions_dataset(input_list):
    dataset = []
    for i in tqdm(range(len(input_list))):
        line = json.loads(input_list[i])
        try:
            questions = line['generated_questions']
            for q_type in questions:
                dataset.append(questions[q_type])
        except KeyError:
            questions = get_questions(line)
            for question in questions:
                dataset.append(question)
    return dataset

def count_challenge(file_list):
    count = {}
    for sample in tqdm(file_list):
        line = json.loads(sample)
        challenge = line['challenge']
        if challenge in count.keys(): count[challenge]+=1
        else: count[challenge] = 1
    for key in count.keys():
        count[key] /= len(file_list)
    return count

def count_lists(files):
    count = {}
    for file_list in files:
        for sample in file_list:
            line = json.loads(sample)
            challenge = line['challenge']
            if challenge in count.keys(): count[challenge]+=1
            else: count[challenge] = 1
    return count

def display_line_print(line):
    formatted_output = f'''
    Question:
        {line["question"]}

    Answer:
        {line["answer"]}

    Generated Questions:
        - Clarification Question: {line["generated_questions"]["clarification question"]}\n
        - Partial Story State Question: {line["generated_questions"]["partial story state question"]}\n
        - Background Knowledge Question: {line["generated_questions"]["background knowledge question"]}\n
        - Next Step Hint: {line["generated_questions"]["next step hint"]}\n
        - Counterfactual Question: {line["generated_questions"]["counterfactual question"]}\n
        - Probing Question: {line["generated_questions"]["probing question"]}\n

    Challenge:
        {line["challenge"]}
        '''
    return formatted_output

def display_line(line):
    formatted_output = f'''
    <div style="font-family: Arial, sans-serif; margin-bottom: 20px;">
        <div style="font-size: 18px;"><strong>Question:</strong></div>
        <div style="font-size: 16px;">{line["question"]}</div>

        <div style="margin-top: 10px; font-size: 18px;"><strong>Answer:</strong></div>
        <div style="font-size: 16px;">{line["answer"]}</div>

        <div style="margin-top: 10px; font-size: 18px;"><strong>Generated Questions:</strong></div>
        <ul style="list-style-type: none; padding: 0;">
            <li style="font-size: 16px;"><em>Clarification Question:</em> {line["generated_questions"]["clarification question"]}</li>
            <li style="font-size: 16px;"><em>Partial Story State Question:</em> {line["generated_questions"]["partial story state question"]}</li>
            <li style="font-size: 16px;"><em>Background Knowledge Question:</em> {line["generated_questions"]["background knowledge question"]}</li>
            <li style="font-size: 16px;"><em>Next Step Hint:</em> {line["generated_questions"]["next step hint"]}</li>
            <li style="font-size: 16px;"><em>Counterfactual Question:</em> {line["generated_questions"]["counterfactual question"]}</li>
            <li style="font-size: 16px;"><em>Probing Question:</em> {line["generated_questions"]["probing question"]}</li>
        </ul>

        <div style="margin-top: 10px; font-size: 18px;"><strong>Challenge:</strong></div>
        <div style="font-size: 16px;">{line["challenge"]}</div>
    </div>
    '''
    return formatted_output

def compute_entropy(dataset):
    # Tokenization
    tokens = ' '.join(dataset).split()
    # Frequency Count
    token_counts = Counter(tokens)
    # Total number of tokens
    total_tokens = len(tokens)
    # Probability Calculation and Entropy Calculation
    entropy = -sum((count/total_tokens) * math.log2(count/total_tokens) for count in token_counts.values())
    
    return entropy

def token_statistics(input_list):
    dataset = " ".join(questions_dataset(input_list))
    tokens = dataset.split()  # Assuming the dataset is a string
    num_tokens = len(tokens)
    unique_tokens = set(tokens)
    num_unique_tokens = len(unique_tokens)
    avg_token_length = sum(len(token) for token in tokens) / num_tokens

    return num_tokens, num_unique_tokens, avg_token_length
