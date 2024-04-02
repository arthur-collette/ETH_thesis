import copy
import json
import os

import torch
import re
from text2digits import text2digits
from base_class import translate_prompt

from functions import *


class IDataset(torch.utils.data.Dataset):
    def __init__(self, input_path):
        self.problem = load_jsonl(input_path)
    
    def __len__(self):
        return len(self.problem)
    
    def __getitem__(self, idx):
        return self.problem[idx]

class QDataset(torch.utils.data.Dataset):
    def __init__(self, input_path):
        self.question, self.answer, self.generated_questions, self.challenge = q_type_dataset(load_jsonl(input_path))
    
    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, idx):
        return dict(question=self.question[idx], answer=self.answer[idx], generated_questions=self.generated_questions[idx], challenge=self.challenge[idx])


class FLANDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, model):
        self.tokenizer = model.tokenizer
        self.max_source_length = 512
        self.max_target_length = 128    
        self.input_seq, self.output_seq, self.q_types = prepare_dataset(load_jsonl(input_path), model)
        self.encoding = self.tokenizer(self.input_seq,
                                       padding='longest',
                                       max_length=self.max_source_length,
                                       truncation=True,
                                       return_tensors="pt")
        self.input_ids, self.attention_mask = self.encoding.input_ids, self.encoding.attention_mask

        self.target_encoding = self.tokenizer(self.output_seq,
                                              padding='longest',
                                              max_length=self.max_target_length,
                                              truncation=True,
                                              return_tensors="pt")
        self.labels = self.target_encoding.input_ids

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_tokens = self.input_ids[idx]
        output_tokens = self.labels[idx]
        mask = self.attention_mask[idx]

        output_tokens = torch.tensor(output_tokens)
        output_tokens[output_tokens == self.tokenizer.pad_token_id] = -100

        return dict(inpt=input_tokens, att_mask=mask, lbl=output_tokens, raw_input=self.input_seq[idx],
                    raw_output=self.output_seq[idx], q_type=self.q_types[idx])
    
class FTDataset(torch.utils.data.Dataset):
    def __init__(self, input_path):
        self.question, self.answer, self.generated_questions, self.challenge = q_type_dataset(load_jsonl(input_path))
    
    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, idx):
        return dict(question=self.question[idx], answer=self.answer[idx], generated_questions=self.generated_questions[idx], challenge=self.challenge[idx])


def load_jsonl(input_name):
    path = f"/Users/arthur/Documents/ETH/Master_Thesis/data/{input_name}.jsonl"
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def prepare_dataset(json_list, model):
    input_seq = []
    output_seq = []
    q_types = []
    for line in json_list:
        for q_type in line['generated_questions']:
            input_seq.append(translate_prompt(model.question_prompt(line, q_type)))
            output_seq.append(line['generated_questions'][q_type])
            q_types.append(q_type)
    return input_seq, output_seq, q_types

def q_type_dataset(json_list):
    question, answer, generated_questions, challenge = [], [], [], []
    for line in json_list:
        question.append(line['question'])
        answer.append(line['answer'])
        generated_questions.append(line['generated_questions'])
        challenge.append(line['challenge'])
    return question, answer, generated_questions, challenge

class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, input_seq, output_seq):
        self.max_source_length = 512
        self.max_target_length = 128
        self.tokenizer = tokenizer
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.encoding = self.tokenizer(self.input_seq,
                                       padding='longest',
                                       max_length=self.max_source_length,
                                       truncation=True,
                                       return_tensors="pt")
        self.input_ids, self.attention_mask = self.encoding.input_ids, self.encoding.attention_mask

        self.target_encoding = self.tokenizer(self.output_seq,
                                              padding='longest',
                                              max_length=self.max_target_length,
                                              truncation=True)
        self.labels = self.target_encoding.input_ids

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_tokens = self.input_ids[idx]
        output_tokens = self.labels[idx]
        mask = self.attention_mask[idx]

        output_tokens = torch.tensor(output_tokens)
        output_tokens[output_tokens == self.tokenizer.pad_token_id] = -100

        #return dict(inpt=input_tokens, att_mask=mask, lbl=output_tokens, raw_input=self.input_seq[idx],
                    #raw_output=self.output_seq[idx])
        return dict(input_ids=input_tokens, attention_mask=mask, labels=output_tokens)


def read_jsonl(split: str):
    """ Reads the input file and parses into a list of question answer pairs.

    Args:
        path (str): Train, test or dev split

    Returns:
        [list]: List of all question answer pairs in dict format.
    """
    path = f"data/{split}_socratic.jsonl"
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def text_to_digits(input_seq: str):
    """Converts text to its digit representation

    Args:
        input_seq [list]: [A list of input sequences]

    Returns:
        input_seq [list]: [A list of input sequences with text as digits]
    """

    t2d = text2digits.Text2Digits()

    try:
        input_seq = t2d.convert(input_seq)
    except:
        pass

    return input_seq


def sentence_planning_window(all_data: list):
    """Create input and output pairs for training the model
    Args:
        all_data ([list]): [all data points: list of dict]
    Returns:
        input_seq, output_seq [list]: [input and output sequence for training the model]
    """

    pairs = []
    for data in all_data:
        out = data["answer"].split("\n")
        ques, ans = [], []
        for samp in out:
            if len(samp.split(" ** ")) == 2:
                ques.append(samp.split(" ** ")[0])
                ans.append(samp.split(" ** ")[1])
        nums = [re.findall("(\d+=)|(\d+\s=)", " ".join(re.findall("<<\S+", sent))) for sent in ans]
        nums = [num[0][0].replace("=", "") for num in nums if num != []]

        for r in range(len(nums)):
            # data["question"] = text_to_digits(data["question"])
            if r == len(nums) - 1:
                pairs.append([data["question"], ques[r]])
            else:
                val = data["question"].find(nums[r])
                punkt = data["question"][val:].find(".")
                if punkt == -1:
                    pairs.append([data["question"][val + 1:], " ".join([ques[left] for left in range(r, len(nums))])])
                    break
                pairs.append([data["question"][:val + punkt + 1], ques[r]])
                data["question"] = data["question"][val + punkt + 1:]

    input_seq = [p[0] for p in pairs]
    output_seq = [p[1] for p in pairs]

    return input_seq, output_seq


def iterative_sentence_planning_with_separators(all_data: list, planning: str = "", reward: str = ""):
    """Create input and output pairs for training the model
    Args:
        all_data ([list]): [all data points: list of dict]
    Returns:
        input_seq, output_seq [list]: [input and output sequence for training the model]
    """
    prompt = "Read the following math problem and generate a socratic question that would help a student with difficulties solve the problem:\n\n"

    pairs = []
    data_point_index = 0
    data_points_indices = []
    for data in all_data:
        data_point_index = data_point_index + 1
        out = data["answer"].split("\n")
        ques, ans = [], []
        for samp in out:
            if len(samp.split(" ** ")) == 2:
                ques.append(samp.split(" ** ")[0])
                ans.append(samp.split(" ** ")[1])
        ques.pop()
        ans.pop()
        equation_sequence = [" ".join(re.findall("<<\S+>>", sent)) for sent in ans]
        operator_sequence = [re.sub('[^-+*/<>]+', ' ', equation, 0, re.I).strip() for equation in equation_sequence]
        nums = [re.findall("(\d+=)|(\d+\s=)|(\d+\.\d+=)|(\d+\.\d+\s=)", " ".join(re.findall("<<\S+", sent))) for sent in
                ans]
        # filter matched groups
        matched_numers = []
        for found_pattern in nums:
            if len(found_pattern) > 0:
                for matched_number in found_pattern[0]:
                    if len(matched_number) > 0:
                        matched_numers.append(matched_number.replace("=", ""))
        nums = matched_numers

        new_operator_sequence = []
        for operator_string in operator_sequence:
            new_string = ""
            for char_in_operator in operator_string:
                new_string += operator_map.get(char_in_operator, char_in_operator)
            new_operator_sequence.append(new_string)
        operator_sequence = new_operator_sequence

        previous_split = 0
        equation_index = 0

        if reward != "qa" or reward == "combined":
            data["question"] = text_to_digits(data["question"])

        modified_question = data["question"]
        additional_information = ""
        for r in range(len(nums)):
            if planning == EQUATION:
                additional_information = " ".join(equation_sequence[equation_index:])
            elif planning == OPERATOR:
                additional_information = " ".join(operator_sequence[equation_index:])

            if r == len(nums) - 1:
                pairs.append([data["question"][
                              :previous_split] + " [SEP] " + prompt + modified_question + " [/SEP] " + additional_information,
                              ques[r]])
                data_points_indices.append(data_point_index)
            else:
                val = modified_question.find(nums[r])
                punkt = modified_question[val:].find(". ")
                split_point = val + punkt + 1
                if punkt == -1:
                    pairs.append([data["question"][
                                  :previous_split] + " [SEP] " + prompt + modified_question + " [/SEP] " + additional_information,
                                  " ".join([ques[left] for left in range(r, len(nums))])])
                    data_points_indices.append(data_point_index)
                    break

                if planning == EQUATION:
                    valid_equations = []
                    for eq in equation_sequence[equation_index:]:
                        is_match = re.search(f"[\+\-\*\/\<]{{1}}{nums[r]}[\+\-\*\/\<\=]{{1}}", eq)
                        if is_match is not None:
                            valid_equations.append(eq)
                            break
                    equation_index = equation_index + len(valid_equations)
                    additional_information = " ".join(valid_equations)
                elif planning == OPERATOR:
                    valid_equations = []
                    for eq in equation_sequence[equation_index:]:
                        is_match = re.search(f"[\+\-\*\/\<]{{1}}{nums[r]}[\+\-\*\/\<\=]{{1}}", eq)
                        if is_match is not None:
                            valid_equations.append(eq)
                            break
                    additional_information = " ".join(
                        operator_sequence[equation_index:equation_index + len(valid_equations)])
                    equation_index = equation_index + len(valid_equations)

                pairs.append([data["question"][:previous_split] + " [SEP] " + prompt + modified_question[
                                                                              :split_point] + " [/SEP] " + additional_information + modified_question[
                                                                                                                                    split_point:],
                              ques[r]])
                data_points_indices.append(data_point_index)
                modified_question = modified_question[split_point:]
                previous_split += split_point

    input_seq = [p[0] for p in pairs]
    output_seq = [p[1] for p in pairs]
    input_seq = [re.sub('\s+', ' ', sentence).strip() for sentence in input_seq]
    return input_seq, output_seq, data_points_indices

 