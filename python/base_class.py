import time
import random 
import json
import re

from langchain.llms import LlamaCpp

from functions import *
from util import initialize_config
from difflib import SequenceMatcher

class Model():
    def __init__(self):
        self.config = initialize_config("/Users/arthur/Documents/ETH/Master_Thesis/python/experiments.conf")
    
    def __call__(self, prompt):
        pass

    def question(self, input, q_type, challenge=False):
        if challenge:
            challenge = self.challenge(input)
            response = self(self.question_prompt_challenge(input, q_type, challenge))
        else: response = self(self.question_prompt(input, q_type))
        result = response.split("?") 
        if result[-1] != "": result.pop() 
        result = [s.strip() for s in result if s.strip()]
        question = ""
        for q in result:
            question += re.split(r'[!.;:]', q)[-1] + "? "
        return question.strip()
    
    def question_mathdial(self, input, q_type, challenge=False):
        if challenge:
            challenge = self.challenge_mathdial(input)
            response = self(self.question_prompt_mathdial_challenge(input, q_type, challenge))
        else: response = self(self.question_prompt_mathdial(input, q_type))
        result = response.split("?") 
        if result[-1] != "": result.pop() 
        result = [s.strip() for s in result if s.strip()]
        question = ""
        for q in result:
            question += re.split(r'[!.;:]', q)[-1] + "? "
        return question.strip()
    
    def all_questions(self, input):
        response = self(self.questions_prompt(input))
        result = response.split("?") 
        if result[-1] != "": result.pop() 
        result = [s.strip() for s in result if s.strip()]
        question = ""
        for q in result:
            question += re.split(r'[!.;:]', q)[-1] + "? "
        return question.strip().split("  ")
    
    def question_w_p(self, prompt):
        response = self(prompt)
        result = response.split("?") 
        if result[-1] != "": result.pop() 
        result = [s.strip() for s in result if s.strip()]
        question = ""
        for q in result:
            question += re.split(r'[!.;:]', q)[-1] + "? "
        return question.strip()
    
    def verify_question(self, question, input, q_type):
        for subquestion in self.config['question_verification'][q_type]:
            response = self(self.verify_question_prompt(question, input, q_type, subquestion))
            print("response: ", response)
            if 'No' in response:
                return False
        return True
    
    def verify_challenge(self, input):
        for subquestion in self.config['challenge_verification'][input['challenge']]:
            response = self(self.verify_challenge_prompt(input, subquestion))
            print("response: ", response)
            if 'No' in response:
                return 0
        return 1
    
    def challenge(self, input):
        response = self(self.challenge_prompt(input))
        for challenge in self.config['challenge_description'].keys():
            if challenge in response.lower():
                return challenge
        print("Challenge not found.")
        return random.choice(list(self.config['challenge_description'].keys()))
    
    def challenge_mathdial(self, input):
        response = self(self.challenge_prompt_mathdial(input))
        for challenge in self.config['challenge_description'].keys():
            if challenge in response.lower():
                return challenge
        print("Challenge not found.")
        return random.choice(list(self.config['challenge_description'].keys()))
    
    def challenge_w_p(self, prompt):
        response = self(prompt)
        for challenge in self.config['challenge_description'].keys():
            if challenge in response.lower():
                return challenge
        return ""

    def question_prompt(self, input, q_type):
        message=[
            #{"role": "system", "content": "A tutor and a student work together to solve the following math word problem. Your role is tutor. The tutor is a soft-spoken empathetic person who dislikes giving out direct answers to students, and instead likes to answer with other questions that would help the student understand the concepts, so that she can solve the problem themselves."},
            {"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor"},
            {"role": "user", "content": f"Math problem: {input['question']}"},
            {"role": "user", "content": f"Answer: {input['answer']}"},
            #{"role": "user", "content": f"Challenge: {input['challenge']}"},
            #{"role": "user", "content": f"Challenge definition: {self.config['challenge_description'][input['challenge']][0]}"},
            {"role": "user", "content": f"A {q_type} is a {self.config['class_description'][q_type]}."}, 
            {"role": "user", "content": f"Generate one {q_type} that would help students solve the problem."},
            {"role": "user", "content": f"Question:"}
        ]
        return message
    
    def question_prompt_challenge(self, input, q_type, challenge):
        message=[
            #{"role": "system", "content": "A tutor and a student work together to solve the following math word problem. Your role is tutor. The tutor is a soft-spoken empathetic person who dislikes giving out direct answers to students, and instead likes to answer with other questions that would help the student understand the concepts, so that she can solve the problem themselves."},
            {"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor"},
            {"role": "user", "content": f"Math problem: {input['question']}"},
            {"role": "user", "content": f"Answer: {input['answer']}"},
            {"role": "user", "content": f"Challenge: {challenge}"},
            {"role": "user", "content": f"Challenge definition: {self.config['challenge_description'][input['challenge']][0]}"},
            {"role": "user", "content": f"A {q_type} is a {self.config['class_description'][q_type]}."}, 
            {"role": "user", "content": f"Generate one {q_type} that would help students solve the problem."},
            {"role": "user", "content": f"Question:"}
        ]
        return message
    
    def questions_prompt(self, input):
        message=[{"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor"}]
        message.append({"role": "system", "content": "List of challenge types faced by a student while solving a math problem."})
        message.append({"role": "user", "content": "Types:"})
        for q_type in self.config['class_description'].keys(): 
            message.append({"role": "user", "content": f"{q_type}: {self.config['class_description'][q_type]}."})
        message.append({"role": "user", "content": f"Math problem: {input['question']}"})
        message.append({"role": "user", "content": f"Answer: {input['answer']}"})
        message.append({"role": "user", "content": f"Challenge: {input['challenge']}"})
        message.append({"role": "user", "content": f"Challenge definition: {self.config['challenge_description'][input['challenge']][0]}"})
        message.append({"role": "user", "content": f"Create a list of 6 questions, one for each question types that would help students solve the problem."})
        #message.append({"role": "user", "content": f"Questions:\n -clarification question:\n -partial story state question:\n -background knowledge question:\n -next step hint\n -counterfactual question\n -probing question"})
        return message
    
    def question_prompt_mathdial(self, input, q_type):
        message=[
            #{"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor. The tutor is a soft-spoken empathetic person who dislikes giving out direct answers to students, and instead likes to answer with other questions that would help the student understand the concepts, so that she can solve the problem themselves."},
            {"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor"},
            {"role": "system", "content": f"Student profile :{input['student_profile']}"},
            {"role": "user", "content": f"Math problem: {input['question']}"},
            {"role": "user", "content": f"Answer: {input['ground_truth']}"},
            #{"role": "user", "content": f"Challenge: {input['challenge']}"},
            {"role": "user", "content": f"History: {get_history(input)}"},
            {"role": "user", "content": f"Challenge: {get_challenge(input)}"},
            {"role": "user", "content": f"A {q_type} is a {self.config['class_description'][q_type]}."}, 
            {"role": "user", "content": f"Generate one {q_type} that would help students solve the problem."},
            {"role": "user", "content": f"Question:"}
        ]
        return message
    
    def question_prompt_mathdial_challenge(self, input, q_type, challenge):

        message=[
            #{"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor. The tutor is a soft-spoken empathetic person who dislikes giving out direct answers to students, and instead likes to answer with other questions that would help the student understand the concepts, so that she can solve the problem themselves."},
            {"role": "system", "content": f"A tutor and a student work together to solve the following math word problem. Your role is tutor"},
            {"role": "system", "content": f"Student profile :{input['student_profile']}"},
            {"role": "user", "content": f"Math problem: {input['question']}"},
            {"role": "user", "content": f"Answer: {input['ground_truth']}"},
            {"role": "user", "content": f"Challenge: {challenge}"},
            {"role": "user", "content": f"History: {get_history(input)}"},
            {"role": "user", "content": f"Challenge: {get_challenge(input)}"},
            {"role": "user", "content": f"A {q_type} is a {self.config['class_description'][q_type]}."}, 
            {"role": "user", "content": f"Generate one {q_type} that would help students solve the problem."},
        ]
        return message
    
    def verify_question_prompt(self, question, input, q_type, subquestion):
        message = [
            {"role": "user", "content": f"Math problem: '{input['question']}'"},
            {"role": "user", "content": f"Question: '{question}'"},
            {"role": "user", "content": f"{subquestion}?"},
            {"role": "user", "content": f"Answer only 'Yes' or 'No':"}
        ]
        return message
    
    def verify_challenge_prompt(self, input, subquestion):
        message = [
            {"role": "user", "content": f"Math problem: '{input['question']}'"},
            {"role": "user", "content": f"Answer: '{input['answer']}'"},
            {"role": "user", "content": f"Challenge: '{input['challenge']}'"},
            {"role": "user", "content": f"{subquestion}?"},
            {"role": "user", "content": f"Answer one word: 'Yes' or 'No'"}
        ]
        return message
    
    def challenge_prompt(self, input):
        message = [{"role": "system", "content": "List of challenge types faced by a student while solving a math problem."}]
        message.append({"role": "user", "content": "Types:"})
        for challenge in self.config['challenge_description'].keys(): 
            message.append({"role": "user", "content": f"{challenge}: {self.config['challenge_description'][challenge][0]}."})
        message.append({"role": "user", "content": f"Math problem: {input['question']}"})
        message.append({"role": "user", "content": f"The correct solution is as follows: {input['answer']}"})
        message.append({"role": "user", "content": f"Choose one main challenge type for this problem."})
        message.append({"role": "user", "content": f"Challenge type: "})
        return message
    
    def challenge_prompt_mathdial(self, input):
        message = [{"role": "system", "content": "You are a tutor trying to find the hardest challenge faced by a student in a math problem."}]
        message.append({"role": "system", "content": "List of challenge types faced by a student while solving a math problem."})
        message.append({"role": "system", "content": "Types:"})
        for challenge in self.config['challenge_description'].keys(): 
            message.append({"role": "system", "content": f"{challenge}: {self.config['challenge_description'][challenge][0]}."})
        message.append({"role": "user", "content": f"Math problem: {input['question']}"})
        message.append({"role": "user", "content": f"The correct solution is as follows: {input['ground_truth']}"})
        message.append({"role": "user", "content": f"Student's answer: {input['student_incorrect_solution']}"})
        message.append({"role": "user", "content": f"Choose the main challenge type in this problem."})
        message.append({"role": "user", "content": f"Challenge type: "})
        return message
    
    def file(self, dataset, outputpath, n_sample):
        if n_sample > len(self.dataset):
            n_sample = len(self.dataset)
            print("n_sample is too big and has been set to max value :", n_sample)
        with open(self.output_path, 'w') as write_file:
            for i in tqdm(range(n_sample)):
                input = self.dataset[i]
                challenge = self.model.challenge(input)
                input['challenge'] = challenge
                questions = {}
                for q_type in self.config["class_description"]:
                    question = self.model.question(input, q_type)
                    questions[q_type] = question
                input['generated_questions'] = questions
                json_object = json.dumps(input)
                write_file.write(json_object) 
                write_file.write('\n')

    def question_file(self, dataset, output_path, n_sample):
        output_path = f"Users/arthur/Documents/ETH/Master_Thesis/data/{output_path}.jsonl"
        if n_sample > len(dataset):
            n_sample = len(dataset)
            print("n_sample is too big and has been set to max value :", n_sample)
        with open(output_path, 'w') as write_file:
            for i in tqdm(range(n_sample)):
                input = dataset[i]
                questions = {}
                for q_type in self.config['class_description'].keys():
                    question = self.question(input, q_type)
                    questions[q_type] = question
                input['generated_questions'] = questions
                json_object = json.dumps(input)
                write_file.write(json_object) 
                write_file.write('\n')
        
    def verify_file(self, dataset, output_path):
        output_path = f"Users/arthur/Documents/ETH/Master_Thesis/data/{output_path}.jsonl"
        with open(output_path, 'w') as output_file:
            for input in tqdm(dataset):
                delete_question = []
                for q_type in input['generated_questions']:
                    if self.verify(input, q_type) == 0:
                        print(f"A {q_type} was removed from the problem: {input['question']}")
                        print(f"The question is {input['generated_questions'][q_type]}.\n")
                        delete_question.append(q_type)
                for q_type in delete_question:
                    del input['generated_questions'][q_type]
                json_object = json.dumps(input)
                output_file.write(json_object) 
                output_file.write('\n')

    def challenge_file(self, dataset, output_path, n_sample, start):
        output_path = f"Users/arthur/Documents/ETH/Master_Thesis/data/{output_path}.jsonl"
        if n_sample > len(dataset):
            n_sample = len(dataset)
            print("n_sample is too big and has been set to max value :", n_sample)
        with open(output_path, 'w') as write_file:
            for i in tqdm(range(start, start+n_sample)):
                input = dataset[i]
                problem = input['question']
                input['challenge'] = self.challenge(input)
                json_object = json.dumps(input)
                write_file.write(json_object) 
                write_file.write('\n')

    def question_ex(self, dataset, q_type):
        n = random.randint(0,len(dataset)-1)
        input = dataset[n]
        print(input)
        return self.question(input, q_type)
    
    def verify_ex(self,  dataset, q_type):
        n = random.randint(0,len(dataset)-1)
        input = dataset[n]
        print(input)
        return self.verify_question(input, q_type)
    
    def challenge_ex(self, dataset):
        n = random.randint(0,len(dataset)-1)
        input = dataset[n]
        print(input)
        return self.challenge(input)
    
    def all(self, dataset, output_path, n_sample, start):
        output_path = f"Users/arthur/Documents/ETH/Master_Thesis/data/{output_path}.jsonl"
        if n_sample > len(dataset):
            n_sample = len(dataset)
            print("n_sample is too big and has been set to max value :", n_sample)
        with open(output_path, 'w') as write_file:
            for i in tqdm(range(start, start+n_sample)):
                input = dataset[i]
                problem = input['question']
                questions = {}
                for q_type in self.config['class_description'].keys(): 
                    question = self.question(input, q_type)
                    questions[q_type] = question
                input['generated_questions'] = questions
                input['challenge'] = self.challenge(input)
                json_object = json.dumps(input)
                write_file.write(json_object) 
                write_file.write('\n')

    def dialogue_answer(self, input):
        """
        Input : math problem + groundtruth + student answer
        Ouput : Socratic question
        """
        challenge = self.challenge(input)
        print(challenge)
        q_type = random.choice(self.config['challenge_description'][challenge][-1])
        print(q_type)
        question = self.question(input, q_type)
        return question

def translate_prompt(prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_string = f"{B_INST} " 
    role = ""
    for message in prompt:
        token = B_SYS if message["role"] == "system" and role == "" else (E_SYS if message["role"] == "user" and role == "system" else "")
        #end_token = E_SYS if message["role"] == "user" and role == "system" else ""
        role = message["role"]
        dialog_string += f"{token}{message['content'].strip()}\n"
    dialog_string += E_INST
    return dialog_string

def get_history(input):
    if 'student_incorrect_solution' in input: return f"Student answer: {input['student_incorrect_solution']}"
    return ""

def get_challenge(input):
    #TODO
    if 'challenge' in input: return f"{input['challenge']}"
    return ""

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()