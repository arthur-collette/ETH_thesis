from base_class import Model
#from definition import *
from functions import *
from util import initialize_config
from openai import OpenAI

import openai
import time

class GPT(Model):
    def __init__(self, name):
        self.config = initialize_config("/Users/arthur/Documents/ETH/Master_Thesis/python/experiments.conf")
        self.name = name
        self.key = self.config['key']
        self.model = self.config[name]
        #self.client = OpenAI(api_key=self.key)
        
    def generic_gpt(self, prompt):
        question = self.client.chat.completions.create(
            model=self.model, 
            messages=prompt,
        )
        return question.choices[0].message.content #['choices'][0]['message']['content']
    
    def __call__(self, prompt):
        return self.generic_gpt(prompt)

    def __call__1(self, prompt):
        """
        Get a gpt_based function and make it run while dealing with all the possible errors
        """
        try: return self.generic_gpt(prompt)
        
        except openai.error.RateLimitError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generic_gpt(prompt)

        except openai.error.APIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"API error occurred. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generic_gpt(prompt)

        except openai.error.ServiceUnavailableError as e:
            retry_time = 10  # Adjust the retry time as needed
            print(f"Service is unavailable. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generic_gpt(prompt)

        except openai.error.Timeout as e:
            retry_time = 10  # Adjust the retry time as needed
            print(f"Request timed out: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generic_gpt(prompt)

        except OSError as e:
            if isinstance(e, tuple) and len(e) == 2 and isinstance(e[1], OSError):
                retry_time = 10  # Adjust the retry time as needed
                print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                return self.generic_gpt(prompt)
            else:
                retry_time = 10  # Adjust the retry time as needed
                print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                return self.generic_gpt(prompt)
    
    