import json
from openai import OpenAI
import re
import os
import ast
from dotenv import load_dotenv

class OpenAIModel:
    def __init__(self, model_name, prompt_system):
        # load API key
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("API key not found.")

        # configs
        self.model_name = model_name
        self.prompt_system = prompt_system

        self.fallback = -1.

    def format_prompt(self, input: str) -> str:
        return f"{self.prompt_prefix}{input}{self.prompt_suffix}"

    def pred(self, input, temperature = 0.0, max_tokens = 512):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": input},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        output = response.choices[0].message.content
        try:
            return ast.literal_eval(output)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse LLM response as a dictionary: {e}")
