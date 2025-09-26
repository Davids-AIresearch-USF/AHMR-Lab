import os
import openai
from dotenv import load_dotenv

load_dotenv("./.env")
openai.api_key = os.environ.get("OPENAI_API_KEY")


client = openai.Client()

def get_gpt_output(model, msg, response_format=None):

    while True:
        try:

            data = {
                "model": model,
                "messages": msg,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
            }

            if (response_format != None):
                data["response_format"] = response_format

            response = client.chat.completions.create(**data)

            return response.choices[0].message.content
        except openai.OpenAIError as e:
            raise e
        