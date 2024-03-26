# pip install python-dotenv
from dotenv import load_dotenv
from openai import OpenAI
import os

# load .env
load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    api_key=API_KEY,
)

async def translate_to_english_gpt4(text):
    response = client.chat.completions.create(
#   model="gpt-4-turbo-preview",
    model="gpt-4-turbo-preview",
  messages=[
    {
      "role": "system",
      "content": 
'''1. 입력받은 문자열은 영어로 번역한다.
2. 다음 Json 형식으로 출력한다. : {"english_text" : "trnaslated text"}'''
    },
    
    {
      "role": "user",
      "content": text,
    },
  ],
  response_format={"type": "json_object"},
  temperature=1,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    textTran = response.choices[0].message.content
    return textTran

async def translate_to_korean_gpt4(text):
    response = client.chat.completions.create(
#   model="gpt-4-turbo-preview",
    model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": 
'''1. 입력받은 문자열은 경어체의 한글로 번역한다.
2. 다음 Json 형식으로 출력한다. : {"korean_text" : "trnaslated text"}'''
    },
    
    {
      "role": "user",
      "content": text,
    },
  ],
  response_format={"type": "json_object"},
  temperature=1,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    textTran = response.choices[0].message.content
    return textTran

