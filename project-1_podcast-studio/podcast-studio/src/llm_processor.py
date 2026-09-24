import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transform_to_podcast_script(text):
    if not text:
        return None
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Rewrite this as a friendly 1-minute podcast script:\n\n{text}"
        }],
        max_tokens=500
    )
    return response.choices[0].message.content
