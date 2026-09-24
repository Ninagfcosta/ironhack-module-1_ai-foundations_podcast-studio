import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("sk proj-9Bl12JnrNjSv5mJb9BccxvKGoMw0qmV-3GDxfLP73Xn77utG05iftqYKdX8Qiw_irrGtbmUQqTT3BlbkFJ6pTChisMd6Ktth6oBM_TPgZRb6stWgcwhnxTkjv_MSBVSYuiQRXId_Xa7svcUWIVpwu6qhxuIA" \
""))

def generate_audio(script, output_path="podcast_output.mp3"):
    if not script:
        return None

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=script
    )

    response.stream_to_file(output_path)
    return output_path