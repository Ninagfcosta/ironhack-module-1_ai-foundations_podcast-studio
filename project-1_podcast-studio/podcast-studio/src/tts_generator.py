import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
