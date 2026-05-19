# 🎙️ Podcast Studio

Automated podcast generation from text input.

## How it works
1. User pastes text into the interface
2. OpenAI GPT rewrites it as a podcast script
3. OpenAI TTS converts the script to speech
4. Audio file is ready to play and download

## Setup
1. Clone this repository
2. Create virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with your `OPENAI_API_KEY`

## Run
cd src
python main.py
