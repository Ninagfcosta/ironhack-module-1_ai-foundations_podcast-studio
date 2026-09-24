import gradio as gr
from data_processor import process_data
from llm_processor import transform_to_podcast_script
from tts_generator import generate_audio

def create_podcast(text_input):
    text = process_data("text", text_input)
    if not text:
        return None, "Error: Please enter some text."
    script = transform_to_podcast_script(text)
    if not script:
        return None, "Error: Could not generate script."
    audio = generate_audio(script)
    if not audio:
        return None, "Error: Could not generate audio."
    return audio, script

with gr.Blocks(title="Podcast Studio") as demo:
    gr.Markdown("# 🎙️ Podcast Studio")
    text_input = gr.Textbox(label="Paste your text here", lines=6)
    generate_btn = gr.Button("Generate Podcast", variant="primary")
    audio_output = gr.Audio(label="Your Podcast")
    script_output = gr.Textbox(label="Generated Script", lines=8)
    generate_btn.click(
        fn=create_podcast,
        inputs=[text_input],
        outputs=[audio_output, script_output]
    )

if __name__ == "__main__":
    demo.launch()
