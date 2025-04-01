import gradio as gr
import torch
import numpy as np
import tempfile
import os
import logging
from model_manager import ensure_model_folder, download_model, get_model_path
from llama_cpp import Llama
import time
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_FOLDER = "models"
MODEL_REPO_ID = "canopylabs/orpheus-3b-0.1-ft"
MODEL_FILENAME = "model.gguf"
MAX_TOKENS = 1000
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000

# Sample prompts
SAMPLE_PROMPTS = {
    "Professional": "Welcome to our quarterly business meeting. Today, we'll be discussing our Q1 performance and strategic initiatives for the upcoming quarter.",
    "Casual": "Hey there! How's it going? I'm just checking in to see how you're doing with the project.",
    "Emotive": "<giggle>That's hilarious!</giggle> I can't believe what just happened. <laugh>This is the funniest thing I've seen all day!</laugh>",
    "Story": "Once upon a time, in a distant galaxy, there was a brave explorer who discovered a mysterious planet.",
    "Technical": "The quantum computing system utilizes superconducting qubits to perform complex calculations at unprecedented speeds.",
    "Poetry": "The stars above, they twinkle bright, casting shadows in the night. A gentle breeze, a soft moonlight, making everything feel right.",
    "News": "Breaking news: Scientists have made a groundbreaking discovery in renewable energy technology.",
    "Educational": "Today, we'll be learning about the fascinating world of marine biology and its impact on our ecosystem."
}

# Initialize model
def init_model():
    try:
        model_path = get_model_path()
        logger.info(f"Initializing model from {model_path}")
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0  # Set to -1 for all layers on GPU
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

# Get system metrics
def get_system_metrics():
    metrics = {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{psutil.virtual_memory().percent}%",
        "Disk Usage": f"{psutil.disk_usage('/').percent}%"
    }

    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics["GPU Usage"] = f"{gpus[0].load*100:.1f}%"
            metrics["GPU Memory"] = f"{gpus[0].memoryUsed}/{gpus[0].memoryTotal} MB"
    except:
        metrics["GPU Usage"] = "N/A"
        metrics["GPU Memory"] = "N/A"

    return metrics

# Text to speech function
def text_to_speech(text, voice, temperature, top_p, repetition_penalty):
    try:
        # Get system metrics before generation
        start_metrics = get_system_metrics()
        start_time = time.time()

        # Format prompt
        prompt = f"<|im_start|>system\nYou are a helpful assistant that converts text to speech using the {voice} voice.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"

        # Generate tokens
        response = model(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stop=["<|im_end|>"],
            echo=False
        )

        # Get system metrics after generation
        end_metrics = get_system_metrics()
        end_time = time.time()
        generation_time = end_time - start_time

        # Extract audio data from response
        audio_data = response['choices'][0]['text'].strip()

        # Process audio data
        try:
            # Convert token strings to numeric IDs
            audio_ids = [int(token.split('_')[-1]) for token in audio_data.split()]
            audio_array = np.array(audio_ids, dtype=np.float32)

            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                import wave
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())

                # Create metrics summary
                metrics_summary = f"""
                Generation Time: {generation_time:.2f}s
                CPU Usage: {start_metrics['CPU Usage']} → {end_metrics['CPU Usage']}
                Memory Usage: {start_metrics['Memory Usage']} → {end_metrics['Memory Usage']}
                GPU Usage: {end_metrics['GPU Usage']}
                GPU Memory: {end_metrics['GPU Memory']}
                """

                return temp_file.name, metrics_summary

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            raise

    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        raise

# Initialize model
model = init_model()

# Create Gradio interface
with gr.Blocks(title="Orpheus TTS Local", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎧 Orpheus TTS Local")
    gr.Markdown("Generate high-quality speech from text using the Orpheus TTS model.")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                text_input = gr.Textbox(
                    label="Enter text to convert to speech",
                    placeholder="Type your text here...",
                    lines=3
                )
                sample_prompts = gr.Dropdown(
                    choices=list(SAMPLE_PROMPTS.keys()),
                    label="Sample Prompts",
                    info="Select a sample prompt to try out"
                )

            voice_dropdown = gr.Dropdown(
                choices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
                value="tara",
                label="Select Voice"
            )

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top P"
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.1,
                    label="Repetition Penalty"
                )

            generate_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio")
            metrics_output = gr.Textbox(
                label="Generation Metrics",
                lines=6,
                interactive=False
            )

    gr.Markdown("""
    ### Emotion Tags
    Add emotion to your speech using these tags:
    ```xml
    <giggle>That's funny!</giggle>
    <laugh>That's hilarious!</laugh>
    <sigh>I'm tired</sigh>
    <gasp>Look at that!</gasp>
    ```
    """)

    # Connect interface components
    generate_btn.click(
        fn=text_to_speech,
        inputs=[text_input, voice_dropdown, temperature, top_p, repetition_penalty],
        outputs=[audio_output, metrics_output]
    )

    # Connect sample prompts dropdown
    sample_prompts.change(
        fn=lambda x: SAMPLE_PROMPTS.get(x, ""),
        inputs=[sample_prompts],
        outputs=[text_input]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True,             # Create public URL
        inbrowser=True          # Auto-open in browser
    )