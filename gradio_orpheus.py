import gradio as gr
import wave
import tempfile
import numpy as np
import torch
from model_manager import get_model_path
from llama_cpp import Llama
from snac import SNAC

# Model parameters
MAX_TOKENS = 1200
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Available voices
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Special token handling
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Initialize SNAC model
print("Initializing SNAC model...")
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {snac_device}")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)

def format_prompt(prompt, voice):
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using 'tara' instead.")
        voice = "tara"
        
    # Format with special tokens
    formatted_prompt = f"{voice}: {prompt}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    
    return f"{special_start}{formatted_prompt}{special_end}"

def turn_token_into_id(token_string, index):
    """Convert token string to numeric ID for audio processing."""
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]  # Remove <custom_token_ and >
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None

def convert_to_audio(multiframe):
    """Convert token frames to audio using SNAC model."""
    print(f"Converting {len(multiframe)} tokens to audio...")
    if len(multiframe) < 7:
        print("Error: Not enough tokens for audio conversion")
        return None
    
    # Prepare code tensors
    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    print(f"Processing {num_frames} frames...")

    for j in range(num_frames):
        i = 7*j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        
        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    
    # Validate token ranges
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or \
       torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or \
       torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        print("Error: Token values out of valid range")
        return None

    print("Decoding audio using SNAC model...")
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    print(f"Generated audio shape: {audio_int16.shape}")
    return audio_int16.tobytes()

def text_to_speech(text, voice, temperature=0.6, top_p=0.9, repetition_penalty=1.1):
    try:
        print(f"Generating speech for text: {text[:50]}...")
        print(f"Using voice: {voice}")
        
        # Initialize the model
        model = Llama(
            model_path=get_model_path(),
            n_ctx=2048,  # Context window
            n_threads=4   # Number of CPU threads to use
        )
        
        # Format the prompt with special tokens
        formatted_prompt = format_prompt(text, voice)
        print(f"Formatted prompt: {formatted_prompt}")
        
        # Generate completion
        print("Generating tokens...")
        result = model(
            formatted_prompt,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stop=["<|eot_id|>"],
            echo=False
        )
        
        # Extract the tokens from the response
        token_text = result['choices'][0]['text']
        print(f"Generated token text: {token_text[:100]}...")
        
        # Process tokens
        tokens = []
        count = 0
        audio_segments = []
        
        # Split the token text into individual tokens
        token_parts = token_text.split('>')
        for part in token_parts:
            if part.strip() and CUSTOM_TOKEN_PREFIX in part:
                # Add back the closing bracket that was removed by split
                token = part + '>'
                token_id = turn_token_into_id(token, count)
                if token_id is not None and token_id > 0:
                    tokens.append(token_id)
                    count += 1
                    print(f"Processed token {count}: {token} -> {token_id}")
                    
                    # Process tokens in chunks of 28 (like the original implementation)
                    if count % 7 == 0 and count > 27:
                        buffer_to_proc = tokens[-28:]
                        audio_data = convert_to_audio(buffer_to_proc)
                        if audio_data is not None:
                            audio_segments.append(audio_data)
        
        print(f"Processed {len(tokens)} valid tokens")
        if not tokens:
            return "Error: No valid tokens generated"
        
        if not audio_segments:
            return "Error: Failed to generate any audio segments"
        
        # Combine all audio segments
        print("Combining audio segments...")
        combined_audio = b''.join(audio_segments)
        
        # Create a temporary WAV file
        print("Saving audio to WAV file...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)  # 24kHz for SNAC model
                wav_file.writeframes(combined_audio)
            
            print(f"Audio saved to: {temp_file.name}")
            return temp_file.name
            
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Orpheus TTS") as demo:
    gr.Markdown("# Orpheus Text-to-Speech")
    gr.Markdown("Generate high-quality speech using the Orpheus TTS model.")
    gr.Markdown("""
    ### Setup Instructions:
    1. The application will automatically download the required models on first run
    2. This may take a few minutes depending on your internet connection
    3. Once downloaded, the models will be cached locally for future use
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Enter your text here...",
                lines=3
            )
            voice_dropdown = gr.Dropdown(
                choices=VOICES,
                value="tara",
                label="Voice"
            )
            
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
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
            
            generate_btn = gr.Button("Generate Speech")
        
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath"
            )
    
    generate_btn.click(
        fn=text_to_speech,
        inputs=[text_input, voice_dropdown, temperature, top_p, repetition_penalty],
        outputs=audio_output
    )

if __name__ == "__main__":
    # Ensure model is downloaded before starting the interface
    try:
        model_path = get_model_path()
        print(f"Using model at: {model_path}")
        print("Initializing models... This may take a few moments.")
        demo.launch()
    except Exception as e:
        print(f"Failed to initialize models: {str(e)}")
        print("Please check your internet connection and try again.") 