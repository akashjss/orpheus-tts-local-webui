import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

MODEL_FOLDER = "models"
REPO_ID = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
FILENAME = "orpheus-3b-0.1-ft-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_FOLDER, FILENAME)

def ensure_model_folder():
    """Create the model folder if it doesn't exist."""
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

def download_model(force=False):
    """Download the model from Hugging Face Hub if it doesn't exist or force is True."""
    ensure_model_folder()
    
    if os.path.exists(MODEL_PATH) and not force:
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH
    
    print(f"Downloading model from Hugging Face Hub...")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Use huggingface_hub to download the model
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODEL_FOLDER,
            force_download=force,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        print(f"Model downloaded successfully to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise

def get_model_path():
    """Get the path to the model file, downloading it if necessary."""
    if not os.path.exists(MODEL_PATH):
        return download_model()
    return MODEL_PATH

if __name__ == "__main__":
    # Test the model download
    try:
        model_path = get_model_path()
        print(f"Model is available at: {model_path}")
    except Exception as e:
        print(f"Failed to download model: {str(e)}") 