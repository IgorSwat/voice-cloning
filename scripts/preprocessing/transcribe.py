import argparse
import torch
from transformers import pipeline

def transcribe(input_file, language):
    """
    Transcribes the given audio file using Whisper Large Turbo, matching omnivoice.py implementation.
    """
    print(f"Transcribing {input_file} in {language}...")

    # Determine device and dtype as in omnivoice.py
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    # Initialize the pipeline exactly like omnivoice.py's load_asr_model
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )

    # Transcribe with the specified language and handle long-form audio
    result = asr_pipe(
        input_file,
        chunk_length_s=30,
        generate_kwargs={"language": language}
    )

    return result["text"].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using MLX Whisper Large")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--lang", type=str, default="en", help="Language for transcription (e.g., 'en', 'pl')")
    
    args = parser.parse_args()
    
    try:
        transcription = transcribe(args.input, args.lang)
        print("\nTranscription Result:")
        print("-" * 20)
        print(transcription)
        print("-" * 20)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
