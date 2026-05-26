#!/usr/bin/env python3
"""
Kokoro TTS: Dataset Preparation Pipeline
========================================
Standardizes a speech dataset for Kokoro training. 

Functionality:
1. Audio: Converts files to 24kHz, mono, s16 PCM in-place.
2. Text: Generates IPA phonemes using the custom Phonemizer.
3. Output: Generates metadata.csv and phonemes.csv for training.

Usage:
    uv run python scripts/prepare_dataset.py \
        --transcriptions path/to/text.txt \
        --audio-dir path/to/wavs \
        --lang pl
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from loguru import logger

# Add project root to sys.path to allow importing tools.phonemes
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

try:
    from tools.phonemes.phonemizer import Phonemizer
except ImportError:
    logger.error("Failed to import Phonemizer from tools.phonemes.phonemizer")
    raise


# ─── Audio Processing ─────────────────────────────────────────────────────────

def standardize_audio(wav_path: Path) -> bool:
    """
    Converts audio to 24kHz, mono, 16-bit PCM in-place using FFmpeg.
    Returns True if successful, False otherwise.
    """
    temp_wav = wav_path.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                str(temp_wav)
            ],
            check=True, capture_output=True
        )
        os.replace(temp_wav, wav_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed for {wav_path.name}: {e.stderr.decode().strip()}")
        if temp_wav.exists():
            temp_wav.unlink()
        return False


# ─── Text Processing ──────────────────────────────────────────────────────────

def load_transcriptions(path: Path) -> List[Tuple[str, str]]:
    """
    Loads transcriptions from a file.
    Expects format: <audio_name>|transcript
    Returns: List of (audio_name, transcript) tuples.
    """
    transcriptions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '|' not in line:
                logger.warning(f"Skipping line: Invalid format (missing '|'): {line}")
                continue
            audio_name, text = line.split('|', 1)
            transcriptions.append((audio_name.strip(), text.strip()))
    return transcriptions


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_pipeline(trans_path: Path, audio_dir: Path, lang: str, out_dir: Path):
    """Orchestrates the conversion and phonemization loop."""
    
    # 1. Initialize Tools
    logger.info(f"Initializing Phonemizer for language: {lang}")
    try:
        phonemizer = Phonemizer(lang_code=lang)
    except Exception as e:
        logger.critical(f"Failed to initialize Phonemizer: {e}")
        sys.exit(1)

    # 2. Load Data and Audio Files
    items = load_transcriptions(trans_path)
    
    if not items:
        logger.error("No valid transcriptions found.")
        return

    # 3. Create Output Paths
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"
    phone_path = out_dir / "phonemes.csv"

    # 4. Processing Loop
    out_meta = []
    out_phone = []
    error_count = 0

    logger.info(f"Processing {len(items)} segments...")
    for audio_name, text in tqdm(items):
        # Look for the audio file
        wav_path = audio_dir / audio_name
        
        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}. Skipping.")
            error_count += 1
            continue

        # Convert audio
        if not standardize_audio(wav_path):
            error_count += 1
            continue

        # Phonemize
        try:
            ps = phonemizer.phonemize(text)
        except Exception as e:
            logger.error(f"Phonemization failed for {audio_name}: {e}")
            error_count += 1
            continue

        # Store results
        # Using format compatible with StyleTTS2 (filename|text|speaker_id)
        out_meta.append(f"{audio_name}|{text}|0")
        out_phone.append(f"{audio_name}|{ps}")

    # 5. Save Outputs
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("filename|text|speaker\n")
        f.write("\n".join(out_meta) + "\n")
    
    with open(phone_path, "w", encoding="utf-8") as f:
        f.write("filename|ipa\n")
        f.write("\n".join(out_phone) + "\n")

    logger.success(f"Pipeline complete!")
    logger.info(f"  Total processed: {len(out_meta)}")
    logger.info(f"  Errors/Missing:  {error_count}")
    logger.info(f"  Output directory: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro TTS: Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument("--transcriptions", required=True, type=Path, help="Path to transcriptions.txt (ID <tab> TEXT)")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory containing audio files to process in-place")
    parser.add_argument("--lang", required=True, help="Language code for phonemization (e.g., 'pl', 'de')")
    
    # Optional Arguments
    parser.add_argument("--output-dir", default=Path("./dataset"), type=Path, help="Directory to save metadata.csv and phonemes.csv")

    args = parser.parse_args()

    # Validation
    if not args.transcriptions.exists():
        logger.error(f"Transcriptions file not found: {args.transcriptions}")
        sys.exit(1)
    if not args.audio_dir.is_dir():
        logger.error(f"Audio directory not found: {args.audio_dir}")
        sys.exit(1)

    try:
        run_pipeline(args.transcriptions, args.audio_dir, args.lang, args.output_dir)
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
