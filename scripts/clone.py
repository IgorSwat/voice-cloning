from tools.phonemes.phonemizer import Phonemizer

import argparse
import os
import random
import soundfile as sf
import time
import torch
import yaml
from tqdm import tqdm

from omnivoice import OmniVoice


def main():
  # ----------------------
  # Read script input args
  # ----------------------

  parser = argparse.ArgumentParser(description="Zero-shot voice cloning with OmniVoice")
  parser.add_argument("--config", type=str, required=True, help="Path to a configuration .yaml file")
  parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., mps, cuda, cpu)")
  parser.add_argument("--output", type=str, required=True, help="Output directory")
  parser.add_argument("--samples", type=int, help="Limit the number of generated audios")

  args = parser.parse_args()
  config = args.config
  device = args.device
  output_dir = args.output

  # ------------------
  # Read configuration
  # ------------------

  with open(config, "r", encoding="utf-8") as f:
      config_data = yaml.safe_load(f)

  lang = config_data.get("language")
  audio_ref = config_data.get("audio_ref")
  text_ref = config_data.get("text_ref")

  data_config = config_data.get("data_config", {})
  data_file = data_config.get("file")
  min_phonemes = data_config.get("min_phonemes")
  max_phonemes = data_config.get("max_phonemes")

  gen_cfg_raw = config_data.get("generation_config", {})
  batch_size =  gen_cfg_raw.pop("batch_size", 1)
  speed = gen_cfg_raw.pop("speed", 1.0)
  gen_kwargs = {k: v for k, v in gen_cfg_raw.items() if v is not None}

  # ------------------------------
  # Read and filter data text file
  # ------------------------------

  if not os.path.exists(data_file):
      raise FileNotFoundError(f"Data file not found at: {data_file}")

  phonemizer = Phonemizer(lang)
  filtered_texts = []

  with open(data_file, "r", encoding="utf-8") as f:
      for line in f:
          text = line.strip()
          if not text:
              continue
          
          phonemes = phonemizer.phonemize(text)
          phoneme_count = len(phonemes)
          
          if min_phonemes <= phoneme_count <= max_phonemes:
              filtered_texts.append(text)

  # Shuffle data
  random.shuffle(filtered_texts)

  if args.samples:
      filtered_texts = filtered_texts[: args.samples]

  print(f"Filtered {len(filtered_texts)} lines from {data_file}")

  # ---------------------
  # Generate voice clones
  # ---------------------

  # Initialize the voice cloning model
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map=device,
      dtype=torch.float16
  )

  os.makedirs(output_dir, exist_ok=True)
  audio_output_dir = os.path.join(output_dir, "audio")
  os.makedirs(audio_output_dir, exist_ok=True)
  
  transcriptions_path = os.path.join(output_dir, "transcriptions.tsv")

  # Process in batches
  with open(transcriptions_path, "w", encoding="utf-8") as tsv_file:
      pbar = tqdm(total=len(filtered_texts), desc="Cloning progress")
      for i in range(0, len(filtered_texts), batch_size):
          batch_texts = filtered_texts[i : i + batch_size]
          
          start_time = time.time()
          audios = model.generate(
              text=batch_texts,
              language=lang,
              ref_audio=audio_ref,
              ref_text=text_ref,
              speed=speed,
              **gen_kwargs
          )
          end_time = time.time()
          
          # Save Each generated audio
          for j, audio in enumerate(audios):
              audio_name = f"clone_{i + j:04d}.wav"
              output_path = os.path.join(audio_output_dir, audio_name)
              sf.write(output_path, audio, model.sampling_rate)
              
              # Write transcription
              tsv_file.write(f"{audio_name}|{batch_texts[j]}\n")
          
          pbar.update(len(batch_texts))
      pbar.close()


if __name__ == "__main__":
    main()
