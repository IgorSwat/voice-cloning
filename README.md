# OmniVoice Voice Cloning

This repository provides a set of tools and scripts for zero-shot voice cloning using the [OmniVoice](https://github.com/k2-fsa/OmniVoice) model. It allows you to generate high-quality speech in various languages (including Polish and German) by cloning a voice from a short reference audio clip.

## Project Structure

- `configuration/`: YAML files containing generation parameters, language settings, and reference audio paths.
- `data/`:
  - `ref/`: Reference audio clips and transcripts for voice cloning.
  - `text/`: Text datasets used for bulk voice generation.
- `scripts/`:
  - `clone.py`: The main script for generating voice clones in batches.
  - `preprocessing/`: Scripts for data preparation (`extract_texts.py`, `visualize_data.py`).
- `tools/`: Internal utilities, including a custom phonemizer with language-specific fixups.

## Features

- **Zero-Shot Voice Cloning**: Clone any voice using just a few seconds of reference audio.
- **Batch Processing**: Efficiently generate large quantities of audio from text files.
- **Phoneme Filtering**: Filter input text based on phoneme length to ensure consistent generation quality.
- **Data Visualization**: Visualize the phoneme distribution of your dataset before generation.

## Quick Start

### 1. Prepare Text
Create a `.txt` file in `data/text/` with one sentence per line.
Example `data/text/my_text.txt`:
```text
Hello world.
This is a test.
```

### 2. Configure
Create a `.yaml` file in `configuration/`.
Example `configuration/my_config.yaml`:
```yaml
language: "en"
audio_ref: "data/ref/speaker.wav"
text_ref: "The text spoken in the speaker.wav file." # Optional: if omitted, Whisper will auto-transcribe.
data_config:
  file: "data/text/my_text.txt"
generation_config:
  batch_size: 4
```



### 3. Run
```bash
python scripts/clone.py \
  --config configuration/my_config.yaml \
  --output data/output/my_run \
  --device mps
```
Add `--respect-phonemes` if you want to filter lines by length (defined in config).
