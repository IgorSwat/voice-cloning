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

1. **Setup Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate Voice Clones**:
   ```bash
   python -m scripts.clone --config configuration/config_polish.yaml --output output/my_clones --device mps
   ```

## Configuration

The generation behavior is controlled via YAML files. See `configuration/config_polish.yaml` for an example of how to set:
- `language`: Target language (e.g., "pl", "de", "en").
- `audio_ref`: Path to the speaker's reference audio.
- `data_config`: Parameters for filtering the input text file.
- `generation_config`: Model hyperparameters like `batch_size`, `speed`, and `num_steps`.
