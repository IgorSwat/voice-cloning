import argparse
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta
from tqdm import tqdm
from tools.phonemes.phonemizer import Phonemizer

def sample_sentences(input_path, output_path, size, alpha, b_param, lang="en", info=False):
    """
    Samples sentences from input such that their phoneme length distribution 
    approximately matches a Beta distribution defined by alpha and beta.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    phonemizer = Phonemizer(lang)
    
    print(f"Reading and phonemizing sentences from {input_path}...")
    sentences_data = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # First pass: Get all sentences and their phoneme lengths
    for text in tqdm(lines, desc="Phonemizing"):
        phonemes = phonemizer.phonemize(text)
        sentences_data.append({
            "text": text,
            "len": len(phonemes)
        })

    if not sentences_data:
        print("No valid sentences found.")
        return

    # Extract lengths and determine range for Beta scaling
    lengths = np.array([d["len"] for d in sentences_data])
    min_len = lengths.min()
    max_len = lengths.max()
    
    # We use the same 'loc' and 'scale' logic as visualize_data
    # This maps the [0, 1] Beta domain to [min_len, max_len]
    loc = min_len
    scale = max_len - min_len

    print(f"Sampling {size} sentences matching Beta(α={alpha}, β={b_param})...")
    
    # Target distribution weights
    # We calculate the probability density of each sentence's length under the target Beta
    weights = []
    for d in sentences_data:
        # beta.pdf expects alpha, beta, loc, scale
        # If scale is 0 (all sentences same length), weight is 1
        if scale > 0:
            w = beta.pdf(d["len"], alpha, b_param, loc=loc, scale=scale)
        else:
            w = 1.0
        weights.append(w)

    weights = np.array(weights)
    
    if info:
        print("\n--- Info Mode ---")
        print(f"Dataset range: {min_len} to {max_len} phonemes")
        print(f"Non-zero weights: {np.count_nonzero(weights)} / {len(weights)}")
        print(f"Weight sum (pre-norm): {weights.sum()}")
        
        plt.figure(figsize=(12, 6))
        # Plot weights as a distribution over lengths
        sorted_indices = np.argsort(lengths)
        plt.plot(lengths[sorted_indices], weights[sorted_indices], 'r-', label='Sampling Weight (Beta PDF)')
        
        # Overlay actual data histogram
        plt.hist(lengths, bins=range(min_len, max_len + 2), density=True, alpha=0.3, label='Actual Data Density')
        
        plt.title(f"Target Distribution: Beta(α={alpha}, β={b_param})")
        plt.xlabel("Phoneme Length")
        plt.ylabel("Probability / Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        return

    # Normalize weights to sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        # Fallback to uniform if PDF is zero everywhere (e.g. out of bounds)
        print("Warning: Target distribution weights are all zero. Falling back to uniform sampling.")
        weights = None

    # Perform weighted sampling
    # We sample indices based on the calculated Beta weights
    indices = np.random.choice(len(sentences_data), size=min(size, len(sentences_data)), replace=False, p=weights)
    
    sampled_sentences = [sentences_data[i]["text"] for i in indices]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in sampled_sentences:
            f.write(s + "\n")

    print(f"Successfully saved {len(sampled_sentences)} sampled sentences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample sentences matching a target phoneme-length Beta distribution.")
    parser.add_argument("--input", required=True, help="Input text file.")
    parser.add_argument("--output", required=True, help="Output text file.")
    parser.add_argument("--size", type=int, required=True, help="Number of sentences to sample.")
    parser.add_argument("--alpha", type=float, required=True, help="Target alpha parameter for Beta distribution.")
    parser.add_argument("--beta", type=float, required=True, help="Target beta parameter for Beta distribution.")
    parser.add_argument("--lang", default="en", help="Language code for phonemization (default: en).")
    parser.add_argument("--info", action="store_true", help="Visualize weight distribution and stats without sampling.")

    args = parser.parse_args()
    
    sample_sentences(args.input, args.output, args.size, args.alpha, args.beta, args.lang, args.info)
