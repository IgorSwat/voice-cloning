import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from tools.phonemes.phonemizer import Phonemizer

def main():
    parser = argparse.ArgumentParser(description="Visualize phoneme distribution in a text dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .txt file (one text per line)")
    parser.add_argument("--lang", type=str, default="en", help="Language code for phonemization (default: en)")
    parser.add_argument("--output", type=str, help="Path to save the histogram plot (e.g., plot.png). If not set, shows the plot.")
    parser.add_argument("--custom-alpha", type=float, help="Custom alpha parameter for Beta distribution comparison.")
    parser.add_argument("--custom-beta", type=float, help="Custom beta parameter for Beta distribution comparison.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    phonemizer = Phonemizer(args.lang)
    phoneme_counts = []

    print(f"Processing {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            text = line.strip()
            if not text:
                continue
            
            phonemes = phonemizer.phonemize(text)
            phoneme_counts.append(len(phonemes))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total_lines} lines...")

    if not phoneme_counts:
        print("No valid texts found to visualize.")
        return

    # Fit Beta distribution
    # beta.fit returns (a, b, loc, scale)
    a, b, loc, scale = beta.fit(phoneme_counts)
    print(f"Beta fit parameters: alpha={a:.4f}, beta={b:.4f}, loc={loc:.4f}, scale={scale:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Using density=True to overlay PDF
    plt.hist(phoneme_counts, bins=range(min(phoneme_counts), max(phoneme_counts) + 2), 
             edgecolor='black', alpha=0.5, density=True, label="Actual distribution")
    
    # Plot Beta PDF
    x = np.linspace(min(phoneme_counts), max(phoneme_counts), 100)
    y = beta.pdf(x, a, b, loc, scale)
    plt.plot(x, y, 'r-', lw=2, label=f'Best-fit Beta (α={a:.2f}, β={b:.2f})')

    # Plot Custom Beta PDF if parameters are provided
    if args.custom_alpha is not None and args.custom_beta is not None:
        y_custom = beta.pdf(x, args.custom_alpha, args.custom_beta, loc, scale)
        plt.plot(x, y_custom, 'b--', lw=2, label=f'Custom Beta (α={args.custom_alpha:.2f}, β={args.custom_beta:.2f})')

    plt.title(f"Phoneme Count Distribution (Lang: {args.lang})")
    plt.xlabel("Number of Phonemes")
    plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Statistical labels
    avg_len = sum(phoneme_counts) / len(phoneme_counts)
    plt.axvline(avg_len, color='green', linestyle='dashed', linewidth=1, label=f'Avg: {avg_len:.2f}')
    plt.legend()

    if args.output:
        plt.savefig(args.output)
        print(f"Histogram saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
