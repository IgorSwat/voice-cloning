import argparse
import os
import matplotlib.pyplot as plt
from tools.phonemes.phonemizer import Phonemizer

def main():
    parser = argparse.ArgumentParser(description="Visualize phoneme distribution in a text dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .txt file (one text per line)")
    parser.add_argument("--lang", type=str, default="en", help="Language code for phonemization (default: en)")
    parser.add_argument("--output", type=str, help="Path to save the histogram plot (e.g., plot.png). If not set, shows the plot.")

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

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(phoneme_counts, bins=range(min(phoneme_counts), max(phoneme_counts) + 2), edgecolor='black', alpha=0.7)
    plt.title(f"Phoneme Count Distribution (Lang: {args.lang})")
    plt.xlabel("Number of Phonemes")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Statistical labels
    avg_len = sum(phoneme_counts) / len(phoneme_counts)
    plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {avg_len:.2f}')
    plt.legend()

    if args.output:
        plt.savefig(args.output)
        print(f"Histogram saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
