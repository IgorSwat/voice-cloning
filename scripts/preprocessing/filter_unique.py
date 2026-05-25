import argparse
import os
from difflib import SequenceMatcher
from tqdm import tqdm

def get_similarity(a, b):
    """
    Returns the similarity ratio between two strings using SequenceMatcher.
    """
    return SequenceMatcher(None, a, b).ratio()

def reduce_similarities(input_path, output_path, threshold=0.8, window_size=50):
    """
    Reduces similarities by filtering out sentences that are too similar to already kept ones.
    
    Args:
        input_path (str): Path to input text file.
        output_path (str): Path to output text file.
        threshold (float): Similarity threshold (0 to 1). Higher means more strict (only near-identical removed).
        window_size (int): How many previously kept sentences to compare against. Larger is more thorough but slower.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kept_sentences = []
    removed_count = 0

    print(f"Processing similarities (Threshold: {threshold}, Window Size: {window_size})...")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for sentence in tqdm(lines, desc="Filtering similarities"):
        is_similar = False
        
        # Compare against a window of recently kept sentences to optimize speed
        # For a small/medium dataset, we could compare against all, but for scale, a window is better
        comparison_subset = kept_sentences[-window_size:] if window_size > 0 else kept_sentences
        
        for existing in reversed(comparison_subset):
            if get_similarity(sentence, existing) > threshold:
                is_similar = True
                break
        
        if not is_similar:
            kept_sentences.append(sentence)
        else:
            removed_count += 1

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for s in kept_sentences:
            outfile.write(s + '\n')

    print(f"\nCompleted!")
    print(f"Total processed: {len(lines)}")
    print(f"Kept: {len(kept_sentences)}")
    print(f"Removed (highly similar): {removed_count}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce similarity by removing sentences with high lexical overlap.")
    parser.add_argument("--input", required=True, help="Input text file (one sentence per line).")
    parser.add_argument("--output", required=True, help="Output text file.")
    parser.add_argument("--threshold", type=float, default=0.85, 
                        help="Similarity threshold between 0.0 and 1.0. Higher values remove only near-duplicates. Default 0.85.")
    parser.add_argument("--window", type=int, default=100, 
                        help="Comparison window size. How many previous sentences to check against. 0 for all. Default 100.")

    args = parser.parse_args()
    reduce_similarities(args.input, args.output, args.threshold, args.window)
