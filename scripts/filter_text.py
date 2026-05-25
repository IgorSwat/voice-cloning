import argparse
import os
import re
import random

def filter_sentences(input_path, output_path, target_size=None):
    """
    Filters sentences based on character composition and length constraints.
    Optionally selects a random subset of size target_size.
    """
    # Regex for Latin alphabetical characters + common special symbols (. , ! ? ; : - ( ))
    # Also including whitespace \s
    allowed_chars_pattern = re.compile(r'^[A-Za-z\s.,!?;:\-\(\)]+$')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_count = 0
    valid_sentences = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            
            # Expecting structure: ID <tab/whitespace> Sentence
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
                
            sentence = parts[1].strip()
            length = len(sentence)
            
            # Check length conditions:
            # - Between 5 and 20 characters
            # - Between 60 and 90 characters
            length_valid = (5 <= length <= 20)
            # length_valid = (5 <= length <= 20) or \
            #               (60 <= length <= 90)
            
            if not length_valid:
                continue
                
            # Check character conditions
            if allowed_chars_pattern.match(sentence):
                valid_sentences.append(sentence)

    # Apply random sampling if target_size is specified
    if target_size is not None and target_size < len(valid_sentences):
        print(f"Selecting {target_size} random sentences from {len(valid_sentences)} valid candidates.")
        valid_sentences = random.sample(valid_sentences, target_size)
    elif target_size is not None:
        print(f"Target size {target_size} is greater than or equal to candidate count {len(valid_sentences)}. Keeping all.")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for s in valid_sentences:
            outfile.write(s + '\n')

    print(f"Processing complete.")
    print(f"Total lines processed: {total_count}")
    print(f"Sentences written to output: {len(valid_sentences)}")
    print(f"Total valid candidates found: {len(valid_sentences) if target_size is None else 'N/A'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter text file based on character set and length constraints.")
    parser.add_argument("--input", required=True, help="Path to the input text file (ID Sentence format).")
    parser.add_argument("--output", required=True, help="Path to the output file.")
    parser.add_argument("--target-size", type=int, help="Number of random sentences to select from the filtered results.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
    else:
        filter_sentences(args.input, args.output, args.target_size)
