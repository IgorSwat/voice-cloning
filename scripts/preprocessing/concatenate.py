import argparse
import os

def concatenate_files(inputs, output):
    """
    Concatenates content from multiple input files into a single output file.
    Each unique line from the input files is written to the output file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    unique_sentences = set()
    total_lines = 0

    with open(output, 'w', encoding='utf-8') as outfile:
        for input_file in inputs:
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found. Skipping.")
                continue
            
            print(f"Processing: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    content = line.strip()
                    if content:
                        total_lines += 1
                        if content not in unique_sentences:
                            unique_sentences.add(content)
                            outfile.write(content + '\n')

    saved_count = total_lines - len(unique_sentences)
    print(f"Successfully concatenated files into: {output}")
    print(f"Total lines processed: {total_lines}")
    print(f"Unique lines: {len(unique_sentences)}")
    print(f"Duplicates removed (saved): {saved_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate multiple text files into one.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of input .txt files to concatenate.")
    parser.add_argument("--output", required=True, help="Path to the output file.")
    
    args = parser.parse_args()
    concatenate_files(args.inputs, args.output)
