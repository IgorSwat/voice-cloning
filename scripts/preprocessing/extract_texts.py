import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract texts from a file with ID|Text pairs.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (id|text)")
    parser.add_argument("--output", type=str, required=True, help="Path to the output .txt file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    extracted_count = 0
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        # Skip header
        next(f_in, None)
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Try splitting by pipe first, then fall back to whitespace
            if "|" in line:
                parts = line.split("|", 1)
            else:
                parts = line.split(None, 1)
                
            if len(parts) == 2:
                text = parts[1].strip()
                if text:
                    f_out.write(text + "\n")
                    extracted_count += 1
            else:
                print(f"Warning: Skipping malformed line: {line}")

    print(f"Successfully extracted {extracted_count} texts to {args.output}")

if __name__ == "__main__":
    main()
