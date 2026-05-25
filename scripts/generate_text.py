import argparse
import sys
import logging
from tools.textgen.generator import Generator

def main():
    parser = argparse.ArgumentParser(description="Automated text generation tool using Gemini.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        required=True, 
        help="Target number of samples to generate."
    )
    
    args = parser.parse_args()

    try:
        # Initialize the generator with the provided config
        generator = Generator(args.config)
        
        # Start the generation process
        generator.generate(args.samples)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
