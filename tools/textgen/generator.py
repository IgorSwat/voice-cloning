import yaml
import time
import random
import os
import re
import logging
from typing import List, Dict, Any

import google.generativeai as genai
from tqdm import tqdm

from tools.textgen.preset import (
    StatementPreset, 
    QuestionPreset, 
    ExclamationPreset, 
    MultiSentencePreset,
    Preset
)

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Generator Class
# =============================================================================
class Generator:
    """
    Handles automated text generation using Gemini LLM based on configured presets.
    
    This class manages the full lifecycle of text generation:
    1. Loading and validating configuration.
    2. Initializing specific generation presets (Statements, Questions, etc.).
    3. Executing batch requests with randomized types and rate-limiting.
    4. Cleaning and persisting output to disk.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Generator with a path to a YAML configuration file.
        
        Args:
            config_path (str): Absolute or relative path to the config.yaml file.
        """
        self.config_path = config_path
        
        # internal state
        self.config = {}
        self.presets: Dict[str, Preset] = {}
        self.styled_presets: Dict[str, Preset] = {}
        self.probs: Dict[str, float] = {}
        
        # 1. Configuration Loading
        self._load_config()
        
        # 2. Preset Initialization
        self._setup_presets()
        
        # 3. Validation
        self._validate_probabilities()
        
        # 4. LLM Backend Setup
        self._setup_llm()

    # -------------------------------------------------------------------------
    # Private Setup Methods
    # -------------------------------------------------------------------------

    def _load_config(self):
        """Parses the YAML configuration and extracts core settings."""
        logger.info(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = self.config.get("api_key")
        self.language = self.config["language"]
        self.batch_size = self.config["batch_size"]
        self.timeout = self.config["timeout"]
        self.output_paths = self.config["output_paths"]
        self.word_limits = self.config.get("word_limits", {})
        
        # Style handling
        style_list = self.config.get("style", [])
        if style_list:
            self.style_str = "\n".join([f"- {s}" for s in style_list])
        else:
            self.style_str = None
        self.style_prob = self.config.get("style_prob", 0.0)

    def _setup_presets(self):
        """Initializes all Preset objects and their corresponding selection weights."""
        probs_conf = self.config["probabilities"]
        
        # Helper to initialize unstyled and styled presets
        def add_preset(key, preset_class, max_len_key):
            max_len = self.word_limits.get(max_len_key)
            self.presets[key] = preset_class(self.language, self.batch_size, max_len)
            self.styled_presets[key] = preset_class(self.language, self.batch_size, max_len, style=self.style_str)

        # Register Single-Sentence Types
        add_preset("statements", StatementPreset, "single_sentence")
        self.probs["statements"] = probs_conf["statements"]
        
        add_preset("questions", QuestionPreset, "single_sentence")
        self.probs["questions"] = probs_conf["questions"]
        
        add_preset("exclamations", ExclamationPreset, "single_sentence")
        self.probs["exclamations"] = probs_conf["exclamations"]
        
        # Register Multi-Sentence Type (Consolidated)
        add_preset("multi_sentence", MultiSentencePreset, "multi_sentence")
        ms_conf = probs_conf["multi_sentence"]
        if isinstance(ms_conf, dict):
            self.probs["multi_sentence"] = sum(ms_conf.values())
        else:
            self.probs["multi_sentence"] = ms_conf

    def _validate_probabilities(self):
        """Ensures that the sum of all generation probabilities is exactly 1.0."""
        total_prob = sum(self.probs.values())
        if not (0.999 <= total_prob <= 1.001):
            raise ValueError(f"Probabilities must sum to 1.0, current sum: {total_prob:.4f}")
        logger.info("Generation probabilities validated successfully.")

    def _setup_llm(self):
        """Configures the Google Generative AI client."""
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No API key found in config or environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _clean_response(self, text: str) -> List[str]:
        """
        Cleans LLM response text into a flat list of items.
        Removes numbering, empty lines, and leading/trailing whitespace.
        """
        lines = text.strip().split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove leading numbering (e.g., "1. ", "1) ", "- ")
            line = re.sub(r'^(\d+[\.\)]|[\*\-])\s*', '', line)
            
            if line:
                cleaned.append(line)
        return cleaned

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(self, size: int):
        """
        Starts the generation process until at least `size` elements are produced.
        
        Args:
            size (int): Targeted total number of generated items across all classes.
        """
        num_requests = max(1, size // self.batch_size)
        logger.info(f"Targeting ~{size} elements via {num_requests} requests.")
        
        preset_keys = list(self.probs.keys())
        weights = [self.probs[k] for k in preset_keys]

        for i in tqdm(range(num_requests), desc="Generating text"):
            # 1. Choose Generation Type
            choice = random.choices(preset_keys, weights=weights, k=1)[0]
            
            # Determine if this request should be styled
            is_styled = random.random() < self.style_prob and self.style_str
            preset = self.styled_presets[choice] if is_styled else self.presets[choice]
            
            # 2. Determine Output Destination
            path = self.output_paths[choice]
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # 3. Execute LLM Request
            try:
                print(preset.prompt)
                response = self.model.generate_content(preset.prompt)
                if not response.text:
                    logger.warning(f"Skipping empty response for: {choice}")
                    continue
                
                # 4. Process and Persist Data
                elements = self._clean_response(response.text)
                with open(path, "a", encoding="utf-8") as f:
                    for item in elements:
                        f.write(item + "\n")
                
            except Exception as e:
                logger.error(f"Failed to generate for {choice}: {e}")

            # 5. Rate Limiting
            if i < num_requests - 1:
                time.sleep(self.timeout)

        logger.info("Generation cycle completed.")
