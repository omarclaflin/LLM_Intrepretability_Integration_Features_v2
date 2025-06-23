#max_token length parameter (default was 100) now using 10; uses all data; chunks up data (not just first parts of data)
# Enhanced version with pre-ReLU hooks and contrast pattern analysis
# python part5c_find_feature_meaning_contrast.py --sae_path ../checkpoints/best_model.pt --model_path ../models/open_llama_3b --features "32616" --output_dir ./NFM_feature_analysis_results_contrast/ --config_dir ../config --max_token_length 10 --num_samples 10000000 --claude_examples 20

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import random
from typing import List, Tuple, Dict, Any, Optional
import time
import datetime
import logging
import os
import requests
import re
import html
import asyncio
from datasets import load_dataset

K=1024

class SparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module - matches training implementation."""
    def __init__(self, input_dim, hidden_dim, k=K):
        super().__init__()
        self.k = k
        # Encoder maps input activation vector to sparse feature vector
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU() # ReLU enforces non-negativity for sparsity
        )
        # Decoder maps sparse feature vector back to original activation space
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        # x: input activation vector (batch_size, seq_len, input_dim)
        features = self.encoder(x) # features: (batch_size, seq_len, hidden_dim)
        
        # Apply TopK sparsity if k is specified
        if self.k is not None:
            sparse_features = self.apply_topk(features)
        else:
            sparse_features = features  # No top-k filtering
            
        # Reconstruction maps features back to the original activation space
        reconstruction = self.decoder(sparse_features) # reconstruction: (batch_size, seq_len, input_dim)

        return sparse_features, reconstruction # Return both the features and the reconstruction
    
    def forward_with_pre_relu(self, x):
        """Forward pass that also returns pre-ReLU activations."""
        # Get raw linear output before ReLU
        linear_output = self.encoder[0](x)  # Just the linear layer
        
        # Apply ReLU for normal features
        features = torch.nn.functional.relu(linear_output)
        
        # Apply TopK sparsity if needed
        if self.k is not None:
            sparse_features = self.apply_topk(features)
        else:
            sparse_features = features
            
        reconstruction = self.decoder(sparse_features)
        
        return sparse_features, reconstruction, linear_output  # Return pre-ReLU values too
    
    def apply_topk(self, features):
        """Apply TopK sparsity - keep only top K activations per sample"""
        batch_size, seq_len, num_features = features.shape
        
        # Reshape to (batch_size * seq_len, num_features) for topk operation
        features_flat = features.reshape(-1, num_features)
        
        # Get TopK values and indices for each sample
        topk_values, topk_indices = torch.topk(features_flat, self.k, dim=1)
        
        # Create sparse feature tensor
        sparse_features_flat = torch.zeros_like(features_flat)
        
        # Scatter the TopK values back to their original positions
        sparse_features_flat.scatter_(1, topk_indices, topk_values)
        
        # Reshape back to original shape
        sparse_features = sparse_features_flat.reshape(batch_size, seq_len, num_features)
        
        return sparse_features


class SAEFeatureAnalyzer:
    """Analyzer for studying features in a Sparse Autoencoder trained on language model activations."""

    def __init__(self, sae_model, tokenizer, base_model, device="cuda", target_layer=16,
                 identification_prompt=None, scoring_prompt=None, claude_api_key=None, max_token_length=100):
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.max_token_length = max_token_length
        self.sae_model.eval()
        self.base_model.eval()

        # Store feature statistics
        self.feature_stats = {}

        # Store prompts
        self.identification_prompt = identification_prompt

        # Claude API key
        self.claude_api_key = claude_api_key

    def find_highest_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10, use_pre_relu=False):
        """
        Find texts with highest token-level activations for a feature.
        
        Args:
            feature_idx: The feature to analyze
            texts: List of text samples
            top_n: Number of top examples to return
            batch_size: Number of texts to process in each batch
            window_size: Number of tokens before and after the highest activation to include
            use_pre_relu: If True, use pre-ReLU activations (can be negative)

        Returns:
            List of dicts with text, max_activation, max_position, and max_token
        """
        results = []

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}

        activation_type = "pre-ReLU" if use_pre_relu else "post-ReLU"
        print(f"Scanning for feature {feature_idx} to find top {top_n} examples ({activation_type}, batch size {batch_size}, window size {window_size}, max_token_length {self.max_token_length})...")

        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Highest Examples for Feature {feature_idx} ({activation_type})"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                if use_pre_relu:
                    _, _, pre_relu_features = self.sae_model.forward_with_pre_relu(hidden_states.to(self.sae_model.encoder[0].weight.dtype))
                    features = pre_relu_features
                else:
                    features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                for b in range(features.shape[0]):
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len == 0: continue

                    token_activations = features[b, :seq_len, feature_idx]
                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)

                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    for pos, token in enumerate(sample_tokens):
                        if token in special_tokens or sample_token_ids[pos] in self.tokenizer.all_special_ids:
                            token_mask[pos] = False

                    if not torch.any(token_mask):
                        continue

                    masked_activations = token_activations.clone()
                    masked_activations[~token_mask] = -1e9 if not use_pre_relu else -1e9

                    max_activation = torch.max(masked_activations).item()

                    # For pre-ReLU, we want actual highest values (could be negative)
                    # For post-ReLU, skip if max activation is <= 0
                    if not use_pre_relu and max_activation <= 0:
                        continue

                    max_position = torch.argmax(masked_activations).item()

                    if max_position < seq_len:
                        max_token_id = token_ids[b, max_position]
                        max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False)

                        start_pos = max(0, max_position - window_size)
                        end_pos = min(seq_len, max_position + window_size + 1)

                        window_token_ids = token_ids[b, start_pos:end_pos]
                        window_text_raw = self.tokenizer.decode(window_token_ids, skip_special_tokens=False)

                        window_text_cleaned = window_text_raw.replace(self.tokenizer.bos_token or "<s>", "") \
                                                         .replace(self.tokenizer.eos_token or "</s>", "") \
                                                         .replace(self.tokenizer.pad_token or "", "") \
                                                         .replace(self.tokenizer.cls_token or "", "") \
                                                         .replace(self.tokenizer.sep_token or "", "")
                        window_text_cleaned = window_text_cleaned.strip()

                        temp_results.append({
                            "text": batch_texts[b],
                            "windowed_text": window_text_cleaned,
                            "max_activation": max_activation,
                            "max_position": int(max_position),
                            "max_token": max_token,
                            "window_start": int(start_pos),
                            "window_end": int(end_pos),
                            "activation_type": activation_type
                        })

        temp_results.sort(key=lambda x: x["max_activation"], reverse=True)
        return temp_results[:top_n]

    def find_lowest_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10, use_pre_relu=False):
        """
        Find texts with LOWEST token-level activations for a feature.
        When use_pre_relu=True, this finds the most negative activations.
        
        Args:
            feature_idx: The feature to analyze
            texts: List of text samples
            top_n: Number of lowest examples to return
            batch_size: Number of texts to process in each batch
            window_size: Number of tokens before and after the lowest activation to include
            use_pre_relu: If True, use pre-ReLU activations (can find negative values)

        Returns:
            List of dicts with text, max_activation, max_position, and max_token
        """
        results = []

        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                            self.tokenizer.pad_token, self.tokenizer.cls_token,
                            self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}

        activation_type = "pre-ReLU" if use_pre_relu else "post-ReLU"
        print(f"Scanning for feature {feature_idx} to find top {top_n} LOWEST activating examples ({activation_type}, batch size {batch_size}, window size {window_size}, max_token_length {self.max_token_length})...")

        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Lowest Examples for Feature {feature_idx} ({activation_type})"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                if use_pre_relu:
                    _, _, pre_relu_features = self.sae_model.forward_with_pre_relu(hidden_states.to(self.sae_model.encoder[0].weight.dtype))
                    features = pre_relu_features
                else:
                    features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                for b in range(features.shape[0]):
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len == 0: continue

                    token_activations = features[b, :seq_len, feature_idx]
                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)

                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    for pos, token in enumerate(sample_tokens):
                        if token in special_tokens or sample_token_ids[pos] in self.tokenizer.all_special_ids:
                            token_mask[pos] = False

                    if not torch.any(token_mask):
                        continue

                    non_special_activations = token_activations[token_mask]
                    
                    if len(non_special_activations) == 0:
                        continue
                    
                    min_activation = torch.min(non_special_activations).item()
                    min_position_among_valid = torch.argmin(non_special_activations).item()
                    
                    valid_positions = torch.where(token_mask)[0]
                    actual_min_position = valid_positions[min_position_among_valid].item()

                    max_activation = min_activation  # Keep variable name for consistency
                    max_position = actual_min_position

                    # For post-ReLU, skip negative activations (shouldn't exist anyway)
                    # For pre-ReLU, accept any activation (including negative)
                    if not use_pre_relu and max_activation < 0:
                        continue

                    if max_position < seq_len:
                        max_token_id = token_ids[b, max_position]
                        max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False)

                        start_pos = max(0, max_position - window_size)
                        end_pos = min(seq_len, max_position + window_size + 1)

                        window_token_ids = token_ids[b, start_pos:end_pos]
                        window_text_raw = self.tokenizer.decode(window_token_ids, skip_special_tokens=False)

                        window_text_cleaned = window_text_raw.replace(self.tokenizer.bos_token or "<s>", "") \
                                                        .replace(self.tokenizer.eos_token or "</s>", "") \
                                                        .replace(self.tokenizer.pad_token or "", "") \
                                                        .replace(self.tokenizer.cls_token or "", "") \
                                                        .replace(self.tokenizer.sep_token or "", "")
                        window_text_cleaned = window_text_cleaned.strip()

                        temp_results.append({
                            "text": batch_texts[b],
                            "windowed_text": window_text_cleaned,
                            "max_activation": max_activation,
                            "max_position": int(max_position),
                            "max_token": max_token,
                            "window_start": int(start_pos),
                            "window_end": int(end_pos),
                            "activation_type": activation_type
                        })

        temp_results.sort(key=lambda x: x["max_activation"], reverse=False)  # LOWEST first
        return temp_results[:top_n]

    def compute_feature_statistics(self, feature_idx, texts, n_samples=5000, batch_size=16, use_pre_relu=False):
        """Compute statistics for a feature across texts."""
        if len(texts) > n_samples:
            sampled_texts = random.sample(texts, n_samples)
        else:
            sampled_texts = texts

        activations = []
        active_token_pct = []

        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}
        special_ids = set(self.tokenizer.all_special_ids)

        activation_type = "pre-ReLU" if use_pre_relu else "post-ReLU"

        for i in tqdm(range(0, len(sampled_texts), batch_size), desc=f"Computing statistics ({activation_type})", leave=False):
            batch_texts = sampled_texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                if use_pre_relu:
                    _, _, pre_relu_features = self.sae_model.forward_with_pre_relu(hidden_states.to(self.sae_model.encoder[0].weight.dtype))
                    features = pre_relu_features
                else:
                    features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                for b in range(features.shape[0]):
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    if seq_len > 0:
                        token_activations = features[b, :seq_len, feature_idx].cpu().numpy()
                        sample_token_ids = token_ids[b][:seq_len]
                        sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                        token_mask = np.ones_like(token_activations, dtype=bool)
                        for k in range(seq_len):
                            if sample_tokens[k] in special_tokens or sample_token_ids[k] in special_ids:
                                token_mask[k] = False

                        non_special_activations = token_activations[token_mask]

                        activations.extend(non_special_activations)
                        if np.sum(token_mask) > 0:
                            if use_pre_relu:
                                # For pre-ReLU, consider any non-zero activation as "active"
                                active_token_pct.append(np.mean(np.abs(non_special_activations) > 1e-6))
                            else:
                                # For post-ReLU, only positive activations
                                active_token_pct.append(np.mean(non_special_activations > 0))
                        else:
                            active_token_pct.append(0.0)

        activations = np.array(activations)
        active_token_pct = np.array(active_token_pct)

        stats = {
            "mean_activation": float(np.mean(activations)) if len(activations) > 0 else 0.0,
            "max_activation": float(np.max(activations)) if len(activations) > 0 else 0.0,
            "min_activation": float(np.min(activations)) if len(activations) > 0 else 0.0,
            "median_activation": float(np.median(activations)) if len(activations) > 0 else 0.0,
            "std_activation": float(np.std(activations)) if len(activations) > 0 else 0.0,
            "percent_active_non_special_tokens": float(np.mean(np.abs(activations) > 1e-6) * 100) if use_pre_relu and len(activations) > 0 else float(np.mean(activations > 0) * 100) if len(activations) > 0 else 0.0,
            "mean_text_active_pct_non_special_tokens": float(np.mean(active_token_pct) * 100) if len(active_token_pct) > 0 else 0.0,
            "feature_idx": feature_idx,
            "activation_type": activation_type
        }

        return stats

    async def identify_feature_contrast_pattern(self, feature_idx, positive_examples, negative_examples, output_dir=None, claude_examples=10):
        """
        Use Claude API to identify the contrasting pattern between positive and negative examples.
        
        Args:
            feature_idx: The feature index
            positive_examples: List of examples with highest activations
            negative_examples: List of examples with lowest/most negative activations
            output_dir: Directory to save results
            claude_examples: Number of examples to send to Claude API

        Returns:
            Identified contrast pattern as a string
        """
        feature_id_logger = logging.getLogger('feature_id')

        # Format positive examples
        positive_formatted = []
        for i, example in enumerate(positive_examples[:claude_examples]):
            if isinstance(example, dict):
                text_content = example.get("text", example.get("windowed_text", ""))
            else:
                text_content = example

            # Clean special tokens
            text_content = text_content.replace(self.tokenizer.bos_token or "<s>", "") \
                                    .replace(self.tokenizer.eos_token or "</s>", "") \
                                    .replace(self.tokenizer.pad_token or "", "") \
                                    .replace(self.tokenizer.cls_token or "", "") \
                                    .replace(self.tokenizer.sep_token or "", "")
            text_content = text_content.strip()

            if text_content:
                positive_formatted.append(f"Positive {i+1}: {text_content}")

        # Format negative examples
        negative_formatted = []
        for i, example in enumerate(negative_examples[:claude_examples]):
            if isinstance(example, dict):
                text_content = example.get("text", example.get("windowed_text", ""))
            else:
                text_content = example

            # Clean special tokens
            text_content = text_content.replace(self.tokenizer.bos_token or "<s>", "") \
                                    .replace(self.tokenizer.eos_token or "</s>", "") \
                                    .replace(self.tokenizer.pad_token or "", "") \
                                    .replace(self.tokenizer.cls_token or "", "") \
                                    .replace(self.tokenizer.sep_token or "", "")
            text_content = text_content.strip()

            if text_content:
                negative_formatted.append(f"Negative {i+1}: {text_content}")

        if not positive_formatted and not negative_formatted:
            feature_id_logger.warning(f"No valid examples to send to Claude for feature {feature_idx} contrast analysis.")
            return "Unknown pattern (No valid examples)"

        # Create the contrast analysis prompt
        separator = '-' * 40
        
        contrast_prompt = f"""I'll analyze examples where a specific neural network feature shows contrasting behavior.

POSITIVE EXAMPLES (High activation):
{chr(10).join(positive_formatted)}

{separator}

NEGATIVE EXAMPLES (Low/inhibited activation):
{chr(10).join(negative_formatted)}

{separator}

Based on these contrasting examples, what is the key semantic or linguistic pattern that distinguishes the positive examples from the negative examples? What concept, structure, or meaning does this feature seem to detect vs. inhibit?

Format your response as:
Pattern: [concise description of what the feature detects in positive examples vs. what it inhibits in negative examples]
EXPLANATION: [brief explanation of the contrast and why this pattern makes sense]"""

        feature_id_logger.debug(f"CLAUDE CONTRAST PATTERN IDENTIFICATION PROMPT (Feature {feature_idx}):\n{contrast_prompt}")

        # Ensure output directory exists
        if output_dir:
            claude_output_dir = os.path.join(output_dir, f"feature_{feature_idx}_contrast")
            os.makedirs(claude_output_dir, exist_ok=True)
        else:
            claude_output_dir = "."

        if not self.claude_api_key:
            feature_id_logger.error("Claude API key is missing. Cannot identify contrast pattern.")
            return "Unknown pattern (Claude API key missing)"

        # Define Claude API URL and headers
        api_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Prepare the request data
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": contrast_prompt}
            ]
        }

        try:
            # Save the prompt
            if output_dir:
                prompt_path = os.path.join(claude_output_dir, "claude_contrast_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(contrast_prompt)

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()
            claude_response = response_data["content"][0]["text"]

            feature_id_logger.debug(f"CLAUDE CONTRAST PATTERN IDENTIFICATION RESPONSE (Feature {feature_idx}):\n{claude_response}")

            # Save the raw response
            if output_dir:
                claude_response_path = os.path.join(claude_output_dir, "claude_contrast_response.txt")
                with open(claude_response_path, "w", encoding="utf-8") as f:
                    f.write(claude_response)

            # Parse the response to extract the pattern
            pattern_parts = claude_response.split("EXPLANATION:", 1)
            concise_pattern = pattern_parts[0].strip() if pattern_parts else "Unknown pattern (parsing failed)"
            explanation = pattern_parts[1].strip() if len(pattern_parts) > 1 else ""

            # Save the parsed pattern
            if output_dir:
                pattern_path = os.path.join(claude_output_dir, "contrast_pattern.txt")
                with open(pattern_path, "w", encoding="utf-8") as f:
                    f.write(f"{concise_pattern}\n{explanation}")

            return concise_pattern

        except Exception as e:
            feature_id_logger.error(f"Error calling Claude API: {e}")
            return f"Error identifying contrast pattern: {str(e)}"

def load_identification_prompt():
    """Load the identification prompt template (kept for compatibility)."""
    prompt_path = Path("feature_identification_prompt.txt")
    if not prompt_path.exists():
        prompt_path = Path("prompts/feature_identification_prompt.txt")

    if not prompt_path.exists():
        logging.warning(f"Prompt template not found at {prompt_path}. Using default contrast analysis.")
        return """This prompt is not used in contrast analysis mode."""

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

async def analyze_feature_contrast_meaning(
    args,
    feature_indices,
    model_path,
    sae_path,
    output_dir,
    num_examples=20,
    num_samples=10000,
    window_size=10,
    batch_size=16,
    config_dir=None,
    max_token_length=100,
    claude_examples=10
):
    """
    Analyze the meaning of SAE features using contrast between positive and negative examples.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "feature_contrast_analysis.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Features argument provided: {args.features}")
    logging.info(f"Analyzing features with contrast method: {feature_indices}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"SAE path: {sae_path}")
    logging.info(f"Number of examples per feature: {num_examples}")
    logging.info(f"Number of dataset samples: {num_samples}")
    logging.info(f"Window size: {window_size}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Max token length: {max_token_length}")
    logging.info(f"Claude examples: {claude_examples}")

    # Load Claude API key
    claude_api_key = None
    if config_dir:
        config_path = Path(config_dir) / "api_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    claude_api_key = config.get("claude_api_key")
                    if claude_api_key:
                        logging.info("Claude API key loaded successfully")
                    else:
                        logging.warning("Claude API key not found in config")
            except Exception as e:
                logging.warning(f"Failed to load config: {e}")
    else:
        logging.warning("Config directory not provided. Skipping API key loading.")

    if not claude_api_key:
        logging.warning("Claude API key not available. Pattern identification will be limited.")

    # Load models
    logging.info("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token for tokenizer")

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    # Load SAE model
    logging.info("Loading SAE model...")
    state_dict = torch.load(sae_path, map_location="cuda")

    # Determine dimensions from state dict
    if 'decoder.weight' in state_dict:
        decoder_weight = state_dict['decoder.weight']
        input_dim = decoder_weight.shape[0]
        hidden_dim = decoder_weight.shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError("Could not determine SAE dimensions from state dict keys")

    logging.info(f"Creating TopK SAE with input_dim={input_dim}, hidden_dim={hidden_dim}, k={K}")
    sae_model = SparseAutoencoder(input_dim, hidden_dim, k=K)
    sae_model.load_state_dict(state_dict)
    sae_model.to(model.device)
    sae_model.eval()

    # Initialize analyzer
    identification_prompt = load_identification_prompt()
    analyzer = SAEFeatureAnalyzer(
        sae_model,
        tokenizer,
        model,
        identification_prompt=identification_prompt,
        claude_api_key=claude_api_key,
        max_token_length=max_token_length
    )

    # Load wikitext dataset
    logging.info(f"Loading wikitext dataset with chunking...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        
        max_indices_to_check = min(len(dataset), num_samples * 3)
        
        filtered_texts = []
        for idx in range(max_indices_to_check):
            text = dataset[idx]["text"]
            if text.strip():
                full_inputs = tokenizer(text, return_tensors="pt", truncation=False)
                full_length = torch.sum(full_inputs["attention_mask"][0]).item()
                
                if full_length > max_token_length:
                    clean_text = tokenizer.decode(full_inputs["input_ids"][0], skip_special_tokens=True)
                    tokens = tokenizer.encode(clean_text, add_special_tokens=False)
                    
                    for chunk_start in range(0, len(tokens), max_token_length):
                        chunk_tokens = tokens[chunk_start:chunk_start + max_token_length]
                        if len(chunk_tokens) > 5:
                            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                            filtered_texts.append(chunk_text)
                            
                            if len(filtered_texts) >= num_samples:
                                break
                else:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_token_length)
                    if torch.sum(inputs["attention_mask"][0]).item() > 5:
                        truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                        filtered_texts.append(truncated_text)
                        
                        if len(filtered_texts) >= num_samples:
                            break
            
            if len(filtered_texts) >= num_samples:
                break

        texts = filtered_texts
        logging.info(f"Loaded {len(texts)} text samples (with chunking) from wikitext-103 (max_token_length={max_token_length})")

    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        logging.info("Please ensure you have an internet connection to download the dataset or use a local path.")
        return

    # Analyze each feature
    results = []

    for feature_idx in feature_indices:
        logging.info(f"\nAnalyzing feature {feature_idx} with contrast method")
        feature_result = {"feature_idx": int(feature_idx)}

        # Create feature directory
        feature_dir = output_dir / f"feature_{feature_idx}"
        feature_dir.mkdir(exist_ok=True)

        # Find positive examples (highest activations using post-ReLU)
        logging.info(f"Finding top {num_examples} positive examples (highest activations)...")
        positive_examples = analyzer.find_highest_activating_examples(
            feature_idx, texts, top_n=num_examples, batch_size=batch_size, 
            window_size=window_size, use_pre_relu=False
        )

        # Find negative examples (most negative activations using pre-ReLU)
        logging.info(f"Finding top {num_examples} negative examples (most inhibited activations)...")
        negative_examples = analyzer.find_lowest_activating_examples(
            feature_idx, texts, top_n=num_examples, batch_size=batch_size, 
            window_size=window_size, use_pre_relu=True
        )

        # Compute feature statistics for both post-ReLU and pre-ReLU
        logging.info("Computing feature statistics (post-ReLU)...")
        stats_post_relu = analyzer.compute_feature_statistics(
            feature_idx, texts, n_samples=min(2000, len(texts)), 
            batch_size=batch_size, use_pre_relu=False
        )
        
        logging.info("Computing feature statistics (pre-ReLU)...")
        stats_pre_relu = analyzer.compute_feature_statistics(
            feature_idx, texts, n_samples=min(2000, len(texts)), 
            batch_size=batch_size, use_pre_relu=True
        )

        feature_result["statistics_post_relu"] = stats_post_relu
        feature_result["statistics_pre_relu"] = stats_pre_relu
        
        logging.info(f"Post-ReLU Statistics: {stats_post_relu}")
        logging.info(f"Pre-ReLU Statistics: {stats_pre_relu}")

        # Save examples to files
        with open(feature_dir / "positive_examples.txt", "w", encoding="utf-8") as f:
            f.write("=== POSITIVE EXAMPLES (Highest Activations) ===\n\n")
            for i, example in enumerate(positive_examples):
                f.write(f"--- Positive Example {i+1} ---\n")
                f.write(f"Activation Score: {example['max_activation']:.4f}\n")
                f.write(f"Max Token: '{example['max_token']}'\n")
                f.write(f"Windowed Text:\n{example['windowed_text']}\n\n")
                f.write(f"Full Text (max {max_token_length} tokens):\n{example['text']}\n\n")
                f.write("-"*50 + "\n\n")

        with open(feature_dir / "negative_examples.txt", "w", encoding="utf-8") as f:
            f.write("=== NEGATIVE EXAMPLES (Most Inhibited Activations) ===\n\n")
            for i, example in enumerate(negative_examples):
                f.write(f"--- Negative Example {i+1} ---\n")
                f.write(f"Activation Score: {example['max_activation']:.4f}\n")
                f.write(f"Max Token: '{example['max_token']}'\n")
                f.write(f"Windowed Text:\n{example['windowed_text']}\n\n")
                f.write(f"Full Text (max {max_token_length} tokens):\n{example['text']}\n\n")
                f.write("-"*50 + "\n\n")

        # Store examples
        feature_result["positive_examples"] = [{
            "full_text": ex["text"],
            "windowed_text": ex["windowed_text"],
            "max_activation": float(ex["max_activation"]),
            "max_token": ex["max_token"],
            "activation_type": ex["activation_type"]
        } for ex in positive_examples]

        feature_result["negative_examples"] = [{
            "full_text": ex["text"],
            "windowed_text": ex["windowed_text"],
            "max_activation": float(ex["max_activation"]),
            "max_token": ex["max_token"],
            "activation_type": ex["activation_type"]
        } for ex in negative_examples]

        # Identify contrast pattern using Claude API
        if claude_api_key:
            logging.info("Identifying feature contrast pattern using Claude API...")
            feature_result["contrast_pattern"] = await analyzer.identify_feature_contrast_pattern(
                feature_idx,
                feature_result["positive_examples"],
                feature_result["negative_examples"],
                output_dir=output_dir,
                claude_examples=claude_examples
            )
            logging.info(f"Identified contrast pattern: {feature_result['contrast_pattern']}")
        else:
            feature_result["contrast_pattern"] = "Contrast pattern identification skipped (no API key)"
            logging.warning("Contrast pattern identification skipped (no API key)")

        # Store results
        results.append(feature_result)

        # Save incremental results
        try:
            with open(output_dir / "feature_contrast_analysis_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Incremental results saved for feature {feature_idx}")
        except Exception as e:
            logging.error(f"Error saving incremental results for feature {feature_idx}: {e}")

    # Create summary report
    logging.info("Creating contrast analysis summary report...")
    with open(output_dir / "contrast_analysis_summary.md", "w", encoding="utf-8") as f:
        f.write("# SAE Feature Contrast Analysis Summary\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"SAE: {sae_path}\n")
        f.write(f"Max Token Length: {max_token_length}\n")
        f.write(f"Analysis Method: Contrast between positive (high activation) and negative (inhibited) examples\n\n")
        f.write("## Analyzed Features\n\n")

        for feature_result in results:
            feature_idx = feature_result["feature_idx"]
            contrast_pattern = feature_result.get("contrast_pattern", "N/A")
            stats_post = feature_result.get("statistics_post_relu", {})
            stats_pre = feature_result.get("statistics_pre_relu", {})

            f.write(f"### Feature {feature_idx}\n\n")
            f.write(f"**Contrast Pattern:** {contrast_pattern}\n\n")

            # Statistics
            f.write(f"**Post-ReLU Statistics:**\n")
            f.write(f"- Max activation: {stats_post.get('max_activation', 0.0):.4f}\n")
            f.write(f"- Mean activation: {stats_post.get('mean_activation', 0.0):.4f}\n")
            f.write(f"- Percent active tokens: {stats_post.get('percent_active_non_special_tokens', 0.0):.2f}%\n\n")

            f.write(f"**Pre-ReLU Statistics:**\n")
            f.write(f"- Max activation: {stats_pre.get('max_activation', 0.0):.4f}\n")
            f.write(f"- Min activation: {stats_pre.get('min_activation', 0.0):.4f}\n")
            f.write(f"- Mean activation: {stats_pre.get('mean_activation', 0.0):.4f}\n")
            f.write(f"- Std activation: {stats_pre.get('std_activation', 0.0):.4f}\n\n")

            # Top positive examples
            f.write("**Top Positive Examples (High Activation):**\n\n")
            positive_examples = feature_result.get("positive_examples", [])
            for i, example in enumerate(positive_examples[:3]):
                f.write(f"Positive {i+1}:\n")
                full_text = example.get('full_text', 'N/A')
                if len(full_text) > 300:
                    full_text = full_text[:300] + "..."
                f.write(f"Context: ```\n{example.get('windowed_text', 'N/A')}\n```\n")
                f.write(f"Max Token: '{example.get('max_token', 'N/A')}'\n")
                f.write(f"Activation: {example.get('max_activation', 0.0):.4f}\n\n")

            # Top negative examples
            f.write("**Top Negative Examples (Inhibited Activation):**\n\n")
            negative_examples = feature_result.get("negative_examples", [])
            for i, example in enumerate(negative_examples[:3]):
                f.write(f"Negative {i+1}:\n")
                full_text = example.get('full_text', 'N/A')
                if len(full_text) > 300:
                    full_text = full_text[:300] + "..."
                f.write(f"Context: ```\n{example.get('windowed_text', 'N/A')}\n```\n")
                f.write(f"Max Token: '{example.get('max_token', 'N/A')}'\n")
                f.write(f"Activation: {example.get('max_activation', 0.0):.4f}\n\n")

            f.write("---\n\n")

    logging.info(f"Contrast analysis complete! Results saved to {output_dir}")
    return results

async def main():
    parser = argparse.ArgumentParser(description="Analyze meaning of SAE features using contrast between positive and negative examples")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--features", type=str, required=True, help="Comma-separated list of feature indices or path to JSON file")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to extract per feature (both positive and negative)")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use from dataset")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for context around activations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--config_dir", type=str, default="../config", help="Directory with API key config")
    parser.add_argument("--max_token_length", type=int, default=100, help="Maximum token length for tokenization")
    parser.add_argument("--claude_examples", type=int, default=10, help="Number of examples to send to Claude API for pattern identification")

    args = parser.parse_args()

    # Parse feature indices
    feature_indices = []
    if os.path.isfile(args.features):
        try:
            with open(args.features, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    feature_indices = data
                elif isinstance(data, dict) and "top_feature_indices" in data:
                    feature_indices = data["top_feature_indices"]
                elif isinstance(data, dict) and "rsa_feature_indices" in data:
                    feature_indices = data["rsa_feature_indices"]
                    logging.info(f"Parsed feature indices from 'rsa_feature_indices' key in {args.features}")
                elif isinstance(data, dict):
                    for item in data.values():
                        if isinstance(item, dict) and "feature_idx" in item:
                            feature_indices.append(item["feature_idx"])
                    if not feature_indices:
                        try:
                            feature_indices = [int(k) for k in data.keys()]
                            logging.info(f"Parsed feature indices from JSON keys: {feature_indices[:10]}{'...' if len(feature_indices) > 10 else ''}")
                        except ValueError:
                            logging.warning(f"Could not parse feature indices from JSON keys in {args.features}.")
                            feature_indices = []
                else:
                    logging.error(f"Unexpected JSON format in {args.features}.")
                    feature_indices = []

        except Exception as e:
            logging.error(f"Error loading feature indices from {args.features}: {e}")
            feature_indices = []
    else:
        try:
            feature_indices = [int(x.strip()) for x in args.features.split(",") if x.strip()]
        except ValueError:
            logging.error(f"Could not parse comma-separated feature indices: {args.features}")
            feature_indices = []

    if not feature_indices:
        print("No feature indices found to analyze. Please check your --features argument or input file.")
        return

    print(f"Analyzing {len(feature_indices)} features with contrast method: {feature_indices[:10]}{'...' if len(feature_indices) > 10 else ''}")
    print(f"Using max token length: {args.max_token_length}")

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"feature_contrast_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Run contrast analysis
    await analyze_feature_contrast_meaning(
        args,
        feature_indices=feature_indices,
        model_path=args.model_path,
        sae_path=args.sae_path,
        output_dir=output_path,
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        window_size=args.window_size,
        batch_size=args.batch_size,
        config_dir=args.config_dir,
        max_token_length=args.max_token_length,
        claude_examples=args.claude_examples
    )

if __name__ == "__main__":
    asyncio.run(main())