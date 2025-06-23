#max_token length parameter (default was 100) now using 10; uses all data; chunks up data (not just first parts of data)
# python part5b_find_feature_meaning_large_wiki.py --sae_path ../checkpoints/best_model.pt --model_path ../models/open_llama_3b --features "32616" --output_dir ./NFM_feature_analysis_results_token10/ --config_dir ../config --max_token_length 10 --num_samples 10000000 --claude_examples 20
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

class SparseAutoencoder(torch.nn.Module):
    """Simple Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
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
        # features are already non-negative due to ReLU

        # Reconstruction maps features back to the original activation space
        reconstruction = self.decoder(features) # reconstruction: (batch_size, seq_len, input_dim)

        return features, reconstruction # Return both the features and the reconstruction

class SAEFeatureAnalyzer:
    """Analyzer for studying features in a Sparse Autoencoder trained on language model activations."""

    def __init__(self, sae_model, tokenizer, base_model, device="cuda", target_layer=16,
                 identification_prompt=None, scoring_prompt=None, claude_api_key=None, max_token_length=100):
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.max_token_length = max_token_length  # NEW: configurable max token length
        self.sae_model.eval()  # Set to evaluation mode
        self.base_model.eval()

        # Store feature statistics
        self.feature_stats = {}

        # Store prompts
        self.identification_prompt = identification_prompt
        self.scoring_prompt = scoring_prompt

        # Claude API key
        self.claude_api_key = claude_api_key

    def find_highest_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10):
        """
        Find texts with highest token-level activations for a feature.
        Uses simple efficient batching.

        Args:
            feature_idx: The feature to analyze
            texts: List of text samples
            top_n: Number of top examples to return
            batch_size: Number of texts to process in each batch
            window_size: Number of tokens before and after the highest activation to include

        Returns:
            List of dicts with text, max_activation, max_position, and max_token
        """
        results = []

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}

        print(f"Scanning for feature {feature_idx} to find top {top_n} examples (batch size {batch_size}, window size {window_size}, max_token_length {self.max_token_length})...")

        # Store results temporarily and sort at the end to get top_n overall
        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Examples for Feature {feature_idx}"):
            batch_texts = texts[i:i+batch_size]
            # Filter empty texts
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            # Tokenize batch - Use configurable max_token_length
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                # Process batch through model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                # Process through SAE
                # Ensure hidden_states dtype matches SAE encoder weight dtype
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                # For each sample in batch
                for b in range(features.shape[0]):
                    # Get valid sequence length (exclude padding)
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    if seq_len == 0: continue

                    # Get activations for this feature for the valid sequence length
                    token_activations = features[b, :seq_len, feature_idx]

                    # Only consider if there's non-zero activation on non-special tokens
                    # Create a mask for non-special tokens
                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)

                    # Convert token IDs to tokens for this sample to check against special tokens
                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    for pos, token in enumerate(sample_tokens):
                        # Check against both token string and token ID for special tokens
                        if token in special_tokens or sample_token_ids[pos] in self.tokenizer.all_special_ids:
                            token_mask[pos] = False

                    # If all tokens are special or no activation on non-special tokens, skip
                    if not torch.any(token_mask) or torch.max(token_activations[token_mask]) <= 0:
                        continue

                    # Apply mask and find max activation among non-special tokens
                    # Set activations for special tokens to a very low value to ensure max is from non-special
                    masked_activations = token_activations.clone()
                    masked_activations[~token_mask] = -1e9

                    max_activation = torch.max(masked_activations).item()

                    # If max activation among non-special tokens is <= 0 after masking, skip
                    if max_activation <= 0:
                         continue

                    # Get the position of the max activation within the valid sequence length
                    max_position = torch.argmax(masked_activations).item()

                    # Ensure max_position is within valid range (should be if derived from seq_len)
                    if max_position < seq_len:
                        # Get the token at the max activation position
                        max_token_id = token_ids[b, max_position]
                        # Decode the single token without skipping special tokens initially
                        max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False)

                        # Calculate window around max activation, limited to max 10 tokens on each side
                        start_pos = max(0, max_position - window_size)
                        end_pos = min(seq_len, max_position + window_size + 1)

                        # Get windowed token ids and decode to text
                        window_token_ids = token_ids[b, start_pos:end_pos]
                        # Decode windowed text *without* skipping special tokens initially, then clean
                        window_text_raw = self.tokenizer.decode(window_token_ids, skip_special_tokens=False)

                        # Clean specific special tokens from the windowed text for presentation
                        window_text_cleaned = window_text_raw.replace(self.tokenizer.bos_token or "<s>", "") \
                                                         .replace(self.tokenizer.eos_token or "</s>", "") \
                                                         .replace(self.tokenizer.pad_token or "", "") \
                                                         .replace(self.tokenizer.cls_token or "", "") \
                                                         .replace(self.tokenizer.sep_token or "", "")
                        window_text_cleaned = window_text_cleaned.strip()

                        temp_results.append({
                            "text": batch_texts[b],  # Full original statement (truncated by max_token_length)
                            "windowed_text": window_text_cleaned,  # Context window
                            "max_activation": max_activation,
                            "max_position": int(max_position),
                            "max_token": max_token, # Keep the decoded max token
                            "window_start": int(start_pos),
                            "window_end": int(end_pos)
                        })

        # Sort all collected results by max activation and return top_n
        temp_results.sort(key=lambda x: x["max_activation"], reverse=True)
        return temp_results[:top_n]

    def find_lowest_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10):
        """
        Find texts with LOWEST token-level activations for a feature.
        Uses simple efficient batching.

        Args:
            feature_idx: The feature to analyze
            texts: List of text samples
            top_n: Number of lowest examples to return
            batch_size: Number of texts to process in each batch
            window_size: Number of tokens before and after the lowest activation to include

        Returns:
            List of dicts with text, max_activation, max_position, and max_token
        """
        results = []

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                            self.tokenizer.pad_token, self.tokenizer.cls_token,
                            self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}

        print(f"Scanning for feature {feature_idx} to find top {top_n} LOWEST activating examples (batch size {batch_size}, window size {window_size}, max_token_length {self.max_token_length})...")

        # Store results temporarily and sort at the end to get top_n overall
        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Lowest Examples for Feature {feature_idx}"):
            batch_texts = texts[i:i+batch_size]
            # Filter empty texts
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            # Tokenize batch - Use configurable max_token_length
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                # Process batch through model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                # Process through SAE
                # Ensure hidden_states dtype matches SAE encoder weight dtype
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                # For each sample in batch
                for b in range(features.shape[0]):
                    # Get valid sequence length (exclude padding)
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    if seq_len == 0: continue

                    # Get activations for this feature for the valid sequence length
                    token_activations = features[b, :seq_len, feature_idx]

                    # Create a mask for non-special tokens
                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)

                    # Convert token IDs to tokens for this sample to check against special tokens
                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    for pos, token in enumerate(sample_tokens):
                        # Check against both token string and token ID for special tokens
                        if token in special_tokens or sample_token_ids[pos] in self.tokenizer.all_special_ids:
                            token_mask[pos] = False

                    # If all tokens are special, skip
                    if not torch.any(token_mask):
                        continue

                    # For lowest activations, we want to find the ACTUAL lowest activation among non-special tokens
                    # Don't mask to extreme values - just get the real activations for non-special tokens
                    non_special_activations = token_activations[token_mask]
                    
                    # Skip if no valid activations
                    if len(non_special_activations) == 0:
                        continue
                    
                    # Find the minimum activation among non-special tokens
                    min_activation = torch.min(non_special_activations).item()
                    min_position_among_valid = torch.argmin(non_special_activations).item()
                    
                    # Convert back to original position in the sequence
                    valid_positions = torch.where(token_mask)[0]
                    actual_min_position = valid_positions[min_position_among_valid].item()

                    # For consistency with the original function, we'll call it "max_activation" 
                    # even though it's actually the minimum for this case
                    max_activation = min_activation
                    max_position = actual_min_position

                    # Accept ANY non-negative activation (including very small ones and zeros)
                    if max_activation < 0:
                        continue

                    # Ensure max_position is within valid range
                    if max_position < seq_len:
                        # Get the token at the min activation position
                        max_token_id = token_ids[b, max_position]
                        # Decode the single token without skipping special tokens initially
                        max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False)

                        # Calculate window around min activation, limited to max 10 tokens on each side
                        start_pos = max(0, max_position - window_size)
                        end_pos = min(seq_len, max_position + window_size + 1)

                        # Get windowed token ids and decode to text
                        window_token_ids = token_ids[b, start_pos:end_pos]
                        # Decode windowed text *without* skipping special tokens initially, then clean
                        window_text_raw = self.tokenizer.decode(window_token_ids, skip_special_tokens=False)

                        # Clean specific special tokens from the windowed text for presentation
                        window_text_cleaned = window_text_raw.replace(self.tokenizer.bos_token or "<s>", "") \
                                                        .replace(self.tokenizer.eos_token or "</s>", "") \
                                                        .replace(self.tokenizer.pad_token or "", "") \
                                                        .replace(self.tokenizer.cls_token or "", "") \
                                                        .replace(self.tokenizer.sep_token or "", "")
                        window_text_cleaned = window_text_cleaned.strip()

                        temp_results.append({
                            "text": batch_texts[b],  # Full original statement (truncated by max_token_length)
                            "windowed_text": window_text_cleaned,  # Context window
                            "max_activation": max_activation,  # Actually the minimum activation
                            "max_position": int(max_position),
                            "max_token": max_token, # Keep the decoded token
                            "window_start": int(start_pos),
                            "window_end": int(end_pos)
                        })

        # Sort all collected results by activation ASCENDING (lowest first) and return top_n
        temp_results.sort(key=lambda x: x["max_activation"], reverse=False)  # LOWEST first
        return temp_results[:top_n]

    def find_lowest_mean_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10):
        """
        Find texts with the lowest MEAN activation across all non-special tokens for a given feature,
        and return a window around the lowest activation token for context.

        Args:
            feature_idx: Feature index to analyze
            texts: List of text samples
            top_n: Number of lowest examples to return
            batch_size: Batch size for processing
            window_size: Number of tokens before and after the lowest activation token to include

        Returns:
            List of dicts with text, mean_activation, windowed_text, max_token
        """
        results = []

        special_tokens = set([
            self.tokenizer.bos_token, self.tokenizer.eos_token,
            self.tokenizer.pad_token, self.tokenizer.cls_token,
            self.tokenizer.sep_token, "<s>", "</s>"
        ])
        special_tokens = {t for t in special_tokens if t is not None}
        special_ids = set(self.tokenizer.all_special_ids)

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Mean Activation (Feature {feature_idx})"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]
            if not batch_texts:
                continue

            # Use configurable max_token_length
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                for b in range(features.shape[0]):
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len == 0:
                        continue

                    token_activations = features[b, :seq_len, feature_idx]
                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)
                    for pos, token in enumerate(sample_tokens):
                        if token in special_tokens or sample_token_ids[pos] in special_ids:
                            token_mask[pos] = False

                    if not torch.any(token_mask):
                        continue

                    non_special_activations = token_activations[token_mask]
                    if len(non_special_activations) == 0:
                        continue

                    mean_activation = torch.mean(non_special_activations).item()

                    # Find the token with minimum activation (just for windowing context)
                    min_val, min_idx = torch.min(token_activations[token_mask], dim=0)
                    valid_positions = torch.where(token_mask)[0]
                    min_position = valid_positions[min_idx].item()
                    max_token_id = sample_token_ids[min_position]
                    max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False)

                    # Build window around min token
                    start = max(0, min_position - window_size)
                    end = min(seq_len, min_position + window_size + 1)
                    window_ids = token_ids[b][start:end]
                    window_text_raw = self.tokenizer.decode(window_ids, skip_special_tokens=False)
                    window_text_cleaned = window_text_raw
                    for tok in special_tokens:
                        window_text_cleaned = window_text_cleaned.replace(tok or "", "")
                    window_text_cleaned = window_text_cleaned.strip()

                    results.append({
                        "text": batch_texts[b],
                        "mean_activation": mean_activation,
                        "max_token": max_token,
                        "windowed_text": window_text_cleaned
                    })

        results.sort(key=lambda x: x["mean_activation"])
        return results[:top_n]

    def find_highest_mean_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10):
        """
        Find texts with the highest MEAN activation across all non-special tokens for a given feature,
        and return a window around the highest activation token for context.

        Args:
            feature_idx: Feature index to analyze
            texts: List of text samples
            top_n: Number of highest examples to return
            batch_size: Batch size for processing
            window_size: Number of tokens before and after the highest activation token to include

        Returns:
            List of dicts with text, mean_activation, windowed_text, max_token
        """
        results = []

        special_tokens = set([
            self.tokenizer.bos_token, self.tokenizer.eos_token,
            self.tokenizer.pad_token, self.tokenizer.cls_token,
            self.tokenizer.sep_token, "<s>", "</s>"
        ])
        special_tokens = {t for t in special_tokens if t is not None}
        special_ids = set(self.tokenizer.all_special_ids)

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Mean Activation (Feature {feature_idx})"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]
            if not batch_texts:
                continue

            # Use configurable max_token_length
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()

            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                for b in range(features.shape[0]):
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len == 0:
                        continue

                    token_activations = features[b, :seq_len, feature_idx]
                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)
                    for pos, token in enumerate(sample_tokens):
                        if token in special_tokens or sample_token_ids[pos] in special_ids:
                            token_mask[pos] = False

                    if not torch.any(token_mask):
                        continue

                    non_special_activations = token_activations[token_mask]
                    if len(non_special_activations) == 0:
                        continue

                    mean_activation = torch.mean(non_special_activations).item()

                    # Find token with max activation (for window)
                    max_val, max_idx = torch.max(token_activations[token_mask], dim=0)
                    valid_positions = torch.where(token_mask)[0]
                    max_position = valid_positions[max_idx].item()
                    max_token_id = sample_token_ids[max_position]
                    max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False)

                    start = max(0, max_position - window_size)
                    end = min(seq_len, max_position + window_size + 1)
                    window_ids = token_ids[b][start:end]
                    window_text_raw = self.tokenizer.decode(window_ids, skip_special_tokens=False)
                    window_text_cleaned = window_text_raw
                    for tok in special_tokens:
                        window_text_cleaned = window_text_cleaned.replace(tok or "", "")
                    window_text_cleaned = window_text_cleaned.strip()

                    results.append({
                        "text": batch_texts[b],
                        "mean_activation": mean_activation,
                        "max_token": max_token,
                        "windowed_text": window_text_cleaned
                    })

        results.sort(key=lambda x: x["mean_activation"], reverse=True)  # highest first
        return results[:top_n]

    def visualize_token_activations(self, text, feature_idx, output_dir=None, window_size=10):
        """
        Create matplotlib visualization of token activations.
        Uses color intensity to show activation strength.

        Args:
            text: Text to visualize
            feature_idx: Feature index to visualize
            output_dir: Directory to save visualization (if None, uses temp dir)
            window_size: Max number of tokens before and after highest activation to visualize
             (Note: This window_size here is for visualization display, not activation extraction)

        Returns:
            Path to the saved image
        """
        # Get token activations for the provided text snippet
        # Tokenize the input text for visualization - use configurable max_token_length
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_token_length).to(self.device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()
        if seq_len == 0:
             logging.warning(f"Visualization input text '{text[:50]}...' tokenized to empty sequence.")
             tokens_to_viz = []
             activations_to_viz = []
        else:
            tokens_raw = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][:seq_len])
            token_ids = inputs["input_ids"][0][:seq_len].cpu().numpy()
            special_tokens_set = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                                  self.tokenizer.pad_token, self.tokenizer.cls_token,
                                  self.tokenizer.sep_token, "<s>", "</s>"])
            special_ids_set = set(self.tokenizer.all_special_ids)

            with torch.no_grad():
                 outputs = self.base_model(**inputs, output_hidden_states=True)
                 hidden_states = outputs.hidden_states[self.target_layer]
                 # Ensure hidden_states dtype matches SAE encoder weight dtype
                 features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))
                 all_activations = features[0, :seq_len, feature_idx].cpu().numpy()

            tokens_to_viz = []
            activations_to_viz = []
            for i in range(seq_len):
                 # Only include non-special tokens for visualization display
                 # Check against both token string and token ID
                 if tokens_raw[i] not in special_tokens_set and token_ids[i] not in special_ids_set:
                     # Clean common prefix for visualization
                     cleaned_token = tokens_raw[i].replace('Ġ', ' ').replace(' ', ' ')
                     tokens_to_viz.append(cleaned_token)
                     activations_to_viz.append(all_activations[i])

        # Skip visualization if no valid tokens remain after filtering
        if not tokens_to_viz:
            logging.warning(f"No non-special tokens to visualize for feature {feature_idx} in text: {text[:50]}...")
            # Create a simple placeholder image
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(0.5, 0.5, "No non-special tokens to visualize",
                     ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Feature {feature_idx} Activation (No Valid Tokens)")
            ax.axis('off')

            # Save placeholder image
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
                img_path = os.path.join(output_dir, img_filename)
            else:
                import tempfile
                temp_dir = tempfile.gettempdir()
                img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
                img_path = os.path.join(temp_dir, img_filename)

            try:
                 plt.savefig(img_path)
                 plt.close(fig)
                 return img_path
            except Exception as e:
                 logging.error(f"Error saving placeholder image: {e}")
                 plt.close(fig)
                 return None

        # Create plot showing activation intensity with color
        fig, ax = plt.subplots(figsize=(max(len(tokens_to_viz)*0.4, 10), 3))

        # Use a heatmap-style visualization
        # Only positive activations get color
        norm_activations = [max(0, a) for a in activations_to_viz]

        max_activation_val = max(norm_activations) if norm_activations else 1.0

        # Create rectangles for each token
        for i, (token, act) in enumerate(zip(tokens_to_viz, norm_activations)):
            # Normalize activation to [0, 1] range based on the max in this snippet
            color_intensity = min(act / max_activation_val if max_activation_val > 0 else 0, 1.0)
            # Orange color scheme with intensity
            color = (1.0, 0.5 * (1 - color_intensity), 0.2 * (1 - color_intensity))

            # Use text with background box
            ax.text(i, 0.5, token, ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.7, pad=3, edgecolor='none'))

        ax.set_xlim(-0.5, len(tokens_to_viz) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"Feature {feature_idx} Activation")
        ax.axis('off')  # Hide axes

        plt.tight_layout()

        # Save image to output directory or temp dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
            img_path = os.path.join(output_dir, img_filename)
        else:
            import tempfile
            temp_dir = tempfile.gettempdir()
            img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
            img_path = os.path.join(temp_dir, img_filename)

        try:
            plt.savefig(img_path)
            plt.close(fig)
            return img_path
        except Exception as e:
            logging.error(f"Error saving visualization image: {e}")
            plt.close(fig)
            return None

    async def identify_feature_pattern(self, feature_idx, example_texts, output_dir=None, example_imgs=None, claude_examples=10):
        """
        Use Claude API to identify the semantic pattern of a feature using full statements.

        Args:
            feature_idx: The feature index
            example_texts: List of example texts (or dicts with text info). Use up to claude_examples examples.
            output_dir: Directory to save results
            example_imgs: Optional list of image paths (not used by Claude-3 Opus text prompt)
            claude_examples: Number of examples to send to Claude API

        Returns:
            Identified pattern as a string
        """
        feature_id_logger = logging.getLogger('feature_id')

        # Format examples properly using FULL TEXTS
        formatted_examples = []
        for i, example in enumerate(example_texts[:claude_examples]):  # Use configurable number of examples
            if isinstance(example, dict):
                # Always prioritize the full text
                text_content = example.get("full_text", example.get("text", example.get("windowed_text", "")))

                # --- REMOVED ACTIVATION SCORE AND MAX TOKEN FROM CLAUDE PROMPT ---
                # The user requested these not be submitted to the Claude API.
                # The following lines are removed:
                # max_token = example.get("max_token", "")
                # activation = example.get("max_activation", 0)
                # if max_token:
                #     text_content += f" [Max activation: {activation:.2f} on token: '{max_token}']"
                # --- END REMOVAL ---
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
                formatted_examples.append(f"Example {i+1}: {text_content}")

        if not formatted_examples:
            feature_id_logger.warning(f"No valid examples to send to Claude for feature {feature_idx} identification.")
            return "Unknown pattern (No valid examples)"

        # Define the separator just like in your original script
        separator = '-' * 40

        # Create a customized Claude prompt that returns a concise pattern
        try:
            claude_prompt = self.identification_prompt.format(
                examples=chr(10).join(formatted_examples),
                separator=separator
            )
        except KeyError as e:
            feature_id_logger.error(f"Prompt template is missing expected placeholder: {e}")
            return f"Error: Prompt template missing placeholder {e}"

        feature_id_logger.debug(f"CLAUDE PATTERN IDENTIFICATION PROMPT (Feature {feature_idx}):\n{claude_prompt}")

        # Ensure output directory exists for Claude response
        if output_dir:
            claude_output_dir = os.path.join(output_dir, f"feature_{feature_idx}_examples")
            os.makedirs(claude_output_dir, exist_ok=True)
        else:
            claude_output_dir = "."

        if not self.claude_api_key:
            feature_id_logger.error("Claude API key is missing. Cannot identify pattern.")
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
                {"role": "user", "content": claude_prompt}
            ]
        }

        try:
            # Save the prompt
            if output_dir:
                prompt_path = os.path.join(claude_output_dir, "claude_pattern_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(claude_prompt)

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()
            claude_response = response_data["content"][0]["text"]

            feature_id_logger.debug(f"CLAUDE PATTERN IDENTIFICATION RESPONSE (Feature {feature_idx}):\n{claude_response}")

            # Save the raw response
            if output_dir:
                claude_response_path = os.path.join(claude_output_dir, "claude_pattern_response.txt")
                with open(claude_response_path, "w", encoding="utf-8") as f:
                    f.write(claude_response)

            # Parse the response to extract the pattern
            pattern_parts = claude_response.split("EXPLANATION:", 1)
            concise_pattern = pattern_parts[0].strip() if pattern_parts else "Unknown pattern (parsing failed)"
            explanation = pattern_parts[1].strip() if len(pattern_parts) > 1 else ""

            # Save the parsed pattern
            if output_dir:
                pattern_path = os.path.join(claude_output_dir, "pattern.txt")
                with open(pattern_path, "w", encoding="utf-8") as f:
                    f.write(f"{concise_pattern}\n{explanation}")

            return concise_pattern

        except Exception as e:
            feature_id_logger.error(f"Error calling Claude API: {e}")
            return f"Error identifying pattern: {str(e)}"

    def compute_feature_statistics(self, feature_idx, texts, n_samples=5000, batch_size=16):
        """Compute statistics for a feature across texts."""
        # Sample texts if needed
        if len(texts) > n_samples:
            sampled_texts = random.sample(texts, n_samples)
        else:
            sampled_texts = texts

        activations = []
        active_token_pct = []

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}
        special_ids = set(self.tokenizer.all_special_ids)

        # Process in batches
        for i in tqdm(range(0, len(sampled_texts), batch_size), desc="Computing statistics", leave=False):
            batch_texts = sampled_texts[i:i+batch_size]
            # Filter empty texts
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            # Tokenize batch - Use configurable max_token_length
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_token_length).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy() # Need token_ids for special token check

            with torch.no_grad():
                # Process batch through model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                # Process through SAE
                # Ensure hidden_states dtype matches SAE encoder weight dtype
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                # For each sample in batch
                for b in range(features.shape[0]):
                    # Get valid sequence length (exclude padding)
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    if seq_len > 0:  # Skip if sequence is empty
                        # Get activations for this feature
                        token_activations = features[b, :seq_len, feature_idx].cpu().numpy()
                        sample_token_ids = token_ids[b][:seq_len]
                        sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                        # Create a mask for non-special tokens
                        token_mask = np.ones_like(token_activations, dtype=bool)
                        for k in range(seq_len):
                            # Check against both token string and token ID
                            if sample_tokens[k] in special_tokens or sample_token_ids[k] in special_ids:
                                token_mask[k] = False

                        # Only consider activations for non-special tokens
                        non_special_activations = token_activations[token_mask]

                        # Accumulate statistics
                        activations.extend(non_special_activations)
                        # Calculate percent active tokens among non-special tokens
                        if np.sum(token_mask) > 0:
                            active_token_pct.append(np.mean(non_special_activations > 0))
                        else:
                            # If only special tokens, 0% active non-special tokens
                            active_token_pct.append(0.0)

        # Convert to numpy arrays
        activations = np.array(activations)
        active_token_pct = np.array(active_token_pct)

        # Compute statistics
        stats = {
            "mean_activation": float(np.mean(activations)) if len(activations) > 0 else 0.0,
            "max_activation": float(np.max(activations)) if len(activations) > 0 else 0.0,
            "median_activation": float(np.median(activations)) if len(activations) > 0 else 0.0,
            "percent_active_non_special_tokens": float(np.mean(activations > 0) * 100) if len(activations) > 0 else 0.0,
            "mean_text_active_pct_non_special_tokens": float(np.mean(active_token_pct) * 100) if len(active_token_pct) > 0 else 0.0,
            "feature_idx": feature_idx
        }

        # Save to instance
        self.feature_stats[feature_idx] = stats

        return stats

    def clamp_feature_intervention(self, feature_idx, text, clamp_values=None):
        """
        Original clamping experiment with a generic prompt.

        Args:
            feature_idx: The feature to clamp
            text: Input text prompt
            clamp_values: List of values to clamp the feature to (default: [0.0, 2*max, 5*max])

        Returns:
            Dictionary with generated text for each clamp value
        """
        results = {}

        # Define clamp values if not provided
        if clamp_values is None:
            # Default: zero, 2x max, 5x max activation observed for this feature
            max_val = self.feature_stats.get(feature_idx, {}).get('max_activation', 1.0)
            clamp_values = [0.0, max_val * 2, max_val * 5]
            print(f"Clamping feature {feature_idx} to default values based on max activation ({max_val:.4f}): {clamp_values}")
        else:
             print(f"Clamping feature {feature_idx} to specified values: {clamp_values}")

        # Get base output (without clamping)
        print("Generating base output (unclamped)...")
        try:
            base_output = self._generate_with_feature_clamped(text, None)
            results["base"] = base_output
        except Exception as e:
            logging.error(f"Error generating base output for clamping feature {feature_idx}: {e}")
            results["base"] = f"Error generating base output: {str(e)}"

        # Get outputs with clamped values
        for value in clamp_values:
            clamp_dict = {feature_idx: value}
            clamp_key = f"clamp_{value:.4f}" # Use more precise key for float values
            print(f"Generating output with feature {feature_idx} clamped to {value:.4f}...")
            try:
                output = self._generate_with_feature_clamped(text, clamp_dict)
                results[clamp_key] = output
            except Exception as e:
                 logging.error(f"Error generating clamped output ({value:.4f}) for feature {feature_idx}: {e}")
                 results[clamp_key] = f"Error generating output clamped to {value:.4f}: {str(e)}"

        return results

    def category_classification_intervention(self, feature_idx, categories, examples, clamp_values=None):
        """
        Run a classification experiment to test if clamping affects category classification.

        Args:
            feature_idx: The feature to clamp
            categories: List of category names (e.g., ["premise accuracy true", "premise accuracy false"])
            examples: List of examples to test (one from each category)
            clamp_values: List of values to clamp the feature to (default: [0.0, 2*max, 5*max])

        Returns:
            Results dictionary with classifications under different clamping conditions
        """
        if len(categories) != 2:
            logging.error(f"Expected exactly 2 categories, got {len(categories)}")
            return {"error": f"Expected exactly 2 categories, got {len(categories)}"}

        if len(examples) != 2:
            logging.error(f"Expected exactly 2 examples (one per category), got {len(examples)}")
            return {"error": f"Expected exactly 2 examples (one per category), got {len(examples)}"}

        results = {}

        # Define clamp values if not provided
        if clamp_values is None:
            # Default: zero, 2x max, 5x max activation observed for this feature
            max_val = self.feature_stats.get(feature_idx, {}).get('max_activation', 1.0)
            clamp_values = [0.0, max_val * 2, max_val * 5]
            print(f"Classification clamping using values: {clamp_values}")

        # For each example, test classification with different clamping
        for i, example in enumerate(examples):
            example_results = {}
            # Construct the classification prompt
            prompt = f"Is this prompt {categories[0]} or {categories[1]}? {example}"

            # Baseline (no clamping)
            print(f"Testing classification of example {i+1} (baseline)...")
            try:
                base_output = self._generate_with_feature_clamped(prompt, None)
                example_results["base"] = base_output
            except Exception as e:
                logging.error(f"Error generating classification (baseline) for example {i+1}: {e}")
                example_results["base"] = f"Error: {str(e)}"

            # Test with different clamping values
            for value in clamp_values:
                clamp_dict = {feature_idx: value}
                clamp_key = f"clamp_{value:.4f}"
                print(f"Testing classification of example {i+1} with feature {feature_idx} clamped to {value:.4f}...")

                try:
                    output = self._generate_with_feature_clamped(prompt, clamp_dict)
                    example_results[clamp_key] = output
                except Exception as e:
                    logging.error(f"Error generating classification with {value:.4f} for example {i+1}: {e}")
                    example_results[clamp_key] = f"Error: {str(e)}"

            # Determine which category this example belongs to for the result key
            # This assumes the order in `examples` matches the order in `categories`
            # A more robust way would be to pass the actual category label with the example
            # For minimal change, we'll assume example 0 is category 0, example 1 is category 1.
            category_label_for_key = categories[i].replace(' ', '_')
            results[f"example_{i+1}_{category_label_for_key}"] = example_results

        return results

    def _generate_with_feature_clamped(self, text, clamp_features=None):
        """
        Generate text with features clamped to specific values using the local base model.

        Args:
            text: Input text
            clamp_features: Dict mapping {feature_idx: value} or None

        Returns:
            Generated text
        """
        # Tokenize the input text - Use configurable max_token_length for generation prompt
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token_length).to(self.device)

        # Set up hooks for clamping features during generation
        hooks = []
        hook = None # Initialize hook outside the if block

        if clamp_features:
            # Define the forward hook that modifies SAE features
            def forward_hook(module, input_tensor, output):
                # The SAE forward is expected to return (features, reconstruction)
                features, reconstruction = output

                # Clamp specified features IN PLACE on the 'features' tensor
                for feat_idx, value in clamp_features.items():
                    if feat_idx < features.shape[-1]: # Ensure feature index is valid within the hidden dimension
                         # Clamp the feature activation value across all tokens in the sequence for this batch item
                         features[:, :, feat_idx] = value # Modify features in place
                    else:
                         logging.warning(f"Attempted to clamp invalid feature index {feat_idx}. Max feature index is {features.shape[-1]-1}")

                # Return the modified features and the original reconstruction
                return (features, reconstruction) # Return the modified output tuple

            try:
                 # Register the hook on the SAE model's forward pass
                 hook = self.sae_model.register_forward_hook(forward_hook)
                 hooks.append(hook)
                 logging.debug(f"Registered forward hook on SAE model for clamping.")

            except Exception as e:
                 logging.error(f"Failed to register forward hook on SAE model for clamping: {e}")

        # Generate with hooks active
        result = ""
        try:
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs, # Pass input_ids and attention_mask from tokenization
                    max_new_tokens=300, # Increase max tokens for generated response length
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True, # Returns a dictionary-like object
                )

            # Get the generated tokens from the output object
            generated_sequence = outputs.sequences[0]

            # Decode the generated sequence, skipping special tokens
            # Start decoding AFTER the input prompt tokens to get only the new text
            input_len = inputs['input_ids'].shape[1]
            # Ensure generated_sequence is longer than input_len before slicing to avoid index errors
            if generated_sequence.shape[0] > input_len:
                result = self.tokenizer.decode(generated_sequence[input_len:], skip_special_tokens=True)
            else:
                # This case happens if the model generates 0 new tokens
                result = "" # No new tokens were generated

        except Exception as e:
            logging.error(f"Error during text generation with clamping hook: {e}")
            result = f"Error during generation: {str(e)}"
        finally:
            # Always remove hooks to avoid side effects on subsequent generations
            if hook is not None:
                 hook.remove()
                 logging.debug(f"Removed generation hook.")

        return result

def load_identification_prompt():
    """Load the identification prompt template."""
    # Check in the current directory first, then a 'prompts' subdirectory
    prompt_path = Path("feature_identification_prompt.txt")
    if not prompt_path.exists():
        prompt_path = Path("prompts/feature_identification_prompt.txt")

    if not prompt_path.exists():
        logging.warning(f"Prompt template not found at {prompt_path}. Using default.")
        return """I'll analyze examples where a specific neural network feature activates strongly.
Please help identify the pattern this feature might be detecting.

{examples}

{separator}

Based on these examples, what's a concise description of the pattern this feature might be detecting?
Format your response as:
Pattern: [concise description]
EXPLANATION: [brief explanation]"""

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

async def analyze_feature_meaning(
    args, # Added args as a parameter here
    feature_indices,
    model_path,
    sae_path,
    output_dir,
    input_json=None,
    num_examples=20,
    num_samples=10000, # Increased default num_samples
    window_size=10,
    batch_size=16,
    config_dir=None,
    run_clamping=True,
    max_token_length=100,  # NEW: Add max_token_length parameter
    claude_examples=10  # NEW: Add claude_examples parameter
):
    """
    Analyze the meaning of a set of SAE features.

    Args:
        args: The parsed command-line arguments object. # Added description
        feature_indices: List of feature indices to analyze
        model_path: Path to base language model
        sae_path: Path to trained SAE model
        output_dir: Directory to save results
        input_json: Path to JSON file with category examples (for classification experiment)
        num_examples: Number of examples to extract per feature
        num_samples: Number of samples to use from dataset
        window_size: Window size for context around activations
        batch_size: Batch size for processing
        config_dir: Directory with API key config
        run_clamping: Whether to run clamping interventions
        max_token_length: Maximum token length for tokenization (NEW)
        claude_examples: Number of examples to send to Claude API (NEW)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "feature_analysis.log"),
            logging.StreamHandler()
        ]
    )

    # Log parameters
    # This line now has access to args
    logging.info(f"Features argument provided: {args.features}")
    logging.info(f"Analyzing features: {feature_indices}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"SAE path: {sae_path}")
    logging.info(f"Number of examples per feature: {num_examples}")
    logging.info(f"Number of dataset samples: {num_samples}")
    logging.info(f"Window size: {window_size}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Run clamping interventions: {run_clamping}")
    logging.info(f"Max token length: {max_token_length}")  # NEW: Log max_token_length
    logging.info(f"Claude examples: {claude_examples}")  # NEW: Log claude_examples

    # Load Claude API key if available
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

    # Load the identification prompt
    identification_prompt = load_identification_prompt()

    # Load models
    logging.info("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Set pad_token to eos_token if it's None
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

    logging.info(f"Creating SAE with input_dim={input_dim}, hidden_dim={hidden_dim}")
    sae_model = SparseAutoencoder(input_dim, hidden_dim)
    sae_model.load_state_dict(state_dict)
    sae_model.to(model.device)
    sae_model.eval()

    # Initialize analyzer with configurable max_token_length
    analyzer = SAEFeatureAnalyzer(
        sae_model,
        tokenizer,
        model,
        identification_prompt=identification_prompt,
        claude_api_key=claude_api_key,
        max_token_length=max_token_length  # NEW: Pass max_token_length
    )

    # Load wikitext dataset - using all data with chunking
    logging.info(f"Loading wikitext dataset with chunking...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        
        # Use all data instead of specific chunks
        # Simply iterate through all indices up to num_samples or dataset length
        max_indices_to_check = min(len(dataset), num_samples * 3)  # Check 3x num_samples to account for filtering
        
        # Collect texts from all data, with optional chunking
        filtered_texts = []
        for idx in range(max_indices_to_check):
            text = dataset[idx]["text"]
            if text.strip():
                # First check if the full text is long enough to potentially yield chunks
                full_inputs = tokenizer(text, return_tensors="pt", truncation=False)
                full_length = torch.sum(full_inputs["attention_mask"][0]).item()
                
                if full_length > max_token_length:
                    # Text is longer than max_token_length, so chunk it
                    # Decode to get clean text, then re-encode in chunks
                    clean_text = tokenizer.decode(full_inputs["input_ids"][0], skip_special_tokens=True)
                    tokens = tokenizer.encode(clean_text, add_special_tokens=False)
                    
                    # Split into non-overlapping chunks
                    for chunk_start in range(0, len(tokens), max_token_length):
                        chunk_tokens = tokens[chunk_start:chunk_start + max_token_length]
                        if len(chunk_tokens) > 5:  # Keep chunks with >5 tokens
                            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                            filtered_texts.append(chunk_text)
                            
                            # Stop if we have enough samples
                            if len(filtered_texts) >= num_samples:
                                break
                else:
                    # Text is short enough, just truncate normally
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_token_length)
                    if torch.sum(inputs["attention_mask"][0]).item() > 5:
                        truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                        filtered_texts.append(truncated_text)
                        
                        # Stop if we have enough samples
                        if len(filtered_texts) >= num_samples:
                            break
            
            # Break outer loop if we have enough samples
            if len(filtered_texts) >= num_samples:
                break

        # Use the collected texts
        texts = filtered_texts

        logging.info(f"Loaded {len(texts)} text samples (with chunking) from wikitext-103 (max_token_length={max_token_length})")

    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        logging.info("Please ensure you have an internet connection to download the dataset or use a local path.")
        return  # Exit if dataset loading fails

    # Load category information if available
    category_examples = None
    # Corrected category names based on the provided JSON file
    category_names = ["premise accuracy true", "premise accuracy false"]

    if input_json:
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract one example from each category for classification test
                # Corrected keys based on the provided JSON file
                true_statements = data.get("premise_accuracy_true", [])
                false_statements = data.get("premise_accuracy_false", [])

                if true_statements and false_statements:
                    # Take the first example from each list
                    category_examples = [true_statements[0], false_statements[0]]
                    logging.info(f"Loaded category examples from {input_json}")
                else:
                    # Log a warning if keys are found but lists are empty
                    if "premise_accuracy_true" in data or "premise_accuracy_false" in data:
                         logging.warning(f"Found keys 'premise_accuracy_true' or 'premise_accuracy_false' in {input_json}, but the lists are empty. Skipping classification experiment.")
                    else:
                         logging.warning(f"Could not find expected keys 'premise_accuracy_true' or 'premise_accuracy_false' in {input_json}. Skipping classification experiment.")

        except Exception as e:
            logging.error(f"Error loading category data from {input_json}: {e}")
            logging.warning("Skipping classification experiment due to error loading input JSON.")

    # Analyze each feature
    results = []

    for feature_idx in feature_indices:
        logging.info(f"\nAnalyzing feature {feature_idx}")
        feature_result = {"feature_idx": int(feature_idx)}

        # Create feature directory
        feature_dir = output_dir / f"feature_{feature_idx}"
        feature_dir.mkdir(exist_ok=True)

        # Find top examples
        logging.info(f"Finding top {num_examples} examples...")
        if args.find_mean_lowest:
            top_examples = analyzer.find_lowest_mean_activating_examples(
                feature_idx, texts, top_n=num_examples, batch_size=batch_size
            )
        elif args.find_weakest:
            top_examples = analyzer.find_lowest_activating_examples(
                feature_idx, texts, top_n=num_examples, batch_size=batch_size, window_size=window_size
            )
        elif args.find_mean_highest:
            top_examples = analyzer.find_highest_activating_examples(
                feature_idx, texts, top_n=num_examples, batch_size=batch_size, window_size=window_size
            )
        else:
            top_examples = analyzer.find_highest_mean_activating_examples(
                feature_idx, texts, top_n=num_examples, batch_size=batch_size, window_size=window_size
            )

        # Compute feature statistics
        logging.info("Computing feature statistics...")
        stats = analyzer.compute_feature_statistics(
            feature_idx,
            texts,
            n_samples=min(2000, len(texts)),  # Use up to 2000 samples for stats
            batch_size=batch_size
        )
        feature_result["statistics"] = stats
        logging.info(f"Statistics: {stats}")

        # Store examples and create visualizations
        example_imgs = []

        # Save examples to a file
        with open(feature_dir / "examples.txt", "w", encoding="utf-8") as f:
            for i, example in enumerate(top_examples):
                f.write(f"--- Example {i+1} ---\n")
                activation_value = example.get("max_activation", example.get("mean_activation", 0.0))
                f.write(f"Activation Score: {activation_value:.4f}\n")
                f.write(f"Max Token: '{example['max_token']}'\n")
                f.write(f"Windowed Text:\n{example['windowed_text']}\n\n")
                f.write(f"Full Text (max {max_token_length} tokens):\n{example['text']}\n\n")
                f.write("-"*50 + "\n\n")

        # Create visualizations
        logging.info("Creating visualizations...")
        for i, example in enumerate(top_examples[:5]):  # Visualize top 5
            img_path = analyzer.visualize_token_activations(
                example["windowed_text"], # Visualize the windowed text
                feature_idx,
                output_dir=feature_dir,
                window_size=window_size # This window_size is for visualization display, not extraction
            )
            if img_path: # Only add if image was successfully created
                 example_imgs.append(os.path.basename(img_path))
        logging.info(f"Created {len(example_imgs)} visualizations.")

        # Store examples with both full and windowed text
        feature_result["examples"] = [{
            "full_text": ex["text"], # This is the text truncated to max_token_length tokens
            "windowed_text": ex["windowed_text"],
            "max_activation": float(ex.get("max_activation", ex.get("mean_activation", 0.0))),
            "max_token": ex["max_token"]
        } for ex in top_examples]

        feature_result["example_imgs"] = example_imgs

        # Identify pattern using Claude API
        if claude_api_key:
            logging.info("Identifying feature pattern using Claude API...")
            feature_result["pattern"] = await analyzer.identify_feature_pattern(
                feature_idx,
                feature_result["examples"], # Pass the examples list
                output_dir=output_dir, # Pass the main output_dir for Claude response saving
                claude_examples=claude_examples  # NEW: Pass claude_examples parameter
            )
            logging.info(f"Identified pattern: {feature_result['pattern']}")
        else:
            feature_result["pattern"] = "Pattern identification skipped (no API key)"
            logging.warning("Pattern identification skipped (no API key)")

        # Clamping interventions
        if run_clamping:
            # Original generic clamping experiment
            logging.info("Running clamping interventions with generic prompt...")
            intervention_prompt = "Human: What's your favorite animal?\n\nAssistant:"
            try:
                clamp_results = analyzer.clamp_feature_intervention(feature_idx, intervention_prompt)
                feature_result["clamp_results"] = clamp_results

                # Save clamping results to file
                with open(feature_dir / "clamping_results.txt", "w", encoding="utf-8") as f:
                    f.write(f"Feature {feature_idx} Clamping Results\n\n")
                    f.write(f"=== GENERIC PROMPT INTERVENTION ===\n")
                    f.write(f"Input: {intervention_prompt}\n\n")
                    f.write(f"Base (Unclamped):\n{clamp_results.get('base', 'N/A')}\n\n")
                    for key, value in clamp_results.items():
                        if key != "base":
                            f.write(f"{key}:\n{value}\n\n")

                logging.info(f"Clamping results obtained with generic prompt")
            except Exception as e:
                logging.error(f"Error running clamping intervention: {e}")
                feature_result["clamp_results"] = {"error": str(e)}

            # Category classification experiment (if category examples available)
            if category_examples:
                logging.info("Running category classification experiment...")
                try:
                    classification_results = analyzer.category_classification_intervention(
                        feature_idx,
                        category_names,
                        category_examples # Pass the loaded examples
                    )
                    feature_result["classification_results"] = classification_results

                    # Save classification results to file
                    with open(feature_dir / "classification_results.txt", "w", encoding="utf-8") as f:
                        f.write(f"Feature {feature_idx} Classification Results\n\n")
                        for example_key, example_results in classification_results.items():
                            f.write(f"=== {example_key.upper()} ===\n")
                            # Find the correct example text based on the key
                            example_text_for_display = "N/A"
                            if "example_1" in example_key and len(category_examples) > 0:
                                example_text_for_display = category_examples[0]
                            elif "example_2" in example_key and len(category_examples) > 1:
                                example_text_for_display = category_examples[1]

                            f.write(f"Prompt: Is this prompt {category_names[0]} or {category_names[1]}? {example_text_for_display}\n\n")
                            f.write(f"Base (Unclamped):\n{example_results.get('base', 'N/A')}\n\n")
                            for key, value in example_results.items():
                                if key != "base":
                                    f.write(f"{key}:\n{value}\n\n")
                            f.write("\n")

                    logging.info(f"Classification results obtained")
                except Exception as e:
                    logging.error(f"Error running classification experiment: {e}")
                    feature_result["classification_results"] = {"error": str(e)}
            else:
                logging.info("Skipping classification experiment (no category examples provided or loaded)")
        else:
            feature_result["clamp_results"] = {"note": "Clamping skipped"}
            logging.info("Clamping interventions skipped")

        # Store results
        results.append(feature_result)

        # Save incremental results
        # Ensure results are convertible to JSON (e.g., handle numpy types)
        try:
            with open(output_dir / "feature_analysis_results.json", "w", encoding="utf-8") as f:
                 # Use the convert_numpy_types function from part3 or define one here if needed
                 # For now, assuming standard types are used or handled by json.dump
                 # If numpy types persist, add a conversion function.
                 json.dump(results, f, indent=2)
            logging.info(f"Incremental results saved for feature {feature_idx}")
        except Exception as e:
             logging.error(f"Error saving incremental results for feature {feature_idx}: {e}")

    # Create summary report
    logging.info("Creating summary report...")
    with open(output_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write("# SAE Feature Analysis Summary\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"SAE: {sae_path}\n")
        f.write(f"Max Token Length: {max_token_length}\n\n")  # NEW: Include max_token_length in summary
        f.write("## Analyzed Features\n\n")

        for feature_result in results:
            feature_idx = feature_result["feature_idx"]
            pattern = feature_result.get("pattern", "N/A")
            stats = feature_result.get("statistics", {})

            f.write(f"### Feature {feature_idx}\n\n")
            f.write(f"**Pattern:** {pattern}\n\n")

            # Statistics
            f.write(f"**Statistics:**\n")
            f.write(f"- Max activation: {stats.get('max_activation', 0.0):.4f}\n")
            f.write(f"- Mean activation: {stats.get('mean_activation', 0.0):.4f}\n")
            f.write(f"- Percent active tokens (non-special): {stats.get('percent_active_non_special_tokens', 0.0):.2f}%\n\n")

            # Top examples
            f.write("**Top Examples:**\n\n")
            examples = feature_result.get("examples", [])
            if examples:
                for i, example in enumerate(examples[:3]):  # Show top 3
                    f.write(f"Example {i+1}:\n")
                    # Show both full text (if not too long) and windowed context
                    full_text = example.get('full_text', 'N/A')
                    if len(full_text) > 300:
                        full_text = full_text[:300] + "..."
                    f.write(f"Full text: ```\n{full_text}\n```\n")
                    f.write(f"Activation context: ```\n{example.get('windowed_text', 'N/A')}\n```\n")
                    f.write(f"Max Token: '{example.get('max_token', 'N/A')}'\n")
                    f.write(f"Activation: {example.get('max_activation', 0.0):.4f}\n\n")
            else:
                f.write("No top examples found.\n\n")

            # Clamping Results
            f.write("**Clamping Results:**\n\n")
            clamp_results = feature_result.get("clamp_results", {})

            if "error" in clamp_results or "note" in clamp_results:
                f.write(f"{clamp_results.get('error', clamp_results.get('note', 'N/A'))}\n\n")
            else:
                f.write("Generic prompt clamping:\n")
                f.write(f"Base: ```\n{clamp_results.get('base', 'N/A')}\n```\n")
                # Use the key format generated in clamp_feature_intervention
                max_val = stats.get('max_activation', 1.0)
                clamp_0_key = f'clamp_{0.0:.4f}'
                clamp_2x_key = f'clamp_{(max_val * 2):.4f}'
                clamp_5x_key = f'clamp_{(max_val * 5):.4f}'

                f.write(f"Clamped to 0.0: ```\n{clamp_results.get(clamp_0_key, 'N/A')}\n```\n\n")
                f.write(f"Clamped to {max_val * 2:.4f}: ```\n{clamp_results.get(clamp_2x_key, 'N/A')}\n```\n\n")
                f.write(f"Clamped to {max_val * 5:.4f}: ```\n{clamp_results.get(clamp_5x_key, 'N/A')}\n```\n\n")

            # Classification Results
            classification_results = feature_result.get("classification_results", {})
            if classification_results and "error" not in classification_results:
                f.write("**Classification Results:**\n\n")
                for example_key, example_results in classification_results.items():
                    f.write(f"{example_key}:\n")
                    f.write(f"Base: ```\n{example_results.get('base', 'N/A')}\n```\n")
                     # Use the key format generated in category_classification_intervention
                    max_val = stats.get('max_activation', 1.0)
                    clamp_0_key = f'clamp_{0.0:.4f}'
                    clamp_2x_key = f'clamp_{(max_val * 2):.4f}'
                    clamp_5x_key = f'clamp_{(max_val * 5):.4f}'

                    f.write(f"Clamped to 0.0: ```\n{example_results.get(clamp_0_key, 'N/A')}\n```\n\n")
                    f.write(f"Clamped to {max_val * 2:.4f}: ```\n{example_results.get(clamp_2x_key, 'N/A')}\n```\n\n")
                    f.write(f"Clamped to {max_val * 5:.4f}: ```\n{example_results.get(clamp_5x_key, 'N/A')}\n```\n\n")

            elif "error" in classification_results:
                 f.write(f"Classification experiment skipped or failed: {classification_results['error']}\n\n")
            else:
                 f.write("Classification experiment skipped (no category examples provided or loaded).\n\n")

            f.write("---\n\n")

    logging.info(f"Analysis complete! Results saved to {output_dir}")
    return results

async def main():
    parser = argparse.ArgumentParser(description="Analyze meaning of specific SAE features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--input_json", type=str, default=None, help="Optional JSON file with category examples")
    parser.add_argument("--features", type=str, required=True, help="Comma-separated list of feature indices or path to JSON file")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to extract per feature")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use from dataset")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for context around activations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--config_dir", type=str, default="../config", help="Directory with API key config")
    parser.add_argument("--no_clamping", action="store_true", help="Skip clamping interventions")
    parser.add_argument("--max_token_length", type=int, default=100, help="Maximum token length for tokenization (default: 100)")  # NEW: Add max_token_length argument

    parser.add_argument("--claude_examples", type=int, default=10, help="Number of examples to send to Claude API for pattern identification")

    #search for weakest
    parser.add_argument("--find_weakest", action="store_true", help="Find weakest activating examples instead of strongest")
    parser.add_argument("--find_mean_lowest", action="store_true", help="Find lowest overall examples instead of weakest token")
    parser.add_argument("--find_mean_highest", action="store_true", help="Find examples with highest mean activation across non-special tokens")

    args = parser.parse_args()

    # Parse feature indices
    feature_indices = []
    if os.path.isfile(args.features):
        # Load from file
        try:
            with open(args.features, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    feature_indices = data
                elif isinstance(data, dict) and "top_feature_indices" in data:
                    feature_indices = data["top_feature_indices"]
                # Added check for "rsa_feature_indices" key
                elif isinstance(data, dict) and "rsa_feature_indices" in data:
                     feature_indices = data["rsa_feature_indices"]
                     logging.info(f"Parsed feature indices from 'rsa_feature_indices' key in {args.features}")
                elif isinstance(data, dict):
                    # Try to find feature indices in a list of dicts format
                    for item in data.values():
                        if isinstance(item, dict) and "feature_idx" in item:
                            feature_indices.append(item["feature_idx"])
                    # Also check if the top level keys are feature indices
                    if not feature_indices:
                         try:
                              feature_indices = [int(k) for k in data.keys()]
                              logging.info(f"Parsed feature indices from JSON keys: {feature_indices[:10]}{'...' if len(feature_indices) > 10 else ''}")
                         except ValueError:
                              logging.warning(f"Could not parse feature indices from JSON keys in {args.features}. Expected integer keys.")
                              feature_indices = [] # Reset if parsing keys failed
                else:
                    logging.error(f"Unexpected JSON format in {args.features}. Expected list or dict with 'top_feature_indices', 'rsa_feature_indices', or feature_idx keys.")
                    feature_indices = [] # Clear indices if format is wrong

        except Exception as e:
             logging.error(f"Error loading or parsing feature indices from {args.features}: {e}")
             feature_indices = [] # Clear indices on error

    else:
        # Parse comma-separated list
        try:
            feature_indices = [int(x.strip()) for x in args.features.split(",") if x.strip()]
        except ValueError:
            logging.error(f"Could not parse comma-separated feature indices: {args.features}. Ensure they are integers.")
            feature_indices = []

    if not feature_indices:
        print("No feature indices found to analyze. Please check your --features argument or input file.")
        return

    print(f"Analyzing {len(feature_indices)} features: {feature_indices[:10]}{'...' if len(feature_indices) > 10 else ''}")
    print(f"Using max token length: {args.max_token_length}")  # NEW: Print max_token_length

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"feature_meaning_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Run analysis
    await analyze_feature_meaning(
        args, # Pass args here
        feature_indices=feature_indices,
        model_path=args.model_path,
        sae_path=args.sae_path,
        output_dir=output_path,
        input_json=args.input_json,
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        window_size=args.window_size,
        batch_size=args.batch_size,
        config_dir=args.config_dir,
        run_clamping=not args.no_clamping,
        max_token_length=args.max_token_length,  # NEW: Pass max_token_length
        claude_examples=args.claude_examples  # NEW: Pass claude_examples
    )

if __name__ == "__main__":
    asyncio.run(main())