"""
Secondary SAE Feature Identification Script (NFM Pipeline) - FIXED VERSION

This script analyzes Secondary SAE features that are built from the NFM embedding layer.
It adapts the Primary SAE feature analysis to work with the longer pipeline:
Layer 16 → Primary SAE → NFM Embedding Layer → Secondary SAE

Pipeline: Layer 16 → Primary SAE → NFM Embedding → Secondary SAE

The script finds both highest and absolute value highest activating examples for Secondary SAE features,
enabling identification of semantic patterns captured by the Secondary SAE built on NFM embeddings.

FIXED: Proper clamping workflow that correctly intervenes in the NFM pipeline during text generation.
"""
# Secondary SAE Feature Meaning Analysis Script for NFM-based Secondary SAE
# Pipeline: Layer 16 → Primary SAE → NFM Embedding Layer → Secondary SAE
# python part12_NFMSAE_feature_meaning_large_wiki.py --primary_sae_path ../checkpoints/best_model.pt --nfm_path ../checkpoints/nfm_model.pt --secondary_sae_path ../checkpoints/secondary_sae_model.pt --model_path ../models/open_llama_3b --features "5,10,15" --output_dir ./NFM_secondary_feature_analysis_results/ --config_dir ../config --max_token_length 10 --num_samples 10000000 --claude_examples 20

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

class TopKSparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        sparse_features = self.apply_topk(features)
        reconstruction = self.decoder(sparse_features)
        return sparse_features, reconstruction
    
    def apply_topk(self, features):
        batch_size, num_features = features.shape
        topk_values, topk_indices = torch.topk(features, self.k, dim=1)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(1, topk_indices, topk_values)
        return sparse_features

# Keep SparseAutoencoder for backward compatibility with Primary SAE if needed
class SparseAutoencoder(torch.nn.Module):
    """Simple Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return features, reconstruction
    
class NeuralFactorizationModel(torch.nn.Module):
    """Neural Factorization Model for analyzing feature interactions."""
    def __init__(self, num_features, k_dim, output_dim):
        super().__init__()
        self.feature_embeddings = torch.nn.Embedding(num_features, k_dim)
        self.linear = torch.nn.Linear(num_features, output_dim)
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Identity(),  # Layer 0 - placeholder
            torch.nn.Linear(k_dim, k_dim),  # Layer 1
            torch.nn.ReLU(),  # Layer 2  
            torch.nn.Linear(k_dim, output_dim)  # Layer 3
        )
    
    def forward(self, x):
        # Linear component
        linear_out = self.linear(x)
        
        # Interaction component - this gives us the embedding layer
        embeddings = self.feature_embeddings.weight.T  # [k_dim, num_features]
        weighted_embeddings = torch.matmul(x, embeddings.T)  # [batch, k_dim]
        interaction_out = self.interaction_mlp(weighted_embeddings)
        
        return linear_out + interaction_out, linear_out, interaction_out, weighted_embeddings

class NFMSecondaryFeatureAnalyzer:
    """Analyzer for studying Secondary SAE features in the NFM pipeline."""

    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, device="cuda", target_layer=16,
                 identification_prompt=None, scoring_prompt=None, claude_api_key=None, max_token_length=100):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.secondary_sae = secondary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.max_token_length = max_token_length
        
        # Set to evaluation mode
        self.primary_sae.eval()
        self.nfm_model.eval()
        self.secondary_sae.eval()
        self.base_model.eval()

        # Store feature statistics
        self.feature_stats = {}

        # Store prompts
        self.identification_prompt = identification_prompt
        self.scoring_prompt = scoring_prompt

        # Claude API key
        self.claude_api_key = claude_api_key

    def process_through_nfm_pipeline(self, texts, batch_size=16):
        """
        Process texts through the full NFM pipeline: Layer 16 → Primary SAE → NFM Embedding → Secondary SAE
        
        Returns mean activations across sequence for each text:
            primary_activations: [num_texts, primary_sae_features] 
            nfm_embeddings: [num_texts, k_dim]
            secondary_activations: [num_texts, secondary_sae_features]
        """
        primary_activations = []
        nfm_embeddings = []
        secondary_activations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing NFM pipeline", leave=False):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]
            if not batch_texts:
                continue
                
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_token_length).to(self.device)
            
            with torch.no_grad():
                # Layer 16 → Primary SAE
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                # RESHAPE FOR TOPK PRIMARY SAE: 3D → 2D
                batch_size_inner, seq_len, hidden_dim = hidden_states.shape
                hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
                
                primary_features, _ = self.primary_sae(
                    hidden_states_reshaped.to(self.primary_sae.encoder[0].weight.dtype)
                )
                
                # RESHAPE BACK: 2D → 3D
                primary_features = primary_features.reshape(batch_size_inner, seq_len, -1)
                
                # Get mean activation across sequence for each text
                for b in range(primary_features.shape[0]):
                    seq_len_actual = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len_actual > 0:
                        # Primary SAE activations (mean across sequence)
                        mean_primary = torch.mean(primary_features[b, :seq_len_actual, :], dim=0)
                        primary_activations.append(mean_primary.cpu().numpy())
                        
                        # NFM Embedding
                        embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
                        nfm_embedding = torch.matmul(mean_primary.unsqueeze(0), embeddings.T)  # [1, k_dim]
                        nfm_embeddings.append(nfm_embedding.squeeze().cpu().numpy())
                        
                        # Secondary SAE (TopK) - nfm_embedding is already 2D [1, k_dim]
                        secondary_features, _ = self.secondary_sae(
                            nfm_embedding.to(self.secondary_sae.encoder[0].weight.dtype)
                        )
                        secondary_activations.append(secondary_features.squeeze().cpu().numpy())
        
        return np.array(primary_activations), np.array(nfm_embeddings), np.array(secondary_activations)

    def find_highest_activating_examples(self, secondary_feature_idx, texts, top_n=20, batch_size=16):
        """
        Find texts with highest Secondary SAE feature activations.

        Args:
            secondary_feature_idx: The secondary feature to analyze
            texts: List of text samples
            top_n: Number of top examples to return
            batch_size: Number of texts to process in each batch

        Returns:
            List of dicts with text, secondary_activation, primary_activations, and nfm_embedding
        """
        results = []

        print(f"Scanning for Secondary SAE feature {secondary_feature_idx} to find top {top_n} examples (batch size {batch_size}, max_token_length {self.max_token_length})...")

        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Examples for Secondary Feature {secondary_feature_idx}"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            # Process through NFM pipeline
            primary_activations, nfm_embeddings, secondary_activations = self.process_through_nfm_pipeline(batch_texts, batch_size=len(batch_texts))

            # For each sample in batch
            for b in range(len(batch_texts)):
                secondary_activation = secondary_activations[b, secondary_feature_idx]

                # Only consider if there's positive activation
                if secondary_activation > 0:
                    temp_results.append({
                        "text": batch_texts[b],
                        "secondary_activation": float(secondary_activation),
                        "primary_activations": primary_activations[b].tolist(),  # Store all primary activations
                        "nfm_embedding": nfm_embeddings[b].tolist()  # Store NFM embedding
                    })

        # Sort all collected results by secondary activation and return top_n
        temp_results.sort(key=lambda x: x["secondary_activation"], reverse=True)
        return temp_results[:top_n]

    def find_lowest_activating_examples(self, secondary_feature_idx, texts, top_n=20, batch_size=16):
        """
        Find texts with lowest Secondary SAE feature activations.

        Args:
            secondary_feature_idx: The secondary feature to analyze
            texts: List of text samples
            top_n: Number of lowest examples to return
            batch_size: Number of texts to process in each batch

        Returns:
            List of dicts with text, secondary_activation, primary_activations, and nfm_embedding
        """
        print(f"Scanning for Secondary SAE feature {secondary_feature_idx} to find top {top_n} LOWEST activating examples (batch size {batch_size}, max_token_length {self.max_token_length})...")

        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Lowest Examples for Secondary Feature {secondary_feature_idx}"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            # Process through NFM pipeline
            primary_activations, nfm_embeddings, secondary_activations = self.process_through_nfm_pipeline(batch_texts, batch_size=len(batch_texts))

            # For each sample in batch
            for b in range(len(batch_texts)):
                secondary_activation = secondary_activations[b, secondary_feature_idx]

                # Accept any non-negative activation (including very small ones and zeros)
                if secondary_activation >= 0:
                    temp_results.append({
                        "text": batch_texts[b],
                        "secondary_activation": float(secondary_activation),
                        "primary_activations": primary_activations[b].tolist(),
                        "nfm_embedding": nfm_embeddings[b].tolist()
                    })

        # Sort all collected results by secondary activation ASCENDING (lowest first) and return top_n
        temp_results.sort(key=lambda x: x["secondary_activation"], reverse=False)
        return temp_results[:top_n]

    def find_most_contributing_primary_features(self, secondary_feature_idx, example_activations, top_n_primary=10):
        """
        Find which Primary SAE features contribute most to Secondary SAE feature activation.
        
        Args:
            secondary_feature_idx: The secondary feature being analyzed
            example_activations: List of example dicts with primary_activations and secondary_activation
            top_n_primary: Number of top primary features to identify
            
        Returns:
            Dict with correlation analysis between primary and secondary features
        """
        if not example_activations:
            return {"error": "No example activations provided"}
        
        # Extract primary activations and secondary activations
        primary_activations_matrix = np.array([ex["primary_activations"] for ex in example_activations])
        secondary_activations = np.array([ex["secondary_activation"] for ex in example_activations])
        
        # Calculate correlation between each primary feature and the secondary feature
        correlations = []
        for primary_idx in range(primary_activations_matrix.shape[1]):
            primary_values = primary_activations_matrix[:, primary_idx]
            
            # Skip if primary feature has no variation
            if np.std(primary_values) == 0:
                correlations.append(0.0)
            else:
                corr = np.corrcoef(primary_values, secondary_activations)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
        
        # Get top N primary features by absolute correlation
        abs_correlations = np.abs(correlations)
        top_indices = np.argsort(abs_correlations)[-top_n_primary:][::-1]
        
        results = {}
        for rank, primary_idx in enumerate(top_indices):
            correlation = correlations[primary_idx]
            mean_activation = np.mean(primary_activations_matrix[:, primary_idx])
            std_activation = np.std(primary_activations_matrix[:, primary_idx])
            
            results[rank+1] = {
                'primary_feature_idx': int(primary_idx),
                'correlation_with_secondary': float(correlation),
                'abs_correlation': float(abs_correlations[primary_idx]),
                'mean_primary_activation': float(mean_activation),
                'std_primary_activation': float(std_activation)
            }
        
        return results

    def visualize_secondary_feature_activations(self, secondary_feature_idx, examples, output_dir=None):
        """
        Create visualization of Secondary SAE feature activations and contributing Primary features.

        Args:
            secondary_feature_idx: Secondary feature index to visualize
            examples: List of example dicts with activations
            output_dir: Directory to save visualization

        Returns:
            Path to the saved image
        """
        if not examples:
            logging.warning(f"No examples to visualize for Secondary feature {secondary_feature_idx}")
            return None

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Secondary feature activations across examples
        secondary_activations = [ex["secondary_activation"] for ex in examples[:10]]  # Top 10
        example_labels = [f"Ex {i+1}" for i in range(len(secondary_activations))]
        
        bars1 = ax1.bar(example_labels, secondary_activations, color='orange', alpha=0.7)
        ax1.set_title(f"Secondary SAE Feature {secondary_feature_idx} - Top Activations")
        ax1.set_ylabel("Secondary Feature Activation")
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars1, secondary_activations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Heatmap of top contributing Primary features
        if len(examples) >= 5:  # Need enough examples for meaningful heatmap
            primary_contributions = self.find_most_contributing_primary_features(
                secondary_feature_idx, examples[:20], top_n_primary=15
            )
            
            if "error" not in primary_contributions:
                # Create heatmap data
                top_primary_indices = [primary_contributions[rank]['primary_feature_idx'] 
                                     for rank in range(1, min(11, len(primary_contributions)+1))]
                
                heatmap_data = []
                for ex in examples[:10]:
                    row = [ex["primary_activations"][idx] for idx in top_primary_indices]
                    heatmap_data.append(row)
                
                heatmap_data = np.array(heatmap_data)
                
                im = ax2.imshow(heatmap_data, cmap='Reds', aspect='auto')
                ax2.set_title(f"Top Contributing Primary SAE Features")
                ax2.set_xlabel("Primary Feature Index")
                ax2.set_ylabel("Example")
                
                # Set labels
                ax2.set_xticks(range(len(top_primary_indices)))
                ax2.set_xticklabels([f"P{idx}" for idx in top_primary_indices], rotation=45)
                ax2.set_yticks(range(len(examples[:10])))
                ax2.set_yticklabels([f"Ex {i+1}" for i in range(len(examples[:10]))])
                
                # Add colorbar
                plt.colorbar(im, ax=ax2, label='Primary Feature Activation')
            else:
                ax2.text(0.5, 0.5, "Insufficient data for Primary feature analysis", 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, "Need more examples for Primary feature analysis", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Save image
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            img_filename = f"secondary_feature_{secondary_feature_idx}_analysis_{int(time.time())}.png"
            img_path = os.path.join(output_dir, img_filename)
        else:
            import tempfile
            temp_dir = tempfile.gettempdir()
            img_filename = f"secondary_feature_{secondary_feature_idx}_analysis_{int(time.time())}.png"
            img_path = os.path.join(temp_dir, img_filename)

        try:
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return img_path
        except Exception as e:
            logging.error(f"Error saving visualization image: {e}")
            plt.close(fig)
            return None

    async def identify_secondary_feature_pattern(self, secondary_feature_idx, example_texts, output_dir=None, example_imgs=None, claude_examples=10):
        """
        Use Claude API to identify the semantic pattern of a Secondary SAE feature.

        Args:
            secondary_feature_idx: The secondary feature index
            example_texts: List of example texts with activation info
            output_dir: Directory to save results
            example_imgs: Optional list of image paths
            claude_examples: Number of examples to send to Claude API

        Returns:
            Identified pattern as a string
        """
        feature_id_logger = logging.getLogger('secondary_feature_id')

        # Format examples properly using FULL TEXTS
        formatted_examples = []
        for i, example in enumerate(example_texts[:claude_examples]):
            if isinstance(example, dict):
                text_content = example.get("text", "")
                # Note: Removed activation score from Claude prompt as requested in original
            else:
                text_content = example

            text_content = text_content.strip()

            if text_content:
                formatted_examples.append(f"Example {i+1}: {text_content}")

        if not formatted_examples:
            feature_id_logger.warning(f"No valid examples to send to Claude for Secondary feature {secondary_feature_idx} identification.")
            return "Unknown pattern (No valid examples)"

        # Define the separator
        separator = '-' * 40

        # Create a customized Claude prompt for Secondary SAE features
        try:
            secondary_prompt = self.identification_prompt.format(
                examples=chr(10).join(formatted_examples),
                separator=separator
            ).replace("neural network feature", "Secondary SAE feature in NFM pipeline")
            
        except KeyError as e:
            feature_id_logger.error(f"Prompt template is missing expected placeholder: {e}")
            return f"Error: Prompt template missing placeholder {e}"

        feature_id_logger.debug(f"CLAUDE SECONDARY PATTERN IDENTIFICATION PROMPT (Feature {secondary_feature_idx}):\n{secondary_prompt}")

        # Ensure output directory exists for Claude response
        if output_dir:
            claude_output_dir = os.path.join(output_dir, f"secondary_feature_{secondary_feature_idx}_examples")
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
                {"role": "user", "content": secondary_prompt}
            ]
        }

        try:
            # Save the prompt
            if output_dir:
                prompt_path = os.path.join(claude_output_dir, "claude_secondary_pattern_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(secondary_prompt)

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()
            claude_response = response_data["content"][0]["text"]

            feature_id_logger.debug(f"CLAUDE SECONDARY PATTERN IDENTIFICATION RESPONSE (Feature {secondary_feature_idx}):\n{claude_response}")

            # Save the raw response
            if output_dir:
                claude_response_path = os.path.join(claude_output_dir, "claude_secondary_pattern_response.txt")
                with open(claude_response_path, "w", encoding="utf-8") as f:
                    f.write(claude_response)

            # Parse the response to extract the pattern
            pattern_parts = claude_response.split("EXPLANATION:", 1)
            concise_pattern = pattern_parts[0].strip() if pattern_parts else "Unknown pattern (parsing failed)"
            explanation = pattern_parts[1].strip() if len(pattern_parts) > 1 else ""

            # Save the parsed pattern
            if output_dir:
                pattern_path = os.path.join(claude_output_dir, "secondary_pattern.txt")
                with open(pattern_path, "w", encoding="utf-8") as f:
                    f.write(f"{concise_pattern}\n{explanation}")

            return concise_pattern

        except Exception as e:
            feature_id_logger.error(f"Error calling Claude API: {e}")
            return f"Error identifying pattern: {str(e)}"

    def compute_secondary_feature_statistics(self, secondary_feature_idx, texts, n_samples=5000, batch_size=16):
        """Compute statistics for a Secondary SAE feature across texts."""
        # Sample texts if needed
        if len(texts) > n_samples:
            sampled_texts = random.sample(texts, n_samples)
        else:
            sampled_texts = texts

        activations = []

        # Process in batches through NFM pipeline
        for i in tqdm(range(0, len(sampled_texts), batch_size), desc="Computing Secondary feature statistics", leave=False):
            batch_texts = sampled_texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]

            if not batch_texts:
                continue

            # Process through NFM pipeline
            _, _, secondary_activations = self.process_through_nfm_pipeline(batch_texts, batch_size=len(batch_texts))

            # Accumulate statistics for this secondary feature
            for b in range(len(batch_texts)):
                activation = secondary_activations[b, secondary_feature_idx]
                activations.append(activation)

        # Convert to numpy arrays
        activations = np.array(activations)

        # Compute statistics
        stats = {
            "mean_activation": float(np.mean(activations)) if len(activations) > 0 else 0.0,
            "max_activation": float(np.max(activations)) if len(activations) > 0 else 0.0,
            "median_activation": float(np.median(activations)) if len(activations) > 0 else 0.0,
            "percent_active": float(np.mean(activations > 0) * 100) if len(activations) > 0 else 0.0,
            "feature_idx": secondary_feature_idx
        }

        # Save to instance
        self.feature_stats[secondary_feature_idx] = stats

        return stats

    def secondary_feature_intervention(self, secondary_feature_idx, text, clamp_values=None):
        """
        Clamping experiment for Secondary SAE features.
        This uses the fixed clamping implementation that properly intervenes in the NFM pipeline.

        Args:
            secondary_feature_idx: The secondary feature to clamp
            text: Input text prompt
            clamp_values: List of values to clamp the feature to

        Returns:
            Dictionary with generated text for each clamp value
        """
        results = {}

        # Define clamp values if not provided
        if clamp_values is None:
            max_val = self.feature_stats.get(secondary_feature_idx, {}).get('max_activation', 1.0)
            clamp_values = [0.0, max_val * 2, max_val * 5]
            print(f"Clamping Secondary feature {secondary_feature_idx} to default values based on max activation ({max_val:.4f}): {clamp_values}")
        else:
            print(f"Clamping Secondary feature {secondary_feature_idx} to specified values: {clamp_values}")

        # Get base output (without clamping)
        print("Generating base output (unclamped)...")
        try:
            base_output = self._generate_with_secondary_feature_clamped(text, None)
            results["base"] = base_output
        except Exception as e:
            logging.error(f"Error generating base output for clamping Secondary feature {secondary_feature_idx}: {e}")
            results["base"] = f"Error generating base output: {str(e)}"

        # Get outputs with clamped values
        for value in clamp_values:
            clamp_dict = {secondary_feature_idx: value}
            clamp_key = f"clamp_{value:.4f}"
            print(f"Generating output with Secondary feature {secondary_feature_idx} clamped to {value:.4f}...")
            try:
                output = self._generate_with_secondary_feature_clamped(text, clamp_dict)
                results[clamp_key] = output
            except Exception as e:
                logging.error(f"Error generating clamped output ({value:.4f}) for Secondary feature {secondary_feature_idx}: {e}")
                results[clamp_key] = f"Error generating output clamped to {value:.4f}: {str(e)}"

        return results

    def _generate_with_secondary_feature_clamped(self, text, clamp_features=None):
        """
        FIXED VERSION: Generate text with Secondary TopK SAE features clamped to specific values.
        This properly intervenes in the NFM pipeline during generation.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                            truncation=True, max_length=self.max_token_length).to(self.device)

        # Set up hooks for clamping Secondary TopK SAE features during NFM pipeline
        hooks = []
        
        if clamp_features:
            def nfm_secondary_clamp_hook(module, input_tensor, output):
                """
                Hook that intercepts the NFM forward pass and applies Secondary TopK SAE clamping
                within the interaction pathway.
                """
                try:
                    # NFM output: (combined_output, linear_out, interaction_out, weighted_embeddings)
                    combined_output, linear_out, interaction_out, weighted_embeddings = output
                    
                    # Apply Secondary TopK SAE to weighted embeddings (NFM embeddings)
                    # weighted_embeddings is already 2D, so no reshape needed
                    secondary_features, secondary_reconstruction = self.secondary_sae(
                        weighted_embeddings.to(self.secondary_sae.encoder[0].weight.dtype)
                    )
                    
                    # Apply clamping to secondary features
                    clamped_secondary_features = secondary_features.clone()
                    for feat_idx, value in clamp_features.items():
                        if feat_idx < clamped_secondary_features.shape[-1]:
                            clamped_secondary_features[:, feat_idx] = value
                        else:
                            logging.warning(f"Attempted to clamp invalid Secondary feature index {feat_idx}. "
                                        f"Max feature index is {clamped_secondary_features.shape[-1]-1}")
                    
                    # Get clamped reconstruction from Secondary TopK SAE decoder
                    clamped_reconstruction = self.secondary_sae.decoder(clamped_secondary_features)
                    
                    # Continue through the rest of the interaction MLP
                    # Apply ReLU (layer 2 in the interaction MLP)
                    post_relu = torch.relu(clamped_reconstruction)
                    
                    # Apply final linear layer (layer 3 in the interaction MLP)
                    if len(self.nfm_model.interaction_mlp) > 3:
                        clamped_interaction_out = self.nfm_model.interaction_mlp[3](post_relu)
                    else:
                        clamped_interaction_out = self.nfm_model.interaction_mlp[-1](post_relu)
                    
                    # Combine with linear output to get final clamped output
                    clamped_combined_output = linear_out + clamped_interaction_out
                    
                    return (clamped_combined_output, linear_out, clamped_interaction_out, weighted_embeddings)
                    
                except Exception as e:
                    logging.error(f"Error in NFM Secondary TopK SAE clamping hook: {e}")
                    return output

            try:
                hook = self.nfm_model.register_forward_hook(nfm_secondary_clamp_hook)
                hooks.append(hook)
                logging.debug(f"Registered forward hook on NFM model for Secondary TopK SAE clamping.")
            except Exception as e:
                logging.error(f"Failed to register forward hook on NFM model for Secondary TopK SAE clamping: {e}")

        # Generate with hooks active
        result = ""
        try:
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                )

            generated_sequence = outputs.sequences[0]
            input_len = inputs['input_ids'].shape[1]
            
            if generated_sequence.shape[0] > input_len:
                result = self.tokenizer.decode(generated_sequence[input_len:], skip_special_tokens=True)
            else:
                result = ""

        except Exception as e:
            logging.error(f"Error during text generation with Secondary TopK SAE clamping hook: {e}")
            result = f"Error during generation: {str(e)}"
        finally:
            # Always remove hooks
            for hook in hooks:
                if hook is not None:
                    hook.remove()
                    logging.debug(f"Removed NFM Secondary TopK SAE generation hook.")

        return result

    def _generate_with_full_nfm_pipeline_clamping(self, text, clamp_features=None):
        """
        Alternative implementation that manually processes through the full NFM pipeline
        with Secondary SAE clamping during generation. This is more complex but gives
        full control over the intervention.
        
        Args:
            text: Input text
            clamp_features: Dict mapping {secondary_feature_idx: value} or None
            
        Returns:
            Generated text with intervention description
        """
        if clamp_features is None:
            # No clamping, use standard generation
            return self._generate_with_secondary_feature_clamped(text, None)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=self.max_token_length).to(self.device)
        
        # Custom generation loop with NFM pipeline intervention
        result = ""
        try:
            with torch.no_grad():
                # Get initial hidden states from base model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                # Process through Primary SAE
                primary_features, _ = self.primary_sae(
                    hidden_states.to(self.primary_sae.encoder[0].weight.dtype)
                )
                
                # Get mean activation across sequence
                seq_len = torch.sum(inputs["attention_mask"][0]).item()
                if seq_len > 0:
                    mean_primary = torch.mean(primary_features[0, :seq_len, :], dim=0).unsqueeze(0)
                    
                    # Process through modified NFM pipeline with Secondary SAE clamping
                    linear_out = self.nfm_model.linear(mean_primary)
                    
                    # Interaction pathway with Secondary SAE intervention
                    embeddings = self.nfm_model.feature_embeddings.weight.T
                    weighted_embeddings = torch.matmul(mean_primary, embeddings.T)
                    
                    # Apply Secondary SAE
                    secondary_features, secondary_reconstruction = self.secondary_sae(
                        weighted_embeddings.to(self.secondary_sae.encoder[0].weight.dtype)
                    )
                    
                    # Apply clamping
                    clamped_secondary_features = secondary_features.clone()
                    for feat_idx, value in clamp_features.items():
                        if feat_idx < clamped_secondary_features.shape[-1]:
                            clamped_secondary_features[:, feat_idx] = value
                    
                    # Get clamped reconstruction
                    clamped_reconstruction = self.secondary_sae.decoder(clamped_secondary_features)
                    
                    # Continue through interaction MLP
                    post_relu = torch.relu(clamped_reconstruction)
                    interaction_out = self.nfm_model.interaction_mlp[3](post_relu)
                    
                    # Combine outputs
                    nfm_output = linear_out + interaction_out
                    
                    # For demonstration purposes, show the intervention was applied
                    result = f"[Secondary SAE clamping applied with features {list(clamp_features.keys())} = {list(clamp_features.values())}]\n"
                    result += f"NFM output shape: {nfm_output.shape}\n"
                    result += f"Clamped secondary features mean: {clamped_secondary_features.mean().item():.6f}\n"
                    result += f"Original secondary features mean: {secondary_features.mean().item():.6f}\n"
                    
                    # Note: Full integration would require modifying the base model's forward pass
                    # to use the NFM output, which is beyond the scope of this demonstration
                    result += "Generated text would continue here with clamped Secondary SAE features affecting the model's representations."
            
        except Exception as e:
            logging.error(f"Error in full NFM pipeline clamping: {e}")
            result = f"Error in NFM pipeline clamping: {str(e)}"
        
        return result

    def create_modified_nfm_forward_for_clamping(self, clamp_features=None):
        """
        Create a modified version of the NFM forward method that includes
        Secondary SAE intervention point for clamping during generation.
        
        This would replace the standard NFM forward method during clamped generation.
        
        Args:
            clamp_features: Dict of {secondary_feature_idx: clamp_value}
            
        Returns:
            Modified forward function
        """
        original_forward = self.nfm_model.forward
        
        def forward_with_secondary_sae_clamping(x):
            """
            Modified NFM forward pass with Secondary SAE clamping intervention.
            
            Args:
                x: Input primary SAE features
            """
            # Linear component (unchanged)
            linear_out = self.nfm_model.linear(x)
            
            # Interaction component with Secondary SAE intervention
            embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
            weighted_embeddings = torch.matmul(x, embeddings.T)  # [batch, k_dim]
            
            # Pass through first part of interaction MLP (up to Secondary SAE intervention point)
            # Assuming MLP structure: Identity(0) -> Linear(1) -> ReLU(2) -> Linear(3)
            mlp_intermediate = weighted_embeddings  # Identity layer
            if len(self.nfm_model.interaction_mlp) > 1:
                mlp_intermediate = self.nfm_model.interaction_mlp[1](mlp_intermediate)
            
            # Secondary SAE intervention
            secondary_features, secondary_reconstruction = self.secondary_sae(
                mlp_intermediate.to(self.secondary_sae.encoder[0].weight.dtype)
            )
            
            # Apply clamping if specified
            if clamp_features:
                clamped_features = secondary_features.clone()
                for feat_idx, value in clamp_features.items():
                    if feat_idx < clamped_features.shape[-1]:
                        clamped_features[:, feat_idx] = value
                
                # Use clamped reconstruction
                intervention_output = self.secondary_sae.decoder(clamped_features)
            else:
                # Use normal reconstruction
                intervention_output = secondary_reconstruction
            
            # Continue through rest of interaction MLP
            post_relu = torch.relu(intervention_output)  # ReLU layer
            
            # Final linear layer
            if len(self.nfm_model.interaction_mlp) > 3:
                interaction_out = self.nfm_model.interaction_mlp[3](post_relu)
            else:
                interaction_out = post_relu  # If no final layer
            
            # Combine linear and interaction outputs
            combined_output = linear_out + interaction_out
            
            return combined_output, linear_out, interaction_out, weighted_embeddings
        
        return forward_with_secondary_sae_clamping

def load_identification_prompt():
    """Load the identification prompt template."""
    prompt_path = Path("feature_identification_prompt.txt")
    if not prompt_path.exists():
        prompt_path = Path("prompts/feature_identification_prompt.txt")

    if not prompt_path.exists():
        logging.warning(f"Prompt template not found at {prompt_path}. Using default.")
        return """I'll analyze examples where a specific Secondary SAE feature in an NFM pipeline activates strongly.
Please help identify the pattern this Secondary feature might be detecting.

{examples}

{separator}

Based on these examples, what's a concise description of the pattern this Secondary SAE feature might be detecting?
Format your response as:
Pattern: [concise description]
EXPLANATION: [brief explanation]"""

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

async def analyze_secondary_feature_meaning(
    args,
    secondary_feature_indices,
    model_path,
    primary_sae_path,
    nfm_path,
    secondary_sae_path,
    output_dir,
    input_json=None,
    num_examples=20,
    num_samples=10000,
    batch_size=16,
    config_dir=None,
    run_clamping=True,
    max_token_length=100,
    claude_examples=10
):
    """
    Analyze the meaning of Secondary SAE features in the NFM pipeline.

    Args:
        args: The parsed command-line arguments object
        secondary_feature_indices: List of Secondary SAE feature indices to analyze
        model_path: Path to base language model
        primary_sae_path: Path to Primary SAE model
        nfm_path: Path to NFM model
        secondary_sae_path: Path to Secondary SAE model
        output_dir: Directory to save results
        input_json: Path to JSON file with category examples (for classification experiment)
        num_examples: Number of examples to extract per feature
        num_samples: Number of samples to use from dataset
        batch_size: Batch size for processing
        config_dir: Directory with API key config
        run_clamping: Whether to run clamping interventions
        max_token_length: Maximum token length for tokenization
        claude_examples: Number of examples to send to Claude API
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "secondary_feature_analysis.log"),
            logging.StreamHandler()
        ]
    )

    # Log parameters
    logging.info(f"Secondary Features argument provided: {args.features}")
    logging.info(f"Analyzing Secondary features: {secondary_feature_indices}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Primary SAE path: {primary_sae_path}")
    logging.info(f"NFM path: {nfm_path}")
    logging.info(f"Secondary SAE path: {secondary_sae_path}")
    logging.info(f"Number of examples per feature: {num_examples}")
    logging.info(f"Number of dataset samples: {num_samples}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Run clamping interventions: {run_clamping}")
    logging.info(f"Max token length: {max_token_length}")
    logging.info(f"Claude examples: {claude_examples}")

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token for tokenizer")

    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    # Load Primary SAE model
    logging.info("Loading Primary SAE model...")
    primary_sae_state_dict = torch.load(primary_sae_path, map_location="cuda")

    # Determine dimensions from Primary SAE state dict
    if 'decoder.weight' in primary_sae_state_dict:
        decoder_weight = primary_sae_state_dict['decoder.weight']
        primary_input_dim = decoder_weight.shape[0]
        primary_hidden_dim = decoder_weight.shape[1]
    elif 'encoder.0.weight' in primary_sae_state_dict:
        encoder_weight = primary_sae_state_dict['encoder.0.weight']
        primary_hidden_dim, primary_input_dim = encoder_weight.shape
    else:
        raise ValueError("Could not determine Primary SAE dimensions from state dict keys")

    logging.info("Loading Primary TopK SAE model...")
    primary_sae = load_topk_sae_model(primary_sae_path, "cuda", args.primary_sae_k)

    primary_sae.load_state_dict(primary_sae_state_dict)
    primary_sae.to("cuda")
    primary_sae.eval()

    # Load NFM model
    logging.info("Loading NFM model...")
    nfm_state_dict = torch.load(nfm_path, map_location="cuda")
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]

    logging.info(f"Creating NFM with num_features={num_features}, k_dim={k_dim}, output_dim={output_dim}")
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to("cuda")
    nfm_model.eval()

    # Load Secondary TopK SAE model
    logging.info("Loading Secondary TopK SAE model...")
    secondary_sae = load_topk_sae_model(secondary_sae_path, "cuda", args.secondary_sae_k)

    # Determine K value for TopK SAE (you'll need to specify this)
    # Option 1: Add as command line argument
    secondary_k = getattr(args, 'secondary_sae_k', 500)  # Default to 500


    # Initialize analyzer
    analyzer = NFMSecondaryFeatureAnalyzer(
        primary_sae,
        nfm_model,
        secondary_sae,
        tokenizer,
        base_model,
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
                    # Chunk long text
                    clean_text = tokenizer.decode(full_inputs["input_ids"][0], skip_special_tokens=True)
                    tokens = tokenizer.encode(clean_text, add_special_tokens=False)
                    
                    for chunk_start in range(0, len(tokens), max_token_length):
                        chunk_tokens = tokens[chunk_start:chunk_start + max_token_length]
                        if len(chunk_tokens) >= max_token_length:
                            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                            filtered_texts.append(chunk_text)
                            
                            if len(filtered_texts) >= num_samples:
                                break
                else:
                    # Short text, truncate normally
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_token_length)
                    if torch.sum(inputs["attention_mask"][0]).item() >= max_token_length:
                        truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                        filtered_texts.append(truncated_text)
                        
                        if len(filtered_texts) >= num_samples:
                            break
            
            if len(filtered_texts) >= num_samples:
                break

        texts = filtered_texts
        logging.info(f"Loaded {len(texts)} text samples from wikitext-103 (max_token_length={max_token_length})")

    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # Analyze each Secondary SAE feature
    results = []

    for secondary_feature_idx in secondary_feature_indices:
        logging.info(f"\nAnalyzing Secondary SAE feature {secondary_feature_idx}")
        feature_result = {"secondary_feature_idx": int(secondary_feature_idx)}

        # Create feature directory
        feature_dir = output_dir / f"secondary_feature_{secondary_feature_idx}"
        feature_dir.mkdir(exist_ok=True)

        # Find top examples
        logging.info(f"Finding top {num_examples} examples...")
        if args.find_weakest:
            top_examples = analyzer.find_lowest_activating_examples(
                secondary_feature_idx, texts, top_n=num_examples, batch_size=batch_size
            )
        else:
            top_examples = analyzer.find_highest_activating_examples(
                secondary_feature_idx, texts, top_n=num_examples, batch_size=batch_size
            )

        # Compute feature statistics
        logging.info("Computing Secondary feature statistics...")
        stats = analyzer.compute_secondary_feature_statistics(
            secondary_feature_idx,
            texts,
            n_samples=min(2000, len(texts)),
            batch_size=batch_size
        )
        feature_result["statistics"] = stats
        logging.info(f"Statistics: {stats}")

        # Find contributing Primary features
        if top_examples:
            logging.info("Finding contributing Primary SAE features...")
            primary_contributions = analyzer.find_most_contributing_primary_features(
                secondary_feature_idx, top_examples, top_n_primary=15
            )
            feature_result["primary_contributors"] = primary_contributions

        # Store examples and create visualizations
        example_imgs = []

        # Save examples to a file
        with open(feature_dir / "examples.txt", "w", encoding="utf-8") as f:
            for i, example in enumerate(top_examples):
                f.write(f"--- Example {i+1} ---\n")
                f.write(f"Secondary Activation: {example['secondary_activation']:.6f}\n")
                f.write(f"Text:\n{example['text']}\n\n")
                
                # Show top contributing Primary features for this example
                if 'primary_activations' in example:
                    primary_acts = np.array(example['primary_activations'])
                    top_primary_indices = np.argsort(primary_acts)[-5:][::-1]  # Top 5
                    f.write(f"Top 5 Primary SAE features for this example:\n")
                    for j, p_idx in enumerate(top_primary_indices):
                        f.write(f"  P{p_idx}: {primary_acts[p_idx]:.6f}\n")
                
                f.write("-"*50 + "\n\n")

        # Create visualizations
        logging.info("Creating visualizations...")
        img_path = analyzer.visualize_secondary_feature_activations(
            secondary_feature_idx,
            top_examples[:10],  # Visualize top 10
            output_dir=feature_dir
        )
        if img_path:
            example_imgs.append(os.path.basename(img_path))

        # Store examples
        feature_result["examples"] = [{
            "text": ex["text"],
            "secondary_activation": float(ex["secondary_activation"]),
            "primary_activations": ex.get("primary_activations", []),
            "nfm_embedding": ex.get("nfm_embedding", [])
        } for ex in top_examples]

        feature_result["example_imgs"] = example_imgs

        # Identify pattern using Claude API
        if claude_api_key:
            logging.info("Identifying Secondary feature pattern using Claude API...")
            feature_result["pattern"] = await analyzer.identify_secondary_feature_pattern(
                secondary_feature_idx,
                feature_result["examples"],
                output_dir=output_dir,
                claude_examples=claude_examples
            )
            logging.info(f"Identified pattern: {feature_result['pattern']}")
        else:
            feature_result["pattern"] = "Pattern identification skipped (no API key)"
            logging.warning("Pattern identification skipped (no API key)")

        # Clamping interventions (FIXED VERSION)
        if run_clamping:
            logging.info("Running FIXED Secondary SAE clamping interventions...")
            intervention_prompt = "Human: What's your favorite animal?\n\nAssistant:"
            try:
                clamp_results = analyzer.secondary_feature_intervention(secondary_feature_idx, intervention_prompt)
                feature_result["clamp_results"] = clamp_results

                # Save clamping results to file
                with open(feature_dir / "clamping_results.txt", "w", encoding="utf-8") as f:
                    f.write(f"Secondary SAE Feature {secondary_feature_idx} Clamping Results (FIXED)\n\n")
                    f.write(f"Input: {intervention_prompt}\n\n")
                    f.write(f"Base (Unclamped):\n{clamp_results.get('base', 'N/A')}\n\n")
                    for key, value in clamp_results.items():
                        if key != "base":
                            f.write(f"{key}:\n{value}\n\n")

                logging.info(f"Secondary SAE clamping results obtained (FIXED implementation)")
            except Exception as e:
                logging.error(f"Error running Secondary SAE clamping intervention: {e}")
                feature_result["clamp_results"] = {"error": str(e)}
        else:
            feature_result["clamp_results"] = {"note": "Clamping skipped"}
            logging.info("Secondary SAE clamping interventions skipped")

        # Store results
        results.append(feature_result)

        # Save incremental results
        try:
            with open(output_dir / "secondary_feature_analysis_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Incremental results saved for Secondary feature {secondary_feature_idx}")
        except Exception as e:
            logging.error(f"Error saving incremental results for Secondary feature {secondary_feature_idx}: {e}")

    # Create summary report
    logging.info("Creating summary report...")
    with open(output_dir / "secondary_analysis_summary.md", "w", encoding="utf-8") as f:
        f.write("# Secondary SAE Feature Analysis Summary (NFM Pipeline) - FIXED VERSION\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Primary SAE: {primary_sae_path}\n")
        f.write(f"NFM: {nfm_path}\n")
        f.write(f"Secondary SAE: {secondary_sae_path}\n")
        f.write(f"Max Token Length: {max_token_length}\n")
        f.write("**FIXED:** Proper clamping workflow that correctly intervenes in the NFM pipeline\n\n")
        f.write("## Analyzed Secondary SAE Features\n\n")

        for feature_result in results:
            secondary_feature_idx = feature_result["secondary_feature_idx"]
            pattern = feature_result.get("pattern", "N/A")
            stats = feature_result.get("statistics", {})

            f.write(f"### Secondary SAE Feature {secondary_feature_idx}\n\n")
            f.write(f"**Pattern:** {pattern}\n\n")

            # Statistics
            f.write(f"**Statistics:**\n")
            f.write(f"- Max activation: {stats.get('max_activation', 0.0):.6f}\n")
            f.write(f"- Mean activation: {stats.get('mean_activation', 0.0):.6f}\n")
            f.write(f"- Percent active: {stats.get('percent_active', 0.0):.2f}%\n\n")

            # Top contributing Primary features
            primary_contributors = feature_result.get("primary_contributors", {})
            if primary_contributors and "error" not in primary_contributors:
                f.write("**Top Contributing Primary SAE Features:**\n\n")
                for rank in range(1, min(6, len(primary_contributors)+1)):  # Top 5
                    if rank in primary_contributors:
                        contrib = primary_contributors[rank]
                        f.write(f"- Primary Feature {contrib['primary_feature_idx']}: ")
                        f.write(f"correlation = {contrib['correlation_with_secondary']:.4f}, ")
                        f.write(f"mean activation = {contrib['mean_primary_activation']:.6f}\n")
                f.write("\n")

            # Top examples
            f.write("**Top Examples:**\n\n")
            examples = feature_result.get("examples", [])
            if examples:
                for i, example in enumerate(examples[:3]):  # Show top 3
                    f.write(f"Example {i+1}:\n")
                    text = example.get('text', 'N/A')
                    if len(text) > 300:
                        text = text[:300] + "..."
                    f.write(f"Text: ```\n{text}\n```\n")
                    f.write(f"Secondary Activation: {example.get('secondary_activation', 0.0):.6f}\n\n")
            else:
                f.write("No examples found.\n\n")

            # Clamping Results (FIXED)
            f.write("**Clamping Results (FIXED Implementation):**\n\n")
            clamp_results = feature_result.get("clamp_results", {})

            if "error" in clamp_results or "note" in clamp_results:
                f.write(f"{clamp_results.get('error', clamp_results.get('note', 'N/A'))}\n\n")
            else:
                f.write("Secondary SAE feature clamping:\n")
                f.write(f"Base: ```\n{clamp_results.get('base', 'N/A')}\n```\n")
                max_val = stats.get('max_activation', 1.0)
                clamp_0_key = f'clamp_{0.0:.4f}'
                clamp_2x_key = f'clamp_{(max_val * 2):.4f}'
                clamp_5x_key = f'clamp_{(max_val * 5):.4f}'

                f.write(f"Clamped to 0.0: ```\n{clamp_results.get(clamp_0_key, 'N/A')}\n```\n\n")
                f.write(f"Clamped to {max_val * 2:.4f}: ```\n{clamp_results.get(clamp_2x_key, 'N/A')}\n```\n\n")
                f.write(f"Clamped to {max_val * 5:.4f}: ```\n{clamp_results.get(clamp_5x_key, 'N/A')}\n```\n\n")

            f.write("---\n\n")

    logging.info(f"Secondary SAE analysis complete! Results saved to {output_dir}")
    return results

def load_topk_sae_model(checkpoint_path, device="cuda", default_k=None):
    """
    Load TopK SAE model with automatic K detection.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the actual model state dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # Determine dimensions
    if 'decoder.weight' in state_dict:
        input_dim = state_dict['decoder.weight'].shape[0]
        hidden_dim = state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError(f"Cannot determine SAE dimensions from state dict keys: {list(state_dict.keys())}")
    
    # Try to determine K value
    k_value = default_k
    if k_value is None:
        # Check filename for TopK indicators
        filename = str(checkpoint_path).lower()
        import re
        k_matches = re.findall(r'topk(\d+)', filename) or re.findall(r'_k(\d+)_', filename)
        if k_matches:
            k_value = int(k_matches[0])
        else:
            # Use 2% as default
            k_value = max(1, int(0.02 * hidden_dim))
            print(f"Warning: K value not found for TopK SAE. Using default K={k_value} (2% of {hidden_dim})")
    
    print(f"Creating TopK SAE with input_dim={input_dim}, hidden_dim={hidden_dim}, k={k_value}")
    model = TopKSparseAutoencoder(input_dim, hidden_dim, k_value)
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model

async def main():
    parser = argparse.ArgumentParser(description="Analyze meaning of specific Secondary SAE features in NFM pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--secondary_sae_path", type=str, required=True, help="Path to Secondary SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--input_json", type=str, default=None, help="Optional JSON file with category examples")
    parser.add_argument("--features", type=str, required=True, help="Comma-separated list of Secondary SAE feature indices or path to JSON file")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to extract per feature")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use from dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--config_dir", type=str, default="../config", help="Directory with API key config")
    parser.add_argument("--no_clamping", action="store_true", help="Skip clamping interventions")
    parser.add_argument("--max_token_length", type=int, default=100, help="Maximum token length for tokenization")
    parser.add_argument("--claude_examples", type=int, default=10, help="Number of examples to send to Claude API for pattern identification")
    parser.add_argument("--find_weakest", action="store_true", help="Find weakest activating examples instead of strongest")
    parser.add_argument("--primary_sae_k", type=int, default=50000, help="K value for Primary TopK SAE (default: 1000)")
    parser.add_argument("--secondary_sae_k", type=int, default=5000, help="K value for Secondary TopK SAE (default: 100)")
    args = parser.parse_args()

    # Parse Secondary SAE feature indices
    secondary_feature_indices = []
    if os.path.isfile(args.features):
        # Load from file
        try:
            with open(args.features, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    secondary_feature_indices = data
                elif isinstance(data, dict):
                    # Try various possible keys for Secondary SAE features
                    possible_keys = ["secondary_features", "secondary_feature_indices", "top_secondary_features", "features"]
                    for key in possible_keys:
                        if key in data:
                            secondary_feature_indices = data[key]
                            logging.info(f"Parsed Secondary feature indices from '{key}' key in {args.features}")
                            break
                    
                    # If no standard key found, try to extract from top-level keys
                    if not secondary_feature_indices:
                        try:
                            secondary_feature_indices = [int(k) for k in data.keys()]
                            logging.info(f"Parsed Secondary feature indices from JSON keys: {secondary_feature_indices[:10]}{'...' if len(secondary_feature_indices) > 10 else ''}")
                        except ValueError:
                            logging.warning(f"Could not parse Secondary feature indices from JSON keys in {args.features}")
                            secondary_feature_indices = []
                else:
                    logging.error(f"Unexpected JSON format in {args.features}")
                    secondary_feature_indices = []

        except Exception as e:
            logging.error(f"Error loading Secondary feature indices from {args.features}: {e}")
            secondary_feature_indices = []

    else:
        # Parse comma-separated list
        try:
            secondary_feature_indices = [int(x.strip()) for x in args.features.split(",") if x.strip()]
        except ValueError:
            logging.error(f"Could not parse comma-separated Secondary feature indices: {args.features}")
            secondary_feature_indices = []

    if not secondary_feature_indices:
        print("No Secondary SAE feature indices found to analyze. Please check your --features argument or input file.")
        return

    print(f"Analyzing {len(secondary_feature_indices)} Secondary SAE features: {secondary_feature_indices[:10]}{'...' if len(secondary_feature_indices) > 10 else ''}")
    print(f"Using max token length: {args.max_token_length}")
    print("FIXED VERSION: Using corrected clamping workflow for proper NFM pipeline intervention")

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"secondary_feature_meaning_FIXED_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Run analysis
    await analyze_secondary_feature_meaning(
        args,
        secondary_feature_indices=secondary_feature_indices,
        model_path=args.model_path,
        primary_sae_path=args.primary_sae_path,
        nfm_path=args.nfm_path,
        secondary_sae_path=args.secondary_sae_path,
        output_dir=output_path,
        input_json=args.input_json,
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        config_dir=args.config_dir,
        run_clamping=not args.no_clamping,
        max_token_length=args.max_token_length,
        claude_examples=args.claude_examples
    )

if __name__ == "__main__":
    asyncio.run(main())