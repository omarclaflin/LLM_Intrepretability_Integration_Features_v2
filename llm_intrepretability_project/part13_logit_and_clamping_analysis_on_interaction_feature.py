"""
part13_logit_and_clamping_analysis_on_interaction_feature.py

Logit Lens and Activation Clamping Analysis for Secondary SAE Features

This script analyzes how clamping Secondary SAE features affects:
1. Logit predictions for target words (formal vs casual vocabulary)
2. Generated text completions across different prompt types

Pipeline: Layer 16 → Primary SAE → NFM → Secondary SAE [CLAMP HERE] → Continue Generation

The script clamps specific Secondary SAE features at different activation levels and measures:
- Logit values for predefined formal/casual words
- Generated text completions for qualitative analysis
"""

import torch
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import logging
from typing import List, Tuple, Dict, Any
import csv

from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Default word lists for logit lens analysis - organized by both dimensions
DEFAULT_FORMAL_LOW_EMOTION_WORDS = ["perhaps", "may", "however", "therefore", "consequently", "moreover", "accordingly", "furthermore"]
DEFAULT_FORMAL_HIGH_EMOTION_WORDS = ["profoundly", "devastated", "magnificent", "appalling", "extraordinary", "overwhelming", "catastrophic", "triumphant"]
DEFAULT_CASUAL_LOW_EMOTION_WORDS = ["yeah", "okay", "kinda", "basically", "anyway", "whatever", "guess", "stuff"]
DEFAULT_CASUAL_HIGH_EMOTION_WORDS = ["OMG", "totally", "literally", "seriously", "honestly", "really", "absolutely", "incredibly"]

# Legacy word lists for backward compatibility
DEFAULT_FORMAL_WORDS = DEFAULT_FORMAL_LOW_EMOTION_WORDS + DEFAULT_FORMAL_HIGH_EMOTION_WORDS
DEFAULT_CASUAL_WORDS = DEFAULT_CASUAL_LOW_EMOTION_WORDS + DEFAULT_CASUAL_HIGH_EMOTION_WORDS

# Prompt sets for different formality/emotion combinations
PROMPT_SETS = {
    "neutral": [
        "I need to inform everyone about",
        "When someone asks me about",
        "The reason this happened is",
        "My thoughts on this matter are",
        "The best way to handle this is"
    ],
    "formal_emotional": [
        "I must express my profound concern regarding",
        "It is with great disappointment that I must address",
        "I am deeply troubled by the implications of",
        "With utmost urgency, I must convey that",
        "I feel compelled to voice my serious objections to"
    ],
    "formal_neutral": [
        "According to the documentation, the procedure requires",
        "The analysis indicates that the optimal approach involves",
        "Pursuant to the established guidelines, it is recommended that",
        "The evaluation demonstrates that the most effective method is",
        "Based on the available evidence, the conclusion suggests that"
    ],
    "casual_emotional": [
        "OMG I can't believe how excited I am about",
        "Honestly, I'm so frustrated with",
        "I'm literally freaking out about",
        "This is seriously the most amazing thing about",
        "I'm totally devastated by"
    ],
    "casual_neutral": [
        "So anyway, I was thinking about",
        "Yeah, the thing is that",
        "Basically what happened was",
        "I guess the main point is",
        "You know, it's kinda interesting how"
    ]
}

class TopKSparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module that can handle k=None."""
    def __init__(self, input_dim, hidden_dim, k=None):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        if self.k is not None:
            sparse_features = self.apply_topk(features)
        else:
            sparse_features = features  # No top-k filtering
        reconstruction = self.decoder(sparse_features)
        return sparse_features, reconstruction
    
    def apply_topk(self, features):
        """Apply TopK sparsity - keep only top K activations per sample"""
        batch_size, num_features = features.shape
        
        # Get TopK values and indices for each sample
        topk_values, topk_indices = torch.topk(features, self.k, dim=1)
        
        # Create sparse feature tensor
        sparse_features = torch.zeros_like(features)
        
        # Scatter the TopK values back to their original positions
        sparse_features.scatter_(1, topk_indices, topk_values)
        
        return sparse_features

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

class InteractionFeatureAnalyzer:
    """Analyzer for Secondary SAE feature clamping and logit analysis."""
    
    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, 
                 device="cuda", target_layer=16):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.secondary_sae = secondary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        
        # Set models to eval mode
        self.primary_sae.eval()
        self.nfm_model.eval()
        self.secondary_sae.eval()
        self.base_model.eval()
        
        # Get vocabulary for logit analysis
        self.vocab_size = len(tokenizer.vocab)

    def get_word_token_ids_by_dimensions(self, formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion):
        """Convert words to their token IDs, organized by formality and emotion dimensions."""
        word_categories = {
            'formal_low_emotion': formal_low_emotion,
            'formal_high_emotion': formal_high_emotion,
            'casual_low_emotion': casual_low_emotion,
            'casual_high_emotion': casual_high_emotion
        }
        
        categorized_tokens = {}
        
        for category, words in word_categories.items():
            word_token_map = {}
            for word in words:
                # Try different tokenization approaches
                tokens_simple = self.tokenizer.encode(word, add_special_tokens=False)
                tokens_space = self.tokenizer.encode(" " + word, add_special_tokens=False)
                
                # Use the single token if available, otherwise take the first token
                if len(tokens_simple) == 1:
                    word_token_map[word] = tokens_simple[0]
                elif len(tokens_space) == 1:
                    word_token_map[word] = tokens_space[0]
                elif len(tokens_space) >= 2:
                    # If " word" gives multiple tokens, take the second one (after space)
                    word_token_map[word] = tokens_space[1]
                elif len(tokens_simple) >= 1:
                    # Fallback to first token of simple tokenization
                    word_token_map[word] = tokens_simple[0]
                else:
                    print(f"Warning: Could not tokenize word '{word}' in category '{category}'")
                    continue
                    
            categorized_tokens[category] = word_token_map
                
        return categorized_tokens

    def get_pipeline_baseline_logits(self, formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion):
        """
        Get baseline logits with full SAE/NFM pipeline but no feature intervention.
        """
        print(f"=== EXTRACTING PIPELINE BASELINE LOGITS ===")
        
        categorized_tokens = self.get_word_token_ids_by_dimensions(
            formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion
        )
        
        test_prompts = [
            "I think that",
            "The situation is", 
            "My opinion on this is",
            "It seems to me that",
            "The best approach would be"
        ]
        
        baseline_results = []
        
        for prompt in tqdm(test_prompts, desc="Pipeline baseline"):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            # Use forward_with_intervention but with no interventions
            logits, _, _ = self.forward_with_intervention(
                inputs["input_ids"], inputs["attention_mask"], 
                target_features=[], clamp_value=1.0  # Empty features list = no intervention
            )
            
            # Extract logits for the last token
            last_token_logits = logits[0, -1, :]
            
            # Record logits for all word categories
            for category, word_token_map in categorized_tokens.items():
                for word, token_id in word_token_map.items():
                    baseline_results.append({
                        'prompt': prompt,
                        'baseline_type': 'pipeline',
                        'word_category': category,
                        'word': word,
                        'token_id': token_id,
                        'logit_value': float(last_token_logits[token_id].cpu())
                    })
        
        return pd.DataFrame(baseline_results)

    def forward_with_intervention(self, input_ids, attention_mask, target_features, clamp_value, clamp_method="multiply"):
        """
        Forward pass with intervention at Secondary SAE features using hooks.
        If target_features is empty, runs the full pipeline without any interventions.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            target_features: List of Secondary SAE feature indices to clamp
            clamp_value: Value to clamp/multiply features by
            clamp_method: "multiply" or "set"
        
        Returns:
            logits: Final logits after intervention
            secondary_features_original: Original Secondary SAE features
            secondary_features_clamped: Clamped Secondary SAE features
        """
        # Store intervention results
        intervention_data = {}
        
        def intervention_hook(module, input, output):
            """Hook to intervene at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Primary SAE processing
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # NFM processing - Linear pathway
            linear_output = self.nfm_model.linear(primary_features)
            
            # NFM processing - Interaction pathway
            # Step 1: Feature embeddings
            embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
            weighted_embeddings = torch.matmul(primary_features, embeddings.T)  # [batch*seq, k_dim]
            
            # Step 2: Interaction MLP Layer 1 [500 x 500]
            mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)  # [batch*seq, 500]
            
            # Step 3: Secondary SAE processing with intervention (AFTER MLP Layer 1, BEFORE ReLU)
            secondary_features_original, _ = self.secondary_sae(
                mlp_layer1_output.to(self.secondary_sae.encoder[0].weight.dtype)
            )
            
            # Apply intervention ONLY if target_features is not empty
            secondary_features_clamped = secondary_features_original.clone()
            if target_features:  # Only intervene if features specified
                for feature_idx in target_features:
                    if clamp_method == "multiply":
                        secondary_features_clamped[:, feature_idx] *= clamp_value
                    elif clamp_method == "set":
                        secondary_features_clamped[:, feature_idx] = clamp_value
                    else:
                        raise ValueError(f"Unknown clamp_method: {clamp_method}")
            
            # Store for later return
            intervention_data['secondary_features_original'] = secondary_features_original
            intervention_data['secondary_features_clamped'] = secondary_features_clamped
            
            # Step 4: Secondary SAE decode back to [500] space
            interaction_reconstructed = self.secondary_sae.decoder(secondary_features_clamped)
            
            # Step 5: Continue through remaining interaction MLP layers
            # ReLU activation (Layer 2)
            relu_output = self.nfm_model.interaction_mlp[2](interaction_reconstructed)  # [batch*seq, 500]
            
            # Final linear layer (Layer 3) 
            interaction_output = self.nfm_model.interaction_mlp[3](relu_output)  # [batch*seq, 3200]
            
            # Step 6: THREE-WAY COMBINATION: Primary SAE reconstruction + Linear + Interaction
            # Reshape primary reconstruction to match
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            linear_output_reshaped = linear_output.view(original_shape)
            interaction_output_reshaped = interaction_output.view(original_shape)
            
            # Combine all three pathways
            nfm_total_output = primary_reconstruction_reshaped + linear_output_reshaped + interaction_output_reshaped
            
            # Return modified activations
            return (nfm_total_output,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(intervention_hook)
        
        try:
            with torch.no_grad():
                # Forward pass with intervention
                outputs = self.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                logits = outputs.logits
                
        finally:
            # Remove hook
            hook_handle.remove()
        
        return (logits, 
                intervention_data.get('secondary_features_original', None),
                intervention_data.get('secondary_features_clamped', None))

    def logit_lens_analysis(self, formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion, target_features, clamp_multipliers):
        """
        Analyze how clamping affects logit values for words across both formality and emotion dimensions.
        
        Args:
            formal_low_emotion: List of formal + low emotion words
            formal_high_emotion: List of formal + high emotion words  
            casual_low_emotion: List of casual + low emotion words
            casual_high_emotion: List of casual + high emotion words
            target_features: List of Secondary SAE feature indices
            clamp_multipliers: List of values to multiply features by
        
        Returns:
            DataFrame with logit analysis results
        """
        print(f"\n=== LOGIT LENS ANALYSIS (2D: Formality × Emotion) ===")
        print(f"Target features: {target_features}")
        print(f"Clamp multipliers: {clamp_multipliers}")
        
        # Get token IDs for all word categories
        categorized_tokens = self.get_word_token_ids_by_dimensions(
            formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion
        )
        
        print(f"Word categories:")
        for category, word_map in categorized_tokens.items():
            print(f"  {category}: {word_map}")
        
        # Use a simple prompt that could go either direction
        test_prompts = [
            "I think that",
            "The situation is", 
            "My opinion on this is",
            "It seems to me that",
            "The best approach would be"
        ]
        
        results = []
        
        for prompt in tqdm(test_prompts, desc="Processing prompts"):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            for clamp_value in clamp_multipliers:
                # Get logits with intervention
                logits, _, _ = self.forward_with_intervention(
                    inputs["input_ids"], inputs["attention_mask"], 
                    target_features, clamp_value
                )
                
                # Extract logits for the last token (where we'd generate next)
                last_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Record logits for all word categories
                for category, word_token_map in categorized_tokens.items():
                    for word, token_id in word_token_map.items():
                        results.append({
                            'prompt': prompt,
                            'clamp_value': clamp_value,
                            'word_category': category,
                            'word': word,
                            'token_id': token_id,
                            'logit_value': float(last_token_logits[token_id].cpu()),
                            'target_features': str(target_features)
                        })
        
        return pd.DataFrame(results)

    def logit_lens_analysis_with_baselines(self, formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion, target_features, clamp_multipliers):
        """
        Enhanced logit analysis with proper baseline comparison.
        """
        print(f"\n=== LOGIT LENS ANALYSIS WITH PIPELINE BASELINE ===")
        
        # 1. Get pipeline baseline logits (NO intervention at all)
        baseline_df = self.get_pipeline_baseline_logits(
            formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion
        )
        
        # 2. Get intervention results for ALL clamp multipliers (including 0.0, etc.)
        print(f"Testing ALL clamp multipliers: {clamp_multipliers}")
        
        # Run original logit analysis with ALL clamp multipliers
        logit_df = self.logit_lens_analysis(
            formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion,
            target_features, clamp_multipliers
        )
        
        # 3. Calculate differences from pipeline baseline
        pipeline_diff_df = calculate_baseline_differences(logit_df, baseline_df, 'pipeline')
        
        return {
            'baseline_df': baseline_df,
            'intervention_df': logit_df,
            'pipeline_differences': pipeline_diff_df
        }

    def generate_with_intervention(self, prompt, target_features, clamp_value, max_length=50):
        """Generate text with Secondary SAE feature intervention."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        
        for _ in range(max_length):
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Get logits with intervention
            logits, _, _ = self.forward_with_intervention(
                input_ids, attention_mask, target_features, clamp_value
            )
            
            # Sample next token (using simple greedy decoding for consistency)
            next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        # Decode the generated part only
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

    def clamping_analysis(self, target_features, clamp_multipliers, generation_length):
        """
        Generate text completions with different clamping values.
        
        Args:
            target_features: List of Secondary SAE feature indices
            clamp_multipliers: List of values to multiply features by  
            generation_length: Number of tokens to generate
        
        Returns:
            Dictionary with generation results
        """
        print(f"\n=== CLAMPING ANALYSIS ===")
        print(f"Target features: {target_features}")
        print(f"Clamp multipliers: {clamp_multipliers}")
        print(f"Generation length: {generation_length}")
        
        results = {}
        
        for prompt_type, prompts in PROMPT_SETS.items():
            print(f"\nProcessing {prompt_type} prompts...")
            results[prompt_type] = {}
            
            for clamp_value in clamp_multipliers:
                results[prompt_type][f"clamp_{clamp_value}"] = []
                
                for prompt in tqdm(prompts, desc=f"Clamp {clamp_value}"):
                    try:
                        generated = self.generate_with_intervention(
                            prompt, target_features, clamp_value, generation_length
                        )
                        
                        results[prompt_type][f"clamp_{clamp_value}"].append({
                            'prompt': prompt,
                            'generated': generated,
                            'full_text': prompt + " " + generated
                        })
                    except Exception as e:
                        print(f"Error generating for prompt '{prompt}': {e}")
                        results[prompt_type][f"clamp_{clamp_value}"].append({
                            'prompt': prompt,
                            'generated': f"ERROR: {str(e)}",
                            'full_text': f"{prompt} ERROR: {str(e)}"
                        })
        
        return results

def calculate_baseline_differences(logit_df, baseline_df, baseline_type='pipeline'):
    """
    Calculate differences from specified baseline for each prompt/word combination.
    
    Args:
        logit_df: DataFrame with intervention results
        baseline_df: DataFrame with baseline logits  
        baseline_type: 'pipeline'
    """
    print(f"Calculating differences from {baseline_type} baseline...")
    
    # Create baseline lookup
    baseline_lookup = baseline_df[baseline_df['baseline_type'] == baseline_type].copy()
    baseline_lookup = baseline_lookup.set_index(['prompt', 'word_category', 'word'])['logit_value']
    
    # Calculate differences
    diff_data = []
    for _, row in logit_df.iterrows():
        # Find corresponding baseline
        key = (row['prompt'], row['word_category'], row['word'])
        if key in baseline_lookup.index:
            baseline_logit = baseline_lookup[key]
            difference = row['logit_value'] - baseline_logit
        else:
            print(f"Warning: No baseline found for {key}")
            baseline_logit = 0.0
            difference = 0.0
            
        diff_data.append({
            'prompt': row['prompt'],
            'clamp_value': row['clamp_value'],
            'word_category': row['word_category'],
            'word': row['word'],
            'token_id': row['token_id'],
            'logit_value': row['logit_value'],
            'baseline_logit': baseline_logit,
            'logit_difference': difference,
            'baseline_type': baseline_type,
            'formality': 'formal' if row['word_category'].startswith('formal') else 'casual',
            'emotion': 'high' if 'high' in row['word_category'] else 'low'
        })
    
    return pd.DataFrame(diff_data)

def run_anova_analysis_with_baselines(analysis_results, baseline_type='pipeline'):
    """
    Run ANOVA analysis on baseline-corrected differences.
    
    Performs:
    1. Separate 2×2 ANOVAs for each clamp level
    2. Overall comparison across clamp values
    3. Effect size calculations
    
    Returns:
        Dictionary with ANOVA results and statistics
    """
    logit_df_diff = analysis_results['pipeline_differences']
    
    print(f"\n=== ANOVA STATISTICAL ANALYSIS (vs {baseline_type} baseline) ===")
    
    print(f"Sample size: {len(logit_df_diff)} observations")
    print(f"Clamp values: {sorted(logit_df_diff['clamp_value'].unique())}")
    
    anova_results = {}
    
    # 1. SEPARATE 2×2 ANOVAs FOR EACH CLAMP LEVEL
    print(f"\n{'='*50}")
    print("2×2 ANOVA BY CLAMP LEVEL (Using Difference from Baseline)")
    print(f"{'='*50}")
    
    clamp_results = {}
    clamp_values = sorted(logit_df_diff['clamp_value'].unique())
    
    for clamp_val in clamp_values:
        print(f"\n--- CLAMP VALUE: {clamp_val} ---")
        
        # Filter data for this clamp level
        clamp_data = logit_df_diff[logit_df_diff['clamp_value'] == clamp_val].copy()
        
        # Create 2×2 groups using DIFFERENCES from baseline
        formal_low = clamp_data[(clamp_data['formality'] == 'formal') & (clamp_data['emotion'] == 'low')]['logit_difference']
        formal_high = clamp_data[(clamp_data['formality'] == 'formal') & (clamp_data['emotion'] == 'high')]['logit_difference']
        casual_low = clamp_data[(clamp_data['formality'] == 'casual') & (clamp_data['emotion'] == 'low')]['logit_difference']
        casual_high = clamp_data[(clamp_data['formality'] == 'casual') & (clamp_data['emotion'] == 'high')]['logit_difference']
        
        # Check sample sizes
        sizes = [len(formal_low), len(formal_high), len(casual_low), len(casual_high)]
        print(f"Sample sizes - Formal-Low: {sizes[0]}, Formal-High: {sizes[1]}, Casual-Low: {sizes[2]}, Casual-High: {sizes[3]}")
        
        if min(sizes) < 3:
            print("WARNING: Sample size too small for reliable ANOVA")
            continue
            
        # Calculate means for DIFFERENCES
        means = {
            'formal_low': formal_low.mean(),
            'formal_high': formal_high.mean(),
            'casual_low': casual_low.mean(),
            'casual_high': casual_high.mean()
        }
        
        print(f"Mean differences from baseline:")
        print(f"  Formal-Low:  {means['formal_low']:.4f}")
        print(f"  Formal-High: {means['formal_high']:.4f}")
        print(f"  Casual-Low:  {means['casual_low']:.4f}")
        print(f"  Casual-High: {means['casual_high']:.4f}")
        
        # Test main effects and interaction using differences
        # Main effect of Formality
        formal_mean = pd.concat([formal_low, formal_high]).mean()
        casual_mean = pd.concat([casual_low, casual_high]).mean()
        formality_effect = abs(formal_mean - casual_mean)
        
        # Main effect of Emotion  
        low_mean = pd.concat([formal_low, casual_low]).mean()
        high_mean = pd.concat([formal_high, casual_high]).mean()
        emotion_effect = abs(low_mean - high_mean)
        
        # Interaction effect (difference in differences)
        formal_diff = means['formal_high'] - means['formal_low']
        casual_diff = means['casual_high'] - means['casual_low']
        interaction_effect = abs(formal_diff - casual_diff)
        
        print(f"Effect sizes (from baseline):")
        print(f"  Formality effect: {formality_effect:.4f}")
        print(f"  Emotion effect: {emotion_effect:.4f}")
        print(f"  Interaction effect: {interaction_effect:.4f}")
        
        # F-tests for main effects and interaction
        try:
            # Test formality main effect
            formality_f, formality_p = stats.f_oneway(
                pd.concat([formal_low, formal_high]), 
                pd.concat([casual_low, casual_high])
            )
            
            # Test emotion main effect
            emotion_f, emotion_p = stats.f_oneway(
                pd.concat([formal_low, casual_low]), 
                pd.concat([formal_high, casual_high])
            )
            
            # Test interaction effect using 2-way ANOVA approach
            # Create data for interaction test
            interaction_data = []
            
            # Add all data points with their factor labels
            for val in formal_low:
                interaction_data.append({'formality': 'formal', 'emotion': 'low', 'value': val})
            for val in formal_high:
                interaction_data.append({'formality': 'formal', 'emotion': 'high', 'value': val})
            for val in casual_low:
                interaction_data.append({'formality': 'casual', 'emotion': 'low', 'value': val})
            for val in casual_high:
                interaction_data.append({'formality': 'casual', 'emotion': 'high', 'value': val})
            
            interaction_df = pd.DataFrame(interaction_data)
            
            # Calculate interaction F-statistic manually
            # This is a simplified approach - proper 2-way ANOVA would be better
            
            # Method: Test if the difference in differences is significant
            # H0: (Formal_High - Formal_Low) = (Casual_High - Casual_Low)
            formal_emotion_effect = formal_high.mean() - formal_low.mean()
            casual_emotion_effect = casual_high.mean() - casual_low.mean()
            
            # Create two groups: differences within formal vs differences within casual
            formal_diffs = []
            casual_diffs = []
            
            # Calculate individual differences (this is approximate)
            min_len = min(len(formal_low), len(formal_high))
            for i in range(min_len):
                if i < len(formal_high) and i < len(formal_low):
                    formal_diffs.append(formal_high.iloc[i] - formal_low.iloc[i])
            
            min_len = min(len(casual_low), len(casual_high))
            for i in range(min_len):
                if i < len(casual_high) and i < len(casual_low):
                    casual_diffs.append(casual_high.iloc[i] - casual_low.iloc[i])
            
            # Test if these difference distributions are different
            if len(formal_diffs) > 0 and len(casual_diffs) > 0:
                interaction_f, interaction_p = stats.f_oneway(formal_diffs, casual_diffs)
            else:
                # Fallback: use variance-based approach
                all_values = list(formal_low) + list(formal_high) + list(casual_low) + list(casual_high)
                predicted_no_interaction = [
                    means['formal_low'], means['formal_high'], 
                    means['casual_low'], means['casual_high']
                ]
                
                # Calculate F-ratio based on interaction sum of squares
                ss_interaction = len(formal_low) * (interaction_effect ** 2)
                ss_error = np.var(all_values) * (len(all_values) - 1)
                
                if ss_error > 0:
                    interaction_f = ss_interaction / (ss_error / (len(all_values) - 4))
                    # Approximate p-value using F-distribution
                    from scipy.stats import f as f_dist
                    interaction_p = 1 - f_dist.cdf(interaction_f, 1, len(all_values) - 4)
                else:
                    interaction_f, interaction_p = 0.0, 1.0
            
            print(f"Statistical tests:")
            print(f"  Formality: F={formality_f:.3f}, p={formality_p:.4f}")
            print(f"  Emotion: F={emotion_f:.3f}, p={emotion_p:.4f}")
            print(f"  Interaction: F={interaction_f:.3f}, p={interaction_p:.4f}")
            
            # Store results
            clamp_results[clamp_val] = {
                'means': means,
                'formality_effect': formality_effect,
                'emotion_effect': emotion_effect,
                'interaction_effect': interaction_effect,
                'formality_f': formality_f,
                'formality_p': formality_p,
                'emotion_f': emotion_f,
                'emotion_p': emotion_p,
                'interaction_f': interaction_f,
                'interaction_p': interaction_p,
                'sample_sizes': sizes
            }
            
            # Check for significant effects
            if formality_p < 0.05:
                print(f"  🎯 SIGNIFICANT FORMALITY EFFECT!")
            if emotion_p < 0.05:
                print(f"  🎯 SIGNIFICANT EMOTION EFFECT!")
            if interaction_p < 0.05:
                print(f"  🎯 SIGNIFICANT INTERACTION EFFECT!")
            if interaction_effect > 0.05:  # Heuristic threshold for effect size
                print(f"  📊 LARGE INTERACTION EFFECT SIZE!")
                
        except Exception as e:
            print(f"Error in statistical tests: {e}")
    
    # 3. CROSS-CLAMP ANCOVA ANALYSIS
    print(f"\n{'='*50}")
    print("ANCOVA: 3-WAY ANALYSIS ACROSS ALL CLAMP VALUES")
    print(f"{'='*50}")
    
    try:
        # Try to run 3-way ANOVA if we have enough data
        if len(clamp_values) >= 2:
            print(f"Running 3-way ANOVA: Formality × Emotion × Clamp_Value")
            
            # Prepare data for 3-way ANOVA
            ancova_data = []
            for _, row in logit_df_diff.iterrows():
                ancova_data.append({
                    'formality': row['formality'],
                    'emotion': row['emotion'], 
                    'clamp_value': row['clamp_value'],
                    'logit_difference': row['logit_difference']
                })
            
            ancova_df = pd.DataFrame(ancova_data)
            
            # Run F-tests for 3-way interactions
            # Test if interaction effects differ across clamp values
            
            # Calculate interaction effects for each clamp
            clamp_interaction_effects = {}
            for clamp_val in clamp_values:
                clamp_subset = ancova_df[ancova_df['clamp_value'] == clamp_val]
                if len(clamp_subset) >= 4:
                    formal_low_mean = clamp_subset[(clamp_subset['formality'] == 'formal') & (clamp_subset['emotion'] == 'low')]['logit_difference'].mean()
                    formal_high_mean = clamp_subset[(clamp_subset['formality'] == 'formal') & (clamp_subset['emotion'] == 'high')]['logit_difference'].mean()
                    casual_low_mean = clamp_subset[(clamp_subset['formality'] == 'casual') & (clamp_subset['emotion'] == 'low')]['logit_difference'].mean()
                    casual_high_mean = clamp_subset[(clamp_subset['formality'] == 'casual') & (clamp_subset['emotion'] == 'high')]['logit_difference'].mean()
                    
                    interaction_effect = abs((formal_high_mean - formal_low_mean) - (casual_high_mean - casual_low_mean))
                    clamp_interaction_effects[clamp_val] = interaction_effect
            
            # Test if interaction effects vary significantly across clamp values
            if len(clamp_interaction_effects) >= 2:
                interaction_values = list(clamp_interaction_effects.values())
                clamp_labels = list(clamp_interaction_effects.keys())
                
                # Simple test of variance in interaction effects
                interaction_variance = np.var(interaction_values)
                interaction_mean = np.mean(interaction_values)
                
                print(f"Interaction effects by clamp value:")
                for clamp, effect in clamp_interaction_effects.items():
                    print(f"  Clamp {clamp}: {effect:.4f}")
                
                print(f"Variance in interaction effects: {interaction_variance:.6f}")
                print(f"Mean interaction effect: {interaction_mean:.4f}")
                
                # Test if any clamp values show particularly strong interactions
                max_interaction = max(interaction_values)
                max_clamp = clamp_labels[interaction_values.index(max_interaction)]
                
                if max_interaction > 0.02:  # Threshold for meaningful interaction
                    print(f"🎯 STRONGEST INTERACTION at clamp {max_clamp}: {max_interaction:.4f}")
                
    except Exception as e:
        print(f"Error in ANCOVA analysis: {e}")
    
    # 2. OVERALL COMPARISON ACROSS CLAMP VALUES
    print(f"\n{'='*50}")
    print("CLAMP LEVEL COMPARISON")
    print(f"{'='*50}")
    
    # Compare interaction effects across clamp levels
    interaction_effects = []
    clamp_labels = []
    
    for clamp_val, results in clamp_results.items():
        interaction_effects.append(results['interaction_effect'])
        clamp_labels.append(clamp_val)
    
    if len(interaction_effects) >= 2:
        # Test if interaction effects differ across clamp levels
        max_interaction = max(interaction_effects)
        min_interaction = min(interaction_effects)
        interaction_range = max_interaction - min_interaction
        
        print(f"Interaction effects by clamp level:")
        for clamp_val, effect in zip(clamp_labels, interaction_effects):
            print(f"  Clamp {clamp_val}: {effect:.4f}")
        
        print(f"\nInteraction effect range: {interaction_range:.4f}")
        
        # Find clamp level with strongest interaction
        max_idx = interaction_effects.index(max_interaction)
        strongest_clamp = clamp_labels[max_idx]
        
        print(f"Strongest interaction at clamp level: {strongest_clamp} (effect = {max_interaction:.4f})")
        
        if interaction_range > 0.02:  # Heuristic threshold
            print(f"🎯 CLAMPING AFFECTS INTERACTION PATTERN!")
            print(f"   This suggests the SAE feature modulates formality×emotion integration!")
    
    # 4. FINAL SUMMARY WITH POSITIVE INTERACTIONS
    print(f"\n{'='*60}")
    print("🎯 FINAL SUMMARY: SIGNIFICANT INTERACTIONS BY CLAMP VALUE")
    print(f"{'='*60}")
    
    significant_interactions = []
    large_effect_interactions = []
    
    for clamp_val, results in clamp_results.items():
        interaction_p = results.get('interaction_p', 1.0)
        interaction_effect = results.get('interaction_effect', 0.0)
        
        # Check for statistical significance
        if interaction_p < 0.05:
            significant_interactions.append({
                'clamp_value': clamp_val,
                'interaction_p': interaction_p,
                'interaction_effect': interaction_effect,
                'type': 'statistically_significant'
            })
            
        # Check for large effect size (even if not statistically significant)
        if interaction_effect > 0.02:  # Threshold for meaningful effect
            large_effect_interactions.append({
                'clamp_value': clamp_val,
                'interaction_p': interaction_p,
                'interaction_effect': interaction_effect,
                'type': 'large_effect_size'
            })
    
    print(f"\n📊 STATISTICALLY SIGNIFICANT INTERACTIONS (p < 0.05):")
    if significant_interactions:
        for result in significant_interactions:
            print(f"   Clamp {result['clamp_value']}: p = {result['interaction_p']:.4f}, effect = {result['interaction_effect']:.4f}")
    else:
        print("   None found")
    
    print(f"\n📈 LARGE INTERACTION EFFECTS (effect > 0.02):")
    if large_effect_interactions:
        for result in large_effect_interactions:
            print(f"   Clamp {result['clamp_value']}: effect = {result['interaction_effect']:.4f}, p = {result['interaction_p']:.4f}")
    else:
        print("   None found")
    
    # Combined: Both significant AND large effect
    both_significant_and_large = []
    for result in significant_interactions:
        if result['interaction_effect'] > 0.02:
            both_significant_and_large.append(result)
    
    print(f"\n🎯 BEST INTERACTIONS (both p < 0.05 AND effect > 0.02):")
    if both_significant_and_large:
        for result in both_significant_and_large:
            print(f"   ⭐ Clamp {result['clamp_value']}: p = {result['interaction_p']:.4f}, effect = {result['interaction_effect']:.4f}")
        
        # Overall conclusion
        best_clamp_values = [r['clamp_value'] for r in both_significant_and_large]
        print(f"\n🏆 CONCLUSION: Feature clamping shows significant formality×emotion interactions at clamp values: {best_clamp_values}")
        print(f"    This suggests the SAE feature modulates formality×emotion integration!")
        
    else:
        print("   None found - no clamp values show both significant p-value AND large effect size")
        
        # Backup: show best available
        if significant_interactions:
            best = max(significant_interactions, key=lambda x: x['interaction_effect'])
            print(f"   💡 Best significant interaction: Clamp {best['clamp_value']} (p = {best['interaction_p']:.4f})")
        elif large_effect_interactions:
            best = min(large_effect_interactions, key=lambda x: x['interaction_p'])
            print(f"   💡 Best large effect: Clamp {best['clamp_value']} (effect = {best['interaction_effect']:.4f})")
    
    # 3. LEGACY COMPARISON SECTION (for backwards compatibility)
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    # Overall patterns
    if clamp_results:
        first_clamp = list(clamp_results.keys())[0]
        baseline_results = clamp_results[first_clamp]
        baseline_means = baseline_results['means']
        
        print(f"Difference patterns (clamp = {first_clamp}):")
        print(f"  Formal words effect: {(baseline_means['formal_low'] + baseline_means['formal_high'])/2:.4f}")
        print(f"  High emotion effect: {(baseline_means['formal_high'] + baseline_means['casual_high'])/2:.4f}")
        print(f"  Interaction present: {baseline_results['interaction_effect'] > 0.02}")
    
    # Store all results
    anova_results = {
        'clamp_results': clamp_results,
        'interaction_effects': dict(zip(clamp_labels, interaction_effects)) if interaction_effects else {},
        'summary': {
            'strongest_interaction_clamp': strongest_clamp if 'strongest_clamp' in locals() else None,
            'interaction_range': interaction_range if 'interaction_range' in locals() else 0.0,
            'significant_modulation': interaction_range > 0.02 if 'interaction_range' in locals() else False
        }
    }
    
    return anova_results

def create_interaction_plots_with_baselines(analysis_results, output_dir, target_features, baseline_type='pipeline'):
    """
    Create visualizations of the 2×2 interaction patterns using baseline-corrected differences.
    
    Generates:
    1. Line plots showing interaction patterns for each clamp level
    2. Heatmaps of mean difference values by condition
    3. Bar plots comparing effect sizes
    4. Individual word scatter plots
    """
    logit_df_diff = analysis_results['pipeline_differences']
    
    print(f"\n=== CREATING INTERACTION PLOTS (vs {baseline_type} baseline) ===")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path(output_dir)
    plots_dir = output_dir / "interaction_plots"
    plots_dir.mkdir(exist_ok=True)
    
    features_str = "_".join(map(str, target_features))
    clamp_values = sorted(logit_df_diff['clamp_value'].unique())
    
    # 1. INTERACTION LINE PLOTS BY CLAMP LEVEL
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Formality × Emotion Interaction Patterns (Difference from Pipeline Baseline)\nSecondary SAE Features: {target_features}', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, clamp_val in enumerate(clamp_values[:4]):  # Show up to 4 clamp levels
        ax = axes[i]
        
        # Calculate means for 2×2 design using DIFFERENCES
        clamp_data = logit_df_diff[logit_df_diff['clamp_value'] == clamp_val]
        means_2x2 = clamp_data.groupby(['formality', 'emotion'])['logit_difference'].mean().unstack()
        
        # Create line plot
        means_2x2.T.plot(kind='line', ax=ax, marker='o', linewidth=3, markersize=8)
        
        ax.set_title(f'Clamp Value: {clamp_val}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emotion Level', fontsize=12)
        ax.set_ylabel('Logit Difference from Baseline', fontsize=12)
        ax.legend(title='Formality', title_fontsize=12, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Reference line at zero
        
        # Add value labels on points
        for formality in means_2x2.index:
            for j, emotion in enumerate(means_2x2.columns):
                value = means_2x2.loc[formality, emotion]
                ax.annotate(f'{value:.3f}', 
                           xy=(j, value), 
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center', fontsize=9)
    
    plt.tight_layout()
    line_plot_path = plots_dir / f"interaction_lines_baseline_features_{features_str}.png"
    plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. HEATMAPS OF MEAN DIFFERENCE VALUES
    fig, axes = plt.subplots(1, len(clamp_values), figsize=(5*len(clamp_values), 4))
    fig.suptitle(f'Mean Logit Differences Heatmap (vs Pipeline Baseline)\nSecondary SAE Features: {target_features}', 
                 fontsize=16, fontweight='bold')
    
    if len(clamp_values) == 1:
        axes = [axes]
    
    for i, clamp_val in enumerate(clamp_values):
        clamp_data = logit_df_diff[logit_df_diff['clamp_value'] == clamp_val]
        heatmap_data = clamp_data.groupby(['formality', 'emotion'])['logit_difference'].mean().unstack()
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0,
                   ax=axes[i],
                   cbar_kws={'label': 'Logit Difference from Baseline'})
        
        axes[i].set_title(f'Clamp: {clamp_val}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Emotion', fontsize=11)
        axes[i].set_ylabel('Formality', fontsize=11)
    
    plt.tight_layout()
    heatmap_path = plots_dir / f"interaction_heatmaps_baseline_features_{features_str}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. EFFECT SIZE COMPARISON BAR PLOT
    # Calculate effect sizes for each clamp level using DIFFERENCES
    effect_data = []
    
    for clamp_val in clamp_values:
        clamp_data = logit_df_diff[logit_df_diff['clamp_value'] == clamp_val]
        
        # Calculate means using DIFFERENCES
        formal_low = clamp_data[(clamp_data['formality'] == 'formal') & (clamp_data['emotion'] == 'low')]['logit_difference'].mean()
        formal_high = clamp_data[(clamp_data['formality'] == 'formal') & (clamp_data['emotion'] == 'high')]['logit_difference'].mean()
        casual_low = clamp_data[(clamp_data['formality'] == 'casual') & (clamp_data['emotion'] == 'low')]['logit_difference'].mean()
        casual_high = clamp_data[(clamp_data['formality'] == 'casual') & (clamp_data['emotion'] == 'high')]['logit_difference'].mean()
        
        # Calculate effects
        formality_effect = abs((formal_low + formal_high) / 2 - (casual_low + casual_high) / 2)
        emotion_effect = abs((formal_high + casual_high) / 2 - (formal_low + casual_low) / 2)
        interaction_effect = abs((formal_high - formal_low) - (casual_high - casual_low))
        
        effect_data.append({
            'clamp_value': clamp_val,
            'Formality Effect': formality_effect,
            'Emotion Effect': emotion_effect,
            'Interaction Effect': interaction_effect
        })
    
    effect_df = pd.DataFrame(effect_data)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(clamp_values))
    width = 0.25
    
    bars1 = ax.bar(x - width, effect_df['Formality Effect'], width, label='Formality Effect', alpha=0.8)
    bars2 = ax.bar(x, effect_df['Emotion Effect'], width, label='Emotion Effect', alpha=0.8)
    bars3 = ax.bar(x + width, effect_df['Interaction Effect'], width, label='Interaction Effect', alpha=0.8)
    
    ax.set_xlabel('Clamp Value', fontsize=12)
    ax.set_ylabel('Effect Size (Absolute Difference from Baseline)', fontsize=12)
    ax.set_title(f'Effect Sizes Across Clamp Levels (vs Pipeline Baseline)\nSecondary SAE Features: {target_features}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{cv}' for cv in clamp_values])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    effects_plot_path = plots_dir / f"effect_sizes_baseline_features_{features_str}.png"
    plt.savefig(effects_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. INDIVIDUAL WORD SCATTER PLOT
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Individual Word Logit Differences by Category (vs Pipeline Baseline)\nSecondary SAE Features: {target_features}', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    categories = ['formal_low_emotion', 'formal_high_emotion', 'casual_low_emotion', 'casual_high_emotion']
    category_labels = ['Formal + Low Emotion', 'Formal + High Emotion', 'Casual + Low Emotion', 'Casual + High Emotion']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (category, label, color) in enumerate(zip(categories, category_labels, colors)):
        ax = axes[i]
        
        category_data = logit_df_diff[logit_df_diff['word_category'] == category]
        
        # Create scatter plot with clamp value as x-axis using DIFFERENCES
        for clamp_val in clamp_values:
            clamp_data = category_data[category_data['clamp_value'] == clamp_val]
            ax.scatter([clamp_val] * len(clamp_data), clamp_data['logit_difference'], 
                      alpha=0.6, s=50, color=color, label=f'Clamp {clamp_val}' if i == 0 else "")
        
        # Add mean line
        means = category_data.groupby('clamp_value')['logit_difference'].mean()
        ax.plot(means.index, means.values, color='black', linewidth=2, marker='D', markersize=8)
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Clamp Value', fontsize=11)
        ax.set_ylabel('Logit Difference from Baseline', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)  # Reference line at zero
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    scatter_plot_path = plots_dir / f"word_scatter_baseline_features_{features_str}.png"
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Interaction plots saved to: {plots_dir}")
    print(f"  - Line plots: {line_plot_path}")
    print(f"  - Heatmaps: {heatmap_path}")
    print(f"  - Effect sizes: {effects_plot_path}")
    print(f"  - Word scatter: {scatter_plot_path}")
    
    return {
        'plots_directory': str(plots_dir),
        'line_plot': str(line_plot_path),
        'heatmap_plot': str(heatmap_path),
        'effects_plot': str(effects_plot_path),
        'scatter_plot': str(scatter_plot_path)
    }

def create_logit_summary_stats_with_baselines(analysis_results):
    """
    Create summary statistics for logit analysis using baseline differences.
    """
    logit_df_diff = analysis_results['pipeline_differences']
    
    summary_stats = []
    
    # Group by prompt, word_category, and word
    grouped = logit_df_diff.groupby(['prompt', 'word_category', 'word'])
    
    for (prompt, word_category, word), group in grouped:
        # Get logit difference values for different clamp values
        diff_values = group['logit_difference'].values
        clamp_values = group['clamp_value'].values
        
        # Calculate statistics on DIFFERENCES
        max_diff = np.max(diff_values)
        min_diff = np.min(diff_values)
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)
        range_diff = max_diff - min_diff
        
        # Calculate max absolute difference
        max_abs_difference = np.max(np.abs(diff_values))
        mean_abs_difference = np.mean(np.abs(diff_values))
        
        # Calculate effect of clamping (difference between max and min clamp values)
        if len(diff_values) >= 2:
            max_clamp_idx = np.argmax(clamp_values)
            min_clamp_idx = np.argmin(clamp_values)
            clamping_effect = diff_values[max_clamp_idx] - diff_values[min_clamp_idx]
        else:
            clamping_effect = 0.0
        
        summary_stats.append({
            'prompt': prompt,
            'word_category': word_category,
            'word': word,
            'token_id': group['token_id'].iloc[0],
            'baseline_logit': group['baseline_logit'].iloc[0],
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'min_difference': min_diff,
            'max_difference': max_diff,
            'range_difference': range_diff,
            'max_abs_difference': max_abs_difference,
            'mean_abs_difference': mean_abs_difference,
            'clamping_effect': clamping_effect,
            'num_clamp_conditions': len(diff_values),
            'target_features': group['target_features'].iloc[0] if 'target_features' in group.columns else ''
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Sort by clamping effect (largest effect first)
    summary_df = summary_df.sort_values('clamping_effect', key=abs, ascending=False)
    
    return summary_df

def create_dimensional_logit_summary_with_baselines(analysis_results):
    """
    Create summary statistics grouped by word category using baseline differences.
    """
    logit_df_diff = analysis_results['pipeline_differences']
    
    dimensional_stats = []
    
    # Group by word_category and clamp_value
    grouped = logit_df_diff.groupby(['word_category', 'clamp_value'])
    
    for (word_category, clamp_value), group in grouped:
        mean_diff = group['logit_difference'].mean()
        std_diff = group['logit_difference'].std()
        median_diff = group['logit_difference'].median()
        num_words = len(group['word'].unique())
        num_prompts = len(group['prompt'].unique())
        
        dimensional_stats.append({
            'word_category': word_category,
            'clamp_value': clamp_value,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'median_difference': median_diff,
            'num_words': num_words,
            'num_prompts': num_prompts,
            'total_observations': len(group)
        })
    
    dimensional_df = pd.DataFrame(dimensional_stats)
    
    # Calculate effect sizes (differences are already relative to baseline)
    effect_sizes = []
    for category in dimensional_df['word_category'].unique():
        category_data = dimensional_df[dimensional_df['word_category'] == category]
        
        for _, row in category_data.iterrows():
            effect_sizes.append({
                'word_category': category,
                'clamp_value': row['clamp_value'],
                'effect_size': row['mean_difference'],  # Already a difference from baseline
                'baseline_difference': 0.0  # Baseline is zero by definition
            })
    
    effect_size_df = pd.DataFrame(effect_sizes)
    
    return dimensional_df, effect_size_df

def load_models(args):
    """Load all required models."""
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Primary SAE
    print("Loading Primary SAE...")
    primary_sae_checkpoint = torch.load(args.primary_sae_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if 'model_state' in primary_sae_checkpoint:
        primary_sae_state_dict = primary_sae_checkpoint['model_state']
    else:
        primary_sae_state_dict = primary_sae_checkpoint
    
    # Determine Primary SAE dimensions
    if 'decoder.weight' in primary_sae_state_dict:
        input_dim = primary_sae_state_dict['decoder.weight'].shape[0]
        hidden_dim = primary_sae_state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in primary_sae_state_dict:
        encoder_weight = primary_sae_state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError("Cannot determine Primary SAE dimensions")
    
    primary_sae = TopKSparseAutoencoder(input_dim, hidden_dim, k=args.primary_k)
    primary_sae.load_state_dict(primary_sae_state_dict)
    primary_sae.to(args.device)
    
    # Load NFM model
    print("Loading NFM model...")
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    # Load Secondary SAE
    print("Loading Secondary SAE...")
    secondary_sae_checkpoint = torch.load(args.secondary_sae_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if 'model_state' in secondary_sae_checkpoint:
        secondary_sae_state_dict = secondary_sae_checkpoint['model_state']
    else:
        secondary_sae_state_dict = secondary_sae_checkpoint
    
    # Determine Secondary SAE dimensions
    if 'decoder.weight' in secondary_sae_state_dict:
        sec_input_dim = secondary_sae_state_dict['decoder.weight'].shape[0]
        sec_hidden_dim = secondary_sae_state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in secondary_sae_state_dict:
        sec_encoder_weight = secondary_sae_state_dict['encoder.0.weight']
        sec_hidden_dim, sec_input_dim = sec_encoder_weight.shape
    else:
        raise ValueError("Cannot determine Secondary SAE dimensions")
    
    secondary_sae = TopKSparseAutoencoder(sec_input_dim, sec_hidden_dim, k=args.secondary_k)
    secondary_sae.load_state_dict(secondary_sae_state_dict)
    secondary_sae.to(args.device)
    
    print(f"Models loaded successfully!")
    print(f"Primary SAE: {input_dim} → {hidden_dim} (k={args.primary_k})")
    print(f"NFM: {num_features} features → {k_dim} embedding dim")
    print(f"Secondary SAE: {sec_input_dim} → {sec_hidden_dim} (k={args.secondary_k})")
    
    return tokenizer, base_model, primary_sae, nfm_model, secondary_sae

def save_anova_results(anova_results, output_dir, target_features):
    """Save ANOVA analysis results to files."""
    output_dir = Path(output_dir)
    anova_dir = output_dir / "anova_analysis"
    anova_dir.mkdir(exist_ok=True)
    
    features_str = "_".join(map(str, target_features))
    
    # Save detailed results as CSV
    clamp_results_list = []
    for clamp_val, results in anova_results['clamp_results'].items():
        row = {
            'clamp_value': clamp_val,
            'formality_effect': results['formality_effect'],
            'emotion_effect': results['emotion_effect'],
            'interaction_effect': results['interaction_effect'],
            'formality_f': results['formality_f'],
            'formality_p': results['formality_p'],
            'emotion_f': results['emotion_f'],
            'emotion_p': results['emotion_p'],
            'interaction_f': results['interaction_f'],
            'interaction_p': results['interaction_p'],
            'formal_low_mean': results['means']['formal_low'],
            'formal_high_mean': results['means']['formal_high'],
            'casual_low_mean': results['means']['casual_low'],
            'casual_high_mean': results['means']['casual_high']
        }
        clamp_results_list.append(row)
    
    anova_df = pd.DataFrame(clamp_results_list)
    anova_csv_path = anova_dir / f"anova_results_baseline_features_{features_str}.csv"
    anova_df.to_csv(anova_csv_path, index=False)
    
    # Save summary (convert numpy/pandas types to native Python types)
    summary_path = anova_dir / f"anova_summary_baseline_features_{features_str}.json"
    
    # Convert all values to JSON-serializable types
    summary_to_save = {}
    for key, value in anova_results['summary'].items():
        if isinstance(value, (np.bool_, bool)):
            summary_to_save[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            summary_to_save[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            summary_to_save[key] = float(value)
        else:
            summary_to_save[key] = value
    
    with open(summary_path, 'w') as f:
        json.dump(summary_to_save, f, indent=2)
    
    print(f"ANOVA results saved to: {anova_csv_path}")
    print(f"ANOVA summary saved to: {summary_path}")

def save_results_with_anova_and_baselines(analysis_results, generation_results, output_dir, target_features, args):
    """Save analysis results INCLUDING ANOVA, PLOTS, and BASELINE COMPARISONS to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save logit analysis
    logit_dir = output_dir / "logit_analysis"
    logit_dir.mkdir(exist_ok=True)
    
    features_str = "_".join(map(str, target_features))
    
    # Save baseline data
    baseline_csv_path = logit_dir / f"baseline_logits_features_{features_str}.csv"
    analysis_results['baseline_df'].to_csv(baseline_csv_path, index=False)
    print(f"Baseline logits saved to: {baseline_csv_path}")
    
    # Save intervention data
    intervention_csv_path = logit_dir / f"intervention_logits_features_{features_str}.csv"
    analysis_results['intervention_df'].to_csv(intervention_csv_path, index=False)
    print(f"Intervention logits saved to: {intervention_csv_path}")
    
    # Save difference data (main analysis)
    difference_csv_path = logit_dir / f"logit_differences_features_{features_str}.csv"
    analysis_results['pipeline_differences'].to_csv(difference_csv_path, index=False)
    print(f"Logit differences saved to: {difference_csv_path}")
    
    # Create and save summary statistics using differences
    logit_summary_df = create_logit_summary_stats_with_baselines(analysis_results)
    logit_summary_path = logit_dir / f"logit_summary_stats_baseline_features_{features_str}.csv"
    logit_summary_df.to_csv(logit_summary_path, index=False)
    print(f"Logit summary statistics saved to: {logit_summary_path}")
    
    # Create and save dimensional analysis using differences
    dimensional_df, effect_size_df = create_dimensional_logit_summary_with_baselines(analysis_results)
    dimensional_path = logit_dir / f"logit_dimensional_analysis_baseline_features_{features_str}.csv"
    effect_size_path = logit_dir / f"logit_effect_sizes_baseline_features_{features_str}.csv"
    dimensional_df.to_csv(dimensional_path, index=False)
    effect_size_df.to_csv(effect_size_path, index=False)
    print(f"Dimensional analysis saved to: {dimensional_path}")
    print(f"Effect sizes saved to: {effect_size_path}")
    
    # Save generation analysis
    gen_dir = output_dir / "generation_analysis"
    gen_dir.mkdir(exist_ok=True)
    
    for prompt_type, clamp_results in generation_results.items():
        for clamp_condition, generations in clamp_results.items():
            filename = f"generations_{prompt_type}_{clamp_condition}_features_{features_str}.txt"
            filepath = gen_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== {prompt_type.upper()} PROMPTS - {clamp_condition.upper()} ===\n\n")
                for i, gen in enumerate(generations):
                    f.write(f"--- Generation {i+1} ---\n")
                    f.write(f"Prompt: {gen['prompt']}\n")
                    f.write(f"Generated: {gen['generated']}\n")
                    f.write(f"Full text: {gen['full_text']}\n\n")
    
    print(f"Generation analysis saved to: {gen_dir}")
    
    # Run and save ANOVA analysis with baselines
    print("\n" + "="*60)
    print("RUNNING ANOVA ANALYSIS WITH BASELINES")
    print("="*60)
    
    anova_results = run_anova_analysis_with_baselines(analysis_results)
    save_anova_results(anova_results, output_dir, target_features)
    
    # Create interaction plots with baselines
    print("\n" + "="*60)
    print("CREATING INTERACTION PLOTS WITH BASELINES")
    print("="*60)
    
    plot_results = create_interaction_plots_with_baselines(analysis_results, output_dir, target_features)
    
    # Save summary
    summary = {
        'target_features': target_features,
        'clamp_multipliers': args.clamp_multipliers,
        'generation_length': args.generation_length,
        'primary_k': getattr(args, 'primary_k', None),
        'secondary_k': getattr(args, 'secondary_k', None),
        'formal_words': args.formal_words,
        'casual_words': args.casual_words,
        'baseline_analysis_file': str(baseline_csv_path),
        'intervention_analysis_file': str(intervention_csv_path),
        'difference_analysis_file': str(difference_csv_path),
        'logit_summary_file': str(logit_summary_path),
        'generation_analysis_dir': str(gen_dir),
        'anova_analysis': {
            'strongest_interaction_clamp': anova_results['summary']['strongest_interaction_clamp'],
            'interaction_range': float(anova_results['summary']['interaction_range']),
            'significant_modulation': bool(anova_results['summary']['significant_modulation'])
        },
        'plots': plot_results,
        'model_info': {
            'model_path': args.model_path,
            'primary_sae_path': getattr(args, 'primary_sae_path', None),
            'nfm_path': getattr(args, 'nfm_path', None),
            'secondary_sae_path': getattr(args, 'secondary_sae_path', None)
        }
    }
    
    summary_path = output_dir / f"analysis_summary_baseline_features_{features_str}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Analysis summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Logit Lens and Activation Clamping Analysis with Proper Baselines (TopK SAE)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary TopK SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--secondary_sae_path", type=str, required=True, help="Path to Secondary TopK SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--target_features", type=int, nargs="+", default=[4607], 
                       help="Secondary SAE feature indices to analyze")
    parser.add_argument("--clamp_multipliers", type=float, nargs="+", default=[-2.0, 0.0, 2.0],
                       help="Multipliers for clamping features")
    parser.add_argument("--generation_length", type=int, default=50,
                       help="Number of tokens to generate")
    
    # TopK parameters
    parser.add_argument("--primary_k", type=int, default=1024, 
                       help="TopK parameter for Primary SAE (default: 1024)")
    parser.add_argument("--secondary_k", type=int, default=512, 
                       help="TopK parameter for Secondary SAE (default: 512)")
    
    parser.add_argument("--formal_low_emotion_words", type=str, nargs="+", default=DEFAULT_FORMAL_LOW_EMOTION_WORDS,
                       help="Formal + low emotion words for logit analysis")
    parser.add_argument("--formal_high_emotion_words", type=str, nargs="+", default=DEFAULT_FORMAL_HIGH_EMOTION_WORDS,
                       help="Formal + high emotion words for logit analysis")
    parser.add_argument("--casual_low_emotion_words", type=str, nargs="+", default=DEFAULT_CASUAL_LOW_EMOTION_WORDS,
                       help="Casual + low emotion words for logit analysis")
    parser.add_argument("--casual_high_emotion_words", type=str, nargs="+", default=DEFAULT_CASUAL_HIGH_EMOTION_WORDS,
                       help="Casual + high emotion words for logit analysis")
    
    # Legacy support
    parser.add_argument("--formal_words", type=str, nargs="+", default=DEFAULT_FORMAL_WORDS,
                       help="Legacy: Formal words for logit analysis")
    parser.add_argument("--casual_words", type=str, nargs="+", default=DEFAULT_CASUAL_WORDS,
                       help="Legacy: Casual words for logit analysis")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--clamp_method", type=str, default="multiply", choices=["multiply", "set"],
                       help="Method for clamping: multiply existing activation or set to fixed value")
    
    args = parser.parse_args()
    
    # Keep ALL clamp multipliers as specified by user
    print(f"Using clamp multipliers as specified: {args.clamp_multipliers}")
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting TopK SAE analysis with PROPER BASELINES for Secondary SAE features: {args.target_features}")
    print(f"Primary SAE TopK: {args.primary_k}")
    print(f"Secondary SAE TopK: {args.secondary_k}")
    print(f"Clamp multipliers: {args.clamp_multipliers}")
    print(f"Generation length: {args.generation_length}")
    print(f"Each clamp value will be compared against NO-intervention baseline")
    
    # Load models
    tokenizer, base_model, primary_sae, nfm_model, secondary_sae = load_models(args)
    
    # Initialize analyzer
    analyzer = InteractionFeatureAnalyzer(
        primary_sae, nfm_model, secondary_sae, tokenizer, base_model, args.device
    )
    
    # Run enhanced logit lens analysis with proper baselines
    print("\n" + "="*60)
    print("RUNNING LOGIT LENS ANALYSIS WITH PIPELINE BASELINE")
    print("="*60)
    
    # Use new dimensional word lists if available, otherwise fall back to legacy
    if hasattr(args, 'formal_low_emotion_words'):
        analysis_results = analyzer.logit_lens_analysis_with_baselines(
            args.formal_low_emotion_words, args.formal_high_emotion_words,
            args.casual_low_emotion_words, args.casual_high_emotion_words,
            args.target_features, args.clamp_multipliers
        )
    else:
        # Legacy support - treat formal_words as formal+low_emotion, casual_words as casual+high_emotion  
        analysis_results = analyzer.logit_lens_analysis_with_baselines(
            args.formal_words, [],  # formal_low_emotion, formal_high_emotion
            [], args.casual_words,  # casual_low_emotion, casual_high_emotion
            args.target_features, args.clamp_multipliers
        )
    
    # Run clamping analysis
    print("\n" + "="*60) 
    print("RUNNING CLAMPING ANALYSIS")
    print("="*60)
    
    generation_results = analyzer.clamping_analysis(
        args.target_features, args.clamp_multipliers, args.generation_length
    )
    
    # Save results with baseline comparisons
    print("\n" + "="*60)
    print("SAVING RESULTS WITH BASELINE ANALYSIS")
    print("="*60)
    
    save_results_with_anova_and_baselines(analysis_results, generation_results, args.output_dir, args.target_features, args)
    
    print(f"\nTopK SAE Analysis with Proper Baselines complete! Results saved to: {args.output_dir}")
    print("\nKey files generated:")
    print(f"  - Baseline logits: baseline_logits_features_*.csv")
    print(f"  - Intervention effects: logit_differences_features_*.csv") 
    print(f"  - ANOVA results: anova_results_baseline_features_*.csv")
    print(f"  - Interaction plots: interaction_plots/")
    print(f"  - Generation samples: generation_analysis/")

if __name__ == "__main__":
    main()