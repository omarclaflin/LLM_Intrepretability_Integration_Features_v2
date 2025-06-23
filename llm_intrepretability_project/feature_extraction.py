import argparse
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Any, Tuple

# Define argument parser
parser = argparse.ArgumentParser(description="Extract features from LLaMA using a trained SAE")
parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts JSON file")
parser.add_argument("--llama_path", type=str, required=True, help="Path to LLaMA model")
parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
parser.add_argument("--layer", type=int, default=16, help="Layer to extract activations from")
parser.add_argument("--top_k", type=int, default=20, help="Number of top features to select")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
model = AutoModelForCausalLM.from_pretrained(args.llama_path, device_map="auto")
sae_model = torch.load(args.sae_path)  # Assumed to be a trained SAE model

# Load prompts
print(f"Loading prompts from {args.prompts_file}")
with open(args.prompts_file, 'r') as f:
    prompt_data = json.load(f)

# Function to extract activations from a specific layer
def extract_activations(texts, layer_idx):
    all_activations = []
    all_sae_features = []
    
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass with hooks to capture activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        # Attach hook to the target layer
        target_layer = model.model.layers[layer_idx]
        hook = target_layer.register_forward_hook(hook_fn)
        
        # Run forward pass
        with torch.no_grad():
            model(**inputs)
        
        # Remove hook
        hook.remove()
        
        # Process through SAE
        layer_activations = activations[0]
        batch_sae_features, _ = sae_model(layer_activations)
        
        all_activations.append(layer_activations)
        all_sae_features.append(batch_sae_features)
    
    # Concatenate results
    layer_activations = torch.cat(all_activations, dim=0)
    sae_features = torch.cat(all_sae_features, dim=0)
    
    return layer_activations, sae_features

# Function to identify top features for a concept
def identify_top_features(positive_examples, negative_examples, top_k=20):
    print(f"Extracting activations for {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    
    # Extract activations
    pos_layer_acts, pos_sae_acts = extract_activations(positive_examples, args.layer)
    neg_layer_acts, neg_sae_acts = extract_activations(negative_examples, args.layer)
    
    # Average over sequence length to get one vector per example
    pos_features = pos_sae_acts.mean(dim=1).cpu().numpy()
    neg_features = neg_sae_acts.mean(dim=1).cpu().numpy()
    
    # Prepare data for classification
    X = np.vstack([pos_features, neg_features])
    y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])
    
    # Train classifier
    print("Training classifier...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    # Get feature importance
    coefs = clf.coef_[0]
    top_indices = np.argsort(np.abs(coefs))[-top_k:][::-1]
    top_coefs = coefs[top_indices]
    
    # Calculate accuracy
    accuracy = clf.score(X, y)
    print(f"Classifier accuracy: {accuracy:.4f}")
    
    return {
        "feature_indices": top_indices.tolist(),
        "feature_weights": top_coefs.tolist(),
        "accuracy": accuracy,
        "num_features": len(coefs)
    }

# Extract and identify features for categories and their states
results = {
    "metadata": {
        "pair_type": prompt_data["pair_type"],
        "pair_name": prompt_data["pair_name"]
    },
    "features": {}
}

# Category 1 features
print(f"Processing Category 1: {prompt_data['category1']['name']}")
cat1_features = identify_top_features(
    prompt_data["category1"]["positive_alone_examples"],
    prompt_data["category1"]["negative_alone_examples"],
    args.top_k
)
results["features"]["category1"] = {
    "name": prompt_data["category1"]["name"],
    "positive_state": prompt_data["category1"]["positive_state"],
    "negative_state": prompt_data["category1"]["negative_state"],
    **cat1_features
}

# Category 2 features
print(f"Processing Category 2: {prompt_data['category2']['name']}")
cat2_features = identify_top_features(
    prompt_data["category2"]["positive_alone_examples"],
    prompt_data["category2"]["negative_alone_examples"],
    args.top_k
)
results["features"]["category2"] = {
    "name": prompt_data["category2"]["name"],
    "positive_state": prompt_data["category2"]["positive_state"],
    "negative_state": prompt_data["category2"]["negative_state"],
    **cat2_features
}

# Combined category analyses
print("Processing combined categories...")
combined_analyses = {}

# Positive 1 vs All others
pos1_examples = prompt_data["combined_examples"]["positive1_positive2"] + prompt_data["combined_examples"]["positive1_negative2"]
others = prompt_data["combined_examples"]["negative1_positive2"] + prompt_data["combined_examples"]["negative1_negative2"]
combined_analyses["positive1_in_combined"] = identify_top_features(pos1_examples, others, args.top_k)

# Positive 2 vs All others
pos2_examples = prompt_data["combined_examples"]["positive1_positive2"] + prompt_data["combined_examples"]["negative1_positive2"]
others = prompt_data["combined_examples"]["positive1_negative2"] + prompt_data["combined_examples"]["negative1_negative2"]
combined_analyses["positive2_in_combined"] = identify_top_features(pos2_examples, others, args.top_k)

# Add combined analyses to results
results["features"]["combined_analyses"] = combined_analyses

# Process all example sets for activation patterns
print("Processing all example sets...")
all_example_sets = {
    "category1_positive": prompt_data["category1"]["positive_alone_examples"],
    "category1_negative": prompt_data["category1"]["negative_alone_examples"],
    "category2_positive": prompt_data["category2"]["positive_alone_examples"],
    "category2_negative": prompt_data["category2"]["negative_alone_examples"],
    "combined_positive1_positive2": prompt_data["combined_examples"]["positive1_positive2"],
    "combined_positive1_negative2": prompt_data["combined_examples"]["positive1_negative2"],
    "combined_negative1_positive2": prompt_data["combined_examples"]["negative1_positive2"],
    "combined_negative1_negative2": prompt_data["combined_examples"]["negative1_negative2"]
}

# Create directory for activations
activations_dir = os.path.join(args.output_dir, "activations")
os.makedirs(activations_dir, exist_ok=True)

# Extract and save activations for all example sets
for set_name, examples in all_example_sets.items():
    print(f"Extracting activations for {set_name}...")
    layer_acts, sae_acts = extract_activations(examples, args.layer)
    
    # Save activations
    torch.save({
        "layer_activations": layer_acts,
        "sae_activations": sae_acts,
        "examples": examples
    }, os.path.join(activations_dir, f"{set_name}_activations.pt"))

# Save final results
print(f"Saving results to {args.output_dir}")
with open(os.path.join(args.output_dir, "feature_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Done!")