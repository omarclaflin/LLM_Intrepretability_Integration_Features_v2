import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
from transformers import AutoTokenizer

class FeatureVisualizer:
    def __init__(self, sae_model, tokenizer, device="cuda"):
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.device = device
        
    def get_feature_activations(self, text: str) -> torch.Tensor:
        """Get feature activations for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            features, _ = self.sae_model(inputs["input_ids"])
        return features
    
    def plot_activation_heatmap(self, text: str, top_k: int = 10):
        """Create a heatmap of top-k feature activations."""
        features = self.get_feature_activations(text)
        
        # Get top-k features
        top_features = torch.topk(features.abs().mean(dim=0), k=top_k)
        feature_indices = top_features.indices
        
        # Create heatmap data
        heatmap_data = features[:, feature_indices].cpu().numpy()
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="YlOrRd", 
                   xticklabels=[f"Feature {i}" for i in feature_indices],
                   yticklabels=range(len(text.split())))
        plt.title("Feature Activation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Token Position")
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distribution(self, activations: torch.Tensor, feature_idx: int):
        """Plot the distribution of activations for a specific feature."""
        feature_acts = activations[:, feature_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(feature_acts, kde=True)
        plt.title(f"Distribution of Feature {feature_idx} Activations")
        plt.xlabel("Activation Value")
        plt.ylabel("Count")
        plt.show()
    
    def create_interactive_activation_plot(self, text: str, top_k: int = 10):
        """Create an interactive plotly visualization of feature activations."""
        features = self.get_feature_activations(text)
        tokens = self.tokenizer.tokenize(text)
        
        # Get top-k features
        top_features = torch.topk(features.abs().mean(dim=0), k=top_k)
        feature_indices = top_features.indices
        
        # Create heatmap data
        heatmap_data = features[:, feature_indices].cpu().numpy()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"Feature {i}" for i in feature_indices],
            y=tokens,
            colorscale="YlOrRd"
        ))
        
        fig.update_layout(
            title="Interactive Feature Activation Heatmap",
            xaxis_title="Features",
            yaxis_title="Tokens",
            height=400 + len(tokens) * 20  # Adjust height based on number of tokens
        )
        
        return fig
    
    def analyze_feature_specificity(self, 
                                 feature_idx: int, 
                                 examples: List[str], 
                                 activation_threshold: float = 0.1) -> Dict[str, Any]:
        """Analyze the specificity of a feature across multiple examples."""
        results = {
            "high_activation_examples": [],
            "activation_stats": {
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0
            }
        }
        
        all_activations = []
        
        for example in examples:
            features = self.get_feature_activations(example)
            feature_acts = features[:, feature_idx].cpu().numpy()
            
            # Store statistics
            all_activations.extend(feature_acts)
            
            # Find high activation regions
            high_act_indices = np.where(feature_acts > activation_threshold)[0]
            if len(high_act_indices) > 0:
                tokens = self.tokenizer.tokenize(example)
                high_act_tokens = [tokens[i] for i in high_act_indices]
                results["high_activation_examples"].append({
                    "text": example,
                    "high_activation_tokens": high_act_tokens,
                    "activation_values": feature_acts[high_act_indices].tolist()
                })
        
        # Calculate overall statistics
        all_activations = np.array(all_activations)
        results["activation_stats"] = {
            "mean": float(np.mean(all_activations)),
            "std": float(np.std(all_activations)),
            "max": float(np.max(all_activations))
        }
        
        return results 