import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from pathlib import Path

class FeatureAnalyzer:
    def __init__(self, sae_model, tokenizer, device="cuda"):
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """Extract features for a list of texts."""
        all_features = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features, _ = self.sae_model(inputs["input_ids"])
                all_features.append(features.mean(dim=0))  # Average over sequence length
        
        return torch.stack(all_features)
    
    def train_feature_classifier(self, 
                               positive_examples: List[str],
                               negative_examples: List[str],
                               feature_idx: int) -> Tuple[float, float]:
        """Train a linear classifier for a specific feature."""
        # Extract features
        pos_features = self.extract_features(positive_examples)
        neg_features = self.extract_features(negative_examples)
        
        # Prepare data
        X = torch.cat([pos_features, neg_features]).cpu().numpy()
        y = np.concatenate([np.ones(len(positive_examples)), 
                          np.zeros(len(negative_examples))])
        
        # Train classifier
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        # Get predictions
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get feature importance
        importance = clf.coef_[0][feature_idx]
        
        return accuracy, importance
    
    def analyze_feature_interactions(self,
                                  feature_a_idx: int,
                                  feature_b_idx: int,
                                  test_examples: List[str]) -> Dict[str, Any]:
        """Analyze interactions between two features."""
        results = {
            "correlation": 0.0,
            "co_activation_examples": [],
            "activation_stats": {
                "feature_a": {"mean": 0.0, "std": 0.0},
                "feature_b": {"mean": 0.0, "std": 0.0}
            }
        }
        
        all_features = self.extract_features(test_examples)
        feature_a_acts = all_features[:, feature_a_idx].cpu().numpy()
        feature_b_acts = all_features[:, feature_b_idx].cpu().numpy()
        
        # Calculate correlation
        correlation = np.corrcoef(feature_a_acts, feature_b_acts)[0, 1]
        results["correlation"] = float(correlation)
        
        # Find co-activation examples
        threshold = 0.1
        co_activation_mask = (feature_a_acts > threshold) & (feature_b_acts > threshold)
        
        for i, (text, is_co_activated) in enumerate(zip(test_examples, co_activation_mask)):
            if is_co_activated:
                results["co_activation_examples"].append({
                    "text": text,
                    "feature_a_activation": float(feature_a_acts[i]),
                    "feature_b_activation": float(feature_b_acts[i])
                })
        
        # Calculate statistics
        results["activation_stats"]["feature_a"] = {
            "mean": float(np.mean(feature_a_acts)),
            "std": float(np.std(feature_a_acts))
        }
        results["activation_stats"]["feature_b"] = {
            "mean": float(np.mean(feature_b_acts)),
            "std": float(np.std(feature_b_acts))
        }
        
        return results
    
    def save_feature_analysis(self, 
                            analysis_results: Dict[str, Any],
                            output_path: str):
        """Save feature analysis results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
    
    def load_feature_analysis(self, input_path: str) -> Dict[str, Any]:
        """Load feature analysis results from a JSON file."""
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def get_feature_importance_ranking(self,
                                     positive_examples: List[str],
                                     negative_examples: List[str],
                                     top_k: int = 100) -> List[Tuple[int, float]]:
        """Rank features by their importance in distinguishing positive from negative examples."""
        # Extract features
        pos_features = self.extract_features(positive_examples)
        neg_features = self.extract_features(negative_examples)
        
        # Prepare data
        X = torch.cat([pos_features, neg_features]).cpu().numpy()
        y = np.concatenate([np.ones(len(positive_examples)), 
                          np.zeros(len(negative_examples))])
        
        # Train classifier
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        # Get feature importances
        importances = np.abs(clf.coef_[0])
        feature_ranking = [(i, imp) for i, imp in enumerate(importances)]
        feature_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return feature_ranking[:top_k] 