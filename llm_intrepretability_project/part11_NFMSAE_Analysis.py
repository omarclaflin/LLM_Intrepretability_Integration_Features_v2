"""
NFM Secondary SAE Analysis Script - TOPK VERSION

This script analyzes how Primary SAE features flow through the COMPLETE NFM pipeline 
(including Primary SAE reconstruction and NFM linear pathway) into Secondary SAE features.

CORRECTED Pipeline: Layer 16 → Primary TopK SAE → [3 Pathways: Primary Reconstruction + NFM Linear + NFM Interaction] → Secondary TopK SAE

ANALYSIS APPROACHES:

1. TOP N SECONDARY SAE ACTIVATION: For a single Primary SAE feature, process its activations 
   through the COMPLETE NFM pipeline to Secondary SAE and rank all Secondary features by activation strength. 
   Identifies which Secondary SAE features are most responsive to the Primary feature.

2. TOP N PRIMARY SAE CONTRIBUTORS: For each top Secondary SAE feature found in #1, analyze 
   the pathway backwards to identify which Primary SAE features contribute most to its activation. 
   Uses gradient/weight analysis to reveal feature dependencies and neighborhoods.

3. STIMULUS-BASED COMBINATION ANALYSIS: Process the full 2x2 stimulus set through the COMPLETE pipeline 
   and compute ANOVA across conditions for each Secondary SAE feature. Identifies Secondary 
   features most sensitive to experimental stimulus manipulations.

4. MODULATION SENSITIVITY: Identical to #3 but framed as finding Secondary SAE features most 
   changed by modulations in the stimulus conditions. Reports the feature with largest variance 
   across experimental conditions.

5. FUTURE CLAMPING APPROACH: Systematic clamping experiments on Primary SAE features with 
   baseline vs. suppressed/enhanced conditions to measure Secondary SAE sensitivity. Would reveal 
   direct dependencies, interaction effects, and threshold responses across the feature hierarchy.

ALTERNATIVE APPROACHES: Instead of stimulus sets, could use (1) gradient weights and/or NFM 
embedding weights for direct pathway analysis, (2) streamed large dataset search for feature 
discovery followed by API calls for identification, (3) correlations/RSA approaches on input 
features across layers, applied to stimulus sets, weights, or discovered subsets of large datasets.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import logging
from scipy import stats
from typing import List, Tuple, Dict, Any

# Expanded stimulus sets with consistent token lengths across ALL categories (50 items each, ~10-12 tokens per item)
STIMULUS_SETS = {
    # Feature1 High (High Formality) + Feature2 High (High Emotion)
    "F1_high_F2_high": [
        "We profoundly regret the devastating error that has tarnished our reputation.",
        "It is with immense sorrow that I convey this tragic news.",
        "My heart is utterly broken by the egregious injustice perpetrated here.",
        "I hereby express my profound disgust at these deplorable findings today.",
        "The egregious suffering endured by citizens demands an immediate passionate response.",
        "With gravest apprehension, we anticipate the calamitous repercussions of this decision.",
        "An overwhelming sense of despair permeates the entire community currently.",
        "Our collective indignation at this flagrant disregard for truth is unyielding.",
        "I am compelled to voice my deepest anguish regarding this loss.",
        "The sheer terror of the unfolding catastrophe is beyond comprehension.",
        "We express our most sincere condolences for the devastating family loss.",
        "The profound injustice witnessed today fills me with insurmountable rage completely.",
        "With utmost reverence, we commemorate the heroic sacrifice of these souls.",
        "The excruciating pain of this betrayal cuts deeper than any wound.",
        "We are deeply mortified by the appalling negligence leading to tragedy.",
        "The overwhelming joy of this momentous achievement brings tears to eyes.",
        "I am absolutely devastated by the unconscionable cruelty displayed in proceedings.",
        "Our hearts overflow with gratitude for the extraordinary compassion you demonstrated.",
        "The horrific nature of these crimes demands swift and uncompromising justice.",
        "We are profoundly moved by the remarkable courage exhibited under circumstances.",
        "The intolerable anguish of separation from loved ones weighs heavily here.",
        "Our deepest sympathies extend to all those affected by this disaster.",
        "The magnificent triumph achieved today will be remembered for generations ahead.",
        "We are utterly appalled by the systematic oppression inflicted upon communities.",
        "The breathtaking beauty of this sacrifice illuminates the darkness of times.",
        "Our hearts are shattered by the senseless violence tearing families apart.",
        "The extraordinary resilience displayed by survivors inspires hope amidst overwhelming despair.",
        "We categorically condemn the heinous atrocities committed against innocent people here.",
        "The overwhelming gratitude we feel cannot adequately express our profound appreciation.",
        "Our souls are tormented by the knowledge of such preventable suffering.",
        "The triumphant victory achieved through sacrifice will never be forgotten completely.",
        "We are consumed with righteous fury at this blatant violation of rights.",
        "The exquisite agony of loss transforms grief into something almost sacred.",
        "Our hearts burst with pride witnessing such extraordinary acts of courage.",
        "The devastating impact of this betrayal will reverberate through future generations.",
        "We are eternally grateful for the supreme sacrifice made on behalf.",
        "The unspeakable horror of these events defies all rational human comprehension.",
        "Our collective mourning unites us in shared sorrow and common purpose.",
        "The brilliant triumph of justice brings hope to the darkest corners.",
        "We are profoundly disturbed by the systematic dehumanization we have witnessed.",
        "The overwhelming relief felt by families brings tears of pure joy.",
        "Our hearts ache with the unbearable weight of collective moral responsibility.",
        "The magnificent courage displayed inspires us all to achieve greater heights.",
        "We are devastated by the cruel indifference shown to human suffering.",
        "The extraordinary love demonstrated transcends all earthly boundaries and physical limitations.",
        "Our righteous indignation at this injustice burns like an eternal flame.",
        "The breathtaking sacrifice made will be honored throughout all of history.",
        "We are overcome with emotion at this unprecedented display of humanity.",
        "The terrifying implications of inaction weigh heavily upon our moral consciences.",
        "Our hearts overflow with compassion for all those who suffer needlessly."
    ],
    
    # Feature1 High (High Formality) + Feature2 Low (Low Emotion)
    "F1_high_F2_low": [
        "The aforementioned data substantiates the initial hypothesis with reasonable statistical confidence.",
        "It is imperative to review the procedural guidelines prior to implementation.",
        "The revised protocol stipulates adherence to stringent safety regulations throughout process.",
        "A comprehensive analysis of statistical variances was subsequently conducted by team.",
        "The committee's deliberations concluded without reaching a unanimous consensus on matter.",
        "This document delineates the parameters for forthcoming operational adjustments within department.",
        "The findings corroborate the established theoretical framework with acceptable precision levels.",
        "Please transmit the requisite documentation to the appropriate department in course.",
        "The stipulated deadlines must be strictly observed to ensure project continuity.",
        "Subsequent to evaluation, modifications to the existing infrastructure are recommended here.",
        "The methodology employed demonstrates adequate adherence to established research protocol standards.",
        "Pursuant to regulatory requirements, all documentation must be submitted by deadline.",
        "The analytical framework provides sufficient foundation for the proposed recommendations made.",
        "Implementation of these measures will require coordination across multiple organizational units.",
        "The assessment indicates satisfactory compliance with institutional standards and expectations currently.",
        "Preliminary observations suggest the need for additional data collection and analysis.",
        "The specified criteria have been met according to predetermined benchmark standards.",
        "Authorization for the proposed modifications must be obtained from appropriate authorities.",
        "The documentation reflects adherence to standard operating procedures throughout the process.",
        "Consultation with relevant stakeholders will be required before proceeding with implementation.",
        "The evaluation process has been completed in accordance with established guidelines.",
        "Distribution of resources will be allocated based on predetermined priority classifications.",
        "The systematic approach ensures consistency with organizational objectives and requirements established.",
        "Coordination between departments will facilitate efficient completion of assigned tasks here.",
        "The assessment protocol has been designed to meet institutional accreditation standards.",
        "Implementation will proceed according to the timeline established in preliminary planning.",
        "The documentation demonstrates compliance with regulatory standards and professional expectations currently.",
        "Evaluation of outcomes will be conducted using established measurement criteria systematically.",
        "The proposed framework aligns with institutional policies and procedural requirements established.",
        "Authorization protocols must be followed to ensure appropriate oversight and accountability.",
        "The methodology incorporates best practices from relevant professional literature and experience.",
        "Systematic documentation will be maintained throughout all phases of the process.",
        "The evaluation criteria reflect established standards for quality assurance and control.",
        "Implementation will require coordination with external agencies and regulatory bodies here.",
        "The assessment framework provides adequate foundation for informed decision-making processes currently.",
        "Documentation standards must be maintained to ensure compliance with archival requirements.",
        "The systematic approach facilitates efficient allocation of resources and personnel available.",
        "Evaluation protocols have been established to monitor progress and identify issues.",
        "The procedural framework ensures consistency with organizational policies and objectives established.",
        "Implementation will proceed in phases to minimize disruption to ongoing operations.",
        "The documentation reflects consideration of relevant factors and stakeholder input received.",
        "Assessment criteria have been developed to ensure objective evaluation of outcomes.",
        "The methodology incorporates standard practices for quality control and assurance measures.",
        "Coordination mechanisms will be established to facilitate communication between departments effectively.",
        "The evaluation framework provides structure for systematic assessment of results obtained.",
        "Implementation procedures have been designed to minimize risk and ensure compliance.",
        "The documentation standards reflect institutional requirements for record-keeping and accountability measures.",
        "Assessment protocols will be applied consistently across all relevant organizational units.",
        "The systematic approach ensures alignment with strategic objectives and operational requirements.",
        "Implementation will be monitored to ensure adherence to established timelines systematically."
    ],
    
    # Feature1 Low (Low Formality) + Feature2 High (High Emotion)  
    "F1_low_F2_high": [
        "OMG, I'm so hyped about that! This is seriously the best news!",
        "I literally can't even right now, this is just way too much!",
        "Dude, that was absolutely insane! I'm totally freaking out about it right!",
        "Seriously, I'm so mad about this I could just scream loudly!",
        "Holy cow, that's absolutely devastating! I feel so awful for them!",
        "I'm so thrilled about this, I might just cry the happiest tears!",
        "No way! This is like, completely mind-blowingly amazing and I can't believe!",
        "Ugh, I'm just so done with all this, it's making me furious!",
        "My heart just aches for everyone involved, it's such a tragic situation.",
        "I'm bouncing off the walls with excitement, you seriously have no idea!",
        "This is blowing my mind! I can't handle how awesome this thing is!",
        "I'm crying tears of joy right now, this is the most beautiful!",
        "Honestly, I'm so pissed off I can barely see straight right now!",
        "This is breaking my heart into a million tiny pieces, I'm totally devastated!",
        "I'm literally shaking with excitement! This is the coolest thing that happened!",
        "I could punch a wall right now, this whole thing makes me livid!",
        "Oh my gosh, I'm so happy I could burst! This is incredible news!",
        "This is making me sick to my stomach, I'm so disgusted right!",
        "I'm over the moon about this! Best day of my entire life!",
        "I'm so heartbroken I can't stop crying, this is absolutely terrible right!",
        "This is giving me chills! I'm so pumped up I could run!",
        "I'm seething with rage! This is the most infuriating thing I've ever seen!",
        "My heart is melting! This is the sweetest, most touching thing ever!",
        "I'm so frustrated I could scream! This is driving me absolutely crazy!",
        "This is making me giddy with excitement! I can't contain my joy!",
        "I'm boiling with anger! This is completely unacceptable and I'm totally furious!",
        "I'm glowing with happiness! This news has made my entire week perfect!",
        "This is crushing my soul! I'm in complete despair about this situation!",
        "I'm jumping for joy! This is the most exciting thing that's happened!",
        "I'm burning with indignation! This injustice is making my blood boil completely!",
        "This is filling me with pure bliss! I've never been happier in!",
        "I'm drowning in sorrow! This tragedy is tearing my heart apart completely!",
        "I'm electric with anticipation! This is going to be absolutely phenomenal and!",
        "I'm consumed with fury! This outrage is making me see red right!",
        "This is warming my heart! Such a beautiful moment that brings tears!",
        "I'm sick with worry! This terrifying situation is keeping me awake nights!",
        "I'm buzzing with energy! This fantastic news has me completely energized and!",
        "I'm choking with emotion! This heartbreaking story is overwhelming my senses completely!",
        "This is lighting me up! I'm radiating happiness and positive energy right!",
        "I'm trembling with fear! This scary situation is absolutely terrifying me completely!",
        "I'm beaming with pride! This achievement is making me glow with satisfaction!",
        "I'm aching with sympathy! This sad story is tugging at my heartstrings!",
        "This is energizing my spirit! I feel like I could conquer the world!",
        "I'm writhing in agony! This painful news is torturing my poor soul!",
        "I'm soaring with elation! This wonderful surprise has lifted my spirits high!",
        "I'm wilting with disappointment! This letdown is crushing all my hopes completely!",
        "This is intoxicating my senses! I'm drunk on happiness and pure euphoria!",
        "I'm withering with shame! This embarrassing situation is mortifying me completely today!",
        "I'm glowing with contentment! This peaceful moment is filling me with serenity!",
        "I'm collapsing with exhaustion! This stressful ordeal is draining all my energy!"
    ],
    
    # Feature1 Low (Low Formality) + Feature2 Low (Low Emotion)
    "F1_low_F2_low": [
        "Yeah, the light switch is on the wall over there by door.",
        "It's raining outside today. What's up with you these days?",
        "He's over there somewhere, I think near the back of room.",
        "I gotta go now, catch ya later. Bye for now everyone.",
        "The cat's asleep on the couch in the living room right now.",
        "Cool, got it. Thanks for letting me know about that thing.",
        "Just chilling at home, nothing much going on today around here.",
        "The food's in the fridge if you want to grab something good.",
        "It's kinda quiet around here today, not much happening at all.",
        "He said okay, I guess that works for everyone involved here.",
        "The meeting's at three, don't forget to bring your notes with.",
        "She's running a bit late but should be here soon enough.",
        "The weather's okay, not too hot and not too cold either today.",
        "I'll check my schedule and get back to you later this afternoon.",
        "The store closes at nine, so we have plenty of time left.",
        "That book's on the shelf, second row from the top there.",
        "The bus comes every twenty minutes or so during rush hour.",
        "I heard the news, seems like things are going alright these days.",
        "The coffee's ready if you want a cup right now this morning.",
        "My phone's charging, I'll call you back in a few minutes here.",
        "The game starts at seven, we should probably leave soon enough.",
        "She mentioned something about changing the plan slightly for us all.",
        "The parking lot's pretty full, but there are spots available still.",
        "I think the answer's in the back of the book somewhere.",
        "The train was on time, so everything worked out fine today.",
        "He's bringing lunch, so we don't need to worry about food.",
        "The computer's running slow, might need to restart it soon enough.",
        "I saw the email, looks like everything's set for tomorrow's meeting.",
        "The keys are on the kitchen counter by the fruit bowl.",
        "She's working late tonight, won't be home until after eight o'clock.",
        "The movie's okay, nothing special but not bad either I guess.",
        "I'll pick up milk on the way home from work today.",
        "The dog needs a walk, probably in an hour or so.",
        "He's got the information we need for the project we're doing.",
        "The internet's working fine, no problems with the connection today at.",
        "I'll send the file over once I finish making the changes.",
        "The package should arrive sometime this week, maybe Thursday or Friday.",
        "She's making dinner, should be ready in about thirty minutes from.",
        "The remote's somewhere on the coffee table under those magazines there.",
        "I'll check the calendar and see what day works best for.",
        "The library's open until six, so we have time to stop by.",
        "He mentioned meeting up for lunch sometime next week maybe Tuesday.",
        "The temperature's supposed to drop a bit tonight according to weather.",
        "I'll grab some snacks for the trip on the way out.",
        "The printer's out of paper, there's more in the supply closet.",
        "She's taking the early flight, should land around ten in morning.",
        "The restaurant's pretty good, we've been there a few times before.",
        "I'll finish this up and then we can head out together.",
        "The music's a bit loud, but it's not bothering me much.",
        "He's got the right idea about how to handle this situation."
    ]
}

# TopK Sparse Autoencoder (from part1c)
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

class NFMSAEAnalyzer:
    """Analyzer for NFM Secondary SAE relationships using COMPLETE pipeline with TopK SAEs."""
    
    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, device="cuda", target_layer=16):
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

    def get_complete_pipeline_activations(self, texts, batch_size=16):
        """
        Process texts through the COMPLETE pipeline: 
        Layer 16 → Primary TopK SAE → [Primary Reconstruction + NFM Linear + NFM Interaction] → Secondary TopK SAE
        
        Returns:
            primary_activations: [num_texts, primary_sae_features] 
            primary_reconstructions: [num_texts, layer_dim]
            nfm_linear_outputs: [num_texts, layer_dim]
            nfm_interaction_outputs: [num_texts, layer_dim]
            secondary_activations: [num_texts, secondary_sae_features]
        """
        primary_activations = []
        primary_reconstructions = []
        nfm_linear_outputs = []
        nfm_interaction_outputs = []
        secondary_activations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing complete TopK pipeline"):
            batch_texts = texts[i:i+batch_size]
            if not batch_texts:
                continue
                
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=100).to(self.device)
            
            with torch.no_grad():
                # Step 1: Layer 16 → Primary TopK SAE
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                # RESHAPE FOR TOPK SAE: 3D → 2D
                batch_size_inner, seq_len, hidden_dim = hidden_states.shape
                hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
                
                primary_features, primary_reconstruction = self.primary_sae(
                    hidden_states_reshaped.to(self.primary_sae.encoder[0].weight.dtype)
                )
                
                # RESHAPE BACK: 2D → 3D
                primary_features = primary_features.reshape(batch_size_inner, seq_len, -1)
                primary_reconstruction = primary_reconstruction.reshape(batch_size_inner, seq_len, -1)
                
                # Step 2: NFM Linear Pathway
                nfm_linear_output = self.nfm_model.linear(primary_features)
                
                # Step 3: NFM Interaction Pathway
                # 3a: Feature embeddings
                embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
                weighted_embeddings = torch.matmul(primary_features, embeddings.T)  # [batch, seq, k_dim]
                
                # 3b: Interaction MLP Layer 1 [500 x 500]
                mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)
                
                # 3c: Secondary TopK SAE (AFTER MLP Layer 1, BEFORE ReLU)
                original_shape = mlp_layer1_output.shape
                mlp_layer1_flat = mlp_layer1_output.view(-1, original_shape[-1])
                
                secondary_features, _ = self.secondary_sae(
                    mlp_layer1_flat.to(self.secondary_sae.encoder[0].weight.dtype)
                )
                secondary_reconstruction = self.secondary_sae.decoder(secondary_features)
                secondary_reconstruction_reshaped = secondary_reconstruction.view(original_shape)
                
                # 3d: Continue through remaining interaction MLP layers
                # ReLU activation (Layer 2)
                relu_output = self.nfm_model.interaction_mlp[2](secondary_reconstruction_reshaped)
                # Final linear layer (Layer 3)
                nfm_interaction_output = self.nfm_model.interaction_mlp[3](relu_output)
                
                # Get mean activation across sequence for each text
                for b in range(primary_features.shape[0]):
                    seq_len_actual = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len_actual > 0:
                        # Store all pathway outputs
                        mean_primary_features = torch.mean(primary_features[b, :seq_len_actual, :], dim=0)
                        mean_primary_reconstruction = torch.mean(primary_reconstruction[b, :seq_len_actual, :], dim=0)
                        mean_nfm_linear = torch.mean(nfm_linear_output[b, :seq_len_actual, :], dim=0)
                        mean_nfm_interaction = torch.mean(nfm_interaction_output[b, :seq_len_actual, :], dim=0)
                        
                        # For secondary features, we need to compute the mean correctly
                        # since secondary_features is already flattened [batch*seq, features]
                        start_idx = b * seq_len_actual
                        end_idx = start_idx + seq_len_actual
                        if end_idx <= secondary_features.shape[0]:
                            mean_secondary_features = torch.mean(secondary_features[start_idx:end_idx, :], dim=0)
                        else:
                            # Fallback if indexing doesn't work
                            secondary_features_reshaped = secondary_features.view(batch_size_inner, seq_len, -1)
                            mean_secondary_features = torch.mean(secondary_features_reshaped[b, :seq_len_actual, :], dim=0)
                        
                        primary_activations.append(mean_primary_features.cpu().numpy())
                        primary_reconstructions.append(mean_primary_reconstruction.cpu().numpy())
                        nfm_linear_outputs.append(mean_nfm_linear.cpu().numpy())
                        nfm_interaction_outputs.append(mean_nfm_interaction.cpu().numpy())
                        secondary_activations.append(mean_secondary_features.cpu().numpy())
        
        return (np.array(primary_activations), np.array(primary_reconstructions), 
                np.array(nfm_linear_outputs), np.array(nfm_interaction_outputs), 
                np.array(secondary_activations))

    def find_top_secondary_features(self, primary_feature_idx, top_n_secondary=10):
        """
        Analysis 1: Find top N Secondary SAE features most activated by a single Primary SAE feature.
        Uses the COMPLETE pipeline including all three pathways with TopK SAEs.
        """
        print(f"\n=== ANALYSIS 1: TOP {top_n_secondary} SECONDARY TopK SAE FEATURES FOR PRIMARY FEATURE {primary_feature_idx} ===")
        
        # Collect all texts for analysis
        all_texts = []
        for condition_texts in STIMULUS_SETS.values():
            all_texts.extend(condition_texts)
        
        # Get complete pipeline activations
        (primary_activations, primary_reconstructions, nfm_linear_outputs, 
         nfm_interaction_outputs, secondary_activations) = self.get_complete_pipeline_activations(all_texts)
        
        # Focus on the specific primary feature
        primary_feature_activations = primary_activations[:, primary_feature_idx]
        
        # For each text, isolate this primary feature and process through complete pipeline
        secondary_isolated_activations = []
        
        for text_idx in range(len(all_texts)):
            # Get primary feature activation for this text
            primary_activation = primary_feature_activations[text_idx]
            
            # Create a sparse vector with only this primary feature activated
            sparse_primary = np.zeros(primary_activations.shape[1])
            sparse_primary[primary_feature_idx] = primary_activation
            sparse_primary_tensor = torch.tensor(sparse_primary, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # Process through complete NFM pipeline
                # Linear pathway
                linear_output = self.nfm_model.linear(sparse_primary_tensor)
                
                # Interaction pathway
                embeddings = self.nfm_model.feature_embeddings.weight.T
                weighted_embeddings = torch.matmul(sparse_primary_tensor, embeddings.T)
                
                # MLP Layer 1
                mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)
                
                # Secondary TopK SAE
                secondary_features, _ = self.secondary_sae(
                    mlp_layer1_output.to(self.secondary_sae.encoder[0].weight.dtype)
                )
                
                secondary_isolated_activations.append(secondary_features.squeeze().cpu().numpy())
        
        # Convert to array and find mean activation for each secondary feature
        secondary_isolated_activations = np.array(secondary_isolated_activations)
        mean_secondary_activations = np.mean(secondary_isolated_activations, axis=0)
        
        # Get top N secondary features
        top_indices = np.argsort(mean_secondary_activations)[-top_n_secondary:][::-1]
        
        print(f"Primary Feature {primary_feature_idx} Statistics:")
        print(f"  Mean activation: {np.mean(primary_feature_activations):.6f}")
        print(f"  Std activation: {np.std(primary_feature_activations):.6f}")
        print(f"  Range: [{np.min(primary_feature_activations):.6f}, {np.max(primary_feature_activations):.6f}]")
        
        # Show TopK sparsity stats for Primary SAE
        if hasattr(self.primary_sae, 'k'):
            primary_total_features = self.primary_sae.encoder[0].out_features
            primary_sparsity = (self.primary_sae.k / primary_total_features) * 100
            print(f"  Primary TopK SAE: {self.primary_sae.k}/{primary_total_features} = {primary_sparsity:.2f}% active")
        
        print(f"\nTop {top_n_secondary} Secondary TopK SAE Features (via complete pipeline):")
        results = {}
        for rank, sec_idx in enumerate(top_indices):
            mean_activation = mean_secondary_activations[sec_idx]
            std_activation = np.std(secondary_isolated_activations[:, sec_idx])
            max_activation = np.max(secondary_isolated_activations[:, sec_idx])
            
            print(f"  Rank {rank+1}: Secondary Feature {sec_idx}")
            print(f"    Mean activation: {mean_activation:.6f} ± {std_activation:.6f}")
            print(f"    Max activation: {max_activation:.6f}")
            
            results[rank+1] = {
                'secondary_feature_idx': int(sec_idx),
                'mean_activation': float(mean_activation),
                'std_activation': float(std_activation),
                'max_activation': float(max_activation)
            }
        
        # Show TopK sparsity stats for Secondary SAE
        if hasattr(self.secondary_sae, 'k'):
            secondary_total_features = self.secondary_sae.encoder[0].out_features
            secondary_sparsity = (self.secondary_sae.k / secondary_total_features) * 100
            print(f"Secondary TopK SAE: {self.secondary_sae.k}/{secondary_total_features} = {secondary_sparsity:.2f}% active")
        
        return results, top_indices

    def find_top_primary_contributors(self, secondary_feature_indices, top_n_primary=10):
        """
        Analysis 2: Find top N Primary SAE features that contribute most to each Secondary SAE feature.
        Uses complete pipeline analysis with TopK SAEs.
        """
        print(f"\n=== ANALYSIS 2: TOP {top_n_primary} PRIMARY TopK SAE CONTRIBUTORS FOR EACH SECONDARY FEATURE ===")
        
        # Collect all texts
        all_texts = []
        for condition_texts in STIMULUS_SETS.values():
            all_texts.extend(condition_texts)
        
        # Get complete pipeline activations for all texts
        (primary_activations, primary_reconstructions, nfm_linear_outputs, 
         nfm_interaction_outputs, secondary_activations) = self.get_complete_pipeline_activations(all_texts)
        
        results = {}
        
        for sec_idx in secondary_feature_indices:
            print(f"\nSecondary TopK Feature {sec_idx}:")
            
            # For each text, find which primary features contribute most when this secondary feature is active
            primary_contributions = []
            
            for text_idx in range(len(all_texts)):
                # Get the secondary feature activation for this text
                secondary_activation = secondary_activations[text_idx, sec_idx]
                
                # If secondary feature is active, record which primary features were active
                if secondary_activation > 0:
                    primary_contributions.append(primary_activations[text_idx, :])
            
            if len(primary_contributions) == 0:
                print(f"  No active instances found for Secondary TopK Feature {sec_idx}")
                continue
                
            # Convert to array and find mean activation when secondary feature is active
            primary_contributions = np.array(primary_contributions)
            mean_primary_when_secondary_active = np.mean(primary_contributions, axis=0)
            
            # Get top N primary features
            top_primary_indices = np.argsort(mean_primary_when_secondary_active)[-top_n_primary:][::-1]
            
            print(f"  Top {top_n_primary} contributing Primary TopK SAE features (via complete pipeline):")
            sec_results = {}
            
            for rank, prim_idx in enumerate(top_primary_indices):
                mean_activation = mean_primary_when_secondary_active[prim_idx]
                std_activation = np.std(primary_contributions[:, prim_idx]) if len(primary_contributions) > 1 else 0.0
                max_activation = np.max(primary_contributions[:, prim_idx])
                
                print(f"    Rank {rank+1}: Primary Feature {prim_idx}")
                print(f"      Mean activation when secondary active: {mean_activation:.6f} ± {std_activation:.6f}")
                print(f"      Max activation: {max_activation:.6f}")
                
                sec_results[rank+1] = {
                    'primary_feature_idx': int(prim_idx),
                    'mean_activation_when_secondary_active': float(mean_activation),
                    'std_activation': float(std_activation),
                    'max_activation': float(max_activation),
                    'num_active_instances': len(primary_contributions)
                }
            
            results[int(sec_idx)] = sec_results
        
        return results

    def find_highest_activated_and_differencing_features(self, top_n_activation=10):
        """
        Analysis 3: Find Secondary SAE features with highest activation in [1,1] vs [0,0] conditions.
        Uses complete pipeline processing with TopK SAEs.
        """
        print(f"\n=== ANALYSIS 3: TOP {top_n_activation} HIGHEST ACTIVATED AND DIFFERENCING SECONDARY TopK SAE FEATURES ===")
        
        # Process [1,1] and [0,0] conditions
        condition_11_texts = STIMULUS_SETS['F1_high_F2_high']  # [1,1]
        condition_00_texts = STIMULUS_SETS['F1_low_F2_low']    # [0,0]
        
        (_, _, _, _, activations_11) = self.get_complete_pipeline_activations(condition_11_texts)
        (_, _, _, _, activations_00) = self.get_complete_pipeline_activations(condition_00_texts)
        
        # Calculate mean activations for each condition
        mean_activations_11 = np.mean(activations_11, axis=0)
        mean_activations_00 = np.mean(activations_00, axis=0)
        
        # Find top N highest activated features in [1,1] condition
        top_11_indices = np.argsort(mean_activations_11)[-top_n_activation:][::-1]
        
        print(f"Top {top_n_activation} highest activated Secondary TopK SAE features in [1,1] condition:")
        for rank, idx in enumerate(top_11_indices):
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx}")
            print(f"    [1,1] activation: {activation_11:.6f}")
            print(f"    [0,0] activation: {activation_00:.6f}")
        
        # Find top N highest activated features in [0,0] condition
        top_00_indices = np.argsort(mean_activations_00)[-top_n_activation:][::-1]
        
        print(f"\nTop {top_n_activation} highest activated Secondary TopK SAE features in [0,0] condition:")
        for rank, idx in enumerate(top_00_indices):
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx}")
            print(f"    [0,0] activation: {activation_00:.6f}")
            print(f"    [1,1] activation: {activation_11:.6f}")
        
        # Find top N features with largest positive difference ([1,1] - [0,0])
        differences = mean_activations_11 - mean_activations_00
        top_diff_indices = np.argsort(differences)[-top_n_activation:][::-1]
        
        print(f"\nTop {top_n_activation} largest differencing Secondary TopK SAE features ([1,1] - [0,0]):")
        for rank, idx in enumerate(top_diff_indices):
            difference = differences[idx]
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx}")
            print(f"    Difference: {difference:.6f}")
            print(f"    [1,1] activation: {activation_11:.6f}")
            print(f"    [0,0] activation: {activation_00:.6f}")
        
        # Find top N features with largest negative difference ([0,0] > [1,1])
        top_neg_diff_indices = np.argsort(differences)[:top_n_activation]
        
        print(f"\nTop {top_n_activation} largest negative differencing Secondary TopK SAE features ([0,0] > [1,1]):")
        for rank, idx in enumerate(top_neg_diff_indices):
            difference = differences[idx]
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx}")
            print(f"    Difference: {difference:.6f}")
            print(f"    [1,1] activation: {activation_11:.6f}")
            print(f"    [0,0] activation: {activation_00:.6f}")
        
        results = {
            'top_11_features': [{'rank': r+1, 'feature_idx': int(idx), 
                                'activation_11': float(mean_activations_11[idx]),
                                'activation_00': float(mean_activations_00[idx])} 
                               for r, idx in enumerate(top_11_indices)],
            'top_00_features': [{'rank': r+1, 'feature_idx': int(idx),
                                'activation_00': float(mean_activations_00[idx]),
                                'activation_11': float(mean_activations_11[idx])}
                               for r, idx in enumerate(top_00_indices)],
            'top_positive_diff_features': [{'rank': r+1, 'feature_idx': int(idx),
                                           'difference': float(differences[idx]),
                                           'activation_11': float(mean_activations_11[idx]),
                                           'activation_00': float(mean_activations_00[idx])}
                                          for r, idx in enumerate(top_diff_indices)],
            'top_negative_diff_features': [{'rank': r+1, 'feature_idx': int(idx),
                                           'difference': float(differences[idx]),
                                           'activation_11': float(mean_activations_11[idx]),
                                           'activation_00': float(mean_activations_00[idx])}
                                          for r, idx in enumerate(top_neg_diff_indices)]
        }
        
        return results

    def find_anova_sensitive_feature(self, top_n_anova=10):
        """
        Analysis 4: Find top N Secondary SAE features most sensitive to 2x2 stimulus manipulations using ANOVA.
        Uses complete pipeline processing with TopK SAEs.
        """
        print(f"\n=== ANALYSIS 4: TOP {top_n_anova} SECONDARY TopK SAE FEATURES MOST SENSITIVE TO MODULATIONS (ANOVA) ===")
        
        # Process each condition separately
        condition_data = {}
        conditions = ['F1_high_F2_high', 'F1_high_F2_low', 'F1_low_F2_high', 'F1_low_F2_low']
        
        for condition in conditions:
            texts = STIMULUS_SETS[condition]
            (_, _, _, _, secondary_activations) = self.get_complete_pipeline_activations(texts)
            condition_data[condition] = secondary_activations
        
        # Compute ANOVA for each secondary feature separately
        num_secondary_features = condition_data[conditions[0]].shape[1]
        f_statistics = []
        p_values = []
        valid_features = []  # Track which features have valid ANOVA results
        
        for sec_idx in range(num_secondary_features):
            # Collect data for this secondary feature across all conditions
            groups = []
            for condition in conditions:
                groups.append(condition_data[condition][:, sec_idx])
            
            # Perform one-way ANOVA on this single feature's activations across conditions
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                # Filter out nan/inf results
                if np.isfinite(f_stat) and np.isfinite(p_val):
                    f_statistics.append(f_stat)
                    p_values.append(p_val)
                    valid_features.append(sec_idx)
            except:
                # Skip features that cause ANOVA errors
                continue
        
        if not valid_features:
            print("No Secondary TopK SAE features produced valid ANOVA results!")
            return {'error': 'No valid ANOVA results'}
        
        # Find top N features with highest F-statistics among valid features
        top_f_indices = np.argsort(f_statistics)[-top_n_anova:][::-1]
        
        print(f"Top {top_n_anova} most ANOVA-sensitive Secondary TopK SAE features:")
        print(f"  Valid features analyzed: {len(valid_features)} out of {num_secondary_features}")
        
        # Show TopK sparsity information
        if hasattr(self.secondary_sae, 'k'):
            secondary_total_features = self.secondary_sae.encoder[0].out_features
            secondary_sparsity = (self.secondary_sae.k / secondary_total_features) * 100
            print(f"  Secondary TopK SAE: {self.secondary_sae.k}/{secondary_total_features} = {secondary_sparsity:.2f}% active")
        
        top_features = []
        for rank, f_idx in enumerate(top_f_indices):
            actual_feature_idx = valid_features[f_idx]
            f_stat = f_statistics[f_idx]
            p_val = p_values[f_idx]
            
            print(f"\n  Rank {rank+1}: Secondary TopK Feature {actual_feature_idx}")
            print(f"    F-statistic: {f_stat:.6f}")
            print(f"    p-value: {p_val:.2e}")
            
            # Show condition means for this feature
            condition_stats = {}
            print(f"    Condition means:")
            for condition in conditions:
                mean_val = np.mean(condition_data[condition][:, actual_feature_idx])
                std_val = np.std(condition_data[condition][:, actual_feature_idx])
                print(f"      {condition}: {mean_val:.6f} ± {std_val:.6f}")
                condition_stats[condition] = {
                    'mean': float(mean_val),
                    'std': float(std_val)
                }
            
            top_features.append({
                'rank': rank+1,
                'feature_idx': int(actual_feature_idx),
                'f_statistic': float(f_stat),
                'p_value': float(p_val),
                'condition_stats': condition_stats
            })
        
        results = {
            'top_anova_features': top_features,
            'valid_features_count': len(valid_features),
            'total_features_count': num_secondary_features
        }
        
        return results

def load_sae_model(checkpoint_path, device="cuda"):
    """
    Load SAE model, detecting whether it's TopK or regular SAE.
    Returns the appropriate model instance.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the actual model state dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        print(f"Loaded checkpoint with keys: {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint
    
    print(f"State dict keys: {list(state_dict.keys())}")
    
    # Determine dimensions
    if 'decoder.weight' in state_dict:
        input_dim = state_dict['decoder.weight'].shape[0]
        hidden_dim = state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        # Try to infer from any available linear layer
        possible_keys = [k for k in state_dict.keys() if 'weight' in k]
        print(f"Available weight keys: {possible_keys}")
        
        # Look for encoder-like layers
        encoder_keys = [k for k in possible_keys if 'encoder' in k.lower()]
        decoder_keys = [k for k in possible_keys if 'decoder' in k.lower()]
        
        if encoder_keys:
            encoder_weight = state_dict[encoder_keys[0]]
            if len(encoder_weight.shape) == 2:
                hidden_dim, input_dim = encoder_weight.shape
            else:
                raise ValueError(f"Unexpected encoder weight shape: {encoder_weight.shape}")
        elif decoder_keys:
            decoder_weight = state_dict[decoder_keys[0]]
            if len(decoder_weight.shape) == 2:
                input_dim, hidden_dim = decoder_weight.shape
            else:
                raise ValueError(f"Unexpected decoder weight shape: {decoder_weight.shape}")
        else:
            raise ValueError(f"Cannot determine SAE dimensions from state dict keys: {possible_keys}")
    
    print(f"Detected SAE dimensions: {input_dim} → {hidden_dim}")
    
    # Check if this is a TopK SAE by looking for TopK-specific parameters or filename
    is_topk = False
    k_value = None
    
    # Check filename for TopK indicators
    filename = str(checkpoint_path).lower()
    if 'topk' in filename or 'top_k' in filename:
        is_topk = True
        # Try to extract K value from filename
        import re
        k_matches = re.findall(r'topk(\d+)', filename)
        if k_matches:
            k_value = int(k_matches[0])
        else:
            # Look for other patterns like _k1024_
            k_matches = re.findall(r'_k(\d+)_', filename)
            if k_matches:
                k_value = int(k_matches[0])
    
    # Check checkpoint metadata for TopK info
    if 'metrics_history' in checkpoint:
        metrics = checkpoint['metrics_history']
        if 'percent_active' in metrics:
            # TopK SAEs typically have very consistent percent_active values
            if len(metrics['percent_active']) > 0:
                recent_active = metrics['percent_active'][-10:]  # Last 10 values
                if len(recent_active) > 1:
                    std_active = np.std(recent_active)
                    mean_active = np.mean(recent_active)
                    # TopK SAEs have very low std in percent_active
                    if std_active < 0.1 and mean_active < 10:  # Less than 10% active
                        is_topk = True
                        # Estimate K from percent active
                        if k_value is None:
                            k_value = int((mean_active / 100.0) * hidden_dim)
    
    # Default K value if not found
    if is_topk and k_value is None:
        # Use a reasonable default (2% of features)
        k_value = max(1, int(0.02 * hidden_dim))
        print(f"Warning: TopK SAE detected but K value not found. Using default K={k_value} (2% of {hidden_dim})")
    
    # Create appropriate model
    if is_topk:
        print(f"Loading as TopK SAE with K={k_value}")
        model = TopKSparseAutoencoder(input_dim, hidden_dim, k_value)
    else:
        print(f"Loading as regular SAE")
        model = SparseAutoencoder(input_dim, hidden_dim)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="NFM Secondary SAE Analysis - TopK VERSION")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--secondary_sae_path", type=str, required=True, help="Path to Secondary SAE model")
    parser.add_argument("--primary_feature1", type=int, required=True, help="First Primary SAE feature index")
    parser.add_argument("--primary_feature2", type=int, required=True, help="Second Primary SAE feature index")
    parser.add_argument("--top_n_secondary", type=int, default=10, help="Number of top Secondary SAE features to report in Analysis 1")
    parser.add_argument("--top_n_primary", type=int, default=10, help="Number of top Primary SAE features to report in Analysis 2")
    parser.add_argument("--top_n_activation", type=int, default=10, help="Number of top features to report in Analysis 3 (activation/differencing)")
    parser.add_argument("--top_n_anova", type=int, default=10, help="Number of top features to report in Analysis 4 (ANOVA sensitivity)")
    parser.add_argument("--output_dir", type=str, default="./nfm_sae_analysis_topk", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "nfm_sae_analysis_topk.log"),
            logging.StreamHandler()
        ]
    )
    
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Primary SAE (auto-detect TopK vs regular)
    print(f"Loading Primary SAE from {args.primary_sae_path}...")
    primary_sae = load_sae_model(args.primary_sae_path, args.device)
    
    # Load NFM model
    print(f"Loading NFM from {args.nfm_path}...")
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    # Load Secondary SAE (auto-detect TopK vs regular)
    print(f"Loading Secondary SAE from {args.secondary_sae_path}...")
    secondary_sae = load_sae_model(args.secondary_sae_path, args.device)
    
    print(f"Models loaded successfully!")
    
    # Print model information
    if hasattr(primary_sae, 'k'):
        primary_total = primary_sae.encoder[0].out_features
        primary_sparsity = (primary_sae.k / primary_total) * 100
        print(f"Primary TopK SAE: {primary_sae.encoder[0].in_features} → {primary_total} (K={primary_sae.k}, {primary_sparsity:.2f}% active)")
    else:
        print(f"Primary Regular SAE: {primary_sae.encoder[0].in_features} → {primary_sae.encoder[0].out_features}")
    
    print(f"NFM: {num_features} features → {k_dim} embedding dim")
    
    if hasattr(secondary_sae, 'k'):
        secondary_total = secondary_sae.encoder[0].out_features
        secondary_sparsity = (secondary_sae.k / secondary_total) * 100
        print(f"Secondary TopK SAE: {secondary_sae.encoder[0].in_features} → {secondary_total} (K={secondary_sae.k}, {secondary_sparsity:.2f}% active)")
    else:
        print(f"Secondary Regular SAE: {secondary_sae.encoder[0].in_features} → {secondary_sae.encoder[0].out_features}")
    
    # Initialize analyzer
    analyzer = NFMSAEAnalyzer(primary_sae, nfm_model, secondary_sae, tokenizer, base_model, args.device)
    
    # Store all results
    all_results = {}
    
    # Analysis 1: For each primary feature separately
    for primary_feature in [args.primary_feature1, args.primary_feature2]:
        print(f"\n{'='*80}")
        print(f"ANALYZING PRIMARY SAE FEATURE {primary_feature}")
        print(f"{'='*80}")
        
        # Analysis 1: Find top secondary features
        analysis1_results, top_secondary_indices = analyzer.find_top_secondary_features(
            primary_feature, top_n_secondary=args.top_n_secondary
        )
        
        # Analysis 2: Find top primary contributors for these secondary features
        analysis2_results = analyzer.find_top_primary_contributors(
            top_secondary_indices, top_n_primary=args.top_n_primary
        )
        
        all_results[f'primary_feature_{primary_feature}'] = {
            'analysis1_top_secondary_features': analysis1_results,
            'analysis2_primary_contributors': analysis2_results
        }
    
    # Analysis 3: Highest activated and differencing features
    analysis3_results = analyzer.find_highest_activated_and_differencing_features(
        top_n_activation=args.top_n_activation
    )
    all_results['analysis3_highest_activated_and_differencing'] = analysis3_results
    
    # Analysis 4: ANOVA sensitivity  
    analysis4_results = analyzer.find_anova_sensitive_feature(
        top_n_anova=args.top_n_anova
    )
    all_results['analysis4_anova_sensitivity'] = analysis4_results
    
    # Save results
    results_path = output_dir / "nfm_sae_analysis_results_topk.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TOPK ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_path}")
    print(f"Log saved to: {output_dir / 'nfm_sae_analysis_topk.log'}")

if __name__ == "__main__":
    main()