# SAE Feature Contrast Analysis Summary

Model: ../models/open_llama_3b
SAE: checkpoints_topk/best_model.pt
Max Token Length: 10
Analysis Method: Contrast between positive (high activation) and negative (inhibited) examples

## Analyzed Features

### Feature 21781

**Contrast Pattern:** Pattern: The positive examples contain the word "real" in the sense of reality, actuality, or the physical world, while the negative examples contain the word "that" introducing a subordinate clause.

**Post-ReLU Statistics:**
- Max activation: 2.5188
- Mean activation: 0.0642
- Percent active tokens: 10.64%

**Pre-ReLU Statistics:**
- Max activation: 2.3303
- Min activation: -12.7768
- Mean activation: 0.0922
- Std activation: 0.4270

**Top Positive Examples (High Activation):**

Positive 1:
Context: ```
iatry student . She retained her real
```
Max Token: 'real'
Activation: 2.6236

Positive 2:
Context: ```
determine the true cause of the clock pause ,
```
Max Token: 'true'
Activation: 2.5188

Positive 3:
Context: ```
; Vargas Llosa weaves real
```
Max Token: 'real'
Activation: 2.4868

**Top Negative Examples (Inhibited Activation):**

Negative 1:
Context: ```
that can be used to acquire and link different
```
Max Token: 'that'
Activation: -12.7768

Negative 2:
Context: ```
that Senjō no Valky
```
Max Token: 'that'
Activation: -12.7768

Negative 3:
Context: ```
that the game had the capacity for downloadable
```
Max Token: 'that'
Activation: -12.7768

---

