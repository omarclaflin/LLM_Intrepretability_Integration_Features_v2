# SAE Feature Contrast Analysis Summary

Model: ../models/open_llama_3b
SAE: checkpoints_topk/best_model.pt
Max Token Length: 10
Analysis Method: Contrast between positive (high activation) and negative (inhibited) examples

## Analyzed Features

### Feature 21607

**Contrast Pattern:** Pattern: The positive examples all contain the word "that" followed by a subordinate clause, while the negative examples do not have this specific structure.

**Post-ReLU Statistics:**
- Max activation: 28.0630
- Mean activation: 0.0121
- Percent active tokens: 0.52%

**Pre-ReLU Statistics:**
- Max activation: 28.0630
- Min activation: -1.1785
- Mean activation: -0.3036
- Std activation: 0.7729

**Top Positive Examples (High Activation):**

Positive 1:
Context: ```
that can be used to acquire and link different
```
Max Token: 'that'
Activation: 28.0630

Positive 2:
Context: ```
that Senjō no Valky
```
Max Token: 'that'
Activation: 28.0630

Positive 3:
Context: ```
that the game had the capacity for downloadable
```
Max Token: 'that'
Activation: 28.0630

**Top Negative Examples (Inhibited Activation):**

Negative 1:
Context: ```
osa takes part in this discourse by recount
```
Max Token: 'discourse'
Activation: -1.2173

Negative 2:
Context: ```
authorized a gold dollar . In its early years
```
Max Token: 'years'
Activation: -1.2125

Negative 3:
Context: ```
the Gateway of India , Mumbai , which has
```
Max Token: 'Mumbai'
Activation: -1.1913

---

