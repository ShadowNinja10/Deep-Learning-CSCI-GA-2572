# Deep Learning — Assignments (Fall 2024)

This repo collects all course assignments. Each section below summarizes **what you build/derive** and the **problem prompts** you tackle.

---

## Homework 1 — Backpropagation

**Theory.**  
You work through the calculus of a two-layer MLP, **Linear → nonlinearity → Linear → output nonlinearity**, with all vectors treated as columns and numerator-layout matrix calculus. You derive forward passes, backprop gradients, and check shapes for both **regression** and **classification** setups. Prompts include: name the 5 SGD programming steps; write forward variables for each layer; compute full gradients for weights/biases; show specific Jacobian entries; then adapt the derivations when switching activations/losses (e.g., from ReLU+MSE to tanh+sigmoid / BCE). You also answer conceptual items on softmax range, drawing a small computational graph, plotting activation **derivative** curves (ReLU, LeakyReLU, Softplus, GELU), and discussing ReLU limitations vs alternatives. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

**Implementation.**  
From scratch (no autograd): implement forward/backward for **Linear**, **ReLU**, **Sigmoid**, **MSE**, **BCE** in `mlp.py`. Separately, implement a simple **gradient-descent-on-input** routine (DeepDream-style) to optimize an image toward a target class in `sgd.py`. Example tests are provided; you’re expected to match their behavior and consider hidden tests. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## Homework 2 — CNNs & RNNs

**Theory — CNNs.**  
Compute output sizes for 2-D convs (given H×W, kernel HK×WK, padding P, stride S, dilation D, and F filters), then dive into **1-D conv vs cross-correlation**: write the discrete formulas, note the sign flip, and derive dimensions/expressions for ∂f/∂W, ∂f/∂x, and then ∂ℓ/∂W given ∂ℓ/∂f. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

**Theory — RNNs.**  
First, analyze a custom RNN with update  
c[t] = σ(W_c x[t] + W_h h[t−1]),  
h[t] = c[t] ⊙ h[t−1] + (1−c[t]) ⊙ W̄_x x[t].  
You draw the unrolled diagram, compute dimensions, and derive Jacobians ∂h[t]/∂h[t−1], ∂c[t]/∂h[t−1]. Then, for a sequence 1…K, express ∂ℓ/∂W_x and discuss forward/backward similarities and gradient stability. :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

Extend this to **AttentionRNN(2)** with query/key/value projections from x[t], h[t−1], h[t−2], compute attention weights a[t] via softargmax, and form h[t] as a weighted sum; then generalize to **AttentionRNN(k)** and **AttentionRNN(∞)** (discussing parameter growth and weight-tying). You derive ∂h[t]/∂h[t−1] and write a general expression for ∂ℓ/∂h[T] with known upstream grads. :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

**Theory — Debugging.**  
Interpret “kinks” in an LSTM loss curve, explain why early spikes can exceed the initial loss, propose training-procedure fixes only (no model/seed changes), and justify the initial accuracy level from the task definition. :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}

**Implementation.**  
Fill in two notebooks: **CNN** and **RNN** (core TODOs). A small **sequence-classification** notebook asks you to adjust the **training procedure** to remove spikes without altering the model or seed. :contentReference[oaicite:19]{index=19}

---

## Homework 3 — Energy-Based Models (EBMs)

**Theory — EBM intuition.**  
Short-answer prompts build intuition: why EBMs naturally model **one-to-many** input→output relations; how EBMs differ from **probabilistic** predictors; how to turn an energy F_W(x,y) into **p(y|x)** (Gibbs); how **β** controls smoothness/variance; roles of **loss** vs **energy**; pitfalls of training on positives only; three ways to **shape** energy; an example loss that uses **negatives**; and the **inference** objective with/without a latent z (argmin over y vs (y,z)). :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

**Theory — Negative log-likelihood loss.**  
Given n-way classification with energies F_W(x,y): write **Gibbs** p(y|x) with β; derive **NLL** (scaled by 1/β for convenience); derive the **gradient** w.r.t. W and discuss why exact computation can be **intractable** and common workarounds; explain why NLL drives the correct example’s energy → −∞ and others → +∞ (over-sharp surfaces for continuous y). :contentReference[oaicite:22]{index=22} :contentReference[oaicite:23]{index=23}

**Theory — Contrastive losses.**  
Compare **simple**, **hinge**, **log (soft-hinge)**, and **square-square** losses: write each definition and its gradient structure (given ∂F/∂W), then discuss how NLL differs from margin-based forms, why soft-hinge can be advantageous, and when you’d prefer simple vs square-square. :contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25}

**Implementation — OCR demo.**  
Build a small **structured-prediction/OCR** system in the provided notebook by filling in neural components and running the end-to-end pipeline. :contentReference[oaicite:26]{index=26}

---

## Homework 4 — Transformers (Attention, MHA, Self-Attention, ViT)

**Attention.**  
Compute dot-product attention H from Q, K, V and reason about output **dimensions** and the effect of the **scale β**; analyze when attention **preserves** a value vector exactly vs when it **diffuses** values; study sensitivity to small **perturbations** of keys/queries and large **scalings** of a key. :contentReference[oaicite:27]{index=27} :contentReference[oaicite:28]{index=28} :contentReference[oaicite:29]{index=29}

**Multi-Headed Attention.**  
Describe MHA computation and benefits; reflect on analogies to “multi-headed” behavior in convs. :contentReference[oaicite:30]{index=30} :contentReference[oaicite:31]{index=31}

**Self-Attention.**  
Define Q/K/V and outputs for **multi-head self-attention** on an input C; explain **positional encoding** (absolute vs relative) and when to use each; construct scenarios where self-attention behaves like **identity/permutation** or like a **wide-kernel convolution** (with positional encodings); discuss **causal** tweaks for real-time ASR. :contentReference[oaicite:32]{index=32} :contentReference[oaicite:33]{index=33} :contentReference[oaicite:34]{index=34}

**Transformer architecture.**  
Summarize key differences from RNN/LSTM seq2seq, the role of self-attention, MHA, and position-wise feed-forwards, and stability techniques (e.g., normalization/residuals) to mitigate exploding/vanishing gradients. :contentReference[oaicite:35]{index=35}

**Vision Transformer (ViT).**  
Contrast ViT vs CNNs in handling images (patchification, presence/absence of conv layers), explain **positional embeddings** vs original Transformer encodings, describe the **classification head** path, and discuss performance across **data regimes**. :contentReference[oaicite:36]{index=36}

**Implementation.**  
Complete a single Colab notebook to implement the attention/Transformer pieces and a mini ViT pipeline (fill TODOs, ensure it runs). :contentReference[oaicite:37]{index=37}
