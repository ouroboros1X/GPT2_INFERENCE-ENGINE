# GPT-2 CPU Inference Engine

A **from-scratch C++17 inference engine for GPT-2 Small (124M)** that executes the complete Transformer forward pass directly on the CPU.

The project intentionally avoids high-level machine-learning frameworks such as **PyTorch, TensorFlow, and ONNX Runtime**. Core neural-network operations are implemented manually in C++, including matrix multiplication, LayerNorm, multi-head causal self-attention, GELU, softmax, residual connections, and vocabulary projection.

The goal is to make the complete inference pipeline explicit:

> **Text → Tokens → Embeddings → Transformer Blocks → Logits → Next Token → Generated Text**

Rather than hiding inference behind a framework, this project exposes the individual computations that turn a sequence of tokens into the next-token prediction.

---

## Overview

GPT-2 is an autoregressive Transformer language model. During inference, it receives a sequence of tokens and estimates the probability of every possible next token.

For example:

```text
Input:
The quick brown

Model:
GPT-2

Output distribution:
fox       → highest score
dog       → lower score
cat       → lower score
...
```

The engine selects the next token using **greedy decoding**, appends it to the sequence, and runs inference again:

```text
The quick brown
        ↓
     predict
        ↓
       fox
        ↓
The quick brown fox
        ↓
     predict
        ↓
      jumps
        ↓
The quick brown fox jumps
        ↓
       ...
```

This autoregressive process continues until the requested number of tokens has been generated.

---

# What This Project Actually Does

It is useful to distinguish between **training** and **inference**.

During training, a neural network learns its parameters:

```text
Training Data
     ↓
Tokenizer
     ↓
Token IDs
     ↓
Transformer
     ↓
Predictions
     ↓
Loss
     ↓
Backpropagation
     ↓
Updated Weights
```

This project does **not** train GPT-2.

Instead, it takes an already-trained GPT-2 checkpoint and performs the forward computation:

```text
Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embedding lookup
  ↓
12 Transformer blocks
  ↓
Final LayerNorm
  ↓
Vocabulary projection
  ↓
Logits for 50,257 tokens
  ↓
Greedy argmax
  ↓
Next token
  ↓
Repeat
```

In other words, the project functions as a small, specialized **neural-network inference runtime** for GPT-2 Small.

---

# Why Build an Inference Engine From Scratch?

Modern ML frameworks make model inference extremely convenient, but they also hide many of the operations occurring inside a Transformer.

For example, a framework may reduce an entire attention calculation to a few high-level API calls.

This project intentionally removes that abstraction.

Instead of:

```python
model(input)
```

the implementation explicitly performs the underlying operations:

```text
Embedding
    ↓
Linear projection
    ↓
Q / K / V
    ↓
Attention scores
    ↓
Scaling
    ↓
Causal masking
    ↓
Softmax
    ↓
Weighted value aggregation
    ↓
Output projection
    ↓
Residual connection
    ↓
LayerNorm
    ↓
MLP
    ↓
GELU
    ↓
Residual connection
```

This makes the implementation useful for understanding:

* Transformer architecture
* GPT-2 internals
* tensor operations
* autoregressive generation
* attention mechanisms
* model weights
* CPU inference
* numerical computation
* memory layout
* performance bottlenecks

---

# Architecture

GPT-2 Small uses the following configuration:

| Parameter              |                 Value |
| ---------------------- | --------------------: |
| Parameters             |                 ~124M |
| Vocabulary             |                50,257 |
| Hidden size            |                   768 |
| Transformer layers     |                    12 |
| Attention heads        |                    12 |
| Head dimension         |                    64 |
| MLP hidden size        |                 3,072 |
| Maximum context length |                 1,024 |
| Activation             |                  GELU |
| Attention              | Causal self-attention |

The model can be viewed as:

```text
                    Input Text
                        │
                        ▼
                  GPT-2 Tokenizer
                        │
                        ▼
                    Token IDs
                        │
                        ▼
              Token Embedding (WTE)
                        │
                        +
              Position Embedding (WPE)
                        │
                        ▼
             ┌─────────────────────┐
             │ Transformer Block 0  │
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Transformer Block 1  │
             └─────────────────────┘
                        │
                       ...
                        │
                        ▼
             ┌─────────────────────┐
             │ Transformer Block 11 │
             └─────────────────────┘
                        │
                        ▼
                 Final LayerNorm
                        │
                        ▼
               Vocabulary Projection
                        │
                        ▼
                 50,257 Logits
                        │
                        ▼
                    Argmax
                        │
                        ▼
                  Next Token
```

---

# End-to-End Inference Pipeline

## 1. Text Input

The process starts with ordinary text:

```text
The quick brown fox
```

The model cannot directly operate on strings.

It first needs a sequence of integer token IDs.

---

## 2. Tokenization

GPT-2 uses a byte-level BPE tokenizer.

For example, a sentence may be converted into something conceptually similar to:

```text
"The quick brown fox"

        ↓

[464, 2068, 7586, 21831]
```

The exact IDs depend on GPT-2's tokenizer vocabulary and merge rules.

This implementation deliberately does not reimplement the full BPE algorithm in C++.

Instead, it invokes the Hugging Face `tokenizers` library through Python so tokenization and decoding remain compatible with the reference GPT-2 tokenizer.

The C++ engine therefore receives token IDs such as:

```text
[464, 2068, 7586, ...]
```

These IDs become the input to the neural network.

---

# 3. Token Embeddings

A token ID is simply an integer.

The Transformer needs a vector representation.

GPT-2 therefore maintains a token embedding matrix:

```text
WTE ∈ R^(50257 × 768)
```

Each of the 50,257 vocabulary entries has a 768-dimensional vector.

For a token with ID `t`:

```text
embedding = WTE[t]
```

So if:

```text
token_id = 464
```

the engine retrieves row `464` from the embedding matrix.

The result is:

```text
768 floating-point values
```

That vector is the numerical representation of the token.

---

# 4. Positional Embeddings

Transformers do not inherently know where a token appears in a sequence.

GPT-2 therefore also contains a positional embedding matrix:

```text
WPE ∈ R^(1024 × 768)
```

For token position `p`:

```text
position_embedding = WPE[p]
```

The token and position representations are added:

```text
hidden[p] = WTE[token_id] + WPE[p]
```

This produces the initial hidden state for every token in the sequence.

If the prompt contains `N` tokens, the initial tensor is approximately:

```text
N × 768
```

---

# 5. Transformer Blocks

GPT-2 Small contains **12 Transformer blocks**.

Each block processes the entire sequence and allows every token to incorporate information from previous positions.

Each block contains two major sublayers:

```text
1. Causal self-attention
2. Feed-forward MLP
```

with residual connections around them.

Conceptually:

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Self-Attention
  │
  ▼
Residual Add
  │
  ▼
LayerNorm
  │
  ▼
MLP
  │
  ▼
Residual Add
  │
  ▼
Output
```

GPT-2 uses a **pre-normalization** structure, meaning LayerNorm is applied before each sublayer.

---

# 6. Layer Normalization

LayerNorm normalizes the hidden representation of each token.

For a vector:

```text
x = [x₁, x₂, ..., x₇₆₈]
```

the implementation computes the mean:

```text
μ = (1 / 768) Σ xᵢ
```

and variance:

```text
σ² = (1 / 768) Σ (xᵢ - μ)²
```

The normalized representation is:

```text
x̂ᵢ = (xᵢ - μ) / sqrt(σ² + ε)
```

GPT-2 then applies learned scale and bias parameters:

```text
outputᵢ = γᵢ x̂ᵢ + βᵢ
```

where:

* `γ` is the learned scale
* `β` is the learned bias
* `ε` prevents division by zero

This operation is performed independently for each token.

---

# 7. Self-Attention

Self-attention is the central mechanism that allows GPT-2 to determine which earlier tokens are relevant to the current token.

The normalized hidden states are projected into:

```text
Q = Query
K = Key
V = Value
```

GPT-2 performs this using a single linear layer whose output contains all three projections.

For hidden size 768:

```text
QKV = XW + b
```

where the output dimension is:

```text
768 × 3 = 2304
```

The resulting vector is split into:

```text
Q ∈ R^768
K ∈ R^768
V ∈ R^768
```

and then divided into 12 attention heads.

---

# 8. Multi-Head Attention

GPT-2 uses:

```text
12 attention heads
```

with:

```text
768 / 12 = 64
```

dimensions per head.

Therefore:

```text
Head 0 → 64 dimensions
Head 1 → 64 dimensions
...
Head 11 → 64 dimensions
```

Each attention head independently computes relationships between tokens.

This allows different heads to learn different types of relationships.

---

# 9. Attention Score Calculation

For a particular head, attention compares each query against every available key.

The raw attention score between token positions `i` and `j` is:

```text
score(i,j) = Qᵢ · Kⱼ
```

The dot product is then scaled by the square root of the head dimension:

```text
score(i,j) =
    (Qᵢ · Kⱼ) / sqrt(64)
```

This scaling keeps the values in a range that makes the softmax numerically more stable.

---

# 10. Causal Attention

GPT-2 is an **autoregressive language model**.

A token cannot use information from future tokens.

For example:

```text
The quick brown fox
```

When processing:

```text
brown
```

the model can attend to:

```text
The
quick
brown
```

but not:

```text
fox
```

The implementation therefore applies a causal mask:

```text
          Key position

          0   1   2   3
Query 0   ✓   ✗   ✗   ✗
Query 1   ✓   ✓   ✗   ✗
Query 2   ✓   ✓   ✓   ✗
Query 3   ✓   ✓   ✓   ✓
```

In implementation terms, attention is only computed where:

```cpp
j <= t
```

Future positions are excluded from the attention calculation.

This is what makes the model suitable for next-token prediction.

---

# 11. Softmax

The masked attention scores are converted into normalized attention weights using softmax:

```text
softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```

The resulting values form a probability-like distribution across the tokens that the current position can attend to.

For example:

```text
Token        Attention Weight

The             0.10
quick           0.20
brown           0.70
```

The weights sum approximately to:

```text
1.0
```

---

# 12. Weighted Value Aggregation

The attention weights are used to combine the value vectors:

```text
Attention(Q,K,V) = softmax(QKᵀ / sqrt(dₖ)) V
```

Conceptually:

```text
Value(The)   × 0.10
Value(quick) × 0.20
Value(brown) × 0.70
               │
               ▼
        weighted sum
               │
               ▼
        attention output
```

Every attention head produces its own output vector.

The 12 heads are then concatenated:

```text
64 × 12 = 768
```

giving the hidden dimension back.

---

# 13. Attention Output Projection

The concatenated attention result is passed through another linear projection:

```text
output = attention_output Wₚ + bₚ
```

This allows information from all attention heads to be mixed back into the model's 768-dimensional hidden representation.

---

# 14. Residual Connection

GPT-2 uses residual connections around the attention layer.

The attention output is added back to the original hidden state:

```text
x = x + Attention(x)
```

Residual connections help information flow through deep Transformer networks.

The same structure is used around the MLP.

---

# 15. Feed-Forward MLP

After self-attention, each token independently passes through a feed-forward neural network.

GPT-2 expands the hidden dimension:

```text
768 → 3072
```

using a linear layer:

```text
h = xW_fc + b_fc
```

Then GELU is applied.

---

# 16. GELU Activation

GPT-2 uses the Gaussian Error Linear Unit (GELU).

The implementation uses the commonly used tanh approximation:

```text
GELU(x) ≈
0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

This introduces non-linearity into the network.

The representation is then projected back:

```text
3072 → 768
```

using another linear transformation.

So the MLP effectively performs:

```text
768
 ↓
3072
 ↓ GELU
3072
 ↓
768
```

followed by another residual connection:

```text
x = x + MLP(x)
```

---

# 17. One Complete Transformer Block

Putting everything together:

```text
                 Input
                   │
                   ▼
              LayerNorm
                   │
                   ▼
           QKV Projection
                   │
                   ▼
        ┌────────────────────┐
        │ 12 Attention Heads │
        └────────────────────┘
                   │
                   ▼
            Causal Mask
                   │
                   ▼
                Softmax
                   │
                   ▼
           Weighted Values
                   │
                   ▼
          Output Projection
                   │
                   ▼
             Residual Add
                   │
                   ▼
              LayerNorm
                   │
                   ▼
            Linear 768→3072
                   │
                   ▼
                 GELU
                   │
                   ▼
            Linear 3072→768
                   │
                   ▼
             Residual Add
                   │
                   ▼
                 Output
```

GPT-2 executes this block **12 times sequentially**.

---

# 18. Final Layer Normalization

After the twelfth Transformer block, the final hidden representation passes through GPT-2's final LayerNorm:

```text
hidden → final LayerNorm
```

The result is the representation used for next-token prediction.

---

# 19. Vocabulary Projection

The model now needs to convert the final hidden state into scores for every vocabulary token.

GPT-2 has:

```text
50,257 vocabulary tokens
```

The final hidden vector has:

```text
768 dimensions
```

Therefore the output must contain:

```text
50,257 scores
```

The projection is approximately:

```text
logits = hidden × WTEᵀ
```

This implementation uses the **token embedding matrix itself** for this projection, meaning the input token embeddings and output vocabulary projection are tied.

The output is therefore:

```text
logits[0]
logits[1]
logits[2]
...
logits[50256]
```

Each value represents the model's unnormalized score for one vocabulary token.

---

# 20. Logits Are Not Yet Probabilities

The values produced by the vocabulary projection are called **logits**.

For example:

```text
Token        Logit

the            4.21
fox            8.94
cat             7.12
dog             6.31
...
```

A larger logit means the model assigns a higher relative preference to that token.

A probability distribution could be obtained using softmax:

```text
P(tokenᵢ) = exp(logitᵢ) / Σ exp(logitⱼ)
```

However, this implementation does not currently perform sampling.

---

# 21. Greedy Decoding

The engine uses **greedy decoding**.

It simply selects the token with the largest logit:

```text
next_token = argmax(logits)
```

For example:

```text
fox → 8.94
cat → 7.12
dog → 6.31
```

Therefore:

```text
next token = fox
```

The selected token is appended to the current sequence.

---

# 22. Autoregressive Generation

Generation then repeats the entire process.

Suppose the user enters:

```text
The quick brown
```

The engine performs:

```text
The quick brown
        ↓
      model
        ↓
       fox
```

The new sequence becomes:

```text
The quick brown fox
```

The engine runs inference again:

```text
The quick brown fox
        ↓
      model
        ↓
      jumps
```

Then:

```text
The quick brown fox jumps
        ↓
      model
        ↓
       over
```

This continues until the requested number of new tokens is produced.

---

# 23. Important Implementation Detail: No KV Cache

The current implementation intentionally performs a full forward pass for every generated token.

Suppose the original prompt contains:

```text
N tokens
```

and the engine generates:

```text
1 token
```

The model processes all `N` positions.

After another token is generated:

```text
N + 1 tokens
```

the engine processes the entire sequence again.

After another token:

```text
N + 2 tokens
```

and so on.

Conceptually:

```text
Step 1 → process N tokens
Step 2 → process N+1 tokens
Step 3 → process N+2 tokens
Step 4 → process N+3 tokens
...
```

Production-grade Transformer inference commonly uses a **KV cache** to avoid recomputing previous keys and values.

This project does not currently implement that optimization.

As a result, generation becomes increasingly expensive as the context grows.

---

# 24. Weight Loading

The model architecture itself does not contain learned knowledge.

The knowledge is encoded in the weights.

This engine loads GPT-2's trained parameters from plain-text tensor files.

For example:

```text
transformer.wte.weight.txt
transformer.wpe.weight.txt
transformer.h.0.ln_1.weight.txt
transformer.h.0.attn.c_attn.weight.txt
...
```

Each file contains whitespace-separated floating-point values.

For example:

```text
0.0123
-0.0841
0.0021
...
```

The engine reads these values into C++ memory.

---

# 25. Matrix Multiplication

A large part of Transformer inference consists of matrix multiplication.

For a linear layer:

```text
y = xW + b
```

the implementation explicitly computes each output value.

Conceptually:

```cpp
for (int i = 0; i < inputSize; ++i) {
    for (int j = 0; j < outputSize; ++j) {
        output[j] += input[i] * weight[i * outputSize + j];
    }
}
```

This is intentionally straightforward.

It makes the mathematical operation easy to inspect, but it is also one of the primary performance bottlenecks of the implementation.

Production inference engines normally use optimized kernels, SIMD instructions, BLAS libraries, GPU kernels, multithreading, or specialized hardware.

---

# 26. Weight Memory Layout

The `.txt` tensors use a flat row-major representation.

For a weight matrix with:

```text
inputSize × outputSize
```

the engine indexes:

```cpp
weight[i * outputSize + j]
```

which corresponds to:

```text
W[i][j]
```

This layout is important because incorrect tensor ordering would produce completely incorrect model outputs even when the mathematical operations themselves are implemented correctly.

---

# 27. Tokenization Architecture

Tokenization is kept separate from the C++ inference implementation.

The engine invokes Python using:

```text
python3
```

and communicates with the tokenizer through temporary files.

The Python process uses Hugging Face's:

```text
tokenizers
```

package and the GPT-2 tokenizer configuration located at:

```text
weights/tokenizer/tokenizer.json
```

The environment variable:

```bash
GPT2_PYTHON
```

can be used to select another Python interpreter.

Example:

```bash
GPT2_PYTHON=python3.11 ./gpt2_infer
```

This separation keeps the neural-network implementation in C++ while using the reference tokenizer for text ↔ token conversion.

---

# 28. Interactive REPL

The executable provides a small interactive interface:

```text
Prompt: The quick brown fox
Number of new tokens: 10
```

The engine then:

1. Tokenizes the prompt.
2. Displays the prompt token IDs.
3. Displays the corresponding GPT-2 token pieces.
4. Runs the Transformer.
5. Finds the highest-scoring next token.
6. Prints the generated token.
7. Adds the token to the sequence.
8. Repeats until the requested number of tokens is generated.
9. Decodes the final token sequence back into text.

Example:

```text
Prompt: The quick brown

Number of new tokens: 5

Token: The
ID: 464

Token: quick
ID: 2068

Token: brown
ID: 7586

Generated:
fox jumps over the lazy
```

The exact output depends on the weights and tokenizer files being used.

---

# 29. Directory Structure

```text
project-root/
│
├── src/
│   └── main.cpp
│
├── include/
│   └── json.hpp
│
├── weights/
│   ├── transformer.wte.weight.txt
│   ├── transformer.wpe.weight.txt
│   ├── transformer.ln_f.weight.txt
│   ├── transformer.ln_f.bias.txt
│   │
│   ├── transformer.h.0/
│   │   └── ...
│   │
│   ├── transformer.h.1/
│   │   └── ...
│   │
│   ├── ...
│   │
│   ├── transformer.h.11/
│   │   └── ...
│   │
│   └── tokenizer/
│       └── tokenizer.json
│
└── build/
    └── gpt2_infer
```

The exact weight filenames follow the flattened naming convention expected by the implementation.

---

# 30. Requirements

## Compiler

A C++17-compatible compiler.

For example:

```bash
g++
```

## C++ dependency

[nlohmann/json](https://github.com/nlohmann/json)

The single-header version is expected at:

```text
include/json.hpp
```

## Python

Python 3 with Hugging Face `tokenizers`:

```bash
pip install tokenizers
```

## Model Weights

GPT-2 Small / 124M weights exported into the plain-text tensor layout expected by the engine.

---

# 31. Building

From the project root:

```bash
g++ -std=c++17 -O3 -I include -o gpt2_infer src/main.cpp
```

The `-O3` optimization level is recommended because inference performs a large number of floating-point operations.

---

# 32. Running

The executable expects the weights directory to resolve as:

```text
../weights
```

from the directory containing the executable.

For example:

```text
project-root/
├── weights/
└── build/
    └── gpt2_infer
```

Run:

```bash
cd build
./gpt2_infer
```

To use a specific Python interpreter:

```bash
GPT2_PYTHON=python3.11 ./gpt2_infer
```

---

# 33. Example Inference Flow

Given:

```text
Prompt:
The quick brown fox
```

the complete process is:

```text
"The quick brown fox"
            │
            ▼
       GPT-2 Tokenizer
            │
            ▼
      Token IDs
            │
            ▼
     Token Embeddings
            +
    Position Embeddings
            │
            ▼
     Transformer Block 0
            │
            ▼
     Transformer Block 1
            │
            ▼
            ...
            │
            ▼
     Transformer Block 11
            │
            ▼
       Final LayerNorm
            │
            ▼
      Vocabulary Projection
            │
            ▼
       50,257 logits
            │
            ▼
        argmax()
            │
            ▼
       Next Token
            │
            ▼
     Append to sequence
            │
            └───────────► Repeat
```

This is the core inference loop implemented by the project.

---

# 34. What Is Actually Being Computed?

At a high level, the model performs a function:

```text
f(tokens) → logits
```

For a sequence:

```text
[t₁, t₂, ..., tₙ]
```

GPT-2 produces a hidden representation for every position:

```text
H = [h₁, h₂, ..., hₙ]
```

The final hidden state:

```text
hₙ
```

contains information aggregated from the preceding context.

The vocabulary projection then produces:

```text
L = hₙ WTEᵀ
```

where:

```text
L ∈ R^50257
```

The engine chooses:

```text
argmax(L)
```

as the next token.

This is the fundamental operation repeated during text generation.

---

# 35. What the Model "Knows"

The engine does not contain explicit rules such as:

```text
if sentence contains "hello":
    respond with "hi"
```

Instead, behavior emerges from numerical parameters learned during training.

The weights encode statistical relationships between tokens.

During inference, the model transforms the current context through many layers of matrix operations until the final representation contains information useful for predicting the next token.

The inference engine's job is simply to execute those transformations accurately.

---

# 36. Performance Characteristics

This implementation prioritizes **clarity and transparency over performance**.

The current engine is:

```text
CPU
Single-threaded
Unbatched
Naive matrix multiplication
No SIMD optimization
No BLAS
No GPU acceleration
No KV cache
```

The result is a deliberately simple implementation that exposes the actual Transformer computations.

A production inference runtime would normally introduce substantially more optimization.

---

# 37. Known Limitations

### Greedy Decoding Only

The current implementation selects:

```text
argmax(logits)
```

and therefore does not currently support:

* temperature
* top-k sampling
* top-p / nucleus sampling
* repetition penalties
* beam search

---

### No KV Cache

Every generation step recomputes attention for the entire current sequence.

This increases computational cost as the context grows.

---

### CPU Only

The engine currently runs on the CPU.

There is no:

* CUDA backend
* GPU kernel implementation
* Metal backend
* accelerator support

---

### Single-Threaded

Inference is performed by a single execution thread.

There is currently no explicit parallelism across:

* tokens
* attention heads
* matrix operations
* transformer layers

---

### No SIMD / BLAS Optimization

Matrix operations use straightforward C++ loops rather than optimized numerical libraries.

This makes the implementation easier to understand but significantly less efficient than optimized inference runtimes.

---

### External Tokenizer Process

Tokenization is performed through a Python subprocess rather than an in-process C++ tokenizer.

The current implementation communicates through temporary files.

This introduces process and I/O overhead.

---

### GPT-2 Small Only

The current implementation is specialized for GPT-2 Small / 124M:

```text
12 layers
768 hidden dimensions
12 heads
3072 MLP dimensions
1024-token context
50257-token vocabulary
```

Supporting GPT-2 Medium, Large, or XL would require adapting the architecture and loading logic to their respective tensor shapes.

---

### Plain-Text Weight Files

Weights are stored as human-readable text files.

This makes the files easy to inspect but significantly increases storage size and loading overhead compared with binary formats.

---

# 38. Potential Future Improvements

Several extensions could turn the project into a significantly more capable inference runtime:

```text
KV cache
    ↓
SIMD-optimized kernels
    ↓
Multithreaded execution
    ↓
BLAS integration
    ↓
Binary weight format
    ↓
Memory-mapped weights
    ↓
In-process C++ tokenizer
    ↓
Temperature sampling
    ↓
Top-k / Top-p sampling
    ↓
Repetition penalty
    ↓
Batch inference
    ↓
Support for additional GPT-2 model sizes
    ↓
GPU backend
```

A particularly important next optimization is **KV caching**, since it removes repeated computation of keys and values for tokens that have already been processed.

---

# 39. Learning Objectives

This project provides a practical implementation of several concepts that are often abstracted away by ML frameworks:

* Transformer architecture
* GPT-2 internals
* tokenization
* embeddings
* matrix multiplication
* LayerNorm
* residual networks
* multi-head attention
* causal masking
* softmax
* GELU
* autoregressive generation
* weight loading
* CPU inference
* tensor memory layout

The implementation is intentionally close to the underlying mathematical structure of GPT-2.

---

# 40. Project Philosophy

The project follows a simple principle:

> **Do not hide the computation. Implement it.**

Instead of treating GPT-2 as a black box, this engine makes the inference process explicit from input text to generated token.

The project is therefore less about building another production-ready chatbot and more about understanding what a Transformer inference engine actually has to execute.

---

# 41. Tech Stack

```text
Language       C++17
Runtime        Native CPU
Model          GPT-2 Small (124M)
Tokenizer      Hugging Face tokenizers
Configuration  JSON
Compiler       GCC / Clang
Optimization   -O3
```

---

# 42. License

Add the license for this repository here.

For example:

```text
MIT License
```

if the repository is intended to be released under MIT.

---

# 43. Summary

This project implements the GPT-2 inference pipeline directly in C++:

```text
                 USER PROMPT
                      │
                      ▼
                  TOKENIZER
                      │
                      ▼
                  TOKEN IDs
                      │
                      ▼
             TOKEN + POSITION
                EMBEDDINGS
                      │
                      ▼
              ┌──────────────┐
              │ Transformer  │
              │   Block ×12  │
              └──────────────┘
                      │
                      ▼
                FINAL NORM
                      │
                      ▼
               LOGIT PROJECTION
                      │
                      ▼
                50,257 LOGITS
                      │
                      ▼
                  ARGMAX
                      │
                      ▼
                NEXT TOKEN
                      │
                      ▼
             APPEND TO CONTEXT
                      │
                      └───────────┐
                                  │
                                  ▼
                               REPEAT
```

At its core, the engine performs one job:

```text
Given a sequence of tokens,
compute the numerical representation of that sequence,
produce a score for every possible next token,
select the next token,
and repeat.
```

Everything between the input string and that next-token decision is implemented explicitly in C++.
