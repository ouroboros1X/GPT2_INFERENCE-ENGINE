# GPT-2 CPU Inference Engine (C++)

A from-scratch, dependency-light C++ implementation of GPT-2 (124M) inference. No PyTorch, no TensorFlow, no ONNX runtime — every op (layer norm, linear projection, multi-head causal self-attention, GELU, softmax) is hand-written and runs on the CPU.

## Features

- **Pure C++ forward pass** — embeddings, 12 transformer blocks, final layer norm, and logit projection (tied with the token embedding matrix) are all implemented manually, with no ML framework in the loop.
- **Text-file weight loading** — weights are read from whitespace-separated `.txt` tensor dumps rather than a binary checkpoint format.
- **Greedy decoding** — generation picks the argmax token at each step (no sampling, temperature, or top-k/top-p yet).
- **Interactive REPL** — prompts you for input text and a token count, then prints both the token-by-token breakdown and the final decoded text.
- **Exact HuggingFace tokenization** — rather than reimplementing BPE, the engine shells out to Python's `tokenizers` library so encode/decode results match the reference GPT-2 tokenizer exactly.

## Model configuration

Hardcoded for GPT-2 small (124M):

| Parameter | Value |
|---|---|
| Vocab size | 50,257 |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Head dim | 64 |
| MLP inner size | 3,072 |
| Max context | 1,024 |

## Requirements

- A C++17-capable compiler (uses `<array>`, `std::max_element`, etc.)
- [nlohmann/json](https://github.com/nlohmann/json) single-header library, expected at `include/json.hpp`
- Python 3 with the [`tokenizers`](https://pypi.org/project/tokenizers/) package installed (`pip install tokenizers`), used only for encoding/decoding text
- GPT-2 124M weights, exported as plain-text tensor files (see layout below)

## Expected directory layout

The binary looks for weights one level up from where it's run (`../weights`), so a typical layout is:

```
project-root/
├── src/
│   └── main.cpp
├── include/
│   └── json.hpp
├── weights/
│   ├── transformer.wte.weight.txt
│   ├── transformer.wpe.weight.txt
│   ├── transformer.ln_f.weight.txt
│   ├── transformer.ln_f.bias.txt
│   ├── transformer.h.0.ln_1.weight.txt
│   ├── transformer.h.0.ln_1.bias.txt
│   ├── transformer.h.0.attn.c_attn.weight.txt
│   ├── transformer.h.0.attn.c_attn.bias.txt
│   ├── transformer.h.0.attn.c_proj.weight.txt
│   ├── transformer.h.0.attn.c_proj.bias.txt
│   ├── transformer.h.0.ln_2.weight.txt
│   ├── transformer.h.0.ln_2.bias.txt
│   ├── transformer.h.0.mlp.c_fc.weight.txt
│   ├── transformer.h.0.mlp.c_fc.bias.txt
│   ├── transformer.h.0.mlp.c_proj.weight.txt
│   ├── transformer.h.0.mlp.c_proj.bias.txt
│   ├── ... (repeated for layers 1 through 11)
│   └── tokenizer/
│       └── tokenizer.json
└── build/           # compiled binary runs from here
```

Each `.txt` file holds a flat list of whitespace-separated floats for that tensor (weight matrices are stored row-major, output-dim-fastest, since `linear()` indexes them as `input[i] * weight[i * outputSize + j]`).

## Building

```bash
g++ -std=c++17 -O3 -o gpt2_infer src/main.cpp
```

Adjust the source path and add `-I include` if `json.hpp` isn't already on your include path.

## Running

Run the binary from a directory where `../weights` resolves correctly (e.g. from a `build/` folder next to `weights/`):

```bash
cd build
./gpt2_infer
```

On startup it loads all weights and the tokenizer vocabulary, then drops into a loop:

```
Prompt: The quick brown fox
Number of new tokens: 10
```

It prints each prompt token (ID, raw GPT-2 piece, and decoded text), generates tokens one at a time via greedy argmax over the logits, prints each as it's produced, and finally prints the fully decoded text. Type `exit` to quit.

By default the tokenizer subprocess is invoked as `python3`; set the `GPT2_PYTHON` environment variable to point at a different interpreter if needed:

```bash
GPT2_PYTHON=python3.11 ./gpt2_infer
```

## How inference works

1. **Embedding** — each input token ID looks up a row in `wte` and adds the corresponding positional row from `wpe`.
2. **Transformer blocks** (×12) — each block does: pre-attention layer norm → causal multi-head self-attention (scaled dot-product, per-head softmax, causal masking via `j <= t`) → residual add → pre-MLP layer norm → 2-layer MLP with GELU (tanh approximation) → residual add.
3. **Final layer norm** on the last token's hidden state.
4. **Logits** — the final hidden state is projected back to vocab space using the (tied) `wte` embedding matrix.
5. **Sampling** — the next token is chosen greedily via argmax; this repeats for the requested number of new tokens, feeding each generated token back into the context.

## Known limitations

- **Greedy-only decoding** — no temperature, top-k, top-p, or repetition penalty.
- **No KV cache** — each generation step reruns the full forward pass over the entire sequence so far, so generation gets slower as the context grows.
- **Single-threaded, unbatched** — one sequence at a time, no SIMD/BLAS acceleration; `linear()` is a plain triple-nested loop.
- **Tokenization depends on an external Python process** — encode/decode round-trip through `/tmp` files and a `python3 -c ...` subprocess call rather than an in-process tokenizer.
- **Hardcoded to GPT-2 124M** — model dimensions are compile-time constants, so other GPT-2 sizes (medium/large/XL) would need the constants changed and would need their weights exported in the same layout.

