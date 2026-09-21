GPT-2 CPU Inference Engine

A from-scratch C++17 implementation of GPT-2 Small (124M) inference that executes the Transformer forward pass directly on the CPU.

This project deliberately avoids high-level machine-learning runtimes such as PyTorch, TensorFlow, and ONNX Runtime for the model computation. Core operations are implemented manually in C++, including:

token and positional embeddings

matrix multiplication / linear layers

LayerNorm

multi-head causal self-attention

softmax

GELU

residual connections

vocabulary projection

greedy decoding

The result is a small, transparent inference runtime where the reader can follow the path from a text prompt all the way to the next predicted token.

Text → Token IDs → Embeddings → Transformer Blocks → Logits → Argmax → Next Token → Repeat

Table of Contents

Project Goals

What Is an Inference Engine?

Model Architecture

End-to-End Data Flow

Repository Layout

Codebase at a Glance

Code Walkthrough

1. Model Constants and Tensor Representation

2. Loading Tensor Weights

3. Representing a Transformer Block

4. Representing and Loading GPT-2

5. LayerNorm

6. Linear Layers

7. GELU

8. Softmax

9. Multi-Head Causal Self-Attention

10. The MLP

11. One Transformer Block

12. The Full Forward Pass

13. Vocabulary Projection

14. Greedy Decoding

15. Tokenizer Integration

16. Token Display

17. Interactive Generation Loop

Tensor Shapes

Weight File Mapping

Numerical Details

Generation and Context Management

Build

Run

Requirements

Example

Performance

Limitations

Future Work

Design Decisions

Troubleshooting

License

Project Goals

This repository is primarily an implementation and learning project.

Modern ML libraries can make inference look like:

logits = model(tokens)

That line hides most of the interesting work.

This project expands that hidden computation into explicit C++ code so that the reader can inspect:

how model weights are stored in memory

how token IDs become vectors

how attention generates Q/K/V

how causal attention prevents looking into the future

how the MLP transforms every token

how 12 Transformer blocks are chained together

how the final hidden state becomes 50,257 vocabulary logits

how the engine turns those logits into a generated token

how generation feeds that token back into the next forward pass

The current implementation is intentionally optimized for clarity and directness, not production inference throughput.

What Is an Inference Engine?

A trained language model consists of an architecture plus learned parameters.

Training looks roughly like:

Training Data
     │
     ▼
Tokenizer
     │
     ▼
Token IDs
     │
     ▼
Transformer
     │
     ▼
Predictions
     │
     ▼
Loss
     │
     ▼
Backpropagation
     │
     ▼
Updated Weights

This project does not perform training.

The weights already exist. The engine executes the forward computation using those weights:

Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
Token + Position Embeddings
  │
  ▼
Transformer Block × 12
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
  │
  └──────────────► Append token and repeat

That execution is inference.

In this repository, the C++ program acts as a specialized inference runtime for GPT-2 Small.

Model Architecture

The implementation targets GPT-2 Small / 124M.

Parameter

Value

Vocabulary size

50,257

Hidden size

768

Transformer blocks

12

Attention heads

12

Head dimension

64

MLP hidden size

3,072

Maximum context

1,024 tokens

Activation

GELU

Attention

Causal self-attention

The architecture is:

flowchart TD
    A["Text Prompt"] --> B["GPT-2 Tokenizer"]
    B --> C["Token IDs"]
    C --> D["Token Embedding WTE"]
    D --> E["Add Position Embedding WPE"]
    E --> F["Transformer Block 0"]
    F --> G["Transformer Block 1"]
    G --> H["..."]
    H --> I["Transformer Block 11"]
    I --> J["Final LayerNorm"]
    J --> K["Vocabulary Projection using WTEᵀ"]
    K --> L["50,257 Logits"]
    L --> M["Argmax"]
    M --> N["Next Token"]
    N --> O["Append to Context"]
    O --> F

End-to-End Data Flow

Suppose the user enters:

The quick brown

The engine performs the following pipeline:

"The quick brown"
       │
       ▼
GPT-2 tokenizer
       │
       ▼
[ token_0, token_1, token_2, ... ]
       │
       ▼
Embedding lookup
       │
       +
Position embedding
       │
       ▼
768-dimensional representation
       │
       ▼
Transformer block 0
       │
       ▼
Transformer block 1
       │
       ▼
...
       │
       ▼
Transformer block 11
       │
       ▼
Final LayerNorm
       │
       ▼
768-dimensional final representation
       │
       ▼
Dot product with every vocabulary embedding
       │
       ▼
50,257 logits
       │
       ▼
argmax
       │
       ▼
next token

If the selected token is fox, the sequence becomes:

The quick brown fox

The engine then runs another forward pass using that larger sequence.

Repository Layout

A typical project layout is:

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
│   │
│   ├── ... same structure for layers 1–11
│   │
│   └── tokenizer/
│       └── tokenizer.json
│
└── build/
    └── gpt2_infer

The current source expects the weights directory to be available as:

../weights

relative to the process working directory.

Codebase at a Glance

The entire model is currently implemented in a single C++ translation unit.

The important functions and types are:

Tensor
  │
  └── std::vector<float>

loadTensor()
  │
  └── Reads one weight tensor from disk

Block
  │
  └── Stores one Transformer block's parameters

GPT2
  │
  ├── Stores WTE / WPE
  ├── Stores 12 Blocks
  └── Loads all model weights

layerNorm()
linear()
gelu()
applyGelu()
softmax()
  │
  └── Core numerical primitives

attention()
  │
  └── QKV + causal multi-head attention

mlp()
  │
  └── 768 → 3072 → 768

transformerBlock()
  │
  └── LayerNorm → Attention → Residual
      LayerNorm → MLP → Residual

forward()
  │
  └── Embeddings → 12 Blocks → Final Norm → Logits

argmax()
  │
  └── Select next token

Tokenizer
  │
  ├── Loads vocabulary for display
  ├── Exact encode through Python
  └── Exact decode through Python

main()
  │
  └── REPL + autoregressive generation

Code Walkthrough

1. Model Constants and Tensor Representation

At the beginning of main.cpp, the architecture is expressed directly as compile-time constants:

constexpr int VOCAB_SIZE = 50257;
constexpr int HIDDEN_SIZE = 768;
constexpr int NUM_LAYERS = 12;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = 64;
constexpr int MLP_SIZE = 3072;
constexpr int MAX_CONTEXT = 1024;
constexpr float EPSILON = 1e-5f;

These constants define the GPT-2 Small configuration used by the entire program.

The important relationships are:

768 / 12 = 64

so every attention head operates on 64 dimensions.

The project defines:

using Tensor = std::vector<float>;

This is an intentionally simple tensor representation.

There is no custom tensor class.

A Tensor is just a contiguous array of floats.

For sequence-level data, the implementation uses:

std::vector<std::vector<float>>

so a sequence can be represented conceptually as:

[
    token_0 hidden vector,
    token_1 hidden vector,
    token_2 hidden vector,
    ...
]

2. Loading Tensor Weights

The fundamental weight-loading function is:

Tensor loadTensor(const std::string& path, size_t expected)

Its responsibility is simple:

file on disk
    ↓
read float
    ↓
append to vector
    ↓
validate element count
    ↓
return Tensor

The function opens a text file:

std::ifstream file(path);

and repeatedly extracts floats:

while (file >> value)
    data.push_back(value);

The important safety check is the expected tensor size.

After loading, the code verifies:

if (data.size() != expected)
    throw std::runtime_error(...);

This catches malformed or incompatible weight files early.

For example, if a matrix should contain:

768 × 768 = 589,824

values but the file contains fewer or more values, the program stops instead of silently executing with a corrupted tensor.

Why this matters

The Transformer is extremely sensitive to tensor shape.

A wrong number of values is not a minor formatting issue—it means the model parameters no longer correspond to the computation graph.

3. Representing a Transformer Block

The Block structure stores the parameters required by one GPT-2 Transformer block:

struct Block
{
    Tensor ln1Weight;
    Tensor ln1Bias;

    Tensor cAttnWeight;
    Tensor cAttnBias;

    Tensor cProjWeight;
    Tensor cProjBias;

    Tensor ln2Weight;
    Tensor ln2Bias;

    Tensor cFcWeight;
    Tensor cFcBias;

    Tensor cProjMlpWeight;
    Tensor cProjMlpBias;
};

This directly mirrors GPT-2's block structure.

The names correspond to the components:

Field

Role

ln1Weight / ln1Bias

LayerNorm before attention

cAttnWeight / cAttnBias

Combined QKV projection

cProjWeight / cProjBias

Attention output projection

ln2Weight / ln2Bias

LayerNorm before MLP

cFcWeight / cFcBias

MLP expansion 768 → 3072

cProjMlpWeight / cProjMlpBias

MLP contraction 3072 → 768

Instead of representing these as separate high-level neural-network objects, the implementation stores the raw parameter arrays.

4. Representing and Loading GPT-2

The GPT2 structure stores the global model weights:

struct GPT2
{
    Tensor wte;
    Tensor wpe;

    std::array<Block, NUM_LAYERS> blocks;

    Tensor lnFinalWeight;
    Tensor lnFinalBias;
};

These are:

wte: token embeddings

wpe: positional embeddings

blocks: all 12 Transformer blocks

lnFinalWeight / lnFinalBias: final LayerNorm

The load() method loads every tensor from disk.

The token embedding matrix has:

50,257 × 768

values.

The position embedding matrix has:

1,024 × 768

values.

For every Transformer block, load() constructs the layer prefix:

std::string p =
    dir +
    "/transformer.h." +
    std::to_string(layer) +
    ".";

Then it loads each parameter file with its expected element count.

This gives the model object a complete in-memory copy of GPT-2's learned parameters.

5. LayerNorm

The implementation of LayerNorm is:

std::vector<float> layerNorm(
    const std::vector<float>& x,
    const Tensor& weight,
    const Tensor& bias
)

For one hidden vector, it first computes the mean:

float mean = 0.0f;

for (float v : x)
    mean += v;

mean /= static_cast<float>(x.size());

Then the variance:

float variance = 0.0f;

for (float v : x)
{
    float d = v - mean;
    variance += d * d;
}

variance /= static_cast<float>(x.size());

The inverse standard deviation is:

float invStd =
    1.0f / std::sqrt(variance + EPSILON);

Finally, every element is normalized and transformed using learned scale and bias:

output[i] =
    ((x[i] - mean) * invStd) * weight[i] +
    bias[i];

Mathematically:

μ  = mean(x)

σ² = mean((x - μ)²)

x̂ = (x - μ) / sqrt(σ² + ε)

y  = γx̂ + β

The same layerNorm() primitive is used for:

attention pre-normalization

MLP pre-normalization

final model normalization

6. Linear Layers

Nearly every major learned transformation in GPT-2 can be expressed as:

y = xW + b

The implementation centralizes this operation in:

std::vector<float> linear(
    const std::vector<float>& input,
    const Tensor& weight,
    const Tensor& bias,
    int inputSize,
    int outputSize
)

The output vector is initialized with the requested number of dimensions.

For each output neuron:

for (int j = 0; j < outputSize; ++j)
{
    float sum = bias[j];

    for (int i = 0; i < inputSize; ++i)
    {
        sum +=
            input[i] *
            weight[
                static_cast<size_t>(i) *
                outputSize +
                j
            ];
    }

    output[j] = sum;
}

This is a direct implementation of a fully connected layer.

For one output index j:

output[j] =
    bias[j]
    + input[0] * W[0,j]
    + input[1] * W[1,j]
    + ...
    + input[inputSize-1] * W[inputSize-1,j]

Weight layout

The weight matrix is stored as a flat array.

The code accesses:

weight[i * outputSize + j]

which represents:

W[i][j]

This is important because the file layout and the code's indexing convention must agree exactly.

Where linear() is used

The same primitive handles:

Attention:
768 → 2304
768 → 768

MLP:
768 → 3072
3072 → 768

Instead of implementing four different matrix-multiplication routines, the program has one reusable linear operator.

7. GELU

The MLP uses GELU.

The scalar implementation is:

float gelu(float x)

using the tanh approximation:

GELU(x) ≈
0.5x [1 + tanh( √(2/π) (x + 0.044715x³) )]

The code defines:

constexpr float c = 0.7978845608028654f;

which is approximately:

sqrt(2 / π)

Then:

return 0.5f *
       x *
       (
           1.0f +
           std::tanh(
               c *
               (
                   x +
                   0.044715f *
                   x * x * x
               )
           )
       );

Because the MLP operates on a vector, the project also defines:

std::vector<float> applyGelu(
    const std::vector<float>& x
)

which simply applies gelu() element-by-element.

8. Softmax

Softmax converts a vector of scores into normalized weights.

The implementation:

std::vector<float> softmax(
    const std::vector<float>& x
)

first handles an empty vector.

Then it finds the maximum value:

float maxValue =
    *std::max_element(
        x.begin(),
        x.end()
    );

The exponentials use the stabilized form:

std::exp(x[i] - maxValue);

and then the values are divided by their sum.

Mathematically:

softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)

but numerically the implementation uses:

exp(xᵢ - max(x))

which helps prevent unnecessarily large exponentials.

The result is a vector whose values approximately sum to:

1.0

In this project, softmax is used inside attention.

9. Multi-Head Causal Self-Attention

The attention implementation lives in:

std::vector<std::vector<float>> attention(
    const std::vector<std::vector<float>>& input,
    const Block& block
)

This function receives the sequence after the first LayerNorm and computes the complete attention output for every position.

9.1 Sequence length

The function begins by determining:

const int seqLen =
    static_cast<int>(input.size());

If the prompt contains N tokens, then:

seqLen = N

9.2 Q, K and V storage

The implementation creates three sequence-sized tensors:

q
k
v

each with:

seqLen × 768

dimensions.

So:

Q = [q₀, q₁, ..., qₙ₋₁]
K = [k₀, k₁, ..., kₙ₋₁]
V = [v₀, v₁, ..., vₙ₋₁]

9.3 Combined QKV projection

For each token position:

std::vector<float> qkv =
    linear(
        input[t],
        block.cAttnWeight,
        block.cAttnBias,
        HIDDEN_SIZE,
        HIDDEN_SIZE * 3
    );

The output size is:

768 × 3 = 2304

The 2304 values are then split into:

0      ... 767   → Q
768    ... 1535  → K
1536   ... 2303  → V

The code performs exactly that split:

q[t][i] = qkv[i];

k[t][i] = qkv[HIDDEN_SIZE + i];

v[t][i] = qkv[2 * HIDDEN_SIZE + i];

So the single GPT-2 c_attn projection simultaneously generates Query, Key, and Value.

9.4 Attention heads

GPT-2 uses:

12 heads

with:

64 dimensions per head

The implementation computes:

const int offset =
    head * HEAD_DIM;

so:

head 0 → dimensions 0..63
head 1 → dimensions 64..127
...
head 11 → dimensions 704..767

No explicit tensor transpose is created. Head slices are accessed directly from the flattened hidden vectors.

9.5 Scaled dot-product attention

For a query at position t and a key at position j, the implementation computes:

for (int d = 0; d < HEAD_DIM; ++d)
{
    score +=
        q[t][offset + d] *
        k[j][offset + d];
}

This is:

Qₜ · Kⱼ

The score is then scaled using:

const float scale =
    1.0f / std::sqrt(
        static_cast<float>(HEAD_DIM)
    );

so mathematically:

score(t,j) =
    (Qₜ · Kⱼ) / sqrt(64)

9.6 Causal masking

This implementation does something simple and important.

Instead of computing scores for every j and then explicitly writing -∞ into future positions, it only allocates:

std::vector<float> scores(t + 1);

and only loops over:

for (int j = 0; j <= t; ++j)

That means position t is allowed to attend only to:

0, 1, 2, ..., t

and never to:

t + 1, t + 2, ...

The causal constraint is therefore enforced by the loop bounds themselves.

The attention matrix is conceptually:

        Key
        0  1  2  3
Query
  0     ✓  ✗  ✗  ✗
  1     ✓  ✓  ✗  ✗
  2     ✓  ✓  ✓  ✗
  3     ✓  ✓  ✓  ✓

This is what makes the Transformer autoregressive.

9.7 Softmax over attention scores

After computing the visible scores:

std::vector<float> probabilities =
    softmax(scores);

the engine obtains attention weights for that position.

For a token at position t, the probabilities only cover positions:

0 ... t

9.8 Weighted sum of values

The implementation then computes:

context[t][offset + d] +=
    probabilities[j] *
    v[j][offset + d];

This is:

head_output(t)
    =
Σⱼ attention_weight(t,j) × Vⱼ

Each head produces a 64-dimensional result.

Across 12 heads:

12 × 64 = 768

so the result returns to the model's hidden dimension.

9.9 Attention output projection

After all heads have been written into context, the code applies another linear layer:

output[t] =
    linear(
        context[t],
        block.cProjWeight,
        block.cProjBias,
        HIDDEN_SIZE,
        HIDDEN_SIZE
    );

This mixes information across the 12 heads and produces the final attention sublayer output.

10. The MLP

The feed-forward network is implemented in:

std::vector<float> mlp(
    const std::vector<float>& input,
    const Block& block
)

The MLP performs:

768
 ↓
3072
 ↓ GELU
3072
 ↓
768

The first projection is:

hidden =
    linear(
        input,
        block.cFcWeight,
        block.cFcBias,
        HIDDEN_SIZE,
        MLP_SIZE
    );

giving:

768 → 3072

Then:

hidden = applyGelu(hidden);

Finally:

return linear(
    hidden,
    block.cProjMlpWeight,
    block.cProjMlpBias,
    MLP_SIZE,
    HIDDEN_SIZE
);

gives:

3072 → 768

The important architectural point is that attention mixes information between token positions, while the MLP processes each token's hidden vector independently.

11. One Transformer Block

The transformerBlock() function combines LayerNorm, attention, residual connections, and MLP.

Its flow is:

Input
  │
  ▼
LayerNorm 1
  │
  ▼
Self-Attention
  │
  ▼
Residual Add
  │
  ▼
LayerNorm 2
  │
  ▼
MLP
  │
  ▼
Residual Add
  │
  ▼
Output

The first normalization is:

ln1[t] =
    layerNorm(
        input[t],
        block.ln1Weight,
        block.ln1Bias
    );

Then attention:

std::vector<std::vector<float>> attn =
    attention(
        ln1,
        block
    );

The first residual connection is:

x[t][i] += attn[t][i];

Then a second LayerNorm is applied:

ln2[t] =
    layerNorm(
        x[t],
        block.ln2Weight,
        block.ln2Bias
    );

The MLP output is generated:

std::vector<float> m =
    mlp(
        ln2[t],
        block
    );

and added back through the second residual connection:

x[t][i] += m[i];

The result is returned as the output of that Transformer block.

12. The Full Forward Pass

The main neural-network execution lives in:

std::vector<float> forward(
    const std::vector<int>& tokenIds,
    const GPT2& model
)

This is the most important function in the engine.

It performs the complete GPT-2 forward pass for a token sequence.

12.1 Input validation

The engine rejects an empty token sequence:

if (tokenIds.empty())
    throw std::runtime_error("No input tokens");

It also enforces the GPT-2 context size:

if (tokenIds.size() > MAX_CONTEXT)
    throw std::runtime_error(
        "Input exceeds GPT-2 context length"
    );

12.2 Build initial hidden states

For every token position, the code computes:

hidden[position][i] =
    model.wte[tokenOffset + i] +
    model.wpe[posOffset + i];

This is:

hidden[p] =
    token_embedding[token_id]
    +
    position_embedding[p]

Every token therefore starts with a 768-dimensional representation.

For N tokens:

hidden shape = N × 768

12.3 Run all 12 Transformer blocks

The core loop is:

for (int layer = 0;
     layer < NUM_LAYERS;
     ++layer)
{
    hidden =
        transformerBlock(
            hidden,
            model.blocks[layer]
        );
}

This is the point where the initial embeddings are repeatedly transformed by GPT-2's learned layers.

The shape remains:

N × 768

throughout the Transformer stack.

Only the contents of the vectors change.

13. Vocabulary Projection

After the 12 Transformer blocks, the engine only needs the representation of the last position to predict the next token.

The code extracts:

std::vector<float> finalHidden =
    layerNorm(
        hidden.back(),
        model.lnFinalWeight,
        model.lnFinalBias
    );

hidden.back() means:

the hidden vector at the last token position

It then creates:

std::vector<float> logits(
    VOCAB_SIZE
);

so there is one output score for every GPT-2 vocabulary entry.

13.1 Tied output projection

For every possible vocabulary token:

for (int token = 0;
     token < VOCAB_SIZE;
     ++token)

the engine retrieves that token's embedding row:

const size_t offset =
    static_cast<size_t>(token) *
    HIDDEN_SIZE;

and computes its dot product with the final hidden state:

for (int i = 0; i < HIDDEN_SIZE; ++i)
{
    sum +=
        finalHidden[i] *
        model.wte[offset + i];
}

So the output operation is:

logit(token)
    =
finalHidden · WTE[token]

or in matrix notation:

logits = finalHidden × WTEᵀ

This is weight tying: the same token embedding matrix used at the input is also used as the output vocabulary projection.

No separate output projection tensor is loaded here.

14. Greedy Decoding

The forward() function returns raw logits.

The next-token decision is handled separately by:

int argmax(
    const std::vector<float>& logits
)

The implementation uses:

std::max_element(
    logits.begin(),
    logits.end()
)

and converts the iterator position into the vocabulary index.

Conceptually:

logits:

token A → 3.12
token B → 7.91
token C → 2.84
token D → 6.44

argmax → token B

This is greedy decoding.

There is no:

temperature

top-k

top-p

sampling

repetition penalty

in the current implementation.

15. Tokenizer Integration

The project uses two tokenizer paths.

C++ vocabulary loading

The Tokenizer structure contains:

std::vector<std::string> vocabulary;

Its load() method reads:

weights/tokenizer/tokenizer.json

using nlohmann::json.

The method navigates into:

model → vocab

and fills:

vocabulary[token_id] = token_piece

This gives the program a fast local mapping for printing token IDs and GPT-2 token pieces.

Exact encoding

The actual prompt encoding is handled by:

std::vector<int> encodeExact(
    const std::string& text
)

The C++ program writes the input prompt to:

/tmp/gpt2_encode_input.txt

then invokes Python:

python3 -c "from tokenizers import Tokenizer; ..."

using the GPT-2 tokenizer JSON.

The environment variable:

GPT2_PYTHON

can replace the default Python executable.

For example:

GPT2_PYTHON=python3.11 ./gpt2_infer

The Python process prints the token IDs as whitespace-separated integers, which the C++ program reads back into:

std::vector<int>

The temporary files are then removed.

Exact decoding

The reverse path is:

std::string decodeExact(
    const std::vector<int>& ids
)

The token IDs are serialized as JSON and passed to the same Hugging Face tokenizer through Python.

The decoded text is read from the temporary output file and returned to the C++ program.

Therefore the model computation remains native C++, while tokenization and decoding use the reference tokenizer implementation.

16. Token Display

The helper:

void printToken(
    int index,
    int id,
    const Tokenizer& tokenizer
)

prints three useful pieces of information:

index
token ID
GPT-2 token piece

plus a friendlier text representation.

For example, the REPL can show:

[0] ID: 464    GPT-2: "The"    Text: "The"

The textToken() helper contains a small amount of display-only handling for GPT-2's token representation, including:

byte-level space rendering

the GPT-2 newline token

This is separate from the actual Python tokenizer encode/decode path.

17. Interactive Generation Loop

main() wires everything together.

At startup:

GPT2 model;
model.load("../weights");

loads all model parameters.

Then:

Tokenizer tokenizer;
tokenizer.load(
    "../weights/tokenizer/tokenizer.json"
);

loads the tokenizer vocabulary.

The program then enters a loop:

while (true)

and asks:

Prompt:
Number of new tokens:

Prompt encoding

The prompt is encoded:

std::vector<int> inputIds =
    encodeExact(prompt);

Then the token IDs are printed using printToken().

Context reservation

If the prompt is already close to the maximum context length:

if (inputIds.size() >= MAX_CONTEXT)
{
    inputIds.resize(
        MAX_CONTEXT - 1
    );
}

The subtraction by one is deliberate: it reserves room for at least one generated token.

The maximum number of generated tokens is then limited with:

const int allowed =
    std::min(
        maxNewTokens,
        MAX_CONTEXT -
        static_cast<int>(
            inputIds.size()
        )
    );

This guarantees that:

prompt tokens + generated tokens <= 1024

for a single generation session.

Autoregressive loop

The actual generation loop is:

for (int step = 0;
     step < allowed;
     ++step)
{
    const std::vector<float> logits =
        forward(
            allTokens,
            model
        );

    const int nextToken =
        argmax(
            logits
        );

    generated.push_back(
        nextToken
    );

    allTokens.push_back(
        nextToken
    );

    printToken(
        step,
        nextToken,
        tokenizer
    );
}

The key operation is:

forward(allTokens, model)

followed by:

argmax(logits)

and then:

allTokens.push_back(nextToken)

This creates the autoregressive feedback loop:

Current context
      │
      ▼
    forward()
      │
      ▼
    logits
      │
      ▼
    argmax()
      │
      ▼
  next token
      │
      ▼
append to context
      │
      └──────────► forward() again

Finally, all tokens are decoded:

decodeExact(allTokens)

and the complete generated text is printed.

Tensor Shapes

Understanding the shapes is one of the easiest ways to understand the code.

Let:

N = current sequence length
H = 768
A = 12
D = 64
F = 3072
V = 50257

Then:

Object

Shape

Token embedding WTE

V × H

Position embedding WPE

1024 × H

Hidden states

N × H

Q

N × H

K

N × H

V

N × H

Combined QKV projection

H × 3H

Per-head Q/K/V slice

N × D

Attention scores for one position/head

t + 1

Context

N × H

MLP expansion

N × F

MLP output

N × H

Final hidden state

H

Logits

V

For example, if:

N = 5

then the hidden state tensor is:

5 × 768

and the vocabulary output is:

50,257

floating-point scores.

Weight File Mapping

The source expects these tensors.

File

Purpose

Expected elements

transformer.wte.weight.txt

Token embeddings

50257 × 768

transformer.wpe.weight.txt

Position embeddings

1024 × 768

transformer.ln_f.weight.txt

Final LayerNorm scale

768

transformer.ln_f.bias.txt

Final LayerNorm bias

768

For every layer L from 0 through 11:

File

Purpose

Expected elements

transformer.h.L.ln_1.weight.txt

Attention LayerNorm scale

768

transformer.h.L.ln_1.bias.txt

Attention LayerNorm bias

768

transformer.h.L.attn.c_attn.weight.txt

Combined QKV projection

768 × 2304

transformer.h.L.attn.c_attn.bias.txt

Combined QKV bias

2304

transformer.h.L.attn.c_proj.weight.txt

Attention output projection

768 × 768

transformer.h.L.attn.c_proj.bias.txt

Attention output bias

768

transformer.h.L.ln_2.weight.txt

MLP LayerNorm scale

768

transformer.h.L.ln_2.bias.txt

MLP LayerNorm bias

768

transformer.h.L.mlp.c_fc.weight.txt

MLP expansion

768 × 3072

transformer.h.L.mlp.c_fc.bias.txt

MLP expansion bias

3072

transformer.h.L.mlp.c_proj.weight.txt

MLP contraction

3072 × 768

transformer.h.L.mlp.c_proj.bias.txt

MLP contraction bias

768

All tensors are loaded as flat float arrays.

Numerical Details

Float representation

The inference engine stores tensors as:

std::vector<float>

so the primary numerical type is 32-bit floating point.

This keeps the implementation simple and avoids introducing a separate half-precision or quantized runtime.

LayerNorm epsilon

LayerNorm uses:

constexpr float EPSILON = 1e-5f;

inside:

sqrt(variance + epsilon)

to avoid numerical instability from division by zero or extremely small denominators.

Stable softmax

Instead of directly evaluating:

exp(x)

the implementation subtracts the maximum input first:

exp(x - max(x))

The resulting softmax distribution is mathematically equivalent while being safer numerically.

Generation and Context Management

GPT-2 has a maximum context length of:

1024 tokens

The source enforces this in two places.

forward()

The forward function rejects sequences longer than 1024.

main()

Before generation, main() truncates the prompt to:

MAX_CONTEXT - 1

when necessary.

It then limits generation using:

std::min(
    maxNewTokens,
    MAX_CONTEXT - input_length
)

This ensures that the generated sequence never exceeds the model's configured context size during the REPL session.

Build

The project requires a C++17-capable compiler.

A build from the repository root can use:

mkdir -p build

g++ \
  -std=c++17 \
  -O3 \
  -I . \
  -o build/gpt2_infer \
  src/main.cpp

The -I . include path matches the source's current include:

#include "include/json.hpp"

If your local layout differs, adjust the include path or source include accordingly.

Run

The executable expects the process working directory to make:

../weights

resolve correctly.

With the layout:

project-root/
├── weights/
└── build/
    └── gpt2_infer

run:

cd build
./gpt2_infer

To use another Python interpreter:

GPT2_PYTHON=python3.11 ./gpt2_infer

Requirements

C++

C++17-compatible compiler

GCC or Clang

JSON

nlohmann/json

The single-header file is expected at:

include/json.hpp

Python

Python 3 with Hugging Face tokenizers:

pip install tokenizers

Model assets

You need:

GPT-2 Small / 124M weights

the expected text tensor files

weights/tokenizer/tokenizer.json

The weight export format must match the tensor order and dimensions expected by the C++ implementation.

Example

Start the executable:

GPT-2 124M CPU Inference Engine
================================
Type 'exit' to quit.

Prompt:

Enter:

The quick brown

Then:

Number of new tokens: 5

The engine will:

1. Encode the prompt
2. Print the input token IDs
3. Run GPT-2 forward
4. Select the highest-scoring token
5. Append it to the context
6. Run forward again
7. Repeat until 5 tokens are generated
8. Decode the complete token sequence

The exact generated text depends on the supplied GPT-2 weights and tokenizer.

Performance

This implementation intentionally favors readability over throughput.

The current code is:

CPU only
Single-threaded
Unbatched
Plain C++ loops
float32 tensors
No SIMD kernels
No BLAS
No GPU backend
No KV cache
Plain-text weight loading
External Python tokenizer process

The largest cost comes from repeated matrix operations and from recomputing the full sequence during autoregressive generation.

The implementation is therefore best viewed as an educational/reference inference engine, not as a replacement for optimized runtimes.

Why Generation Gets Slower

The current generation path performs:

Step 1:
forward(prompt)

Step 2:
forward(prompt + token_1)

Step 3:
forward(prompt + token_1 + token_2)

Step 4:
forward(prompt + token_1 + token_2 + token_3)

...

Nothing is cached between steps.

In particular, previously computed attention keys and values are recomputed.

A production runtime would typically use a KV cache:

Past tokens
   │
   ├── cached K
   └── cached V

New token
   │
   ▼
compute only new K/V
   │
   ▼
reuse cached history

That is one of the major performance improvements that could be added to this project.

Limitations

Greedy-only decoding

Current decoding is:

nextToken = argmax(logits);

There is no stochastic sampling.

Unsupported decoding strategies include:

temperature

top-k

top-p / nucleus sampling

repetition penalty

beam search

No KV cache

The engine recomputes the Transformer for the whole current context at each generation step.

CPU only

There is no CUDA, GPU, or accelerator backend.

Single-threaded

The current implementation does not explicitly parallelize matrix operations, attention heads, or token positions.

Naive matrix multiplication

linear() uses direct nested loops rather than optimized numerical kernels.

Plain-text weights

Human-readable tensor files are convenient for inspection but are much larger and slower to load than a binary representation.

External tokenizer process

Encoding and decoding launch Python subprocesses and use temporary files.

This adds overhead and requires Python to be installed at runtime.

GPT-2 Small only

The architecture is hardcoded for:

50257 vocab
768 hidden
12 layers
12 heads
64 head dimension
3072 MLP
1024 context

Supporting other GPT-2 sizes requires generalized configuration and compatible weight files.

Future Work

The code is intentionally structured so that major inference improvements can be added incrementally.

Potential extensions include:

1. KV cache

Avoid recomputing previous keys and values during generation.

2. Optimized matrix kernels

Replace the naive linear() implementation with:

SIMD

cache-aware kernels

BLAS

threaded matrix multiplication

3. Binary weight format

Replace whitespace-separated text files with a compact binary representation.

4. Memory mapping

Allow large weight tensors to be memory-mapped rather than fully parsed from text.

5. Native C++ tokenizer

Remove the Python subprocess and temporary-file round trip.

6. Sampling

Add:

temperature
top-k
top-p
repetition penalty

7. Batch inference

Support multiple prompts at once.

8. Model configuration

Replace compile-time GPT-2 constants with runtime model metadata.

9. GPU backend

Introduce CUDA or another accelerator backend after the CPU reference implementation is optimized and validated.

Design Decisions

Why std::vector<float>?

It keeps the implementation understandable.

The project does not need a custom tensor framework to demonstrate the core operations.

Why one linear() function?

Attention projections and MLP projections all reduce to:

y = xW + b

Using one primitive avoids duplicated matrix-multiplication code.

Why compute Q, K and V together?

GPT-2's attention projection is stored as one combined c_attn matrix.

Therefore the implementation computes:

QKV = XW + b

once and splits the result into Q, K and V.

Why use Python for tokenization?

The C++ project focuses on model inference.

Using the provided Hugging Face tokenizer keeps the tokenization path aligned with the reference GPT-2 tokenizer rather than maintaining a second BPE implementation in C++.

Why use plain-text weights?

It makes the tensors easy to inspect and debug.

The tradeoff is significantly larger files and slower loading.

Full Inference Pipeline

The entire program can be summarized as:

                           USER
                            │
                            ▼
                     Text Prompt
                            │
                            ▼
                     encodeExact()
                            │
                            ▼
                        Token IDs
                            │
                            ▼
                 Token + Position Embeddings
                            │
                            ▼
                 ┌─────────────────────────┐
                 │ Transformer Block × 12  │
                 │                         │
                 │ LayerNorm                │
                 │    ↓                    │
                 │ QKV Projection          │
                 │    ↓                    │
                 │ 12 Attention Heads      │
                 │    ↓                    │
                 │ Causal Attention        │
                 │    ↓                    │
                 │ Output Projection       │
                 │    ↓                    │
                 │ Residual                │
                 │    ↓                    │
                 │ LayerNorm                │
                 │    ↓                    │
                 │ MLP 768→3072→768        │
                 │    ↓                    │
                 │ Residual                │
                 └────────────┬────────────┘
                              │
                              ▼
                       Final LayerNorm
                              │
                              ▼
                 Vocabulary Projection
                     using WTEᵀ
                              │
                              ▼
                       50,257 Logits
                              │
                              ▼
                          argmax()
                              │
                              ▼
                         Next Token
                              │
                              ▼
                       append token
                              │
                              └───────────────┐
                                              │
                                              ▼
                                         forward()
                                          again

The Core Idea in One Function

The most important conceptual function is forward().

At a high level, it implements:

tokens
  ↓
embeddings
  ↓
Transformer × 12
  ↓
final hidden state
  ↓
vocabulary scores

or mathematically:

H₀ = WTE(tokens) + WPE(positions)

H₁  = Block₀(H₀)
H₂  = Block₁(H₁)
...
H₁₂ = Block₁₁(H₁₁)

h = LayerNorm(H₁₂[last_position])

logits = h × WTEᵀ

The decoding layer then performs:

next_token = argmax(logits)

Generation repeats this computation after appending the newly selected token.

That is the entire inference engine in conceptual form.

Understanding the Code in Reading Order

For someone reading the source for the first time, the recommended order is:

1. Constants / Tensor
        ↓
2. loadTensor()
        ↓
3. Block
        ↓
4. GPT2::load()
        ↓
5. layerNorm()
        ↓
6. linear()
        ↓
7. gelu()
        ↓
8. softmax()
        ↓
9. attention()
        ↓
10. mlp()
        ↓
11. transformerBlock()
        ↓
12. forward()
        ↓
13. argmax()
        ↓
14. Tokenizer
        ↓
15. encodeExact() / decodeExact()
        ↓
16. main()

This order mirrors the dependency structure of the implementation.

Start with the small mathematical primitives, then move upward into attention, then the Transformer block, then the complete forward pass, and finally the REPL and generation loop.

What Makes This an Inference Engine?

The program does not "understand text" through a collection of handwritten rules.

There is no logic such as:

if (prompt contains "hello")
    return "hi";

Instead, the behavior comes from the learned numerical parameters.

The C++ engine's responsibility is to execute those parameters through the GPT-2 computation graph correctly.

That means the project sits at the boundary between:

Model parameters
      +
Model architecture
      +
Numerical kernels
      ↓
Inference

The model weights provide the learned behavior.

The Transformer architecture defines how those weights are used.

The C++ implementation performs the arithmetic.

Educational Value

This project is useful for studying:

Transformer architecture

GPT-2 internals

autoregressive language modeling

attention

causal masking

embeddings

matrix multiplication

LayerNorm

residual connections

feed-forward networks

softmax

tokenization

model weight formats

CPU inference

memory layout

inference bottlenecks

KV-cache design

The implementation intentionally keeps these concepts visible instead of hiding them behind a framework.