#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/json.hpp"

using json = nlohmann::json;

constexpr int VOCAB_SIZE = 50257;
constexpr int HIDDEN_SIZE = 768;
constexpr int NUM_LAYERS = 12;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = 64;
constexpr int MLP_SIZE = 3072;
constexpr int MAX_CONTEXT = 1024;
constexpr float EPSILON = 1e-5f;

using Tensor = std::vector<float>;

Tensor loadTensor(const std::string& path, size_t expected)
{
    std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Cannot open weight file: " + path);

    Tensor data;
    data.reserve(expected);

    float value;

    while (file >> value)
        data.push_back(value);

    if (data.size() != expected)
    {
        throw std::runtime_error(
            "Wrong tensor size for " + path +
            ". Expected " + std::to_string(expected) +
            ", got " + std::to_string(data.size())
        );
    }

    return data;
}

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

struct GPT2
{
    Tensor wte;
    Tensor wpe;

    std::array<Block, NUM_LAYERS> blocks;

    Tensor lnFinalWeight;
    Tensor lnFinalBias;

    void load(const std::string& dir)
    {
        std::cout << "Loading GPT-2 weights...\n";

        wte = loadTensor(
            dir + "/transformer.wte.weight.txt",
            static_cast<size_t>(VOCAB_SIZE) * HIDDEN_SIZE
        );

        wpe = loadTensor(
            dir + "/transformer.wpe.weight.txt",
            static_cast<size_t>(MAX_CONTEXT) * HIDDEN_SIZE
        );

        for (int layer = 0; layer < NUM_LAYERS; ++layer)
        {
            Block& b = blocks[layer];

            std::string p =
                dir +
                "/transformer.h." +
                std::to_string(layer) +
                ".";

            b.ln1Weight = loadTensor(
                p + "ln_1.weight.txt",
                HIDDEN_SIZE
            );

            b.ln1Bias = loadTensor(
                p + "ln_1.bias.txt",
                HIDDEN_SIZE
            );

            b.cAttnWeight = loadTensor(
                p + "attn.c_attn.weight.txt",
                static_cast<size_t>(HIDDEN_SIZE) * HIDDEN_SIZE * 3
            );

            b.cAttnBias = loadTensor(
                p + "attn.c_attn.bias.txt",
                static_cast<size_t>(HIDDEN_SIZE) * 3
            );

            b.cProjWeight = loadTensor(
                p + "attn.c_proj.weight.txt",
                static_cast<size_t>(HIDDEN_SIZE) * HIDDEN_SIZE
            );

            b.cProjBias = loadTensor(
                p + "attn.c_proj.bias.txt",
                HIDDEN_SIZE
            );

            b.ln2Weight = loadTensor(
                p + "ln_2.weight.txt",
                HIDDEN_SIZE
            );

            b.ln2Bias = loadTensor(
                p + "ln_2.bias.txt",
                HIDDEN_SIZE
            );

            b.cFcWeight = loadTensor(
                p + "mlp.c_fc.weight.txt",
                static_cast<size_t>(HIDDEN_SIZE) * MLP_SIZE
            );

            b.cFcBias = loadTensor(
                p + "mlp.c_fc.bias.txt",
                MLP_SIZE
            );

            b.cProjMlpWeight = loadTensor(
                p + "mlp.c_proj.weight.txt",
                static_cast<size_t>(MLP_SIZE) * HIDDEN_SIZE
            );

            b.cProjMlpBias = loadTensor(
                p + "mlp.c_proj.bias.txt",
                HIDDEN_SIZE
            );

            std::cout
                << "Loaded layer "
                << layer + 1
                << "/"
                << NUM_LAYERS
                << "\n";
        }

        lnFinalWeight = loadTensor(
            dir + "/transformer.ln_f.weight.txt",
            HIDDEN_SIZE
        );

        lnFinalBias = loadTensor(
            dir + "/transformer.ln_f.bias.txt",
            HIDDEN_SIZE
        );

        std::cout << "Weights loaded.\n";
    }
};

std::vector<float> layerNorm(
    const std::vector<float>& x,
    const Tensor& weight,
    const Tensor& bias
)
{
    float mean = 0.0f;

    for (float v : x)
        mean += v;

    mean /= static_cast<float>(x.size());

    float variance = 0.0f;

    for (float v : x)
    {
        float d = v - mean;
        variance += d * d;
    }

    variance /= static_cast<float>(x.size());

    float invStd =
        1.0f / std::sqrt(variance + EPSILON);

    std::vector<float> output(x.size());

    for (size_t i = 0; i < x.size(); ++i)
    {
        output[i] =
            ((x[i] - mean) * invStd) * weight[i] +
            bias[i];
    }

    return output;
}

std::vector<float> linear(
    const std::vector<float>& input,
    const Tensor& weight,
    const Tensor& bias,
    int inputSize,
    int outputSize
)
{
    std::vector<float> output(
        outputSize,
        0.0f
    );

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

    return output;
}

float gelu(float x)
{
    constexpr float c = 0.7978845608028654f;

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
}

std::vector<float> applyGelu(
    const std::vector<float>& x
)
{
    std::vector<float> output(x.size());

    for (size_t i = 0; i < x.size(); ++i)
        output[i] = gelu(x[i]);

    return output;
}

std::vector<float> softmax(
    const std::vector<float>& x
)
{
    if (x.empty())
        return {};

    std::vector<float> output(x.size());

    float maxValue =
        *std::max_element(
            x.begin(),
            x.end()
        );

    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); ++i)
    {
        output[i] =
            std::exp(x[i] - maxValue);

        sum += output[i];
    }

    if (sum <= 0.0f)
        throw std::runtime_error("Softmax failure");

    for (float& value : output)
        value /= sum;

    return output;
}

std::vector<std::vector<float>> attention(
    const std::vector<std::vector<float>>& input,
    const Block& block
)
{
    const int seqLen =
        static_cast<int>(input.size());

    std::vector<std::vector<float>> q(
        seqLen,
        std::vector<float>(HIDDEN_SIZE)
    );

    std::vector<std::vector<float>> k(
        seqLen,
        std::vector<float>(HIDDEN_SIZE)
    );

    std::vector<std::vector<float>> v(
        seqLen,
        std::vector<float>(HIDDEN_SIZE)
    );

    for (int t = 0; t < seqLen; ++t)
    {
        std::vector<float> qkv =
            linear(
                input[t],
                block.cAttnWeight,
                block.cAttnBias,
                HIDDEN_SIZE,
                HIDDEN_SIZE * 3
            );

        for (int i = 0; i < HIDDEN_SIZE; ++i)
        {
            q[t][i] = qkv[i];

            k[t][i] =
                qkv[HIDDEN_SIZE + i];

            v[t][i] =
                qkv[2 * HIDDEN_SIZE + i];
        }
    }

    std::vector<std::vector<float>> context(
        seqLen,
        std::vector<float>(
            HIDDEN_SIZE,
            0.0f
        )
    );

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(HEAD_DIM)
        );

    for (int head = 0; head < NUM_HEADS; ++head)
    {
        const int offset =
            head * HEAD_DIM;

        for (int t = 0; t < seqLen; ++t)
        {
            std::vector<float> scores(
                t + 1
            );

            for (int j = 0; j <= t; ++j)
            {
                float score = 0.0f;

                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    score +=
                        q[t][offset + d] *
                        k[j][offset + d];
                }

                scores[j] =
                    score * scale;
            }

            std::vector<float> probabilities =
                softmax(scores);

            for (int j = 0; j <= t; ++j)
            {
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    context[t][offset + d] +=
                        probabilities[j] *
                        v[j][offset + d];
                }
            }
        }
    }

    std::vector<std::vector<float>> output(
        seqLen,
        std::vector<float>(HIDDEN_SIZE)
    );

    for (int t = 0; t < seqLen; ++t)
    {
        output[t] =
            linear(
                context[t],
                block.cProjWeight,
                block.cProjBias,
                HIDDEN_SIZE,
                HIDDEN_SIZE
            );
    }

    return output;
}

std::vector<float> mlp(
    const std::vector<float>& input,
    const Block& block
)
{
    std::vector<float> hidden =
        linear(
            input,
            block.cFcWeight,
            block.cFcBias,
            HIDDEN_SIZE,
            MLP_SIZE
        );

    hidden =
        applyGelu(hidden);

    return linear(
        hidden,
        block.cProjMlpWeight,
        block.cProjMlpBias,
        MLP_SIZE,
        HIDDEN_SIZE
    );
}

std::vector<std::vector<float>> transformerBlock(
    const std::vector<std::vector<float>>& input,
    const Block& block
)
{
    const int seqLen =
        static_cast<int>(input.size());

    std::vector<std::vector<float>> ln1(
        seqLen
    );

    for (int t = 0; t < seqLen; ++t)
    {
        ln1[t] =
            layerNorm(
                input[t],
                block.ln1Weight,
                block.ln1Bias
            );
    }

    std::vector<std::vector<float>> attn =
        attention(
            ln1,
            block
        );

    std::vector<std::vector<float>> x =
        input;

    for (int t = 0; t < seqLen; ++t)
    {
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            x[t][i] += attn[t][i];
    }

    std::vector<std::vector<float>> ln2(
        seqLen
    );

    for (int t = 0; t < seqLen; ++t)
    {
        ln2[t] =
            layerNorm(
                x[t],
                block.ln2Weight,
                block.ln2Bias
            );
    }

    for (int t = 0; t < seqLen; ++t)
    {
        std::vector<float> m =
            mlp(
                ln2[t],
                block
            );

        for (int i = 0; i < HIDDEN_SIZE; ++i)
            x[t][i] += m[i];
    }

    return x;
}

std::vector<float> forward(
    const std::vector<int>& tokenIds,
    const GPT2& model
)
{
    if (tokenIds.empty())
        throw std::runtime_error("No input tokens");

    if (tokenIds.size() > MAX_CONTEXT)
        throw std::runtime_error(
            "Input exceeds GPT-2 context length"
        );

    std::vector<std::vector<float>> hidden(
        tokenIds.size(),
        std::vector<float>(HIDDEN_SIZE)
    );

    for (size_t position = 0;
         position < tokenIds.size();
         ++position)
    {
        const int tokenId =
            tokenIds[position];

        if (tokenId < 0 ||
            tokenId >= VOCAB_SIZE)
        {
            throw std::runtime_error(
                "Invalid token ID: " +
                std::to_string(tokenId)
            );
        }

        const size_t tokenOffset =
            static_cast<size_t>(tokenId) *
            HIDDEN_SIZE;

        const size_t posOffset =
            position *
            HIDDEN_SIZE;

        for (int i = 0; i < HIDDEN_SIZE; ++i)
        {
            hidden[position][i] =
                model.wte[tokenOffset + i] +
                model.wpe[posOffset + i];
        }
    }

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

    std::vector<float> finalHidden =
        layerNorm(
            hidden.back(),
            model.lnFinalWeight,
            model.lnFinalBias
        );

    std::vector<float> logits(
        VOCAB_SIZE
    );

    for (int token = 0;
         token < VOCAB_SIZE;
         ++token)
    {
        const size_t offset =
            static_cast<size_t>(token) *
            HIDDEN_SIZE;

        float sum = 0.0f;

        for (int i = 0; i < HIDDEN_SIZE; ++i)
        {
            sum +=
                finalHidden[i] *
                model.wte[offset + i];
        }

        logits[token] = sum;
    }

    return logits;
}

int argmax(
    const std::vector<float>& logits
)
{
    return static_cast<int>(
        std::max_element(
            logits.begin(),
            logits.end()
        ) -
        logits.begin()
    );
}

struct Tokenizer
{
    std::vector<std::string> vocabulary;

    void load(const std::string& path)
    {
        std::ifstream file(path);

        if (!file.is_open())
            throw std::runtime_error(
                "Cannot open tokenizer.json: " + path
            );

        json data;
        file >> data;

        if (!data.contains("model"))
            throw std::runtime_error(
                "Tokenizer model missing"
            );

        if (!data["model"].contains("vocab"))
            throw std::runtime_error(
                "Tokenizer vocabulary missing"
            );

        const auto& vocab =
            data["model"]["vocab"];

        vocabulary.resize(
            VOCAB_SIZE
        );

        size_t count = 0;

        for (auto it = vocab.begin();
             it != vocab.end();
             ++it)
        {
            const int id =
                it.value().get<int>();

            if (id >= 0 && id < VOCAB_SIZE)
            {
                vocabulary[id] =
                    it.key();

                ++count;
            }
        }

        if (count != VOCAB_SIZE)
        {
            throw std::runtime_error(
                "Unexpected GPT-2 vocabulary size"
            );
        }

        std::cout
            << "Tokenizer vocabulary loaded: "
            << count
            << "\n";
    }

    const std::string& token(int id) const
    {
        if (id < 0 ||
            id >= static_cast<int>(vocabulary.size()))
        {
            throw std::runtime_error(
                "Invalid vocabulary ID"
            );
        }

        return vocabulary[id];
    }

    std::string textToken(int id) const
    {
        std::string t =
            token(id);

        if (t.size() >= 2 &&
            t[0] == static_cast<char>(0xC4) &&
            t[1] == static_cast<char>(0xA0))
        {
            return " " + t.substr(2);
        }

        if (t == "Ċ")
            return "\\n";

        return t;
    }
};

std::string getPython()
{
    const char* value =
        std::getenv("GPT2_PYTHON");

    if (value != nullptr &&
        value[0] != '\0')
    {
        return value;
    }

    return "python3";
}

std::vector<int> encodeExact(
    const std::string& text
)
{
    const std::string inputPath =
        "/tmp/gpt2_encode_input.txt";

    const std::string outputPath =
        "/tmp/gpt2_encode_output.txt";

    {
        std::ofstream file(inputPath);

        if (!file.is_open())
            throw std::runtime_error(
                "Cannot create tokenizer input"
            );

        file << text;
    }

    const std::string command =
        getPython() +
        " -c \""
        "from tokenizers import Tokenizer;"
        "import sys;"
        "t=Tokenizer.from_file('../weights/tokenizer/tokenizer.json');"
        "s=sys.stdin.read();"
        "print(' '.join(str(x) for x in t.encode(s).ids))"
        "\" < " +
        inputPath +
        " > " +
        outputPath;

    const int status =
        std::system(
            command.c_str()
        );

    if (status != 0)
    {
        std::remove(inputPath.c_str());
        std::remove(outputPath.c_str());

        throw std::runtime_error(
            "Tokenizer failed. Run: pip install tokenizers"
        );
    }

    std::ifstream file(outputPath);

    if (!file.is_open())
    {
        std::remove(inputPath.c_str());
        std::remove(outputPath.c_str());

        throw std::runtime_error(
            "Cannot read tokenizer output"
        );
    }

    std::vector<int> ids;
    int id;

    while (file >> id)
        ids.push_back(id);

    std::remove(inputPath.c_str());
    std::remove(outputPath.c_str());

    return ids;
}

std::string decodeExact(
    const std::vector<int>& ids
)
{
    const std::string inputPath =
        "/tmp/gpt2_decode_input.json";

    const std::string outputPath =
        "/tmp/gpt2_decode_output.txt";

    {
        std::ofstream file(inputPath);

        if (!file.is_open())
            throw std::runtime_error(
                "Cannot create decode input"
            );

        json data = ids;
        file << data.dump();
    }

    const std::string command =
        getPython() +
        " -c \""
        "from tokenizers import Tokenizer;"
        "import sys,json;"
        "t=Tokenizer.from_file('../weights/tokenizer/tokenizer.json');"
        "ids=json.load(sys.stdin);"
        "print(t.decode(ids,skip_special_tokens=False),end='')"
        "\" < " +
        inputPath +
        " > " +
        outputPath;

    const int status =
        std::system(
            command.c_str()
        );

    if (status != 0)
    {
        std::remove(inputPath.c_str());
        std::remove(outputPath.c_str());

        throw std::runtime_error(
            "Tokenizer decode failed"
        );
    }

    std::ifstream file(outputPath);

    if (!file.is_open())
    {
        std::remove(inputPath.c_str());
        std::remove(outputPath.c_str());

        throw std::runtime_error(
            "Cannot read decoded output"
        );
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    std::remove(inputPath.c_str());
    std::remove(outputPath.c_str());

    return buffer.str();
}

void printToken(
    int index,
    int id,
    const Tokenizer& tokenizer
)
{
    std::cout
        << "["
        << index
        << "] ID: "
        << id
        << "    GPT-2: \""
        << tokenizer.token(id)
        << "\"    Text: \""
        << tokenizer.textToken(id)
        << "\"\n";
}

int main()
{
    try
    {
        GPT2 model;
        model.load("../weights");

        Tokenizer tokenizer;
        tokenizer.load(
            "../weights/tokenizer/tokenizer.json"
        );

        std::cout << "\n";
        std::cout
            << "GPT-2 124M CPU Inference Engine\n";
        std::cout
            << "================================\n";
        std::cout
            << "Type 'exit' to quit.\n";

        while (true)
        {
            std::string prompt;

            std::cout
                << "\nPrompt: "
                << std::flush;

            if (!std::getline(
                    std::cin,
                    prompt
                ))
            {
                break;
            }

            if (prompt == "exit")
                break;

            if (prompt.empty())
            {
                std::cout
                    << "Prompt cannot be empty.\n";
                continue;
            }

            int maxNewTokens;

            std::cout
                << "Number of new tokens: "
                << std::flush;

            if (!(std::cin >>
                  maxNewTokens))
            {
                std::cin.clear();

                std::cin.ignore(
                    std::numeric_limits<
                        std::streamsize
                    >::max(),
                    '\n'
                );

                std::cout
                    << "Invalid token count.\n";

                continue;
            }

            std::cin.ignore(
                std::numeric_limits<
                    std::streamsize
                >::max(),
                '\n'
            );

            if (maxNewTokens <= 0)
            {
                std::cout
                    << "Token count must be greater than zero.\n";
                continue;
            }

            std::vector<int> inputIds =
                encodeExact(prompt);

            if (inputIds.empty())
            {
                std::cout
                    << "Tokenizer returned no tokens.\n";
                continue;
            }

            if (inputIds.size() >= MAX_CONTEXT)
            {
                inputIds.resize(
                    MAX_CONTEXT - 1
                );
            }

            std::cout
                << "\nPrompt tokens:\n";

            for (size_t i = 0;
                 i < inputIds.size();
                 ++i)
            {
                printToken(
                    static_cast<int>(i),
                    inputIds[i],
                    tokenizer
                );
            }

            std::vector<int> allTokens =
                inputIds;

            std::vector<int> generated;

            const int allowed =
                std::min(
                    maxNewTokens,
                    MAX_CONTEXT -
                    static_cast<int>(
                        inputIds.size()
                    )
                );

            std::cout
                << "\nGenerated tokens:\n";

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

            std::cout
                << "\nGenerated text:\n";

            std::cout
                << decodeExact(
                    allTokens
                )
                << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "\nERROR: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}