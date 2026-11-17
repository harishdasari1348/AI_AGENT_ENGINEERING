"""Build the generative_ai.ipynb notebook using nbformat and execute it with nbclient."""

from __future__ import annotations

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbclient import NotebookClient

FOLDER = Path(__file__).resolve().parent
NOTEBOOK_PATH = FOLDER / "generative_ai.ipynb"


def make_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.5",
        },
    }

    cells = []

    # Title & Overview
    cells.append(
        new_markdown_cell(
            """# Generative AI from Scratch: Probability, Bigrams, and Sampling

A foundational, self-contained educational notebook exploring **Generative Artificial Intelligence** for AI Agent Engineers.

### What you will explore:
1. **Discriminative vs Generative Models**: The fundamental mathematical divide ($P(Y \mid X)$ vs $P(X)$).
2. **Tokenization and Vocabulary**: Word indexing, special sequence delimiters (`<BOS>`, `<EOS>`).
3. **Bigram Language Model**: Transition count matrices and Laplace (add-1) smoothing.
4. **Sampling and Temperature Control**: Converting logits to probabilities; greedy ($T \\to 0$), balanced ($T = 0.7$), and exploratory ($T = 1.5$) decoding.
5. **Evaluation**: Negative Log-Likelihood (NLL) and Perplexity (PPL) on in-domain vs out-of-domain sequences.
6. **Higher-Order Markov Chains**: Extending context windows ($k=1, 2, 3$).
7. **The Bridge to Modern LLMs**: How Transformer architectures scale these exact same principles across billions of parameters.

*Zero external API keys required. Implemented completely in standard Python and NumPy.*"""
        )
    )

    # Section 1: Environment Setup
    cells.append(
        new_markdown_cell(
            """## 1. Environment Verification and Imports

We import standard library tools and NumPy, fixing a seed for deterministic reproducibility."""
        )
    )

    cells.append(
        new_code_cell(
            """import sys
import math
import random
from collections import defaultdict, Counter
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"Python version : {sys.version.split()[0]}")
print(f"NumPy version  : {np.__version__}")
print(f"Random seed    : {SEED}")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: confirm libraries and environment
assert sys.version_info >= (3, 10), "Requires Python 3.10+"
assert np.__version__ is not None, "NumPy must be loaded"
print("Environment checks passed.")"""
        )
    )

    # Section 2: Discriminative vs Generative
    cells.append(
        new_markdown_cell(
            """## 2. Discriminative vs Generative Models

Machine learning models fall into two core statistical paradigms:

| Attribute | Discriminative Model | Generative Model |
|---|---|---|
| **Mathematical Goal** | Estimates $P(Y \\mid X)$ directly | Estimates $P(X \\mid Y)P(Y)$ or $P(X)$ |
| **Learns** | Decision boundary between classes | Distribution and structure of the data itself |
| **Primary Task** | Classification, regression, labeling | Creating synthetic data, simulation, text generation |
| **Core Question** | "Is this input cat or dog?" | "What does a cat look like? Generate one." |
| **Classic Examples** | Logistic Regression, SVM, Decision Trees | Naive Bayes, Markov Chains, VAEs, GANs, LLMs |

Below, we construct a 1D synthetic experiment to see the difference numerically."""
        )
    )

    cells.append(
        new_code_cell(
            """# Create synthetic 1D data from two Gaussian distributions
# Class 0: mean = -2.0, std = 0.8 (e.g. normal behavior)
# Class 1: mean = +2.5, std = 0.9 (e.g. anomalous behavior)
n_samples = 200
x0 = np.random.normal(loc=-2.0, scale=0.8, size=n_samples)
x1 = np.random.normal(loc=2.5, scale=0.9, size=n_samples)

X = np.concatenate([x0, x1])
y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])

# A. Discriminative Model: Find decision boundary P(Y=1 | X) via Logistic Regression
w = 0.0
b = 0.0
lr = 0.05
for _ in range(500):
    logits = w * X + b
    preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    grad_w = np.mean((preds - y) * X)
    grad_b = np.mean(preds - y)
    w -= lr * grad_w
    b -= lr * grad_b

decision_boundary = -b / w
print(f"Discriminative Model (P(Y|X)):")
print(f"  Learned boundary: x = {decision_boundary:.3f}")
print(f"  Decision rule   : If x > {decision_boundary:.3f} predict Class 1, else Class 0.\\n")

# B. Generative Model: Learn class-conditional data distributions P(X | Y=k)
mean_0, std_0 = float(np.mean(x0)), float(np.std(x0))
mean_1, std_1 = float(np.mean(x1)), float(np.std(x1))
prior_0 = len(x0) / len(X)
prior_1 = len(x1) / len(X)

print(f"Generative Model (P(X|Y) and P(Y)):")
print(f"  Class 0 distribution: Normal(mean={mean_0:.3f}, std={std_0:.3f}), Prior={prior_0:.2f}")
print(f"  Class 1 distribution: Normal(mean={mean_1:.3f}, std={std_1:.3f}), Prior={prior_1:.2f}")

# Generate brand new synthetic data points from scratch
new_synthetic_c0 = np.random.normal(loc=mean_0, scale=std_0, size=5)
new_synthetic_c1 = np.random.normal(loc=mean_1, scale=std_1, size=5)

print("\\nGenerated Synthetic Samples from Generative Model:")
print(f"  Synthetic Class 0: {np.round(new_synthetic_c0, 3).tolist()}")
print(f"  Synthetic Class 1: {np.round(new_synthetic_c1, 3).tolist()}")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: generative model captured distribution statistics
assert abs(mean_0 - (-2.0)) < 0.25, "Class 0 mean should be near -2.0"
assert abs(mean_1 - 2.5) < 0.25, "Class 1 mean should be near 2.5"
assert len(new_synthetic_c0) == 5
assert len(new_synthetic_c1) == 5
assert np.all(new_synthetic_c0 < decision_boundary)
assert np.all(new_synthetic_c1 > decision_boundary)
print("Discriminative vs Generative validation checks passed.")"""
        )
    )

    # Section 3: Tokenization & Vocabulary
    cells.append(
        new_markdown_cell(
            """## 3. Tokenization and Vocabulary Building

In generative natural language processing, text generation is modeled as predicting the probability distribution over a sequence of **tokens**:
$$P(w_1, w_2, \\dots, w_T) = \\prod_{t=1}^T P(w_t \\mid w_1, \\dots, w_{t-1})$$

To allow the model to know when a sequence starts and ends, we add special delimiter tokens:
- `<BOS>`: Beginning of Sequence
- `<EOS>`: End of Sequence

Below, we process a focused domain corpus of AI Agent principles."""
        )
    )

    cells.append(
        new_code_cell(
            """# Domain Corpus: Agent lifecycle, reasoning, and tool execution
CORPUS = [
    "the intelligent agent perceives its environment and takes action",
    "the agent learns from experience to maximize cumulative reward",
    "an intelligent agent communicates with tools and other agents",
    "the system perceives incoming data and takes action",
    "an autonomous agent plans each action and executes tools",
    "the neural network learns representations from raw data",
    "an intelligent system takes action to solve complex tasks",
    "the agent observes the environment and plans its next step",
    "the autonomous agent executes tools to solve complex tasks",
    "an agent learns from data to improve cumulative performance",
]

BOS = "<BOS>"
EOS = "<EOS>"

# Tokenize into word sequences wrapped in sequence markers
tokenized_sentences = []
all_words = [BOS, EOS]

for sent in CORPUS:
    words = sent.strip().lower().split()
    tokenized_sentences.append([BOS] + words + [EOS])
    all_words.extend(words)

# Build vocabulary mappings
unique_tokens = sorted(list(set(all_words)))
vocab_size = len(unique_tokens)
token2id = {tok: idx for idx, tok in enumerate(unique_tokens)}
id2token = {idx: tok for idx, tok in enumerate(unique_tokens)}

print(f"Corpus sentences   : {len(CORPUS)}")
print(f"Total token count  : {sum(len(s) for s in tokenized_sentences)}")
print(f"Vocabulary size (V): {vocab_size} unique tokens (including <BOS>, <EOS>)")
print(f"Vocabulary sample  : {unique_tokens[:8]} ...")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: vocabulary mappings must be consistent and reversible
assert BOS in token2id and EOS in token2id
assert len(token2id) == vocab_size == len(id2token)
assert all(id2token[token2id[tok]] == tok for tok in unique_tokens)
print(f"Vocabulary validation passed: {vocab_size} tokens indexed bijectively.")"""
        )
    )

    # Section 4: Bigram Model
    cells.append(
        new_markdown_cell(
            """## 4. The Bigram Language Model

The Bigram model makes the **first-order Markov assumption**:
$$P(w_t \\mid w_1, \\dots, w_{t-1}) \\approx P(w_t \\mid w_{t-1})$$
The probability of the next word depends exclusively on the single word immediately preceding it.

We build a transition count matrix $C[u, v]$ and convert it into a probability matrix using **Laplace (add-1) smoothing**:
$$P(v \\mid u) = \\frac{C[u, v] + 1}{\\sum_{v'} (C[u, v'] + 1)} = \\frac{C[u, v] + 1}{\\sum_{v'} C[u, v'] + |V|}$$

Smoothing guarantees that no transition has zero probability, preventing mathematical instability."""
        )
    )

    cells.append(
        new_code_cell(
            """# Build bigram count matrix
count_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

for sent in tokenized_sentences:
    for u, v in zip(sent[:-1], sent[1:]):
        idx_u = token2id[u]
        idx_v = token2id[v]
        count_matrix[idx_u, idx_v] += 1

# Convert counts to Laplace-smoothed transition probabilities
alpha = 1.0  # Laplace smoothing parameter
smoothed_counts = count_matrix + alpha
transition_probs = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)

# Inspect high-probability transitions
print("Top 3 transitions following '<BOS>':")
bos_idx = token2id[BOS]
top_bos = np.argsort(transition_probs[bos_idx])[::-1][:3]
for idx in top_bos:
    print(f"  P({id2token[idx]:<12} | <BOS>) = {transition_probs[bos_idx, idx]:.4f}")

print("\\nTop 3 transitions following 'agent':")
agent_idx = token2id["agent"]
top_agent = np.argsort(transition_probs[agent_idx])[::-1][:3]
for idx in top_agent:
    print(f"  P({id2token[idx]:<12} | agent) = {transition_probs[agent_idx, idx]:.4f}")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: rows in transition matrix must sum to 1.0
row_sums = transition_probs.sum(axis=1)
assert np.allclose(row_sums, 1.0), "All transition probability rows must sum to 1.0"
assert transition_probs[token2id[EOS], token2id[BOS]] > 0, "Laplace smoothing guarantees non-zero probability"
print("Bigram transition probability matrix validation passed.")"""
        )
    )

    # Section 5: Sampling and Temperature
    cells.append(
        new_markdown_cell(
            """## 5. Sampling and Temperature Control

When generating text, we do not want to merely pick the single highest probability token every step (greedy decoding), as this leads to repetitive and robotic phrasing. Instead, we **sample** from the conditional distribution.

### Temperature ($T$)
The temperature hyperparameter modifies the entropy (sharpness) of the probability distribution:
$$P_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$

- **$T \\to 0$ (Greedy / Argmax)**: Always selects the single most probable token. Fully deterministic.
- **$T = 0.7$ (Low Temperature)**: Focuses probability on top candidates; reduces erratic tokens while allowing modest variation.
- **$T = 1.0$ (Neutral)**: Samples directly according to the natural data distribution.
- **$T = 1.5$ (High Temperature)**: Flattens the distribution, giving rare words a much higher chance to be selected (increases creativity and diversity, but risks incoherent nonsense)."""
        )
    )

    cells.append(
        new_code_cell(
            """def sample_next_token(probs: np.ndarray, temperature: float = 1.0) -> int:
    \"\"\"Sample next token index from probability distribution with temperature scaling.\"\"\"
    if temperature <= 1e-4:
        # Greedy decoding: argmax
        return int(np.argmax(probs))
    
    # Convert probabilities to log-scale logits: z = log(p)
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled_logits = logits / temperature
    shifted = scaled_logits - np.max(scaled_logits)  # numerical stability
    exp_logits = np.exp(shifted)
    tempered_probs = exp_logits / np.sum(exp_logits)
    
    # Sample token index from tempered distribution
    return int(np.random.choice(len(tempered_probs), p=tempered_probs))

def generate_sentence(trans_matrix: np.ndarray, temperature: float = 1.0, max_len: int = 20) -> str:
    \"\"\"Autoregressively generate a sentence starting from <BOS> until <EOS> or max_len.\"\"\"
    curr_idx = token2id[BOS]
    generated_tokens = []
    
    for _ in range(max_len):
        next_probs = trans_matrix[curr_idx]
        next_idx = sample_next_token(next_probs, temperature=temperature)
        next_tok = id2token[next_idx]
        
        if next_tok == EOS:
            break
        generated_tokens.append(next_tok)
        curr_idx = next_idx
        
    return " ".join(generated_tokens)

print("--- Text Generation Across Temperatures ---")
for temp in [0.001, 0.7, 1.5]:
    np.random.seed(SEED)
    label = "Greedy (T -> 0)" if temp < 0.01 else f"Temperature T = {temp}"
    print(f"\\n{label}:")
    for i in range(3):
        sent = generate_sentence(transition_probs, temperature=temp)
        print(f"  [{i+1}] {sent}")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: Greedy decoding is deterministic across seeds
np.random.seed(111)
s1 = generate_sentence(transition_probs, temperature=0.0001)
np.random.seed(999)
s2 = generate_sentence(transition_probs, temperature=0.0001)
assert s1 == s2, "Greedy decoding must be identical regardless of random seed"

# Quick check: Temperature alters entropy
logits = np.array([2.5, 1.2, 0.1])
p_low = np.exp(logits / 0.2) / np.sum(np.exp(logits / 0.2))
p_high = np.exp(logits / 2.0) / np.sum(np.exp(logits / 2.0))
ent_low = -np.sum(p_low * np.log2(p_low))
ent_high = -np.sum(p_high * np.log2(p_high))
assert ent_low < ent_high, "Lower temperature must have lower entropy"
print(f"Temperature sampling checks passed (Entropy: low T = {ent_low:.3f} vs high T = {ent_high:.3f}).")"""
        )
    )

    # Section 6: Negative Log-Likelihood and Perplexity
    cells.append(
        new_markdown_cell(
            """## 6. Evaluating Generative Models: Cross-Entropy and Perplexity

How do we quantitatively score how well a generative model explains a piece of text?
We evaluate the **Negative Log-Likelihood (NLL)** and **Perplexity (PPL)**:
$$\\text{Average Cross-Entropy (NLL)} = -\\frac{1}{N} \\sum_{i=1}^N \\log_2 P(w_i \\mid w_{i-1})$$
$$\\text{Perplexity} = 2^{\\text{Average NLL}}$$

- **Perplexity** intuitively measures the model's uncertainty—how many equally likely tokens the model is choosing among on average at each step.
- Lower perplexity indicates that the text matches the learned patterns of the model.
- High perplexity indicates high "surprise" (atypical or nonsensical transitions)."""
        )
    )

    cells.append(
        new_code_cell(
            """def calculate_perplexity(sentence: str, trans_matrix: np.ndarray) -> tuple[float, float]:
    \"\"\"Calculate Cross-Entropy (bits/token) and Perplexity for a sentence.\"\"\"
    words = [BOS] + sentence.strip().lower().split() + [EOS]
    total_nll = 0.0
    valid_transitions = 0
    
    for u, v in zip(words[:-1], words[1:]):
        if u in token2id and v in token2id:
            u_idx = token2id[u]
            v_idx = token2id[v]
            p = trans_matrix[u_idx, v_idx]
        else:
            # Fallback uniform probability for unseen out-of-vocab tokens
            p = 1.0 / vocab_size
            
        total_nll += -math.log2(p)
        valid_transitions += 1
        
    avg_nll = total_nll / max(1, valid_transitions)
    ppl = 2.0 ** avg_nll
    return avg_nll, ppl

# Test sentences
sent_in = "the agent learns from experience to solve complex tasks"
sent_out = "quantum bananas orbit supersonic umbrellas in space"

nll_in, ppl_in = calculate_perplexity(sent_in, transition_probs)
nll_out, ppl_out = calculate_perplexity(sent_out, transition_probs)

print(f"In-domain sentence    : '{sent_in}'")
print(f"  Cross-Entropy Loss  : {nll_in:.3f} bits/token | Perplexity: {ppl_in:.2f}")

print(f"\\nOut-of-domain sentence: '{sent_out}'")
print(f"  Cross-Entropy Loss  : {nll_out:.3f} bits/token | Perplexity: {ppl_out:.2f}")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: In-domain sentence must have lower perplexity than out-of-domain sentence
assert ppl_in < ppl_out, "In-domain sentence must have lower perplexity than out-of-domain sentence"
print(f"Perplexity check passed: In-domain ({ppl_in:.2f}) vs Out-of-domain ({ppl_out:.2f}).")"""
        )
    )

    # Section 7: Context Windows and N-gram Markov Chains
    cells.append(
        new_markdown_cell(
            """## 7. Context Windows and Higher-Order Markov Chains

The fundamental limitation of the Bigram model is its **single-token memory** ($k=1$).
Because it only knows the single preceding word, it quickly drifts into ungrammatical or disconnected statements.

To capture richer phrasing, we extend to an **N-gram model** or **$k$-th order Markov Chain**:
$$P(w_t \\mid w_{t-k}, \\dots, w_{t-1})$$
- $k=1$: Bigram (1 word context)
- $k=2$: Trigram (2 words context)
- $k=3$: 4-gram (3 words context)

Below, we build an `NGramMarkovGenerator` that accepts arbitrary context size $k$."""
        )
    )

    cells.append(
        new_code_cell(
            """class NGramMarkovGenerator:
    \"\"\"N-gram Markov Chain text generator with variable context window k.\"\"\"
    def __init__(self, order_k: int = 2):
        self.k = order_k
        self.transitions = defaultdict(Counter)
        
    def fit(self, sentences: list[str]):
        \"\"\"Train Markov chain by counting transitions from k-token prefix contexts.\"\"\"
        for sent in sentences:
            tokens = [BOS] * self.k + sent.strip().lower().split() + [EOS]
            for i in range(len(tokens) - self.k):
                prefix = tuple(tokens[i : i + self.k])
                next_tok = tokens[i + self.k]
                self.transitions[prefix][next_tok] += 1
                
    def generate(self, max_length: int = 25, seed: int = None) -> str:
        \"\"\"Autoregressively generate text conditioned on the previous k tokens.\"\"\"
        if seed is not None:
            random.seed(seed)
            
        context = [BOS] * self.k
        output = []
        
        for _ in range(max_length):
            prefix = tuple(context[-self.k :])
            next_counts = self.transitions.get(prefix)
            
            if not next_counts:
                break
                
            candidates = list(next_counts.keys())
            weights = list(next_counts.values())
            total = sum(weights)
            probs = [w / total for w in weights]
            
            next_word = random.choices(candidates, weights=probs, k=1)[0]
            if next_word == EOS:
                break
                
            output.append(next_word)
            context.append(next_word)
            
        return " ".join(output)

print("--- N-gram Markov Generation by Context Window (Order k) ---")
for order in [1, 2, 3]:
    generator = NGramMarkovGenerator(order_k=order)
    generator.fit(CORPUS)
    print(f"\\nOrder k={order} (Context size = {order} token{'s' if order > 1 else ''}):")
    for sample_i in range(2):
        text = generator.generate(seed=SEED + sample_i * 10)
        print(f"  Sample {sample_i + 1}: {text}")"""
        )
    )

    cells.append(
        new_code_cell(
            """# Quick check: Higher-order Markov generator produces coherent training-consistent text
m2 = NGramMarkovGenerator(order_k=2)
m2.fit(CORPUS)
gen_text = m2.generate(seed=42)
assert len(gen_text) > 0
for w in gen_text.split():
    assert w in token2id, f"Generated token '{w}' must exist in vocabulary"
print("NGramMarkovGenerator validation passed.")"""
        )
    )

    # Section 8: Bridge to Modern LLMs
    cells.append(
        new_markdown_cell(
            """## 8. From Markov Chains to Modern Transformer Large Language Models

Every Large Language Model (GPT-4, Llama 3, Claude, Gemini) performs the **exact same fundamental task** we built here:
$$\\text{Given a context sequence } (w_1, \\dots, w_t), \\text{ estimate } P(w_{t+1} \\mid w_1, \\dots, w_t) \\text{ and sample the next token.}$$

### The Three Fundamental Bottlenecks Solved by LLMs:
1. **The Combinatorial Explosion of N-grams**:
   An N-gram model requires storing $|V|^k$ combinations. For a typical modern vocabulary $|V| = 128,000$ and context $k = 10$, a lookup table would require $128,000^{10} \\approx 1.18 \\times 10^{51}$ entries!
   *Solution:* Neural networks replace the discrete lookup table with continuous vector embeddings and parameter matrices ($W_Q, W_K, W_V, W_{FFN}$).
2. **Context Window Limitations**:
   Markov chains discard all history prior to $k$ tokens.
   *Solution:* **Self-Attention** in Transformers computes pairwise relationships across the entire context window (up to 1,000,000+ tokens).
3. **Semantic Generalization**:
   A Markov chain treats "agent" and "bot" as completely unrelated numbers ($ID_7$ and $ID_{29}$).
   *Solution:* Continuous vector embeddings place semantically similar concepts together in high-dimensional vector space."""
        )
    )

    cells.append(
        new_code_cell(
            """# Final summary metrics
print("==================================================")
print("Generative AI Foundational Walkthrough Complete!")
print(f"Total vocabulary size  : {vocab_size} tokens")
print(f"Bigram matrix shape    : {transition_probs.shape}")
print(f"In-domain perplexity   : {ppl_in:.2f}")
print(f"Out-of-domain perplex  : {ppl_out:.2f}")
print("==================================================")"""
        )
    )

    nb.cells = cells
    return nb


def main():
    nb = make_notebook()
    print(f"Writing notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Executing notebook with nbclient to populate actual outputs...")
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(FOLDER)}},
    )
    client.execute()

    print(f"Saving populated notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Notebook built and executed successfully!")


if __name__ == "__main__":
    main()
