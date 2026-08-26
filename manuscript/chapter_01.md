# Chapter 1: The Data Pipeline — Text, Tensors, and GPU

Before a language model learns anything, your code has to answer one question that sounds almost too simple to ask: what form does text take inside a computer?

Not a philosophical question. A practical one. A neural network cannot read a sentence. It reads numbers. So the first job is to build a reliable pipeline that turns raw characters into numbers, packages them into tensors, and moves them onto a GPU where the heavy lifting will happen. That pipeline is what this chapter builds.

We are working with Shakespeare. Specifically, the *tinyshakespeare* dataset — about 1.1 million characters of Shakespeare plays, freely available and small enough to fit in memory with room to spare. It is a good starting corpus because it is rich, structured, and familiar enough that you can sanity-check outputs by eye.

By the end of this chapter you will have:

- A 95-character vocabulary built from the corpus
- Two lookup tables that translate between characters and integers in both directions
- A PyTorch tensor containing the entire encoded corpus
- A function that samples training batches on demand
- A trained bigram model and your first honest evaluation of what it can and cannot do

Open [Chapter1\_Predict\_Next\_Token.ipynb](https://github.com/asbassan/AmarLMFP/blob/main/Chapter1_Predict_Next_Token.ipynb) and follow along. Every code block below corresponds to a cell in the notebook.

## What You Will Learn

By the end of this chapter you will be able to:

- Explain how raw text becomes a sequence of integers a model can process
- Build a character vocabulary from scratch — and understand why design choices like sorting and expanding it matter for reproducibility
- Describe what a PyTorch tensor is, how it differs from a Python list, and why `dtype=torch.long` is required for token indices
- Construct a training batch: two tensors (`x`, `y`) shifted by one position, ready for next-token prediction
- Write and understand every line of a neural network training loop — the same four-step structure used in models of any size
- Interpret cross-entropy loss and perplexity as measures of model quality
- Explain *why* greedy decoding produces repetitive output and what that reveals about the nature of language modelling

These are not just Chapter 1 concepts. Every subsequent chapter in this series builds directly on them. The data pipeline, training loop, and evaluation pattern established here carry forward unchanged as we move from a 9,025-parameter bigram to a multi-million-parameter transformer.

## Getting the Corpus

```python
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "tinyshakespeare.txt")
```

One line. The file lands in your working directory as `tinyshakespeare.txt`. We strip newlines when reading it, since we are treating this as a flat stream of characters rather than a collection of lines:

```python
with open("tinyshakespeare.txt", "r", encoding="utf-8") as dataset:
    txt = dataset.read().replace('\n', '')
```

The result is a single Python string of roughly 1.07 million characters. Everything from here works on that string.

## Building the Vocabulary

A vocabulary is the set of symbols your model knows about. At the character level, that means every distinct character that appears in the corpus.

```python
from collections import Counter
import string

char_counts = Counter(txt)
chars = set(txt) | set(string.ascii_letters) | set(string.digits) | set(string.punctuation)
chars = sorted(chars)
```

Two things to notice here.

**We expand beyond what is in the text.** The `|` operator unions three sets: the characters actually in Shakespeare, all ASCII letters, and all digits and punctuation. Why include characters that do not appear in tinyshakespeare? Because we want a *stable* vocabulary. If you later run this pipeline on a different text — or even a slightly different download of the same file — you do not want the vocabulary to shift. By including the full ASCII printable range, you guarantee that character `'a'` always maps to the same integer regardless of whether `'a'` happened to appear in this particular corpus.

**We sort the result.** Sets in Python are unordered — two runs of the same code can produce different iteration orders. Sorting by ASCII value makes the vocabulary deterministic: the same text always produces the same mapping, every run, on every machine.

The output is 95 characters, from space (ASCII 32) to tilde (ASCII 126). This is our complete vocabulary.

A> **Why ASCII ordering?**
A>
A> When you call `sorted()` on a list of characters, Python sorts by the integer value of each character's Unicode code point — which for standard printable characters is identical to their ASCII value. Space comes first (32), then punctuation, then digits, then uppercase letters, then lowercase letters. The ordering is arbitrary from a linguistic standpoint, but it is consistent, and that is all that matters here.

## The Lookup Tables

With a sorted vocabulary in hand, the next step is two dictionaries that translate in both directions.

```python
vocab_size = len(chars)              # 95

stoi = {ch: i for i, ch in enumerate(chars)}   # character → integer
itos = {i: ch for i, ch in enumerate(chars)}   # integer → character

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

`stoi` means *string to integer*. `itos` means *integer to string*. They are exact inverses of each other. `encode` and `decode` apply them to full strings.

Test it immediately:

```python
encode('hello')              # → [72, 69, 76, 76, 79]
decode([72, 69, 76, 76, 79]) # → 'hello'
```

The round-trip works. `'h'` is at index 72 because ASCII 104 lands at sorted position 72 in our 95-character vocabulary. Every character has exactly one integer, and every integer maps back to exactly one character. No information is lost.

This is the simplest possible tokenizer. Modern language models — whether small, large, or in between — use subword tokenizers (BPE, WordPiece) that operate on chunks of characters rather than individual ones. The principle — assigning an integer to each vocabulary item — is identical.

## From List to Tensor

A Python list of integers is not what PyTorch needs. We convert the entire encoded corpus into a single 1D tensor:

```python
data = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
```

The `dtype=torch.long` matters. PyTorch's `nn.Embedding` — which we will use shortly — requires integer indices, and specifically 64-bit integers (`torch.long`, also called `torch.int64`). If you pass `torch.float32` by accident, you will get an error. Long integers are the standard dtype for token indices throughout PyTorch.

The tensor contains 1,075,394 integers. The entire corpus is now a single PyTorch object.

A> **What is a tensor?**
A>
A> A tensor is a multi-dimensional array with a fixed dtype and a device (CPU or GPU). A Python list is flexible but slow — it can hold any Python object, and each access is a Python operation. A tensor is a contiguous block of memory with a single dtype, which means PyTorch can operate on millions of values with a single GPU instruction. When you later call `loss.backward()`, PyTorch knows exactly which tensors participated in the computation and can differentiate through the entire chain automatically. Lists cannot do that.

## Train and Validation Split

We split the data 90 / 10:

```python
n = int(0.9 * len(data))
train_data = data[:n]    # 967,854 tokens
val_data   = data[n:]    #  107,540 tokens
```

Training data is what the model learns from. Validation data is what we use to check that learning is actually generalising rather than memorising. We never train on validation data.

The split is sequential, not random — the first 90% is training, the last 10% is validation. For text, random splitting would break context: a sequence from the middle of a play would land in training while the surrounding context landed in validation. Sequential splitting avoids that.

## Moving to the GPU

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

This line checks whether a CUDA-capable GPU is available. If it is, we use it. If not, we fall back to CPU without changing any other code. The notebook ran on an NVIDIA RTX 4060 Laptop GPU with 8.6 GB of VRAM — adequate for everything in this chapter and several that follow.

The data itself does not move to the GPU yet. We move individual batches during training, which keeps GPU memory usage low and predictable.

## Building Batches

A language model does not train on the full corpus at once. It trains on small windows called batches. Two hyperparameters control this:

```python
block_size = 8    # how many tokens the model sees at once
batch_size = 32   # how many sequences per gradient step
```

`block_size` is the context window — the maximum number of preceding characters the model can use to predict the next one. For our bigram model, this is largely irrelevant (the bigram only looks at one preceding character), but the batch machinery we build here will scale directly to any model — small or large — that uses context windows of 512, 1024, or beyond.

`batch_size` is how many independent sequences we process in parallel. Processing 32 sequences simultaneously is far more efficient than processing them one at a time, because GPUs are designed for parallel computation.

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i  : i + block_size  ] for i in ix])
    y = torch.stack([data[i+1 : i + block_size+1] for i in ix])
    return x.to(device), y.to(device)
```

`ix` is a tensor of 32 random start positions. For each position, `x` takes the 8 characters starting there, and `y` takes the 8 characters starting one step later. The result is that `y[i]` is always the correct next character after `x[i]` — the model's job is to predict `y` from `x`.

```python
xb, yb = get_batch('train')
# xb.shape → torch.Size([32, 8])
# yb.shape → torch.Size([32, 8])
# xb.device → cuda:0
```

Both tensors live on the GPU. Every training step will call `get_batch` to pull a fresh random batch.

A> **Why shift by one?**
A>
A> Language modelling is next-token prediction. Given the sequence `[h, e, l, l, o]`, we want the model to learn: after `h`, predict `e`; after `h, e`, predict `l`; and so on. Shifting `y` by one position achieves this — every position in `x` has its correct target in the corresponding position of `y`. This single shift is the core supervision signal for all autoregressive language models, from bigrams to GPT.

## A First Model: The Bigram

A bigram model predicts the next character using only the single preceding character. It is the most constrained language model you can build — and a useful one precisely because its limitations are completely transparent.

```python
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss
```

The model has exactly one layer: `nn.Embedding(vocab_size, vocab_size)`. This creates a matrix of shape (95, 95). Each row corresponds to one input character; its values are the raw scores (logits) for each possible next character. When the model sees character index 72 (`'h'`), it looks up row 72 and returns a vector of 95 numbers — one score per possible next character.

That is it. No attention, no feedforward layers, no position encodings. Just a lookup table.

The total parameter count is 95 × 95 = 9,025. This model fits in any environment, trains in seconds, and establishes a clear baseline.

**Cross-entropy loss** is the standard objective for next-token prediction. Under the hood, `F.cross_entropy` does three things in one call: it applies **softmax** to convert the raw logits into a probability distribution over the vocabulary, takes the **log** of the probability assigned to the correct character, and **averages** the negative log-probabilities across the batch. A random model over 95 characters would assign equal probability to all of them, giving a cross-entropy of log(95) ≈ 4.55 bits. Our starting loss of 4.98 is close to that — the untrained model is nearly random.

```python
model = BigramModel(vocab_size).to(device)
# Parameters: 9,025
```

## Training

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(20000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step:5d} | loss: {loss.item():.4f}")
```

The training loop has four lines that repeat at every step, and these four lines are the same in every neural network training loop you will ever write:

1. **Get a batch** — pull fresh data
2. **Forward pass** — compute predictions and loss
3. **Backward pass** — compute gradients (`loss.backward()`)
4. **Update** — adjust weights in the direction that reduces loss (`optimizer.step()`)

`optimizer.zero_grad()` clears accumulated gradients before each backward pass. PyTorch accumulates gradients by default, so without this line, gradients from previous steps contaminate the current one.

**AdamW** is the standard optimiser for modern neural networks. It adapts the learning rate for each parameter individually based on gradient history, which makes it more robust than plain gradient descent. The `lr=1e-3` (0.001) is a standard starting point.

After 20,000 steps the loss settles around 2.45:

```
step     0 | loss: 4.9827
step  1000 | loss: 4.0146
step  2000 | loss: 3.3127
step  3000 | loss: 2.8138
...
step 19000 | loss: 2.5632
Final loss: 2.4484
```

The loss dropped from near-random (4.98) to about 2.45. That is meaningful learning. But what does 2.45 actually mean?

## Evaluation: Loss and Perplexity

A single batch's loss is a noisy estimate. For a reliable evaluation we average over many batches, and we compute it separately for training and validation data:

```python
@torch.no_grad()
def estimate_loss(eval_iters=200):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

`@torch.no_grad()` tells PyTorch not to track gradients during evaluation — we are only doing inference, so there is no need to store the computation graph. This saves memory and speeds up evaluation.

`model.eval()` and `model.train()` toggle behaviour that differs between training and inference (dropout, batch normalisation). Our bigram model does not use either, but the pattern is correct and should become habit.

```
train loss: 2.5106 | perplexity: 12.31
val loss  : 2.5600 | perplexity: 12.94
```

**Perplexity** is `exp(loss)` — a more intuitive measure of model uncertainty. A perplexity of 12.31 means the model is roughly as uncertain as if it were choosing uniformly among 12 characters at each position. Our vocabulary has 95 characters; an untrained model has perplexity 95. We have brought that down to 12.

The gap between training loss (2.51) and validation loss (2.56) is small. That is good news: the model is not memorising the training data; it is generalising. The bigram is simple enough that overfitting is not really a risk.

## The Greedy Collapse

Now we generate text. The simplest generation strategy is greedy decoding: at each step, pick the character with the highest score.

```python
def generate_greedy(model, start_char, max_new_tokens=50):
    model.eval()
    idx = torch.tensor([[stoi[start_char]]], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        next_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_idx], dim=1)
    return decode(idx[0].tolist())

print(generate_greedy(model, 't'))
```

```
the the the the the the the the the the the the the
```

The model is stuck in a loop. Every time it sees `'e'`, the highest-scoring next character is a space. Every time it sees a space, the highest-scoring next character is `'t'`. So `'t' → 'h' → 'e' → ' ' → 't' → ...` forever.

This is not a bug. It is exactly what the bigram learned. In Shakespeare, `'t'` is very often followed by `'h'`, `'h'` by `'e'`, `'e'` by space, and space by `'t'` for "the". Given only one preceding character and a greedy selection, this is the most probable path.

The fix is **sampling**: instead of always picking the highest-scoring character, draw from the full probability distribution. That introduces variety and breaks the deterministic loop. Chapter 01c covers sampling strategies in detail — temperature scaling, top-k, and nucleus sampling.

A> **The bigram's known horizon**
A>
A> The bigram model has exactly one character of context. When predicting the character after `'e'`, it has no idea whether that `'e'` came from "the", "she", "he", "be", or any other word. It cannot distinguish a comma following the end of a clause from a comma inside a number. Every `'e'` in every context looks identical to it.
A>
A> This limitation is useful to understand clearly, because the rest of this book is about systematically extending that horizon — first to a fixed context window, then to a learned attention mechanism that can weight distant characters differently. The bigram gives us a concrete floor to measure against.
A>
A> We ran a controlled experiment on this corpus (and two others) to quantify exactly how much each character benefits from context beyond one preceding character. Structural characters like punctuation benefit more from longer context than letters do, at least in prose. The bigram is the model whose limitations that paper measures.
A> Paper: *Which Characters Need Context?* — [https://doi.org/10.5281/zenodo.22074823](https://doi.org/10.5281/zenodo.22074823)

## How This Connects to Modern LMs and Inference Engineering

Everything built in this chapter appears, in scaled form, in production language models. The concepts are not simplified analogies — they are the actual mechanisms, just at a smaller scale.

### Tokenization

Our character vocabulary has 95 tokens. GPT-4's BPE tokenizer has roughly 100,000. The mechanism is identical: assign an integer to each vocabulary item, encode text as a sequence of integers, decode integers back to text. The `stoi` and `itos` dictionaries you built are the conceptual equivalent of `tokenizer.encode()` and `tokenizer.decode()` in any Hugging Face tokenizer. What changes between character-level and subword tokenization is the granularity of the tokens — not the lookup-table principle underneath.

### The Embedding Table

`nn.Embedding(vocab_size, vocab_size)` in our bigram is the same layer that sits at the input of every transformer. In GPT-2 it is `nn.Embedding(50257, 768)` — 50,257 vocabulary items each mapped to a 768-dimensional vector. In LLaMA 3 it is larger still. The lookup operation — given a token index, return its row — is identical. What differs is what the embedding vectors represent and how deep the network is behind them.

### The Training Loop

The four-step loop — get batch → forward → backward → step — is universal. Training a 9,025-parameter bigram and training a 7-billion-parameter model use the same structure. The loop does not change when you scale. What changes is what happens inside the forward pass, how many GPUs share the work, and how long each step takes.

### Context Window and `block_size`

`block_size = 8` in this chapter. In LLaMA 3 the context window is 128,000 tokens. The concept is the same: how many preceding tokens can the model attend to when predicting the next one? This number has direct consequences for inference. A longer context window means larger KV-caches, more memory bandwidth consumed per token, and higher latency per request. The bigram's effective context of one character is the limiting case — it tells you exactly what a model loses when it can see almost nothing.

### Autoregressive Generation and Why Sampling Matters

The `generate_greedy()` function is the foundation of autoregressive inference. Every production language model generates text the same way: predict the next token, append it, predict again. What changes in production is:

- **Sampling strategy** — temperature, top-k, top-p instead of argmax, to avoid the greedy collapse you observed
- **KV-cache** — storing past attention computations to avoid recomputing them for every new token
- **Batched inference** — processing multiple requests simultaneously to saturate GPU throughput

The "the the the the" loop is not just a curiosity. It is the reason temperature and top-p parameters exist in every inference API you will ever use.

### Perplexity in Production

The perplexity metric you computed — `exp(loss)` — is used to benchmark production models. When a paper reports that a model achieves perplexity of 6.1 on WikiText-103, it is computing the same quantity: averaged cross-entropy, exponentiated. Your bigram's perplexity of 12.3 gives you a concrete anchor for what that number means.

### Inference Engineering

Inference engineering is the discipline of making generation fast, cheap, and reliable at scale. Context length management, batching strategy, decoding algorithm, and memory footprint all trace directly back to the mechanisms in this chapter.

## Chapter Summary

| Step | What we built | Key concept |
|---|---|---|
| Load corpus | `txt` (1.07M chars) | Raw text as a flat string |
| Build vocabulary | `chars` (95 items) | Sort for determinism |
| Lookup tables | `stoi`, `itos` | encode / decode round-trip |
| Tensor | `data` (1.07M longs) | `dtype=torch.long` for embeddings |
| Split | 90% train / 10% val | Sequential, not random |
| Device | `cuda` or `cpu` | Move batches, not the full dataset |
| Batching | `get_batch` | `x`, `y` shifted by one |
| Model | `BigramModel` | 9,025 parameters, one embedding table |
| Training | 20,000 steps, AdamW | Forward → backward → step |
| Evaluation | `estimate_loss` | Train 2.51 / Val 2.56, perplexity ~12 |
| Generation | Greedy → loop | Deterministic decoding collapses |

## What's Next

The data pipeline built in this chapter does not change as we move to more powerful models. The same `get_batch` function, the same train/val split, the same device pattern carries through every subsequent chapter. What changes is what happens between `get_batch` and `loss.backward()`.

**Chapter 01b** replaces the bigram's single-character lookup with a feedforward model that sees multiple preceding characters at once. The loss drops further and the generated text starts to look less random — because the model now has more signal to work with.

**Chapter 01c** fixes the greedy collapse directly. Temperature scaling, top-k, and nucleus sampling each give the model room to be creative rather than deterministic. By the end of Chapter 01c you will have a model that generates plausible Shakespeare-like text.

Chapters 01b and 01c together lay the groundwork for the step that changes everything: from Chapter 4 onwards, we replace the feedforward with a **self-attention layer** — the mechanism that makes transformers transformers. Self-attention lets each token look at every other token in the context window simultaneously, weighted by relevance. Everything you have built here — the batching, the training loop, the evaluation — remains exactly the same. Only the model architecture changes.

## The Journey

Every chapter in this series is one station on the way to a single destination: deploying a language model to Hugging Face.

| Station | Chapter | What we gain |
|---|---|---|
| **Station 1 — Bigram** | Chapter 1 (this chapter) | Data pipeline · training loop · baseline model |
| **Station 2 — Feedforward** | Chapter 01b | Multi-character context · lower loss |
| **Station 3 — Sampling** | Chapter 01c | Creative generation · no greedy collapse |
| **Station 4 — Self-Attention** | Chapter 4 | Transformers · long-range dependencies |
| **Station 5 — Scaling** | Later chapters | Deeper networks · better language |
| **Final Destination** | Last chapter | Model deployed to Hugging Face |

**Chapter 1 takeaways:**

- Text becomes integers through a vocabulary lookup — the same principle used in every production tokenizer
- The 4-step training loop (get batch → forward → backward → step) is universal across all model sizes
- Cross-entropy loss measures how surprised the model is by the correct answer; perplexity makes that intuitive
- Greedy decoding always picks the single most likely next character — and collapses into loops as a result
- The fix is sampling: choosing from the distribution rather than from the argmax

*The destination is a language model — whatever scale suits your goals — that you can push to Hugging Face. The path there is the data pipeline you just built, one layer at a time.*

## Review Questions

Test your understanding before moving to Chapter 01b. Try answering each question without looking back at the chapter.

1. Why do we call `sorted()` on the character set rather than using it directly? What goes wrong if you skip the sort?

2. We union three sets to build `chars`: the characters in the text, `string.ascii_letters`, and `string.digits | string.punctuation`. What problem does this solve?

3. What is the difference between `stoi` and `itos`? Write out what `encode('hi')` and `decode([72, 73])` would return given our vocabulary.

4. Why must the tensor use `dtype=torch.long`? What error would you get if you used `dtype=torch.float32` instead?

5. We split the corpus sequentially (first 90% train, last 10% val) rather than randomly. Why does random splitting cause a problem for text data?

6. In `get_batch`, why is `y` constructed as `data[i+1 : i+block_size+1]` rather than `data[i : i+block_size]`? What would happen to training if `x` and `y` were identical?

7. `F.cross_entropy` does three operations internally. Name them in order and explain what each one does.

8. A model trained on a 95-character vocabulary starts with a loss near 4.98 and ends at 2.45. What would the perplexity be at each point? What does the improvement in perplexity tell you about the model?

9. The greedy generator produces `the the the the the...` when starting from `'t'`. Trace the exact sequence of decisions that produces this loop. Why does the bigram model converge to this specific cycle?

10. The training loop calls `optimizer.zero_grad()` before `loss.backward()`. What happens if you remove that line? Would the model still train? Would it train correctly?
