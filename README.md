# LM from First Principles
### A Notebook-Driven Guide to Building and Deploying Language Models

**Author:** Amarpreet Singh Bassan
**Book:** [Leanpub — LM from First Principles](https://leanpub.com/lm-from-first-principles) *(coming soon)*

---

## What This Is

This repository contains all notebooks, code, and resources for the book
**LM from First Principles: A Notebook-Driven Guide to Building and Deploying Language Models**.

The book builds a language model step by step — starting from a 9,025-parameter
bigram and ending with a model deployed to Hugging Face. Every chapter is driven
by a runnable Jupyter notebook. The goal is not just to show you the code, but to
make sure you understand the *why* behind every decision.

You do not need to be an ML researcher to follow along. You need Python, curiosity,
and a willingness to run the notebooks yourself.

---

## Prerequisites

### Knowledge

| Topic | Required level |
|---|---|
| Python | Comfortable with functions, classes, dictionaries, list comprehensions |
| Linear algebra | Basic — vectors, matrices, dot products |
| Probability | Basic — what a probability distribution is |
| Neural networks | Helpful but not required — we build intuition from scratch |

If you can read and write Python functions and you know what a matrix is,
you are ready to start.

### Hardware

A GPU is strongly recommended for chapters involving training. All notebooks
detect hardware automatically and fall back to CPU if no GPU is available —
but training on CPU is significantly slower for later chapters.

The notebooks were developed and tested on:
- **GPU:** NVIDIA RTX 4060 Laptop (8.6 GB VRAM)
- **OS:** Windows 11

Any CUDA-capable GPU with 6 GB+ VRAM will work. Google Colab (free tier)
is a viable alternative if you do not have a local GPU.

### Software

**Python 3.10 or later** is required.

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter numpy matplotlib
```

> If you do not have a CUDA GPU, install the CPU-only version of PyTorch:
> ```bash
> pip install torch torchvision
> pip install jupyter numpy matplotlib
> ```

To verify your setup:

```python
import torch
print(torch.__version__)
print("GPU available:", torch.cuda.is_available())
```

---

## Repository Layout

```
Chapter1_Predict_Next_Token.ipynb   ← Data pipeline: text → tensors → GPU
                                       Bigram model, training, evaluation,
                                       greedy generation
```

More chapters will be added as the book is written.

---

## Chapter Overview

| Chapter | Notebook | Topic |
|---|---|---|
| 1 | `Chapter1_Predict_Next_Token.ipynb` | Data pipeline · Bigram model · Training · Evaluation |
| 2 | *(coming soon)* | Extending context · Feedforward model |
| 3 | *(coming soon)* | Self-attention from scratch |
| 4 | *(coming soon)* | The transformer block |
| 5+ | *(coming soon)* | Scaling · Fine-tuning · Deploying to Hugging Face |

---

## Running the Notebooks

```bash
git clone https://github.com/asbassan/LMFP.git
cd LMFP
jupyter notebook
```

Open the chapter notebook and run cells top to bottom.
Each notebook is self-contained — datasets are downloaded automatically.

---

## Citation

```
@book{bassan2026lmfp,
  title  = {LM from First Principles: A Notebook-Driven Guide to
             Building and Deploying Language Models},
  author = {Bassan, Amarpreet Singh},
  year   = {2026},
  url    = {https://github.com/asbassan/LMFP}
}
```

---

## License

Code in this repository is released under the [MIT License](LICENSE).
Book text and prose content are copyright Amarpreet Singh Bassan, 2026.
