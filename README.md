# GPT_basic



````markdown
# GPT-Adder: A Deep Dive into Learning Arithmetic with Transformers

## Overview  
**GPT-Adder** explores how a GPT-style transformer can be trained to perform integer addition as a classification task—mapping complete “_a + b =_” prompts directly to their numeric sum. Rather than the usual autoregressive, token-by-token generation, we treat each addition problem as a single-step classification, where every possible answer is its own class. :contentReference[oaicite:0]{index=0}

## Features  
- **Sequence-to-Single-Token Prediction**: Input is the full prompt (`"2+3="`), output is a single token (`"5"`). :contentReference[oaicite:1]{index=1}  
- **Error Analysis**: In-depth breakdown of when and why the model errs, including “carry” cases.  
- **Hyperparameter Sweeps**: Five controlled experiments adjusting learning rate, embedding size, number of layers, and dropout.  

## Requirements  
- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [Transformers](https://huggingface.co/transformers/)  
- NumPy  
- Pandas  
- Matplotlib  

Install with:

```bash
pip install torch transformers numpy pandas matplotlib
````

## Configuration

All key hyperparameters are defined at the top of the notebook:

```python
batch_size     = 32       # sequences per batch  
max_iters      = 10000    # total training iterations  
eval_interval  = 250      # evaluate every N steps  
learning_rate  = 5e-4     # optimizer LR  
device         = 'cpu'    # or 'cuda'  
n_embd         = 128      # embedding dimension  
n_head         = 4        # attention heads  
n_layer        = 4        # transformer layers  
dropout        = 0.2      # dropout probability  
```



## Project Sections

1. **Part 1: In-depth Error Analysis**

   * Detects when addition requires carrying and analyzes model mistakes.

2. **Part 2: GPT-Adder Tutorial**

   * Walks through training a transformer for the classification-style addition task.

3. **Part 3: Hyperparameter Experiments**

   * Runs five ablations (A–E) to see how each choice affects accuracy and loss.

## Results Summary

| Config               | Train Acc | Val Acc | Train Loss | Val Loss |
| -------------------- | --------: | ------: | ---------: | -------: |
| **A: baseline**      |    0.8574 |  0.8573 |     0.4916 |   0.4916 |
| **B: lr = 1e-3**     |    0.8574 |  0.8499 |     0.4956 |   0.5018 |
| **C: embd = 256**    |    0.8558 |  0.8521 |     0.4940 |   0.4963 |
| **D: layers = 6**    |    0.8544 |  0.8529 |     0.4915 |   0.4961 |
| **E: dropout = 0.0** |    0.8573 |  0.8548 |     0.4941 |   0.4950 |

*All experiments run 5,000 training steps, evaluated every 250 steps (500 batches per eval).*&#x20;

## Observations

* **Carrying Digits** significantly increases error rate compared to “no-carry” cases.
* **Larger Embeddings** (256 vs 128) yield only marginal gains.
* **Higher Learning Rates** can hurt stability on the validation set.

*(See notebook for full plots and per-digit breakdowns.)*

## Usage

1. Clone this repo.
2. Install requirements.
3. Launch JupyterLab and open `GPT-Adder_Final_Project.html` (or the accompanying `.ipynb`).
4. Run all cells to reproduce experiments and figures.

## License

MIT © 2025

```

Feel free to adjust any sections—such as adding badges, dataset details, or links to pre-trained checkpoints—as needed!
```
