# Sparsely Wired Mortal LLM Inference

Official implementation of **Moira**, from the NeurIPS 2026 paper *Sparsely Wired Mortal LLM Inference*.

<!-- TODO: add paper link (arXiv / OpenReview) once available -->

Moira prepares an LLM to be hardwired as shift-accumulate circuits. Each FP16 weight is
factorized into signed power-of-two bases, so multiplication becomes a shift and each base
becomes one physical wire. The catch is that representing weights this way makes the number of
wires explode. Moira picks only the bases that are worth keeping under a wire budget
`N_target` (the average number of wires per weight). It needs no retraining.

- **Greedy weight factorization (Algorithm 1).** Each weight is rebuilt from its residual one
  base at a time: `b = sign(r) * 2^round(log2|r|)`. A new base is kept only if it lowers the
  activation-weighted residual by more than the wire price `λ`. A weight that keeps zero bases
  is pruned, and one that keeps only a few bases is effectively quantized.
- **Wire-budget fitting (Algorithm 2).** For each linear layer, `λ` is found by binary search so
  that the average number of selected bases `N̄` matches `N_target`.

With about two bases per weight on average, Moira stays near-lossless on LLaMA / LLaMA-2 models
from 7B to 70B / Mistral 7B / Qwen3.5 27B.

## Repository structure

| File | Contents |
|---|---|
| `moira.py` | The `Moira` class: activation statistics collection, greedy weight factorization (Alg. 1), wire-budget fitting (Alg. 2) |
| `main.py` | End-to-end pipeline: load model → vanilla PPL → Moira → Moira PPL → (optional) zero-shot evaluation |
| `utils.py` | Per-basis wire distribution export and result logging |

## Installation

Tested with Python 3.10.12.

```bash
pip install -r requirements.txt
```

LLaMA-2 checkpoints are gated on the Hugging Face Hub. Request access, then log in:

```bash
huggingface-cli login   # or: export HF_TOKEN=<your token>
```

## Quick start

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --data wikitext2 \
    --target_wires 2.0
```

Results are written to three directories, created automatically:

| Directory | Contents |
|---|---|
| `base_dist/` | Wire count per power-of-two basis, before and after Moira |
| `converge/` | λ binary search log for each layer |
| `results/` | Vanilla and Moira PPL, wire counts, and zero-shot accuracy |

## Reproducing the paper

### Target wire budgets

The paper uses `N_target = 1.5` for the comparison with pruning and `N_target = 1.1` for the
comparison with quantization. To sweep several budgets, including zero-shot evaluation:

```bash
MODEL=meta-llama/Llama-2-7b-hf
for N in 3.0 2.5 2.0 1.5 1.1 1.0; do
    python main.py --model $MODEL --data wikitext2 --target_wires $N \
        --save_model ./ckpt/${MODEL##*/}-moira-$N --eval_zero_shot
done
```

Each run saves a full FP16 checkpoint (about 13 GB for a 7B model).

### Models and hardware

| Model | Hugging Face ID | GPUs used in the paper (H100) |
|---|---|---|
| LLaMA-2 7B | `meta-llama/Llama-2-7b-hf` | 1 |
| LLaMA-2 13B | `meta-llama/Llama-2-13b-hf` | 1 |
| LLaMA 30B | `huggyllama/llama-30b` | 4 |
| LLaMA 65B | `huggyllama/llama-65b` | 8 |
| LLaMA-2 70B | `meta-llama/Llama-2-70b-hf` | 8 |
| Mistral 7B | `mistralai/Mistral-7B-v0.3` | 1 |
| Qwen3.5 27B | `Qwen/Qwen3.5-27B` | 8 |

The model is split across all visible GPUs. Choose which ones with `CUDA_VISIBLE_DEVICES`. The
Moira search itself processes one layer at a time on a single GPU (`cuda:0`). It falls back to
the CPU if no GPU is available, which is much slower.

### Zero-shot evaluation

```bash
python main.py --model meta-llama/Llama-2-7b-hf --data wikitext2 --target_wires 1.5 \
    --save_model ./ckpt/Llama-2-7b-hf-moira-1.5 --eval_zero_shot
```

The reconstructed model is saved first and then evaluated with LM-Eval-Harness on
BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. The run fails at this
step if `--save_model` is not given. All earlier results are already saved by then.

## Notes

- The evaluated model holds the reconstructed weights `W̃` in FP16. 
- This pre-simulates the model performance of the hardwired circuit. The circuit itself is not part of this repository.

## License

This repository is released under the **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)** license. Read [LICENSE.md](LICENSE.md) for the full terms
before using this code.

- **NonCommercial.** Commercial use is not permitted under this license.
- **Attribution.** Credit this work by citing the paper below.
- **ShareAlike.** Modified versions must be distributed under the same license.

Model weights downloaded from the Hugging Face Hub (e.g., LLaMA-2, Mistral, Qwen3.5) are subject
to their own licenses.

## Citation

<!-- TODO: fill in proceedings details after camera-ready -->
```bibtex
@article{park2026sparsely,
  title     = {Sparsely Wired Mortal {LLM} Inference},
  author    = {Park, Mincheol and Lee, Sunwoo and Lee, Sukjin and Yang, Wooram and
               Tai, Yongmin and Kim, Sang Joon and Yu, Jaehoon},
  journal = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
