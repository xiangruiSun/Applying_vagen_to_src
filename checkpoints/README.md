# Trained checkpoints (Git LFS)

Hugging Face format, bf16, full fine-tunes of `Qwen/Qwen3-0.6B` on the SRC `approach` subtask
(GRPO, LR 5e-6, 2026-11-25). Results: `results/approach_20261125/`.

| Directory | Run | Step | Success on 100 random-needle seeds |
|---|---|---|---|
| `qwen3-0.6b-compact-step30` | `VARIANT=compact` (best validation reward) | 30 | 100/100 |
| `qwen3-0.6b-rag-step60` | `VARIANT=rag` | 60 | 100/100 |

Fetch the weights with `git lfs install && git lfs pull` (a plain clone only gets pointer files).

```bash
MODEL_PATH=$PWD/checkpoints/qwen3-0.6b-compact-step30 MEM_FRACTION=0.3 MAX_MODEL_LEN=4096 bash scripts/serve_model.sh
MODEL=$PWD/checkpoints/qwen3-0.6b-compact-step30 bash scripts/eval_vagen.sh \
  configs/eval_src_approach_text_compact.yaml envs.0.harness=no_concat
```

Not included: the optimizer states (2.4 GB each, above GitHub's LFS file-size limit) and the
verl `model_*.pt` copies of the same weights, so training cannot be resumed from these.
