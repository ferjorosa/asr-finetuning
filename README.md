# asr-finetuning

A learning repo for fine-tuning automatic speech recognition (ASR) models. Currently supporting only Whisper-like models because it's the most common architecture, but the goal is to explore and implement others over time. 

The project started with me testing this [Unsloth's Whisper notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Whisper.ipynb) but then decided to create a proper repo using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## What is currently implemented

- **Pipeline for fine-tuning and LoRA**. Both full parameter fine-tuning and Low-Rank Adaptation are supported. Pytorch lightning handles the training loop, providing abstractions like  gradient clipping, and checkpointing.
- **Custom LoRA implementation**. A from-scratch LoRA implementation. This was chosen over PEFT or Unsloth due observed to compatibility issues with PyTorch Lightning with respect to gradient checkpointing.
- **Trackio logger**. Used for experiment tracking (metrics, configs, logs). There is not currently an official Lightning implementation, [using the one from this PR](https://github.com/Lightning-AI/pytorch-lightning/pull/21521).

## Quick start

```bash
# Run training with default configs
bash run_training.sh

# Or run with custom configs
uv run python scripts/train.py --model-config configs/models/whisper-large-v3-lora.yaml --training-config configs/training/whisper-large-v3-lora-elise.yaml --data-config configs/data/elise.yaml
```

The default setup trains `openai/whisper-large-v3` with LoRA on the [Elise dataset](https://huggingface.co/datasets/MrDragonFox/Elise):

## Next steps

- I am interested in exploring other architectures like the one used in [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2).
- I would like to make Unsloth / PEFT work with the project.
