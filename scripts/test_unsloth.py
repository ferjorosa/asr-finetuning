from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
import torch

# Load the original model
base_model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/whisper-large-v3",
    dtype = None, # Leave as None for auto detection
    load_in_4bit = False, # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "English",
    whisper_task = "transcribe",
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

# Load the LoRA adapters so we only need to update the weights of the LoRA adapters
peft_model = FastModel.get_peft_model(
    base_model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = None, # ** MUST set this for Whisper **
)

# Data prep
import numpy as np
import tqdm

#Set this to the language you want to train on
peft_model.generation_config.language = "<|en|>"
peft_model.generation_config.task = "transcribe"
peft_model.config.suppress_tokens = []
peft_model.generation_config.forced_decoder_ids = None

def formatting_prompts_func(example):
    audio_arrays = example['audio']['array']
    sampling_rate = example["audio"]["sampling_rate"]
    features = tokenizer.feature_extractor(
        audio_arrays, sampling_rate = sampling_rate
    )
    tokenized_text = tokenizer.tokenizer(example["text"])
    return {
        "input_features": features.input_features[0],
        "labels": tokenized_text.input_ids,
    }

from datasets import load_dataset, Audio
dataset = load_dataset("MrDragonFox/Elise", split = "train")

dataset = dataset.cast_column("audio", Audio(sampling_rate = 16000))
dataset = dataset.train_test_split(test_size = 0.06)
train_dataset = [formatting_prompts_func(example) for example in tqdm.tqdm(dataset['train'], desc = 'Train split')]
test_dataset = [formatting_prompts_func(example) for example in tqdm.tqdm(dataset['test'], desc = 'Test split')]

# Create compute_metrics and datacollator

# @title Create compute_metrics and datacollator
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pdb

metric = evaluate.load("wer")
def compute_metrics(pred):

    pred_logits = pred.predictions[0]
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id


    pred_ids = np.argmax(pred_logits, axis = -1)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens = True)

    wer = 100 * metric.compute(predictions = pred_str, references = label_str)

    return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors = "pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors = "pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Train model

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bf16_supported
trainer = Seq2SeqTrainer(
    model = peft_model,
    train_dataset = train_dataset,
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor = tokenizer),
    eval_dataset = test_dataset,
    tokenizer = tokenizer.feature_extractor,
    compute_metrics = compute_metrics,
    args = Seq2SeqTrainingArguments(
        # predict_with_generate = True,
        per_device_train_batch_size = 96,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        weight_decay = 0.001,
        remove_unused_columns = False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        eval_steps = 5 ,
        eval_strategy = "steps",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc

    ),
)

trainer_stats = trainer.train()