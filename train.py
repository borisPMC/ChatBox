from datasets import load_dataset
from transformers import AutoTokenizer, BarkModel, TrainingArguments, Trainer, BertTokenizer
import numpy as np
import evaluate
import librosa
import torch

dataset = load_dataset("mozilla-foundation/common_voice_17_0", "yue")
model_id = "suno/bark-small"

pretrained_model = BarkModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id) # https://github.com/suno-ai/bark/tree/main/bark/assets/prompts

print(dataset["train"])

def preprocess(row):
    # preprocess text
    row["input_ids"] = tokenizer.encode(row["sentence"], padding="max_length", truncation=True, return_tensors="pt")
    row["labels"] = torch.from_numpy(librosa.resample(row["audio"]["array"], orig_sr=row["audio"]["sampling_rate"], target_sr=16000))
    return row

remove_columns = ['client_id', 'sentence', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant']

train_dataset = dataset["train"].shuffle(seed=42).map(preprocess, remove_columns=remove_columns)
eval_dataset = dataset["test"].shuffle(seed=42).map(preprocess, remove_columns=remove_columns)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="cantonese_bark",
    eval_strategy="epoch",
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./log",
    logging_steps=10,
)

trainer = Trainer(
    model=pretrained_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("Start training...")

trainer.train()

print("Complete")