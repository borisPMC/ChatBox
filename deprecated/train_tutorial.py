from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate, numpy as np

# To push the model online, login by typing "huggingface-cli login" on cmd


# We are going to fine-tune this model to achieve the result we want.
PRETRAINED_MODEL = "google-bert/bert-base-cased"


# PART 1: Preprocess Data (For text, use tokenization)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Method 1: Raw method
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]

# Method 2: Load from HuggingFace Hub
dataset = load_dataset("yelp_review_full")
def tokenize(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize, batched=True)

# Randomly select partial for training and testing
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# PART 1 END


# PART 2: Setting training arguments
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=5)
training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    num_train_epochs=1,
    push_to_hub=True,           # Push trained model online
    # Tune hyperparameters for the training
    )
# PART 2 END


# PART 3: Evaluation
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# PART 3 END


# MAIN FUNCTION
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("test_trainer")  # Save locally
model.push_to_hub("test_trainer")       # Push model to HuggingFace
tokenizer.push_to_hub("test_trainer")   # Push Tokenizer