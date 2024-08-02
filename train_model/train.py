import os
from datasets import load_dataset
from transformers import (
    Wav2Vec2Processor,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)
# Load the Common Voice dataset
cv_17 = load_dataset(
    'mozilla-foundation/common_voice_17_0',
    'yue',
    split='train', 
    token=True,
    trust_remote_code=True
)

def prepare_dataset(batch):
  """Function to preprocess the dataset with the .map method"""
  transcription = batch["sentence"]
  
  if transcription.startswith('"') and transcription.endswith('"'):
    # we can remove trailing quotation marks as they do not affect the transcription
    transcription = transcription[1:-1]
  
  if transcription[-1] not in [".", "?", "!"]:
    # append a full-stop to sentences that do not end in punctuation
    transcription = transcription + "."
  
  batch["sentence"] = transcription
  
  return batch

cv_17 = cv_17.map(prepare_dataset, desc="preprocess dataset")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

