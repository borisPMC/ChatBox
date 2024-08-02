# With pipeline, just specify the task and the model id from the Hub.
# ALL models are stored in C:\Users\BSIT\.cache\huggingface\hub (and quite large)
from transformers import pipeline, LlamaTokenizer
import torch
MODEL_NAME = "WayneLinn/Whisper-Cantonese" 
lang = "zh"
audio_pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
)
audio_pipe.model.config.forced_decoder_ids = audio_pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
text = audio_pipe("audio_files\世一.mp3")["text"]
