from transformers import pipeline
MODEL_NAME = "alvanlii/whisper-small-cantonese" 
lang = "zh"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
text = pipe("data_source\\audio_files\\世一.mp3")["text"]