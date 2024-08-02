import sounddevice as sd
import samplerate
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM, AutoTokenizer # version < 4.43
import librosa
import asyncio
import torch
import numpy as np

# Models are stored at C:\Users\{your username}}\.cache\huggingface\hub

# To whom looking at the screen, plz don't edit this

DEVICE = 'cpu'

class ChatboxEar:
    # Configurate input
    INPUT_SAMPLE_RATE = 48000
    MODEL_SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK = 1024
    SECONDS = 5
    D_TYPE = 'float32'

    # Configurate Sounddevice
    sd.default.samplerate = INPUT_SAMPLE_RATE
    sd.default.channels = 1

    #TODO: Fix audio input
    
    def record_audio(self, duration=5.0, fs=INPUT_SAMPLE_RATE):
        """Record audio for a specified duration."""
        def preprocess_audio(audio: np.ndarray):
            
            # Flatten data
            audio = audio.flatten()
            
            ratio = self.MODEL_SAMPLE_RATE / self.INPUT_SAMPLE_RATE
            
            resampled_audio = samplerate.resample(audio, ratio=ratio, converter_type='sinc_best')
            return resampled_audio
        
        print("Start recording...")
        audio = sd.rec(int(duration * fs), blocking=True, dtype=self.D_TYPE)
        print("End recording")
        
        audio = preprocess_audio(audio)
        
        return audio

    def save_audio(self, audio: np.ndarray):
        # Save the recording to a WAV file
        from scipy.io.wavfile import write
        output_filename = "output.wav"
        write(output_filename, self.MODEL_SAMPLE_RATE, audio.astype(self.D_TYPE))
        print(f"Recording saved to {output_filename}")

    # def import_audio(filepath: str):
    #     y, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    #     return y

    def transcribe_audio(self, audio):
        
        model_id = "simonl0909/whisper-large-v2-cantonese"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.D_TYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        model.to(self.DEVICE)

        processor = AutoProcessor.from_pretrained(model_id)

        assistant_model_id = "alvanlii/whisper-small-cantonese"
        assistant_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            assistant_model_id,
            torch_dtype=self.D_TYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )

        assistant_model.to(self.DEVICE)
        
        processed_in = processor(audio, sampling_rate=self.MODEL_SAMPLE_RATE, return_tensors="pt")
        
        gout = model.generate(
            input_features=processed_in.input_features,
            output_scores=True, return_dict_in_generate=True
        )
        
        # Decode token ids to text
        transcription = processor.batch_decode(gout.sequences, skip_special_tokens=True)[0]
        return transcription

    async def main(self):
        audio = self.record_audio()
        # save_audio(audio)
        # audio = import_audio("data_source\\audio_files\\世一.mp3")
        transcription = self.transcribe_audio(audio)  # Transcribe audio to text
        return transcription

DIALOGUE = [
    {"role": "system", "content": "你叫做櫻子，你要同用家北原伊織進行對話，你同北原伊織係情侶關係。"},
    # {"role": "user", "content": "櫻子，令日你會去邊度玩呀？"}
]

def append_memory(role, content):
    DIALOGUE.append(
        {"role": role, "content": content}
    )

def process_word(temperature=0.9, max_new_tokens=200):
    
    model_id = "hon9kon9ize/CantoneseLLMChat-v0.5"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype = torch.bfloat16,
        device_map = 'auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    input_ids = tokenizer.apply_chat_template(
        conversation=DIALOGUE, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors='pt'
    ).to(DEVICE)
    
    output_ids = model.generate(input_ids, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        num_return_sequences=1, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        num_beams=3, 
        repetition_penalty=1.18
    )
    
    print(output_ids)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    return response

def main():
    input = "櫻子，令日你會去邊度玩呀？"
    append_memory("user", input)
    res = process_word()
    print("BOT: {}".format(res))
    append_memory("assistant", res)

"""Full version main()"""
# def main():
#     ear = ChatboxEar() # ASR
#     #TODO: Text Generation + Audio Generation + Interface (optional)
#     try:
#         while True:
#             text = asyncio.run(ear.main())
#             print(f"Transcription: {text}")
#     except KeyboardInterrupt:
#             print("\nExiting...")

if __name__ == "__main__":
    main()