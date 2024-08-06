import sounddevice as sd
import samplerate
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModel
) # version < 4.43
from datasets import load_dataset
import librosa
import asyncio
import torch # version < 2.3
import numpy as np
import scipy



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

    def __init__(self) -> None:
        
        model_id = "simonl0909/whisper-large-v2-cantonese"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.D_TYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        self.model.to(DEVICE)

        self.processor = AutoProcessor.from_pretrained(model_id)

        assistant_model_id = "alvanlii/whisper-small-cantonese"

        if assistant_model_id != None:
            self.assistant_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                assistant_model_id,
                torch_dtype=self.D_TYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            )

            self.assistant_model.to(DEVICE)
    
    def listen_audio(self, duration=5.0, fs=INPUT_SAMPLE_RATE):
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
        
        processed_in = self.processor(audio, sampling_rate=self.MODEL_SAMPLE_RATE, return_tensors="pt")
        
        gout = self.model.generate(
            input_features=processed_in.input_features,
            output_scores=True, return_dict_in_generate=True
        )
        
        # Decode token ids to text
        transcription = self.processor.batch_decode(gout.sequences, skip_special_tokens=True)[0]
        return transcription

    async def main(self):
        audio = self.listen_audio()
        # save_audio(audio)
        # audio = import_audio("data_source\\audio_files\\世一.mp3")
        transcription = self.transcribe_audio(audio)  # Transcribe audio to text
        return transcription

class ChatboxBrain:

    DIALOGUE = [
        {"role": "system", "content": "你是名爲波子的家務助理AI，和六至十二歲小孩子聊天，並保持其心理健康。"},
        # {"role": "user", "content": "你好啊，我要食糖！"},
        # {"role": "assitant", "content": "好的，我馬上給你一粒糖。吃糖可以讓人心情愉快，但請記住，吃太多糖會對身體造成負擔。"},
        # {"role": "user", "content": "多謝你啊波子！"},
        # {"role": "assitant", "content": "不用謝！很高興能夠幫助你。如果你還有其他需要幫助的事情，請隨時告訴我。"}
    ]

    def __init__(self) -> None:

        model_id = "hon9kon9ize/CantoneseLLMChat-v0.5"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.bfloat16,
            device_map = 'auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.input_ids = self.tokenizer.apply_chat_template(
            conversation=self.DIALOGUE, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        ).to(DEVICE)

    def append_memory(self, role, content):
        self.DIALOGUE.append(
            {"role": role, "content": content}
        )

    def process_word(self, temperature=0.9, max_new_tokens=200):
        
        input_ids = self.tokenizer.apply_chat_template(
            conversation=self.DIALOGUE, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        ).to(DEVICE)

        output_ids = self.model.generate(input_ids, 
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
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        return response
    
    def main(self, input):
        input = "櫻子，令日你會去邊度玩呀？"
        self.append_memory("user", input)
        res = self.process_word()
        print("BOT: {}".format(res))
        self.append_memory("assistant", res)

class ChatboxMouth:

    D_TYPE = 'float32'

    def __init__(self) -> None:
        
        model_id = "suno/bark-small"
        
        # Load model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        
        self.model.to(DEVICE)

    def speak(self, text):
        
        processed_in = self.processor(
            text=[text], 
            return_tensors="pt"
        )
        
        # Decode token ids to text
        speech_values = self.model.generate(**processed_in)

        print(type(speech_values))
        return speech_values
    
    def save_speech(self, speech) -> None:
        audio_array = speech.cpu().numpy().squeeze()
        sampling_rate = self.model.generation_config.sample_rate
        scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=audio_array)
        return

    def main(self, text):
        speech = self.speak(text)
        self.save_speech(speech)
        

def main():
    m = ChatboxMouth()
    input = "Good morning, how are you today?"
    m.main(input)

# """Full version main()"""
# def main():
#     ear = ChatboxEar() # ASR
#     #TODO: Text Generation + Audio Generation + Interface (optional)
#     try:
#         while True:
#             text = asyncio.run(ear.main())
#             print(f"Input: {text}")
#     except KeyboardInterrupt:
#             print("\nExiting...")

if __name__ == "__main__":
    main()