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

    def __init__(self, model_id, assistant_model_id) -> None:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.D_TYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        self.model.to(DEVICE)

        self.processor = AutoProcessor.from_pretrained(model_id)

        if assistant_model_id != None:
            self.assistant_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                assistant_model_id,
                torch_dtype=self.D_TYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            )

            self.assistant_model.to(DEVICE)

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
        
        processed_in = self.processor(audio, sampling_rate=self.MODEL_SAMPLE_RATE, return_tensors="pt")
        
        gout = self.model.generate(
            input_features=processed_in.input_features,
            output_scores=True, return_dict_in_generate=True
        )
        
        # Decode token ids to text
        transcription = self.processor.batch_decode(gout.sequences, skip_special_tokens=True)[0]
        return transcription

    async def main(self):
        audio = self.record_audio()
        # save_audio(audio)
        # audio = import_audio("data_source\\audio_files\\世一.mp3")
        transcription = self.transcribe_audio(audio)  # Transcribe audio to text
        return transcription

class ChatbotWernicke:
    
    DIALOGUE = [
        {"role": "system", "content": "你是名爲波子的家務助理AI，和六至十二歲小孩子聊天，並保持其心理健康。"},
        {"role": "user", "content": "你好啊，我要食糖！"},
        {"role": "assitant", "content": "好的，我馬上給你一粒糖。吃糖可以讓人心情愉快，但請記住，吃太多糖會對身體造成負擔。"},
        {"role": "user", "content": "多謝你啊波子！"},
        {"role": "assitant", "content": "不用謝！很高興能夠幫助你。如果你還有其他需要幫助的事情，請隨時告訴我。"}
    ]
    
    def __init__(self, model_id) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.bfloat16,
            device_map = 'auto',
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            device_map = 'auto',
            trust_remote_code=True,
        )
        
    def process_dialogue(self, temperature=0.9, max_new_tokens=200):
        
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
        
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        return response
    
    def append_memory(self, role, content):
        self.DIALOGUE.append(
            {"role": role, "content": content}
        )
        return
        
    def main(self, input):
        self.append_memory("user", input)
        res = self.process_dialogue()
        self.append_memory("assistant", res)
        return res

def main():
    
    wernicke = ChatbotWernicke("stvlynn/Qwen-7B-Chat-Cantonese")
    # mouth = ChatboxMouth()
    
    try:
        while True:
            input_text = input("\nSpeak to AI: ")
            res = wernicke.main(input_text)
            print("BOT: {}".format(res))
    except KeyboardInterrupt:
        print("\nExiting...")

"""Full version main()"""
# def main():
#     ear = ChatboxEar("simonl0909/whisper-large-v2-cantonese", "alvanlii/whisper-small-cantonese") # ASR
#     #TODO: Text Generation + Audio Generation + Interface (optional)
#     try:
#         while True:
#             text = asyncio.run(ear.main())
#             print(f"Transcription: {text}")
#     except KeyboardInterrupt:
#         print("\nExiting...")

if __name__ == "__main__":
    main()