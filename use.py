from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEVICE = 'cpu'

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
    print(DIALOGUE)