import torch
import tiktoken
from The_Guts import GPTModel, format_input, text_to_token_ids, token_ids_to_text, generate

# Use EXACTLY the same config as in training
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,   # or 128 if you changed it
    "drop_rate": 0.1,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}

model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("gpt2/gpt-files/gpt2-small124M-sft-standalone(p0).pth", map_location='cpu'))
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# test a few known instructions
test_instructions = [
    "Why are the pumpkins on my plant rotting before they mature?",
]

for instr in test_instructions:
    prompt = format_input({"instruction": instr, "input": ""})
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=50,
        context_size=BASE_CONFIG["context_length"],
        temperature=0.0,
        top_k=None,
        eos_id=50256,
        stop_strings=None,
        tokenizer=tokenizer,
    )
    output = token_ids_to_text(token_ids, tokenizer)
    response = output[len(prompt):].strip()
    print(f"{instr:50s}  ->  {response}")