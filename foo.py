from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "Once upon a time there was a little orange cat named"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

# output = model(input_ids, max_length = 1000, num_beams=1, output_hidden_states=True)
output = model(input_ids, output_hidden_states=True)
print(type(output))

# output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print(output_text)
