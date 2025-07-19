from llama_cpp import Llama

model_path = "llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

prompt = "Write a Python code to add two numbers."
output = model(prompt, max_tokens=128)
print(output["choices"][0]["text"])