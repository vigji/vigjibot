from openai import OpenAI

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key="uXt3RgNMGA1zlNdqHMEJrjgrpvmSpqtl",
    base_url="https://api.deepinfra.com/v1/openai",
)

input = "The food was delicious and the waiter...", # or an array ["hello", "world"]

embeddings = openai.embeddings.create(
  model="BAAI/bge-m3",
  input=input,
  encoding_format="float"
)

if isinstance(input, str):
    print(embeddings.data[0].embedding)
else:
    for i in range(len(input)):
        print(embeddings.data[i].embedding)

print(embeddings.usage.prompt_tokens)