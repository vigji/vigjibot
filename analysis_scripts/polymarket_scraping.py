# %%
import os
import dotenv
from pathlib import Path
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from regex import D
import requests
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import numpy as np
from openai import OpenAI

# %%

# Set your private key as an environment variable or replace os.getenv("PK") with your key string
dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
private_key = os.getenv("PK")
host = "https://clob.polymarket.com"
chain_id = POLYGON  # Polygon Mainnet

# Initialize the client
client = ClobClient(host, key=private_key, chain_id=chain_id)

# Generate API credentials
api_creds = client.create_or_derive_api_creds()

# Access the credentials
print("API Key:", api_creds.api_key)
print("Secret:", api_creds.api_secret)
print("Passphrase:", api_creds.api_passphrase)


from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

# Assuming client is already initialized as shown above

# %% 
# %%
# Fetch open markets
markets = client.get_markets()
all_markets = markets['data']

max_pages = 100
page = 0
for _ in tqdm(range(max_pages)):
    if not markets.get('next_cursor') or markets.get('next_cursor') == "LTE=":
        print(f"No more pages to fetch")
        break
    page += 1
    if page > max_pages:
        print(f"Reached max pages: {max_pages}")
        break
    next_cursor = markets['next_cursor']
    print(f"Fetching page {page} with cursor {next_cursor}")
    markets = client.get_markets(next_cursor=next_cursor)
    all_markets.extend(markets['data'])

df = pd.DataFrame(all_markets)
active_df = df[df["active"] & ~df["closed"]].reset_index(drop=True)
active_df["num_tokens"] = active_df.apply(lambda x: len(x["tokens"]), axis=1)
print(active_df["num_tokens"].value_counts())

# %%
active_df["question"].to_list()  #["question_text"] = active_df["question"].apply(lambda x: x["text"])
# %%
# %%
model = "BAAI/bge-m3"

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_TOKEN"),
    base_url="https://api.deepinfra.com/v1/openai",
)


os.getenv("DEEPINFRA_TOKEN")
questions_list = active_df["question"].to_list()
sanitized_questions = [q.strip() for q in questions_list]

# all_embeddings = []
# for q in tqdm(sanitized_questions):
#     if not q or len(q) < 2 or len(q) > 1000:
#         print(q)
#         all_embeddings.append(np.zeros(1024)*np.nan)
#     else:
#         embeddings = openai.embeddings.create(
#             model=model, input=q, encoding_format="float"
#         )
#         all_embeddings.append(embeddings.data[0].embedding)
cache_file = f"embeddings_polymarket_cache_{model.replace('/', '.')}.csv"
chunk_size = 100
if (Path(__file__).parent / cache_file).exists():
    embeddings_df = pd.read_csv(cache_file, index_col=0)
    embeddings_array = embeddings_df.to_numpy()
    
else:
    embeddings_list = []
    for i in tqdm(range(0, len(sanitized_questions), chunk_size)):
        chunk = sanitized_questions[i:i+chunk_size]
        embeddings = openai.embeddings.create(
            model=model, input=chunk, encoding_format="float"
        )
        embeddings_list.extend(embeddings.data)

    embeddings_array = np.array([embedding.embedding for embedding in embeddings_list])
    embeddings_array.shape
# %%
embeddings_df = pd.DataFrame(embeddings_array, index=active_df["question_id"])
embeddings_df.to_csv(cache_file)
print(embeddings_array.shape)

# %%
meta_questions_df = pd.read_csv("questions_df.csv")
# %%
meta_questions_df[0]

# %%
