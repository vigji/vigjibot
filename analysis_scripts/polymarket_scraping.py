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

# %%
sel_q = active_df.loc[0, :]
sel_q
# %%
df["rewards"]
# %%
sel_q["tokens"]
# %%
q_id = 1746093315664

# %%
sel_q["question_id"]
# %%
df["question_id"]
# %%# %%
sel_q = active_df[active_df["market_slug"] == "us-recession-in-2025"]
# %%
sel_q.to_dict()
# %%
