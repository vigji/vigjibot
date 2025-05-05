# %%
%load_ext autoreload
%autoreload 2
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

def initialize_client():
    """Initialize Polymarket CLOB client and print API credentials"""
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
    private_key = os.getenv("PK")
    host = "https://clob.polymarket.com"
    chain_id = POLYGON  # Polygon Mainnet

    client = ClobClient(host, key=private_key, chain_id=chain_id)
    api_creds = client.create_or_derive_api_creds()

    print("API Key:", api_creds.api_key)
    print("Secret:", api_creds.api_secret)
    print("Passphrase:", api_creds.api_passphrase)

    return client

def fetch_all_markets(client, max_pages=200, return_active=True):
    """Fetch all markets from Polymarket CLOB API"""
    markets = client.get_markets()
    all_markets = markets['data']
    
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
        markets = client.get_markets(next_cursor=next_cursor)
        all_markets.extend(markets['data'])
    
    df = pd.DataFrame(all_markets)
    if not return_active:
        return df
    else:
        return df[df["active"] & ~df["closed"]].reset_index(drop=True)

# Initialize client and fetch markets
client = initialize_client()
df = fetch_all_markets(client)

active_df = df[df["active"] & ~df["closed"]].reset_index(drop=True)
