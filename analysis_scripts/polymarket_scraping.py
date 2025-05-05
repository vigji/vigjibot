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
from functools import lru_cache
from datetime import datetime, timedelta

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

@lru_cache(maxsize=1)
def fetch_all_markets(client, max_pages=200, return_active=True, cache_key=None):
    """
    Fetch all markets from Polymarket CLOB API with caching.
    
    Args:
        client: ClobClient instance
        max_pages: Maximum number of pages to fetch
        return_active: Whether to return only active markets
        cache_key: Optional cache key for invalidation (timestamp)
    """
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
        return df, markets
    else:
        return df[df["active"] & ~df["closed"]].reset_index(drop=True), markets

def get_markets_with_cache(client, max_pages=200, return_active=True, cache_duration_minutes=30):
    """
    Get markets with caching, automatically invalidating cache after specified duration.
    
    Args:
        client: ClobClient instance
        max_pages: Maximum number of pages to fetch
        return_active: Whether to return only active markets
        cache_duration_minutes: How long to keep the cache valid
    """
    current_time = datetime.now()
    cache_key = current_time.replace(
        minute=current_time.minute - (current_time.minute % cache_duration_minutes),
        second=0,
        microsecond=0
    )
    
    return fetch_all_markets(client, max_pages, return_active, cache_key)

def get_market_trades(client, market_id, max_pages=200):
    """
    Fetch all trades for a specific market from Polymarket CLOB API.
    
    Args:
        client: ClobClient instance
        market_id: The ID of the market to fetch trades for
        max_pages: Maximum number of pages to fetch
        
    Returns:
        DataFrame containing all trades for the market
    """
    trades = client.get_trades(market_id=market_id)
    all_trades = trades['data']
    
    page = 0
    for _ in tqdm(range(max_pages)):
        if not trades.get('next_cursor') or trades.get('next_cursor') == "LTE=":
            print(f"No more pages to fetch")
            break
        page += 1
        if page > max_pages:
            print(f"Reached max pages: {max_pages}")
            break
        next_cursor = trades['next_cursor']
        trades = client.get_trades(market_id=market_id, next_cursor=next_cursor)
        all_trades.extend(trades['data'])
    
    return pd.DataFrame(all_trades)

# %%
# Initialize client and fetch markets
client = initialize_client()
df, markets = get_markets_with_cache(client)

active_df = df[df["active"] & ~df["closed"]].reset_index(drop=True)

# %%
active_df.columns
# %%
for i in range(100):
    # print(active_df.iloc[i, :]["tokens"])
    print(len(active_df.iloc[i, :]["tokens"]))
# %%
markets
# %%

# Get trades for a specific market
trades_df = get_market_trades(client, "your_market_id")
print(f"Number of trades: {len(trades_df)}")
