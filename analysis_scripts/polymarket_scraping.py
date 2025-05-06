# %%
%load_ext autoreload
%autoreload 2
import os
import dotenv
from pathlib import Path
from py_clob_client.client import ClobClient, TradeParams
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

# Insert new Gamma API functions here
GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"

@lru_cache(maxsize=1)
def fetch_all_markets_gamma(return_active=True, cache_key=None, limit_per_page=100, max_requests=200):
    """
    Fetch all markets from Polymarket Gamma API.

    Args:
        return_active: Whether to return only active markets (where closed is False).
        cache_key: Optional cache key for invalidation (timestamp), used by lru_cache.
                       This argument is present to be compatible with the caching strategy.
        limit_per_page: Number of markets to fetch per API call.
        max_requests: Maximum number of API requests to prevent infinite loops.
    """
    all_markets_data = []
    offset = 0
    
    print("Fetching markets from Gamma API...")
    # Using range for loop control, tqdm can be wrapped around it if desired
    for i in range(max_requests): 
        params = {"limit": limit_per_page, "offset": offset}
        try:
            response = requests.get(f"{GAMMA_API_BASE_URL}/markets", params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if not data: 
                print(f"No more markets to fetch after {i+1} requests (offset {offset}).")
                break
            
            all_markets_data.extend(data)
            
            if len(data) < limit_per_page: 
                print(f"Fetched last page of markets ({len(data)} items) in request {i+1}.")
                break
                
            offset += limit_per_page
        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets from Gamma API on request {i+1} (offset {offset}): {e}")
            break
        except ValueError as e: 
            print(f"Error decoding JSON from Gamma API on request {i+1} (offset {offset}): {e}")
            break
        if i == max_requests -1: # Check if loop finished due to max_requests
            print(f"Reached max_requests limit ({max_requests}) for Gamma API.")


    if not all_markets_data:
        print("No market data fetched from Gamma API.")
        return pd.DataFrame(), None

    df = pd.DataFrame(all_markets_data)
    
    if return_active:
        if "closed" in df.columns:
            df = df[~df["closed"]].reset_index(drop=True)
        else:
            print("Warning: 'closed' column not found in Gamma API market data. Cannot filter for active markets.")
            
    return df, None # Second element is None for consistency with original tuple return, meaning changed.

def get_markets_with_cache_gamma(return_active=True, cache_duration_minutes=30, limit_per_page=100, max_requests=200):
    """
    Get markets from Gamma API with caching, automatically invalidating cache after specified duration.
    
    Args:
        return_active: Whether to return only active markets.
        cache_duration_minutes: How long to keep the cache valid.
        limit_per_page: Number of markets to fetch per API call for fetch_all_markets_gamma.
        max_requests: Maximum number of API requests for fetch_all_markets_gamma.
    """
    current_time = datetime.now()
    cache_key_time = current_time.replace(
        minute=current_time.minute - (current_time.minute % cache_duration_minutes),
        second=0,
        microsecond=0
    )
    
    return fetch_all_markets_gamma(
        return_active=return_active, 
        cache_key=cache_key_time, 
        limit_per_page=limit_per_page, 
        max_requests=max_requests
    )

# %%
# Initialize client and fetch markets
# client = initialize_client() # This initializes ClobClient. Keep/uncomment if other parts of the script use it.

# Old way using CLOB API (commented out):
# df, markets = get_markets_with_cache(client) # Default return_active=True, so df was active markets
# active_df_clob = df[df["active"] & ~df["closed"]].reset_index(drop=True) # This was somewhat redundant but ensured name and copy

# New way using Gamma API:
# Fetches active markets (where 'closed' is False) from the Gamma API.
# The first element of the tuple is the DataFrame of active markets.
active_df, _ = get_markets_with_cache_gamma(return_active=True)

# Now 'active_df' holds the active markets fetched from the Gamma API.
# The rest of your script can proceed using this 'active_df'.
# For example, if the original script used 'df' as the active dataframe:
# df = active_df


# %%
active_df["tokens"][1]

# %%
for i in range(100):
    # print(active_df.iloc[i, :]["tokens"])
    print(len(active_df.iloc[i, :]["tokens"]))

# %%

def get_order_book_data(token_id: str) -> dict | None:
    """Fetches the order book data for a given token_id."""
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching order book for token {token_id}: {e}")
        return None

def calculate_token_liquidity(order_book: dict) -> dict:
    """Calculates liquidity metrics from a token's order book."""
    metrics = {
        "bid_quantity": 0.0,
        "bid_value": 0.0,
        "ask_quantity": 0.0,
        "ask_value": 0.0,
    }

    if not order_book:
        return metrics

    for order_type, orders in [("bid", order_book.get("bids", [])), ("ask", order_book.get("asks", []))]:
        for order in orders:
            try:
                price = float(order.get("price", 0))
                size = float(order.get("size", 0))
                metrics[f"{order_type}_quantity"] += size
                metrics[f"{order_type}_value"] += price * size
            except (ValueError, TypeError) as e:
                print(f"Skipping order due to data issue: {order}, error: {e}")
    return metrics

def calculate_market_liquidity_metrics(market_tokens_list: list) -> dict:
    """
    Calculates aggregated liquidity metrics for a market given its list of tokens.
    Each item in market_tokens_list is expected to be a dict with a 'token_id'.
    """
    market_metrics = {
        "total_market_bid_quantity": 0.0,
        "total_market_bid_value": 0.0,
        "total_market_ask_quantity": 0.0,
        "total_market_ask_value": 0.0,
        "liquidity_fetch_success_count": 0,
    }

    if not isinstance(market_tokens_list, list):
        print("Warning: market_tokens_list is not a list. Skipping liquidity calculation.")
        return market_metrics

    for token_info in market_tokens_list:
        token_id = token_info.get("token_id")
        if not token_id:
            print(f"Warning: Missing token_id in token_info: {token_info}")
            continue

        order_book = get_order_book_data(token_id)
        if order_book:
            market_metrics["liquidity_fetch_success_count"] += 1
            token_liquidity = calculate_token_liquidity(order_book)
            market_metrics["total_market_bid_quantity"] += token_liquidity["bid_quantity"]
            market_metrics["total_market_bid_value"] += token_liquidity["bid_value"]
            market_metrics["total_market_ask_quantity"] += token_liquidity["ask_quantity"]
            market_metrics["total_market_ask_value"] += token_liquidity["ask_value"]
            # Consider adding a small delay if making many requests rapidly
            # import time
            # time.sleep(0.1) 

    return market_metrics

# %%
# Calculate liquidity metrics for each market in active_df
# This might take some time if there are many markets, due to API calls for each token

liquidity_data = []
for index, row in tqdm(active_df.iterrows(), total=active_df.shape[0], desc="Calculating Market Liquidity"):
    market_tokens = row["tokens"]
    liquidity_metrics = calculate_market_liquidity_metrics(market_tokens)
    liquidity_metrics["question_id"] = row["question_id"] # To merge back
    liquidity_data.append(liquidity_metrics)

liquidity_df = pd.DataFrame(liquidity_data)

# Merge liquidity data back into active_df
if not liquidity_df.empty:
    active_df = pd.merge(active_df, liquidity_df, on="question_id", how="left")

print("Liquidity metrics added to active_df.")
pprint(active_df[[col for col in active_df.columns if 'market' in col or 'question_id' in col]].head())


# %%
import requests
from pprint import pprint

# Replace with the actual token ID for the outcome
i = 2 # Example index, ensure it's valid for your active_df
# Ensure the market at index 'i' has at least two tokens if accessing tokens[1]
if len(active_df) > i and isinstance(active_df.iloc[i, :]["tokens"], list) and len(active_df.iloc[i, :]["tokens"]) > 1:
    token_id = active_df.iloc[i, :]["tokens"][1]["token_id"]

    # Construct the API URL
    url = f"https://clob.polymarket.com/book?token_id={token_id}"

    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        order_book = response.json()
        print("Order Book Data for a sample token:")
        pprint(order_book)
        # Example of calculating liquidity for this single token
        sample_token_liquidity = calculate_token_liquidity(order_book)
        print("Liquidity for sample token:")
        pprint(sample_token_liquidity)
    else:
        print(f"Failed to fetch data for sample token. Status code: {response.status_code}")
else:
    print(f"Could not select sample token_id at index {i}, token index 1. Check active_df content and length.")


# %%

# Ensure the market at index 'i' has at least one token if accessing tokens[0]
if len(active_df) > i and isinstance(active_df.iloc[i, :]["tokens"], list) and len(active_df.iloc[i, :]["tokens"]) > 0:
    pprint(f"Token ID from active_df sample: {active_df.iloc[i, :]['tokens'][0]['token_id']}")
else:
    print(f"Could not print sample token_id at index {i}, token index 0.")
# %% 