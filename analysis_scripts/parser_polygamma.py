# %%
%load_ext autoreload
%autoreload 2
import os
import dotenv
from pathlib import Path
from py_clob_client.client import ClobClient, TradeParams
from py_clob_client.constants import POLYGON
# Remove regex import if D is not used, or clarify its use.
# from regex import D 
import requests
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import numpy as np
from openai import OpenAI
from functools import lru_cache
from datetime import datetime, timedelta
import time
import json # For parsing outcomes string
from typing import List, Optional, Any, Dict # For type hinting
from dataclasses import dataclass

# %%

# Insert new Gamma API functions here
GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"

# Dataclasses for structuring market data
@lru_cache(maxsize=None) # To cache parsing of outcome strings if they repeat
def parse_outcomes_string(outcomes_str: str) -> List[str]:
    """Safely parses the outcomes string which is a JSON array."""
    if not outcomes_str or not isinstance(outcomes_str, str):
        return []
    try:
        parsed_outcomes = json.loads(outcomes_str)
        if isinstance(parsed_outcomes, list):
            return [str(o) for o in parsed_outcomes]
        return []
    except json.JSONDecodeError:
        return []

def format_outcomes(outcomes: List[str], prices: Optional[List[float]] = None) -> str:
    """Formats outcomes and their prices (as probabilities) into a readable string."""
    if not outcomes:
        return "N/A"
    if not prices:
        return "; ".join([f"{name}: N/A" for name in outcomes])
    assert len(outcomes) == len(prices), f"Lengths of outcomes and prices must match, but got {outcomes} and {prices}"
    return "; ".join([f"{name}: {round(price * 100)}% prob" for name, price in zip(outcomes, prices)])

def safe_float(value: Any, default: float = 0.0) -> float:
    """Converts a value to float, returning a default if conversion fails or value is None."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "") -> str:
    """Converts a value to str, returning a default if value is None."""
    if value is None:
        return default
    return str(value)

def parse_datetime_optional(datetime_str: Optional[str]) -> Optional[datetime]:
    """Parses an ISO format datetime string, returns None if input is None or invalid."""
    if not datetime_str:
        return None
    try:
        # Attempt to parse with or without milliseconds/Z
        if '.' in datetime_str and 'Z' in datetime_str:
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        elif 'Z' in datetime_str: # No milliseconds
             return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=None) # Make naive
        return datetime.fromisoformat(datetime_str) # Should handle cases without Z as naive
    except ValueError:
        # print(f"Warning: Could not parse datetime string: {datetime_str}")
        return None

@dataclass
class PolymarketMarket:
    id: str
    question: str
    slug: str
    description: str
    outcomes: List[str]
    outcome_prices: Optional[List[float]] # Prices corresponding to outcomes
    formatted_outcomes: str
    url: str
    total_volume: float # Changed from volume_24hr to represent total volume
    liquidity: float # Primary liquidity metric (e.g., sum of AMM and CLOB)
    end_date: Optional[datetime]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    active: bool
    closed: bool
    resolution_source: Optional[str]
    # Add other fields as necessary, simplifying from the original list

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "PolymarketMarket":
        raw_outcomes = data.get("outcomes", "[]") # Default to empty JSON array string
        parsed_outcomes_list = parse_outcomes_string(raw_outcomes)
        
        raw_outcome_prices = data.get("outcomePrices")
        parsed_outcome_prices: Optional[List[float]] = None

        if isinstance(raw_outcome_prices, str):
            try:
                # Attempt to parse the string as a JSON list
                potential_list = json.loads(raw_outcome_prices)
                if isinstance(potential_list, list):
                    raw_outcome_prices = potential_list # Now it's a list
                else:
                    # The string was valid JSON, but not a list
                    raw_outcome_prices = [] 
            except json.JSONDecodeError:
                # The string was not valid JSON
                raw_outcome_prices = []
        
        if isinstance(raw_outcome_prices, list):
            parsed_outcome_prices = [safe_float(p) for p in raw_outcome_prices]
        elif pd.isna(raw_outcome_prices) or raw_outcome_prices is None: # Handle explicit NaN or None
             parsed_outcome_prices = [] 
        else: # If it's something unexpected (not a string, not a list, not None/NaN)
            parsed_outcome_prices = []

        formatted_outcomes_str = format_outcomes(parsed_outcomes_list, parsed_outcome_prices)
        
        # Simplify volume: prioritize 'volume' (total), then 'volumeNum'
        total_volume = safe_float(data.get("volume"))
        if total_volume == 0.0: # if 'volume' is not present or zero, try 'volumeNum'
            total_volume = safe_float(data.get("volumeNum"))
        # 'volume24hr' is no longer the primary source for this field.

        # Simplify liquidity: sum of AMM and CLOB, or use 'liquidity' if available
        liquidity = safe_float(data.get("liquidityAmm")) + safe_float(data.get("liquidityClob"))
        if liquidity == 0.0: # If AMM and CLOB are zero or not present, try 'liquidity'
            liquidity = safe_float(data.get("liquidity"))
        if liquidity == 0.0: # if 'liquidity' is not present, try 'liquidityNum'
            liquidity = safe_float(data.get("liquidityNum"))


        slug = safe_str(data.get("slug"))
        market_url = f"https://polymarket.com/event/{slug}" if slug else ""

        return cls(
            id=safe_str(data.get("id")),
            question=safe_str(data.get("question")),
            slug=slug,
            description=safe_str(data.get("description")),
            outcomes=parsed_outcomes_list,
            outcome_prices=parsed_outcome_prices if parsed_outcome_prices else None,
            formatted_outcomes=formatted_outcomes_str,
            url=market_url,
            total_volume=total_volume, # Updated field name and value
            liquidity=liquidity,
            end_date=parse_datetime_optional(data.get("endDate")),
            created_at=parse_datetime_optional(data.get("createdAt")),
            updated_at=parse_datetime_optional(data.get("updatedAt")),
            active=bool(data.get("active", False)), # Default to False if not present
            closed=bool(data.get("closed", False)), # Default to False if not present
            resolution_source=safe_str(data.get("resolutionSource")) if data.get("resolutionSource") else None,
        )

@lru_cache(maxsize=1)
def fetch_all_markets_gamma(return_active=True, cache_key=None, max_requests=200):
    """
    Fetch all markets from Polymarket Gamma API and parse them into PolymarketMarket objects.

    Args:
        return_active: Whether to return only active markets (where closed is False).
        cache_key: Optional cache key for invalidation (timestamp), used by lru_cache.
        limit_per_page: Number of markets to fetch per API call.
        max_requests: Maximum number of API requests to prevent infinite loops.
    """
    LIMIT_PER_PAGE = 500  # heuristically found to be the maximum that can be fetched in a single request
    all_market_objects: List[PolymarketMarket] = []
    offset = 0
    
    print("Fetching markets from Gamma API...")
    pbar = tqdm(range(max_requests))
    for i in pbar: 
        params = {"limit": LIMIT_PER_PAGE, "offset": offset}
        try:
            response = requests.get(f"{GAMMA_API_BASE_URL}/markets", params=params, timeout=20)
            response.raise_for_status()
            raw_data_list = response.json() # List of market dictionaries
            
            if not raw_data_list: 
                print(f"No more markets to fetch after {i+1} requests (offset {offset}).")
                break
            
            # Parse each market dictionary into a PolymarketMarket object
            for market_data in raw_data_list:
                try:
                    market_obj = PolymarketMarket.from_api_data(market_data)
                    if return_active:
                        if not market_obj.closed: # Assuming 'active' also implies not closed, but 'closed' is more explicit for filtering resolved markets
                            all_market_objects.append(market_obj)
                    else:
                        all_market_objects.append(market_obj)
                except Exception as e:
                    print(f"Error parsing market data for market ID {market_data.get('id', 'Unknown')}: {e}")
                    # Optionally, append a partially parsed object or skip
            
            if len(raw_data_list) < LIMIT_PER_PAGE: 
                print(f"Fetched last page of markets ({len(raw_data_list)} items) in request {i+1}.")
                break
            # print(f"Fetching markets from Gamma API: {i+1} of {max_requests}; offset {offset}; fetched {len(all_market_objects)} relevant markets")
            offset += LIMIT_PER_PAGE
        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets from Gamma API on request {i+1} (offset {offset}): {e}")
            break
        except ValueError as e: # JSONDecodeError inherits from ValueError
            print(f"Error decoding JSON from Gamma API on request {i+1} (offset {offset}): {e}")
            break
        if i == max_requests - 1:
            print(f"Reached max_requests limit ({max_requests}) for Gamma API.")
        
        pbar.set_description(f"Fetched {len(all_market_objects)} markets")

    if not all_market_objects:
        print("No market data fetched or parsed from Gamma API.")
        return pd.DataFrame(), None # Return empty DataFrame

    # Create DataFrame from the list of dataclass objects
    # This ensures that columns are derived from dataclass fields and values are direct attributes
    df = pd.DataFrame([market_obj.__dict__ for market_obj in all_market_objects])
            
    return df # Second element is None for consistency

def get_markets_with_cache_gamma(return_active=True, cache_duration_minutes=30, max_requests=500):
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
        max_requests=max_requests
    )

# %%
# Initialize client and fetch markets
start_time = time.time()
active_df = get_markets_with_cache_gamma(return_active=True, max_requests=150)
end_time = time.time()
print(f"Fetching markets from Gamma API took {end_time - start_time:.2f} seconds")
print(f"Found {len(active_df)} active markets.")
# Now 'active_df' holds the active markets fetched from the Gamma API.
# The rest of your script can proceed using this 'active_df'.

if not active_df.empty:
    print("\nSample of processed market data (first market):")
    # Pretty print the first market's details if available
    # Accessing via .iloc[0].to_dict() if you want to see it as a dict
    # Or directly access attributes if you have the Pydantic model instance
    # For DataFrame, to see the structure:
    pprint(active_df.iloc[0].to_dict())
    
    print("\nColumns in the DataFrame:")
    print(sorted(list(active_df.columns)))

    print("\nExample of formatted outcomes for the first market (if available):")
    if 'formatted_outcomes' in active_df.columns and len(active_df) > 0:
        print(active_df.iloc[0]['formatted_outcomes'])
    
    print("\nExample of URL for the first market (if available):")
    if 'url' in active_df.columns and len(active_df) > 0:
        print(active_df.iloc[0]['url'])
else:
    print("No active markets fetched or DataFrame is empty.")



# %%
active_df.columns
# %%
active_df.formatted_outcomes
# %%
active_df.outcome_prices.apply(lambda x: max(x) if x else None).hist()

# %%
