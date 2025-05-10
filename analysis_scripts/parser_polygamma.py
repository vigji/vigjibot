# %%

import os
import dotenv
from pathlib import Path
# from py_clob_client.client import ClobClient, TradeParams # Not used in this version
# from py_clob_client.constants import POLYGON # Not used in this version
import requests
from tqdm import tqdm
from pprint import pprint
import pandas as pd
# import numpy as np # Not explicitly used
# from openai import OpenAI # Not used
from functools import lru_cache
from datetime import datetime #, timedelta # timedelta not used
import time
import json
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from common_markets import PooledMarket, BaseMarket, BaseScraper

# Load environment variables if .env file exists
dotenv.load_dotenv()

GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"

@lru_cache(maxsize=None)
def parse_outcomes_string(outcomes_str: str) -> List[str]:
    if not outcomes_str or not isinstance(outcomes_str, str):
        return []
    try:
        parsed_outcomes = json.loads(outcomes_str)
        if isinstance(parsed_outcomes, list):
            return [str(o) for o in parsed_outcomes]
        return []
    except json.JSONDecodeError:
        return []

def format_outcomes_polymarket(outcomes: List[str], prices: Optional[List[float]] = None) -> str:
    if not outcomes:
        return "N/A"
    if not prices or len(outcomes) != len(prices):
        # print(f"Warning: Outcomes and prices length mismatch or prices missing. Outcomes: {outcomes}, Prices: {prices}")
        return "; ".join([f"{name}: N/A" for name in outcomes])
    return "; ".join([f"{name}: {(price * 100):.1f}% prob" if price is not None else f"{name}: N/A" for name, price in zip(outcomes, prices)])

def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)

@dataclass
class PolymarketMarket(BaseMarket):
    id: str
    question: str
    slug: str
    description: str
    outcomes: List[str]
    outcome_prices: Optional[List[Optional[float]]] # Prices corresponding to outcomes, can have None
    formatted_outcomes: str
    url: str
    total_volume: float
    liquidity: float
    end_date: Optional[datetime]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    active: bool # From API
    closed: bool # From API, used for is_resolved
    resolution_source: Optional[str]
    # Store raw API type if available for more accurate original_market_type
    raw_market_type: Optional[str] = None 

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "PolymarketMarket":
        raw_outcomes = data.get("outcomes", "[]")
        parsed_outcomes_list = parse_outcomes_string(raw_outcomes)
        
        raw_outcome_prices_payload = data.get("outcomePrices")
        parsed_outcome_prices_list: List[Optional[float]] = []

        temp_prices_for_parsing = []
        if isinstance(raw_outcome_prices_payload, str):
            try:
                potential_list = json.loads(raw_outcome_prices_payload)
                if isinstance(potential_list, list):
                    temp_prices_for_parsing = potential_list
            except json.JSONDecodeError:
                pass # Keep temp_prices_for_parsing empty
        elif isinstance(raw_outcome_prices_payload, list):
            temp_prices_for_parsing = raw_outcome_prices_payload
        
        # Ensure prices list matches outcomes length, padding with None if necessary
        if temp_prices_for_parsing:
            num_outcomes = len(parsed_outcomes_list)
            parsed_outcome_prices_list = [(safe_float(p) if p is not None else None) for p in temp_prices_for_parsing[:num_outcomes]]
            # Pad with None if API provided fewer prices than outcomes
            if len(parsed_outcome_prices_list) < num_outcomes:
                parsed_outcome_prices_list.extend([None] * (num_outcomes - len(parsed_outcome_prices_list)))
        else: # No prices provided or parse failed, fill with Nones matching outcomes length
            parsed_outcome_prices_list = [None] * len(parsed_outcomes_list)
            
        formatted_outcomes_str = format_outcomes_polymarket(parsed_outcomes_list, parsed_outcome_prices_list)
        
        total_volume = safe_float(data.get("volume"))
        if total_volume == 0.0:
            total_volume = safe_float(data.get("volumeNum"))

        liquidity = safe_float(data.get("liquidityAmm")) + safe_float(data.get("liquidityClob"))
        if liquidity == 0.0:
            liquidity = safe_float(data.get("liquidity"))
        if liquidity == 0.0:
            liquidity = safe_float(data.get("liquidityNum"))

        slug_val = safe_str(data.get("slug"))
        market_url = f"https://polymarket.com/event/{slug_val}" if slug_val else ""

        return cls(
            id="polymarket_" + safe_str(data.get("id")),
            question=safe_str(data.get("question")),
            slug=slug_val,
            description=safe_str(data.get("description")),
            outcomes=parsed_outcomes_list,
            outcome_prices=parsed_outcome_prices_list if any(p is not None for p in parsed_outcome_prices_list) else None,
            formatted_outcomes=formatted_outcomes_str,
            url=market_url,
            total_volume=total_volume,
            liquidity=liquidity,
            end_date=BaseMarket.parse_datetime_flexible(data.get("endDate")),
            created_at=BaseMarket.parse_datetime_flexible(data.get("createdAt")),
            updated_at=BaseMarket.parse_datetime_flexible(data.get("updatedAt")),
            active=bool(data.get("active", False)),
            closed=bool(data.get("closed", False)),
            resolution_source=safe_str(data.get("resolutionSource")) if data.get("resolutionSource") else None,
            raw_market_type=safe_str(data.get("category")) # Assuming 'category' might be 'Sports', 'Politics' etc. or sometimes 'Binary'
        )

    def to_pooled_market(self) -> PooledMarket:
        # Determine original_market_type based on raw_market_type or outcomes
        market_type = self.raw_market_type
        if not market_type:
            if len(self.outcomes) == 2:
                # Basic check for common binary outcome names (case-insensitive)
                norm_outcomes = [o.lower() for o in self.outcomes]
                if ("yes" in norm_outcomes and "no" in norm_outcomes) or \
                   ("true" in norm_outcomes and "false" in norm_outcomes):
                    market_type = "BINARY"
                else:
                    market_type = "CATEGORICAL" # Or other generic term for 2-outcome non-binary
            elif len(self.outcomes) > 2:
                market_type = "CATEGORICAL" # Multiple choice
            else:
                market_type = "UNKNOWN" # Or None

        return PooledMarket(
            id=self.id,
            question=self.question,
            outcomes=self.outcomes,
            outcome_probabilities=self.outcome_prices if self.outcome_prices else [None]*len(self.outcomes),
            formatted_outcomes=self.formatted_outcomes,
            url=self.url,
            published_at=self.created_at, # Use created_at as published_at
            source_platform="Polymarket",
            volume=self.total_volume,
            n_forecasters=None,  # Not directly available in the provided Polymarket API data struct
            comments_count=None, # Not directly available
            original_market_type=market_type,
            is_resolved=self.closed,
            raw_market_data=self
        )

class PolymarketGammaScraper(BaseScraper):
    BASE_URL = GAMMA_API_BASE_URL

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    # Note: Making this async, but requests call is synchronous.
    async def _fetch_page_data(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        params = {"limit": limit, "offset": offset}
        try:
            # Synchronous call
            response = requests.get(f"{self.BASE_URL}/markets", params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets from Gamma API (offset {offset}): {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gamma API (offset {offset}): {e}")
            return []

    # Note: Async, but internal calls are effectively synchronous due to _fetch_page_data
    async def _fetch_all_raw_markets(self, max_requests: int = 200, limit_per_page: int = 100) -> List[Dict[str, Any]]:
        all_raw_market_data: List[Dict[str, Any]] = []
        offset = 0
        
        # print("Fetching all raw market data from Gamma API...")
        # tqdm is not async-friendly by default, consider alternatives or careful usage in async.
        # For this refactor, keeping it but noting potential issues in highly concurrent scenarios.
        for i in tqdm(range(max_requests), desc="Fetching Polymarket pages"):
            raw_data_list = await self._fetch_page_data(limit=limit_per_page, offset=offset)
            if not raw_data_list:
                # print(f"No more markets to fetch after {i+1} requests (offset {offset}).")
                break
            all_raw_market_data.extend(raw_data_list)
            if len(raw_data_list) < limit_per_page:
                # print(f"Fetched last page of markets ({len(raw_data_list)} items) in request {i+1}.")
                break
            offset += limit_per_page
            if i == max_requests - 1:
                print(f"Reached max_requests limit ({max_requests}) for Gamma API.")
        return all_raw_market_data

    async def fetch_markets(self, only_open: bool = True, **kwargs: Any) -> List[PolymarketMarket]:
        """
        Fetch markets from Polymarket Gamma API and parse them into PolymarketMarket objects.

        Args:
            only_open: Whether to return only open markets (where closed is False).
            **kwargs: Supports 'max_requests' (int, default 200) and 
                      'limit_per_page' (int, default 100).
        Returns:
            A list of PolymarketMarket objects.
        """
        max_requests = kwargs.get('max_requests', 200)
        limit_per_page = kwargs.get('limit_per_page', 100)

        raw_markets_data = await self._fetch_all_raw_markets(max_requests=max_requests, limit_per_page=limit_per_page)
        
        parsed_markets: List[PolymarketMarket] = []
        if not raw_markets_data:
            # print("No raw market data fetched from Gamma API.")
            return []

        for market_data_dict in raw_markets_data:
            try:
                market_obj = PolymarketMarket.from_api_data(market_data_dict)
                if only_open:
                    if not market_obj.closed:
                        parsed_markets.append(market_obj)
                else:
                    parsed_markets.append(market_obj)
            except Exception as e:
                print(f"Error parsing market data for market ID {market_data_dict.get('id', 'Unknown')}: {e}")
        
        # print(f"Successfully parsed {len(parsed_markets)} markets.")
        return parsed_markets


if __name__ == "__main__":
    import asyncio # For async main

    async def run_polymarket_scraper():
        print("Starting PolymarketGammaScraper example...")
        scraper = PolymarketGammaScraper()

        fetch_active = True
        max_api_req = 3 # Reduced for quick example
        limit_p_page = 50 # Reduced for quick example
        print(f"Fetching {'active' if fetch_active else 'all'} markets from Polymarket (max_requests={max_api_req}, limit_per_page={limit_p_page})...")
        start_time = time.time()
        
        polymarket_list = await scraper.fetch_markets(
            only_open=fetch_active, 
            max_requests=max_api_req, 
            limit_per_page=limit_p_page
        )
        
        end_time = time.time()

        print(f"Fetching took {end_time - start_time:.2f} seconds.")
        print(f"Fetched {len(polymarket_list)} Polymarket markets.")

        if polymarket_list:
            # Convert to PooledMarket format using the inherited method
            print("\nConverting fetched Polymarket markets to PooledMarket format using get_pooled_markets...")
            pooled_markets = await scraper.get_pooled_markets(
                only_open=fetch_active, 
                max_requests=max_api_req, 
                limit_per_page=limit_p_page
            )
            print(f"Converted {len(pooled_markets)} markets to PooledMarket format.")

            if pooled_markets:
                print("Details of the first pooled market (Polymarket):")
                pprint(pooled_markets[0].__dict__)

                # Optional: Create a DataFrame
                # df_pooled = pd.DataFrame([pm.__dict__ for pm in pooled_markets])
                # print(f"Created DataFrame with {len(df_pooled)} pooled Polymarket markets.")
                # print(df_pooled.head())
            else:
                print("No Polymarket markets were successfully converted to PooledMarket format.")
        else:
            print("No markets were fetched from Polymarket.")

    asyncio.run(run_polymarket_scraper())


