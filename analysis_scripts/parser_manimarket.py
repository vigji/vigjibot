from pathlib import Path
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm


@dataclass
class ManifoldAnswer:
    text: str
    probability: float
    volume: float
    number_of_bets: int
    created_time: datetime

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "ManifoldAnswer":
        return cls(
            text=data["text"],
            probability=data.get("probability", 0),
            volume=data.get("volume", 0),
            number_of_bets=len(data.get("bets", [])),
            created_time=datetime.fromtimestamp(data["createdTime"] / 1000),
        )


@dataclass
class MarketMetadata:
    """Base class with common market fields"""
    id: str
    question: str
    outcome_type: str
    created_time: datetime
    creator_name: str
    creator_username: str
    slug: str
    volume: float
    unique_bettor_count: int
    total_liquidity: float
    close_time: Optional[datetime]
    last_updated_time: datetime
    tags: List[str]
    group_slugs: List[str]
    visibility: str
    resolution: Optional[str]
    resolution_time: Optional[datetime]

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "MarketMetadata":
        """Returns a dict of metadata fields from API data"""
        return cls(
            id="manifold_" + data["id"],
            question=data["question"],
            outcome_type=data["outcomeType"],
            created_time=datetime.fromtimestamp(data["createdTime"] / 1000),
            creator_name=data["creatorName"],
            creator_username=data["creatorUsername"],
            slug=data["slug"],
            volume=data.get("volume", 0),
            unique_bettor_count=data.get("uniqueBettorCount", 0),
            total_liquidity=data.get("totalLiquidity", 0),
            close_time=datetime.fromtimestamp(data["closeTime"] / 1000)
            if data.get("closeTime")
            else None,
            last_updated_time=datetime.fromtimestamp(data["lastUpdatedTime"] / 1000),
            tags=data.get("tags", []),
            group_slugs=data.get("groupSlugs", []),
            visibility=data.get("visibility", "public"),
            resolution=data.get("resolution"),
            resolution_time=datetime.fromtimestamp(data["resolutionTime"] / 1000)
            if data.get("resolutionTime")
            else None,
        )


@dataclass
class ManifoldMarket:
    """Unified class for Manifold markets, handling different outcome types."""

    id: str
    question: str
    outcome_type: str
    created_time: datetime
    creator_name: str
    creator_username: str
    slug: str
    volume: float
    unique_bettor_count: int
    total_liquidity: float
    close_time: Optional[datetime]
    last_updated_time: datetime
    tags: List[str]
    group_slugs: List[str]
    visibility: str
    resolution: Optional[str]
    resolution_time: Optional[datetime]
    outcomes: List[str]
    outcome_prices: List[float]
    formatted_outcomes: str
    # Binary specific
    probability: Optional[float] = None
    initial_probability: Optional[float] = None
    p: Optional[float] = None
    total_shares: Optional[float] = None
    pool: Optional[float] = None
    # Multi-choice specific
    answers: Optional[List[ManifoldAnswer]] = None

    def get_url(self) -> str:
        return f"https://manifold.markets/{self.metadata.slug}"

    def __str__(self) -> str:
        return f"{self.metadata.question} ({self.metadata.outcome_type})"

    @staticmethod
    def _format_outcomes(outcomes: List[str], prices: List[float]) -> str:
        """Format outcomes and prices into a string"""
        return "; ".join([f"{o}: {(p * 100):.1f}%" for o, p in zip(outcomes, prices)])

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> Optional["ManifoldMarket"]:
        metadata = MarketMetadata.from_api_data(data)
        outcome_type = metadata.outcome_type

        outcomes = []
        outcome_prices = []
        probability = None
        initial_probability = None
        p_val = None
        total_shares = None
        pool_val = None
        answers_obj = None

        if outcome_type == "BINARY":
            probability = data.get("probability")
            initial_probability = data.get("initialProbability")
            p_val = data.get("p")
            total_shares = data.get("totalShares")
            pool_val = data.get("pool")
            outcomes = ["Yes", "No"]
            outcome_prices = [
                probability if probability is not None else 0.5,
                1 - probability if probability is not None else 0.5,
            ]
        elif outcome_type == "MULTIPLE_CHOICE":
            answers_data = data.get("answers", [])
            answers_obj = [ManifoldAnswer.from_api_data(ans) for ans in answers_data]
            outcomes = [ans.text for ans in answers_obj]
            outcome_prices = [ans.probability for ans in answers_obj]
        else:
            # Potentially handle other types or return None if unsupported
            return None

        formatted_outcomes_str = cls._format_outcomes(outcomes, outcome_prices)

        return cls(
            id="manifold_" + data["id"],
            question=data["question"],
            outcome_type=data["outcomeType"],
            created_time=datetime.fromtimestamp(data["createdTime"] / 1000),
            creator_name=data["creatorName"],
            creator_username=data["creatorUsername"],
            slug=data["slug"],
            volume=data.get("volume", 0),
            unique_bettor_count=data.get("uniqueBettorCount", 0),
            total_liquidity=data.get("totalLiquidity", 0),
            close_time=datetime.fromtimestamp(data["closeTime"] / 1000)
            if data.get("closeTime")
            else None,
            last_updated_time=datetime.fromtimestamp(data["lastUpdatedTime"] / 1000),
            tags=data.get("tags", []),
            group_slugs=data.get("groupSlugs", []),
            visibility=data.get("visibility", "public"),
            resolution=data.get("resolution"),
            resolution_time=datetime.fromtimestamp(data["resolutionTime"] / 1000)
            if data.get("resolutionTime")
            else None,
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            formatted_outcomes=formatted_outcomes_str,
            probability=probability,
            initial_probability=initial_probability,
            p=p_val,
            total_shares=total_shares,
            pool=pool_val,
            answers=answers_obj,
        )


class ManifoldMarketClient:
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
 
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @staticmethod
    def _create_market(data: Dict[str, Any]) -> Optional[ManifoldMarket]:
        """Create appropriate market type from API data."""
        try:
            return ManifoldMarket.from_api_data(data)
        except Exception as e:
            print(f"Error creating market {data.get('id')}: {e}")
            return None

    async def _get_open_markets(self, limit: int = 1000, max_pages: int = 100) -> List[Dict[str, Any]]:
        """Fetch open markets from API."""
        base_url = "https://api.manifold.markets/v0/markets"
        params = {
            "limit": limit,
            "sort": "created-time",
            "order": "desc"
        }
        
        all_markets = []
        pbar = tqdm(range(0, limit, max_pages), desc=f"Fetched {len(all_markets)} open markets")
        try:
            for i in pbar:
                async with self.session.get(base_url, params=params) as response:
                    response.raise_for_status()
                    markets = await response.json()
                    if not markets:
                        break
                        
                    # Filter out resolved markets
                    open_markets = [m for m in markets if not m.get("isResolved", False)]
                    all_markets.extend(open_markets)
                    pbar.set_description(f"Fetched {len(all_markets)} open markets")
                    
                    # Get the ID of the last market for pagination
                    if len(markets) < limit:
                        break
                    params["before"] = markets[-1]["id"]
                    
        except aiohttp.ClientError as e:
            print(f"Error fetching open markets: {e}")
            return []
            
        print(f"Total open markets found: {len(all_markets)}")
        return all_markets

    async def _get_market_details(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific market."""
        base_url = f"https://api.manifold.markets/v0/market/{market_id.replace('manifold_', '')}"
        try:
            async with self.session.get(base_url) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error fetching details for market {market_id}: {e}")
            return None

    async def _process_market(self, market_data: Dict[str, Any], min_unique_bettors: int, min_volume: float) -> Optional[ManifoldMarket]:
        """Process a single market's data."""
        if market_data.get("uniqueBettorCount", 0) < min_unique_bettors:
            return None
        if market_data.get("volume", 0) < min_volume:
            return None
            
        full_data = await self._get_market_details(market_data["id"])
        if not full_data:
            return None
            
        return self._create_market(full_data)

    async def get_filtered_markets(
        self,
        min_unique_bettors: int = 20,
        min_volume: float = 0,
        limit: int = 1000
    ) -> List[ManifoldMarket]:
        """Get filtered markets that meet the criteria."""
        # First get all open markets
        raw_markets = await self._get_open_markets(limit)
        print(f"Total open markets fetched: {len(raw_markets)}")
        if not raw_markets:
            return []
            
        # Filter markets based on basic criteria
        # Create a list of market IDs to fetch details for, avoiding fetching for already filtered out markets
        markets_to_fetch_details_for = [
            m["id"] for m in raw_markets
            if m.get("uniqueBettorCount", 0) >= min_unique_bettors
            and m.get("volume", 0) >= min_volume
            and m.get("outcomeType") in ["BINARY", "MULTIPLE_CHOICE"]
        ]
        
        print(f"Initial open markets matching basic criteria: {len(markets_to_fetch_details_for)}")

        markets = []
        for i in tqdm(range(0, len(markets_to_fetch_details_for), self.max_concurrent), desc="Processing batches of market details"):
            batch_ids = markets_to_fetch_details_for[i:i + self.max_concurrent]
            tasks = [self._get_market_details(market_id) for market_id in batch_ids]
            results = await asyncio.gather(*tasks)
            
            for market_id, full_data in zip(batch_ids, results):
                if not full_data:
                    # print(f"Failed to fetch details for market {market_id}") # Already printed in _get_market_details
                    continue
                try:
                    # Ensure the ID passed to _create_market is the original one from Manifold, not "manifold_" + id
                    market_obj = self._create_market(full_data)
                    if market_obj:
                        # Additional check, though most filtering is done before fetching details
                        if market_obj.unique_bettor_count >= min_unique_bettors and \
                           market_obj.volume >= min_volume:
                            markets.append(market_obj)
                        pass
                except Exception as e:
                    print(f"Error processing market {full_data.get('id')} after fetching details: {e}")
                    continue
            
        print(f"Final number of processed markets: {len(markets)}")
        return markets


async def main():
   #  try:
        async with ManifoldMarketClient(max_concurrent=10) as client:
            markets = await client.get_filtered_markets(
                min_unique_bettors=50,
                min_volume=500,
                # limit=200 # Reduced limit for faster testing
            )
            print(f"Found {len(markets)} markets matching criteria")
            
            # Convert list of ManifoldMarket objects to list of dicts for DataFrame
            market_dicts = []
            for market in markets:
                market_dict = market.__dict__  # Start with metadata
                # Add other top-level fields from ManifoldMarket
                market_dict['outcomes'] = market.outcomes
                market_dict['outcome_prices'] = market.outcome_prices
                market_dict['formatted_outcomes'] = market.formatted_outcomes
                market_dict['probability'] = market.probability
                market_dict['initial_probability'] = market.initial_probability
                market_dict['p'] = market.p
                market_dict['total_shares'] = market.total_shares
                market_dict['pool'] = market.pool
                # For answers, store their text representation or similar if needed
                if market.answers:
                    market_dict['answers'] = [ans.text for ans in market.answers] # Or ans.__dict__
                else:
                    market_dict['answers'] = None
                market_dicts.append(market_dict)

            if market_dicts:
                df = pd.DataFrame(market_dicts)
                print(df.head())
                # Optional: print columns to verify
                # print(df.columns)
            else:
                print("No markets to display in DataFrame.")


if __name__ == "__main__":
    asyncio.run(main())
