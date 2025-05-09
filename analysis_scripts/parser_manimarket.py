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
    def from_api_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a dict of metadata fields from API data"""
        return {
            "id": "manifold_"+data["id"],
            "question": data["question"],
            "outcome_type": data["outcomeType"],
            "created_time": datetime.fromtimestamp(data["createdTime"] / 1000),
            "creator_name": data["creatorName"],
            "creator_username": data["creatorUsername"],
            "slug": data["slug"],
            "volume": data.get("volume", 0),
            "unique_bettor_count": data.get("uniqueBettorCount", 0),
            "total_liquidity": data.get("totalLiquidity", 0),
            "close_time": datetime.fromtimestamp(data["closeTime"] / 1000) if data.get("closeTime") else None,
            "last_updated_time": datetime.fromtimestamp(data["lastUpdatedTime"] / 1000),
            "tags": data.get("tags", []),
            "group_slugs": data.get("groupSlugs", []),
            "visibility": data.get("visibility", "public"),
            "resolution": data.get("resolution"),
            "resolution_time": datetime.fromtimestamp(data["resolutionTime"] / 1000) if data.get("resolutionTime") else None,
        }


@dataclass
class OutcomeMixin:
    """Mixin for outcome-related functionality"""
    outcomes: List[str]
    outcome_prices: List[float]
    formatted_outcomes: str

    @staticmethod
    def format_outcomes(outcomes: List[str], prices: List[float]) -> str:
        """Format outcomes and prices into a string"""
        return "; ".join([f"{o}: {(p * 100):.1f}%" for o, p in zip(outcomes, prices)])


@dataclass
class ManifoldMarket(MarketMetadata, ABC):
    """Abstract base class for all market types"""

    def get_url(self) -> str:
        return f"https://manifold.markets/{self.slug}"

    def __str__(self) -> str:
        return f"{self.question} ({self.outcome_type})"

    @classmethod
    @abstractmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "ManifoldMarket":
        pass


@dataclass
class BinaryManifoldMarket(ManifoldMarket, OutcomeMixin):
    probability: Optional[float]
    initial_probability: Optional[float]
    p: Optional[float]
    total_shares: Optional[float]
    pool: Optional[float]

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "BinaryManifoldMarket":
        base_fields = MarketMetadata.from_api_data(data)
        probability = data.get("probability")
        
        # Set binary outcomes
        outcomes = ["Yes", "No"]
        outcome_prices = [probability if probability is not None else 0.5, 
                        1 - probability if probability is not None else 0.5]
        
        return cls(
            **base_fields,
            probability=probability,
            initial_probability=data.get("initialProbability"),
            p=data.get("p"),
            total_shares=data.get("totalShares"),
            pool=data.get("pool"),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            formatted_outcomes=cls.format_outcomes(outcomes, outcome_prices)
        )


@dataclass
class MultiChoiceManifoldMarket(ManifoldMarket, OutcomeMixin):
    answers: List[ManifoldAnswer]

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "MultiChoiceManifoldMarket":
        base_fields = MarketMetadata.from_api_data(data)
        answers = [ManifoldAnswer.from_api_data(ans) for ans in data.get("answers", [])]
        
        # Extract outcomes and probabilities from answers
        outcomes = [ans.text for ans in answers]
        outcome_prices = [ans.probability for ans in answers]
        
        return cls(
            **base_fields,
            answers=answers,
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            formatted_outcomes=cls.format_outcomes(outcomes, outcome_prices)
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
    def _create_market(data: Dict[str, Any]) -> Optional[Union[BinaryManifoldMarket, MultiChoiceManifoldMarket]]:
        """Create appropriate market type from API data."""
        try:
            if data["outcomeType"] == "BINARY":
                return BinaryManifoldMarket.from_api_data(data)
            elif data["outcomeType"] == "MULTIPLE_CHOICE":
                return MultiChoiceManifoldMarket.from_api_data(data)
            return None
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
        base_url = f"https://api.manifold.markets/v0/market/{market_id}"
        try:
            async with self.session.get(base_url) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error fetching details for market {market_id}: {e}")
            return None

    async def _process_market(self, market_data: Dict[str, Any], min_unique_bettors: int, min_volume: float) -> Optional[Union[BinaryManifoldMarket, MultiChoiceManifoldMarket]]:
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
    ) -> List[Union[BinaryManifoldMarket, MultiChoiceManifoldMarket]]:
        """Get filtered markets that meet the criteria."""
        # First get all open markets
        raw_markets = await self._get_open_markets(limit)
        print(f"Total open markets fetched: {len(raw_markets)}")
        if not raw_markets:
            return []
            
        # Filter markets based on basic criteria
        filtered_markets = [
            m for m in raw_markets
            if m.get("uniqueBettorCount", 0) >= min_unique_bettors
            and m.get("volume", 0) >= min_volume
            and m.get("outcomeType") in ["BINARY", "MULTIPLE_CHOICE"]
        ]
        
        # Then fetch full details for filtered markets
        markets = []
        for i in tqdm(range(0, len(filtered_markets), self.max_concurrent), desc="Processing batches"):
            batch = filtered_markets[i:i + self.max_concurrent]
            tasks = [self._get_market_details(market_data["id"]) for market_data in batch]
            results = await asyncio.gather(*tasks)
            
            for market_data, full_data in zip(batch, results):
                if not full_data:
                    print(f"Failed to fetch details for market {market_data['id']}")
                    continue
                try:
                    market = self._create_market(full_data)
                    if market:
                        markets.append(market)
                    else:
                        print(f"Failed to create market object for {market_data['id']}")
                except Exception as e:
                    print(f"Error processing market {market_data['id']}: {e}")
                    continue
            
        print(f"Final number of processed markets: {len(markets)}")
        return markets


async def main():
   #  try:
        async with ManifoldMarketClient(max_concurrent=10) as client:
            markets = await client.get_filtered_markets(
                min_unique_bettors=50,
                min_volume=500
            )
            print(f"Found {len(markets)} markets matching criteria")
            
            df = pd.DataFrame(markets) # ([market.__dict__ for market in markets])
            print(df.head())



if __name__ == "__main__":
    asyncio.run(main())
