from pathlib import Path
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Answer:
    text: str
    probability: float
    volume: float
    number_of_bets: int
    created_time: datetime

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "Answer":
        return cls(
            text=data["text"],
            probability=data.get("probability", 0),
            volume=data.get("volume", 0),
            number_of_bets=len(data.get("bets", [])),
            created_time=datetime.fromtimestamp(data["createdTime"] / 1000),
        )


@dataclass
class Market:
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
    def from_api_data(cls, data: Dict[str, Any]) -> "Market":
        return cls(
            id=data["id"],
            question=data["question"],
            outcome_type=data["outcomeType"],
            created_time=datetime.fromtimestamp(data["createdTime"] / 1000),
            creator_name=data["creatorName"],
            creator_username=data["creatorUsername"],
            slug=data["slug"],
            volume=data.get("volume", 0),
            unique_bettor_count=data.get("uniqueBettorCount", 0),
            total_liquidity=data.get("totalLiquidity", 0),
            close_time=datetime.fromtimestamp(data["closeTime"] / 1000) if data.get("closeTime") else None,
            last_updated_time=datetime.fromtimestamp(data["lastUpdatedTime"] / 1000),
            tags=data.get("tags", []),
            group_slugs=data.get("groupSlugs", []),
            visibility=data.get("visibility", "public"),
            resolution=data.get("resolution"),
            resolution_time=datetime.fromtimestamp(data["resolutionTime"] / 1000) if data.get("resolutionTime") else None,
        )

    def get_url(self) -> str:
        return f"https://manifold.markets/{self.slug}"

    def __str__(self) -> str:
        return f"{self.question} ({self.outcome_type})"


@dataclass
class BinaryMarket(Market):
    probability: Optional[float]
    initial_probability: Optional[float]
    p: Optional[float]
    total_shares: Optional[float]
    pool: Optional[float]

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "BinaryMarket":
        base_market = Market.from_api_data(data)
        return cls(
            **base_market.__dict__,
            probability=data.get("probability"),
            initial_probability=data.get("initialProbability"),
            p=data.get("p"),
            total_shares=data.get("totalShares"),
            pool=data.get("pool"),
        )

    def get_probability_str(self) -> str:
        return f"{self.probability:.1%}" if self.probability is not None else "N/A (no bets yet)"


@dataclass
class MultiChoiceMarket(Market):
    answers: List[Answer]

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "MultiChoiceMarket":
        base_market = Market.from_api_data(data)
        answers = [Answer.from_api_data(ans) for ans in data.get("answers", [])]
        return cls(**base_market.__dict__, answers=answers)

    def get_probability_str(self) -> str:
        if not self.answers:
            return "No options defined yet"
        return "\n".join(
            f"  - {ans.text}: {ans.probability:.1%} (volume: {ans.volume:,.0f} M$)"
            for ans in sorted(self.answers, key=lambda x: x.probability, reverse=True)
        )


class ManifoldMarket:
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @staticmethod
    def _create_market(data: Dict[str, Any]) -> Optional[Union[BinaryMarket, MultiChoiceMarket]]:
        """Create appropriate market type from API data."""
        try:
            if data["outcomeType"] == "BINARY":
                return BinaryMarket.from_api_data(data)
            elif data["outcomeType"] == "MULTIPLE_CHOICE":
                return MultiChoiceMarket.from_api_data(data)
            return None
        except Exception as e:
            print(f"Error creating market {data.get('id')}: {e}")
            return None

    async def _get_open_markets(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch open markets from API."""
        base_url = "https://api.manifold.markets/v0/markets"
        params = {
            "limit": min(limit, 1000),
            "sort": "created-time",
            "order": "desc"
        }
        
        try:
            async with self.session.get(base_url, params=params) as response:
                response.raise_for_status()
                markets = await response.json()
                return [m for m in markets if not m.get("isResolved", False)]
        except aiohttp.ClientError as e:
            print(f"Error fetching open markets: {e}")
            return []

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

    async def _process_market(self, market_data: Dict[str, Any], min_unique_bettors: int, min_volume: float) -> Optional[Union[BinaryMarket, MultiChoiceMarket]]:
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
    ) -> List[Union[BinaryMarket, MultiChoiceMarket]]:
        """Get filtered markets that meet the criteria."""
        raw_markets = await self._get_open_markets(limit)
        if not raw_markets:
            return []
            
        markets = []
        for i in range(0, len(raw_markets), self.max_concurrent):
            batch = raw_markets[i:i + self.max_concurrent]
            tasks = [self._process_market(market_data, min_unique_bettors, min_volume) for market_data in batch]
            results = await asyncio.gather(*tasks)
            markets.extend([m for m in results if m is not None])
            
        return markets

    @staticmethod
    def print_market_details(market: Union[BinaryMarket, MultiChoiceMarket]):
        """Print details for a market."""
        print("\n=== Market Details ===")
        print(f"ID: {market.id}")
        print(f"Question: {market.question}")
        print(f"Type: {market.outcome_type}")
        print(f"Created: {market.created_time}")
        print(f"Creator: {market.creator_name} (@{market.creator_username})")
        print(f"URL: {market.get_url()}")
        
        print("\n=== Market Activity ===")
        print(f"Total volume: {market.volume:,.0f} M$")
        print(f"Unique bettors: {market.unique_bettor_count}")
        print(f"Total liquidity: {market.total_liquidity:,.0f} M$")
        print(f"Close time: {market.close_time if market.close_time else 'None'}")
        print(f"Last updated: {market.last_updated_time}")
        
        if isinstance(market, BinaryMarket):
            print("\n=== Probabilities ===")
            print(market.get_probability_str())
        elif isinstance(market, MultiChoiceMarket):
            print("\n=== Answer Details ===")
            print(market.get_probability_str())
        
        print("\n=== Additional Data ===")
        print(f"Resolution: {market.resolution}")
        print(f"Resolution time: {market.resolution_time if market.resolution_time else 'None'}")
        print(f"Tags: {', '.join(market.tags)}")
        print(f"Group slugs: {', '.join(market.group_slugs)}")
        print(f"Visibility: {market.visibility}")
        
        print("\n" + "="*80)


async def main():
    try:
        async with ManifoldMarket(max_concurrent=10) as client:
            markets = await client.get_filtered_markets(
                min_unique_bettors=40,
                min_volume=500
            )
            print(f"Found {len(markets)} markets matching criteria")
            
            for market in markets[:5]:  # Print details for first 5 markets
                ManifoldMarket.print_market_details(market)
                
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
