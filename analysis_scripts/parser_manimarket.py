from pathlib import Path
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm
from common_markets import PooledMarket, BaseMarket, BaseScraper


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
        return f"https://manifold.markets/{self.creator_username}/{self.slug}"

    def __str__(self) -> str:
        return f"{self.question} ({self.outcome_type})"

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

    def to_pooled_market(self) -> PooledMarket:
        is_res = bool(self.resolution and self.resolution != "MKT")

        return PooledMarket(
            id=self.id,
            question=self.question,
            outcomes=self.outcomes,
            outcome_probabilities=self.outcome_prices,
            formatted_outcomes=self.formatted_outcomes,
            url=f"https://manifold.markets/{self.creator_username}/{self.slug}",
            published_at=self.created_time,
            source_platform="Manifold",
            volume=self.volume,
            n_forecasters=self.unique_bettor_count,
            comments_count=None,
            original_market_type=self.outcome_type,
            is_resolved=is_res,
            raw_market_data=self
        )


class ManifoldScraper(BaseScraper):
    def __init__(self, max_concurrent: int = 5, api_key: Optional[str] = None):
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = api_key
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Key {self.api_key}"
 
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
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

    async def _fetch_raw_markets_list(self, limit: int = 1000, before: Optional[str] = None, only_open: bool = True) -> List[Dict[str, Any]]:
        """Fetch a list of markets from API, with optional pagination and open status filter."""
        base_url = "https://api.manifold.markets/v0/markets"
        params: Dict[str, Any] = {
            "limit": limit,
            "sort": "created-time",
            "order": "desc"
        }
        if before:
            params["before"] = before
        
        all_markets_batch = []
        try:
            async with self.session.get(base_url, params=params, headers=self.headers) as response:
                response.raise_for_status()
                markets_page = await response.json()
                if not markets_page:
                    return [] # No more markets
                
                if only_open:
                    all_markets_batch = [m for m in markets_page if not m.get("isResolved", False)]
                else:
                    all_markets_batch = markets_page
                    
        except aiohttp.ClientError as e:
            print(f"Error fetching markets list: {e}")
            return []
        return all_markets_batch

    async def _get_market_details(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific market."""
        # Ensure we use the original ID for the API call
        original_id = market_id.replace("manifold_", "")
        base_url = f"https://api.manifold.markets/v0/market/{original_id}"
        try:
            async with self.session.get(base_url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error fetching details for market {market_id} (original_id: {original_id}): {e}")
            return None

    async def fetch_markets(
        self,
        only_open: bool = True,
        limit: int = 1000,
        max_pages_to_fetch: int = 10,
        min_unique_bettors: int = 0,
        min_volume: float = 0,
        **kwargs: Any
    ) -> List[ManifoldMarket]:
        """Get filtered markets that meet the criteria."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)

        raw_markets_list_paginated: List[Dict[str, Any]] = []
        last_market_id: Optional[str] = None
        markets_fetched_count = 0

        pbar_overall = tqdm(total=limit, desc=f"Fetching Manifold markets (aiming for {limit})")

        for page_num in range(max_pages_to_fetch):
            if markets_fetched_count >= limit:
                break

            batch_limit = min(1000, limit - markets_fetched_count)
            if batch_limit <= 0:
                break

            current_batch = await self._fetch_raw_markets_list(
                limit=batch_limit, 
                before=last_market_id, 
                only_open=only_open
            )
            
            if not current_batch:
                break

            raw_markets_list_paginated.extend(current_batch)
            markets_fetched_count += len(current_batch)
            pbar_overall.update(len(current_batch))
            
            if len(current_batch) < batch_limit:
                break 
            
            last_market_id = current_batch[-1]["id"]
            
        pbar_overall.close()
        
        if not raw_markets_list_paginated:
            return []
            
        markets_to_fetch_details_for_ids: List[str] = []
        for m_summary in raw_markets_list_paginated:
            if only_open and m_summary.get("isResolved", False):
                continue
            if m_summary.get("uniqueBettorCount", 0) < min_unique_bettors:
                continue
            if m_summary.get("volume", 0) < min_volume:
                continue
            if m_summary.get("outcomeType") not in ["BINARY", "MULTIPLE_CHOICE"]:
                continue
            markets_to_fetch_details_for_ids.append(m_summary["id"])
        
        processed_markets: List[ManifoldMarket] = []
        for i in tqdm(range(0, len(markets_to_fetch_details_for_ids), self.max_concurrent), desc="Fetching Manifold market details"):
            batch_ids = markets_to_fetch_details_for_ids[i:i + self.max_concurrent]
            tasks = [self._get_market_details(market_id) for market_id in batch_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for market_id_original, full_data_or_exc in zip(batch_ids, results):
                if isinstance(full_data_or_exc, Exception) or not full_data_or_exc:
                    continue
                
                market_obj = self._create_market(full_data_or_exc)
                if market_obj:
                    if only_open and (market_obj.resolution and market_obj.resolution != "MKT"):
                        continue
                    if market_obj.unique_bettor_count >= min_unique_bettors and \
                       market_obj.volume >= min_volume:
                        processed_markets.append(market_obj)
        
        return processed_markets


async def main():
    print("Starting ManifoldScraper example...")
    async with ManifoldScraper(max_concurrent=5) as client:
        print("Fetching ALL markets (including resolved ones)... Min 0 bettors, 0 volume, limit 50")
        all_markets = await client.fetch_markets(
            only_open=False, 
            limit=50,
            max_pages_to_fetch=2,
            min_unique_bettors=0, 
            min_volume=0
        )
        print(f"Found {len(all_markets)} ALL markets matching criteria.")
        if all_markets:
            print(f"First market (all): {all_markets[0].question}, Resolved: {all_markets[0].resolution is not None}")

        print("\nFetching OPEN markets... Min 0 bettors, 0 volume, limit 50")
        open_markets = await client.fetch_markets(
            only_open=True, 
            limit=50, 
            max_pages_to_fetch=2,
            min_unique_bettors=0, 
            min_volume=0
        )
        print(f"Found {len(open_markets)} OPEN markets matching criteria.")
        if open_markets:
            print(f"First market (open): {open_markets[0].question}, Resolved: {open_markets[0].resolution is not None}")

        print("\nFetching OPEN markets with min 50 bettors, min 500 volume, limit 50")
        filtered_open_markets = await client.fetch_markets(
            only_open=True,
            limit=50,
            max_pages_to_fetch=2,
            min_unique_bettors=50,
            min_volume=500
        )
        print(f"Found {len(filtered_open_markets)} filtered OPEN markets.")
        if filtered_open_markets:
            pprint(filtered_open_markets[0].__dict__)

            print("\nConverting filtered open markets to PooledMarket format...")
            pooled_markets_data = await client.get_pooled_markets(
                only_open=True,
                limit=50,
                max_pages_to_fetch=2,
                min_unique_bettors=50,
                min_volume=500
            )
            print(f"Got {len(pooled_markets_data)} pooled markets.")
            if pooled_markets_data:
                pprint(pooled_markets_data[0].__dict__)

            manually_pooled = [m.to_pooled_market() for m in filtered_open_markets]
            print(f"Manually converted {len(manually_pooled)} markets.")


if __name__ == "__main__":
    asyncio.run(main())
