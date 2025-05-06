from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class PredictItContract:
    id: int
    name: str
    short_name: str
    status: str
    last_trade_price: float
    best_buy_yes_cost: float
    best_buy_no_cost: float
    best_sell_yes_cost: float
    best_sell_no_cost: float
    last_close_price: float
    display_order: int
    image: Optional[str] = None
    date_end: Optional[str] = None

    @property
    def spread(self) -> float:
        """Calculate the spread between best buy and sell prices."""
        if self.best_buy_yes_cost is not None and self.best_sell_yes_cost is not None:
            return self.best_buy_yes_cost - self.best_sell_yes_cost
        return 0.0

    @property
    def liquidity(self) -> float:
        """Calculate a rough estimate of liquidity based on price spreads."""
        if self.spread == 0:
            return 0.0
        return 1.0 / self.spread  # Inverse of spread as a rough liquidity measure

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "PredictItContract":
        return cls(
            id=data["id"],
            name=data["name"],
            short_name=data["shortName"],
            status=data["status"],
            last_trade_price=data["lastTradePrice"],
            best_buy_yes_cost=data["bestBuyYesCost"],
            best_buy_no_cost=data["bestBuyNoCost"],
            best_sell_yes_cost=data["bestSellYesCost"],
            best_sell_no_cost=data["bestSellNoCost"],
            last_close_price=data["lastClosePrice"],
            display_order=data["displayOrder"],
            image=data.get("image"),
            date_end=data.get("dateEnd")
        )

 
@dataclass
class PredictItMarket:
    id: int
    name: str
    short_name: str
    url: str
    contracts: List[PredictItContract]
    image: Optional[str] = None
    timestamp: Optional[str] = None
    status: Optional[str] = None

    @property
    def total_liquidity(self) -> float:
        """Calculate total market liquidity."""
        return sum(c.liquidity for c in self.contracts)

    @property
    def avg_spread(self) -> float:
        """Calculate average spread across all contracts."""
        spreads = [c.spread for c in self.contracts if c.spread > 0]
        return sum(spreads) / len(spreads) if spreads else 0.0

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "PredictItMarket":
        return cls(
            id=data["id"],
            name=data["name"],
            short_name=data["shortName"],
            url=data["url"],
            contracts=[PredictItContract.from_api_data(c) for c in data["contracts"]],
            image=data.get("image"),
            timestamp=data.get("timeStamp"),
            status=data.get("status")
        )

    def get_probability_str(self) -> str:
        if not self.contracts:
            return "No contracts available"
        return "\n".join(
            f"  - {c.name}: {c.last_trade_price:.1%} (spread: {c.spread:.1%}, liquidity: {c.liquidity:.1f})"
            for c in sorted(self.contracts, key=lambda x: x.last_trade_price, reverse=True)
        )


def parse_predictit_response(response_data: Dict[str, Any]) -> List[PredictItMarket]:
    """Parse the raw PredictIt API response into a list of PredictItMarket objects."""
    return [PredictItMarket.from_api_data(market) for market in response_data.get("markets", [])]


def print_market_details(market: PredictItMarket):
    """Print details for a PredictIt market."""
    print("\n=== Market Details ===")
    print(f"ID: {market.id}")
    print(f"Name: {market.name}")
    print(f"Short Name: {market.short_name}")
    print(f"URL: {market.url}")
    print(f"Status: {market.status}")
    print(f"Last Updated: {market.timestamp}")
    
    print("\n=== Market Metrics ===")
    print(f"Total Liquidity: {market.total_liquidity:.1f}")
    print(f"Average Spread: {market.avg_spread:.1%}")
    
    print("\n=== Contract Details ===")
    print(market.get_probability_str())
    
    print("\n" + "="*80) 


def get_predictit_markets():
    import requests
    url = "https://www.predictit.org/api/marketdata/all/"
    response = requests.get(url)
    response.raise_for_status()
    return parse_predictit_response(response.json())


def main():
    markets = get_predictit_markets()
    print(f"Found {len(markets)} markets")
    
    # Print details for first 3 markets
    for market in markets[:10]:
        print_market_details(market)


if __name__ == "__main__":
    main()