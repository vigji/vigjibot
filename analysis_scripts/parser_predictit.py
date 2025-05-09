from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd
from pprint import pprint

@dataclass
class PredictItContract:
    id: int
    name: str
    #short_name: str
    # status: str
    last_trade_price: float
    best_buy_yes_cost: float
    # best_buy_no_cost: float
    best_sell_yes_cost: float
    # best_sell_no_cost: float
    # last_close_price: float
    # display_order: int
    # image: Optional[str] = None
    # date_end: Optional[str] = None

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
            # short_name=data["shortName"],
            # status=data["status"],
            last_trade_price=data["lastTradePrice"],
            best_buy_yes_cost=data["bestBuyYesCost"],
            # best_buy_no_cost=data["bestBuyNoCost"],
            best_sell_yes_cost=data["bestSellYesCost"],
            # best_sell_no_cost=data["bestSellNoCost"],
            # last_close_price=data["lastClosePrice"],
            # display_order=data["displayOrder"],
            # image=data.get("image"),
            # date_end=data.get("dateEnd")
        )
    
    

 
@dataclass
class PredictItMarket:
    id: int
    name: str
    # short_name: str
    url: str
    contracts: List[PredictItContract]
    timestamp: Optional[str]
    status: Optional[str]
    outcomes: List[str]
    outcomes_prices: List[float] 
    formatted_outcomes: List[str]
    total_liquidity: float
    avg_spread: float
    volume: float = None
    n_forecasters: int = None

    @classmethod
    def total_liquidity(cls, contracts: List[PredictItContract]) -> float:
        """Calculate total market liquidity."""
        return sum(c.liquidity for c in contracts)

    @classmethod
    def avg_spread(cls, contracts: List[PredictItContract]) -> float:
        """Calculate average spread across all contracts."""
        spreads = [c.spread for c in contracts if c.spread > 0]
        return sum(spreads) / len(spreads) if spreads else 0.0

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "PredictItMarket":
        # pprint(data)
        # pprint(data["contracts"][0])
        contracts = [PredictItContract.from_api_data(c) for c in data["contracts"]]

        if len(contracts) > 1:
            outcomes = [c.name for c in contracts]
            raw_prices = [c.last_trade_price for c in contracts]
            total = sum(raw_prices)
            outcomes_prices = [p/total for p in raw_prices] if total > 0 else [0] * len(raw_prices)
        else:   # means this is a binary market
            outcomes = ["Yes", "No"]
            outcomes_prices = [contracts[0].last_trade_price, 1-contracts[0].last_trade_price]
            # formatted_outcomes = [f"{o}: {(p*100):.1f}%" for o, p in zip(outcomes, outcomes_prices)]

        formatted_outcomes = [f"{o}: {(p*100):.1f}%" for o, p in zip(outcomes, outcomes_prices)]
        return cls(
            id="predictit_"+str(data["id"]),
            name=data["name"],
            # short_name=data["shortName"],
            url=data["url"],
            contracts=contracts,
            timestamp=data.get("timeStamp"),
            status=data.get("status"),
            outcomes=outcomes,
            outcomes_prices=outcomes_prices,
            formatted_outcomes=formatted_outcomes,
            total_liquidity=cls.total_liquidity(contracts),
            avg_spread=cls.avg_spread(contracts)
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
    all_markets = [PredictItMarket.from_api_data(market) for market in response_data.get("markets", [])]

    return pd.DataFrame(all_markets)


def get_predictit_markets():
    import requests
    url = "https://www.predictit.org/api/marketdata/all/"
    response = requests.get(url)
    response.raise_for_status()
    return parse_predictit_response(response.json())


def main():
    markets = get_predictit_markets()
    print(f"Found {len(markets)} markets")
    
    print(markets.columns)
    i = 2
    print(markets.loc[i, "name"])
    pprint(markets.loc[i, "contracts"])
    print(markets.sort_values(by="total_liquidity", ascending=False))


if __name__ == "__main__":
    main()