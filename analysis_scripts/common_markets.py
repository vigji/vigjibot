from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
from datetime import datetime, timezone


@dataclass
class PooledMarket:
    id: str  # Platform-prefixed ID, e.g., "gjopen_123", "polymarket_abc"
    question: str
    outcomes: List[
        str
    ]  # List of outcome names, e.g., ["Yes", "No"] or ["Candidate A", "Candidate B"]
    outcome_probabilities: List[
        Optional[float]
    ]  # Corresponding probabilities for outcomes
    formatted_outcomes: str  # Single string representation, e.g., "Yes: 60.0%; No: 40.0%"
    url: str  # Direct URL to the market
    published_at: Optional[
        datetime
    ]  # Publication/creation time of the market (UTC if possible)
    source_platform: str  # Name of the source platform, e.g., "GJOpen", "Polymarket"

    # Optional fields, common across platforms
    volume: Optional[float] = None  # Trading volume
    n_forecasters: Optional[int] = None  # Number of unique predictors/bettors
    comments_count: Optional[int] = None
    original_market_type: Optional[
        str
    ] = None  # Platform-specific type, e.g., "BINARY", "MULTIPLE_CHOICE"
    is_resolved: Optional[
        bool
    ] = None  # True if the market is resolved/closed, False if open, None if unknown

    # To store the original market object for further details if needed, not part of repr
    raw_market_data: Optional[Any] = field(default=None, repr=False, compare=False)


class BaseMarket(ABC):
    """
    Abstract base class for platform-specific market data classes.
    Ensures that each platform-specific market can be converted to a PooledMarket.
    """

    @abstractmethod
    def to_pooled_market(self) -> PooledMarket:
        """
        Converts the platform-specific market data to the common PooledMarket format.
        """
        pass

    @classmethod
    def parse_datetime_flexible(cls, dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None

        # Try ISO format with 'Z' (UTC)
        if isinstance(dt_str, datetime):  # Already a datetime object
            return dt_str

        if not isinstance(dt_str, str):
            # print(f"Warning: Expected string for datetime parsing, got {type(dt_str)}")
            return None

        try:
            # Handle timezone-aware strings ending with Z
            if dt_str.endswith("Z"):
                # For formats like '2023-10-26T00:00:00Z' or '2023-07-15T20:38:13.044Z'
                # Stripping Z and adding UTC timezone info for fromisoformat
                # Or, ensure it's compatible by adding +00:00 if needed
                if "." in dt_str:
                    dt_obj = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                else:
                    dt_obj = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
                return dt_obj.replace(tzinfo=timezone.utc)
            # Handle timezone-aware strings with offset
            elif (
                "+" in dt_str[10:] or "-" in dt_str[10:]
            ):  # Check for +/- in the time part
                return datetime.fromisoformat(dt_str)
            # Handle naive datetime strings
            else:
                # Try ISO format for naive datetime (e.g. fromisoformat handles '2023-10-26T00:00:00')
                dt_obj = datetime.fromisoformat(dt_str)
                # If it was truly naive, it remains naive. If it needs to be UTC, caller should specify.
                return dt_obj
        except ValueError:
            # Fallback for other common formats if needed
            # Example: '2021-07-20 16:00:00' (less common in APIs)
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # print(f"Warning: Could not parse datetime string: {dt_str}")
                return None


class BaseScraper(ABC):
    """
    Abstract base class for platform-specific scrapers.
    """

    @abstractmethod
    async def fetch_markets(self, only_open: bool = True, **kwargs) -> List[Any]:
        """
        Fetches markets from the specific platform.

        Args:
            only_open: If True, fetches only open/active markets.
            **kwargs: Additional platform-specific parameters.

        Returns:
            A list of platform-specific market objects.
        """
        pass

    async def get_pooled_markets(
        self, only_open: bool = True, **kwargs
    ) -> List[PooledMarket]:
        """
        Fetches markets and converts them to the PooledMarket format.

        Args:
            only_open: If True, fetches only open/active markets.
            **kwargs: Additional platform-specific parameters.

        Returns:
            A list of PooledMarket objects.
        """
        platform_specific_markets = await self.fetch_markets(
            only_open=only_open, **kwargs
        )
        pooled_markets = []
        for market in platform_specific_markets:
            if hasattr(market, "to_pooled_market") and callable(
                market.to_pooled_market
            ):
                try:
                    pooled_markets.append(market.to_pooled_market())
                except Exception as e:
                    market_id = getattr(market, "id", "unknown_id")
                    print(
                        f"Warning: Could not convert market {market_id} to PooledMarket: {e}"
                    )
            else:
                market_id = getattr(market, "id", "unknown_id")
                print(
                    f"Warning: Market object {market_id} of type {type(market)} does not have a to_pooled_market method."
                )
        return pooled_markets
