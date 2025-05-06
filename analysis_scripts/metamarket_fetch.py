from typing import List, Dict, Optional, Tuple
import time
import requests
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class Market:
    id: str
    title: str
    url: str
    platform: str
    options: List[Dict[str, float]]
    probability: Optional[float] = None
    first_seen: Optional[datetime] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None


class MetaforecastClient:
    """Client for fetching markets from the Metaforecast GraphQL API using cursor‐based pagination."""

    API_URL = "https://metaforecast.org/api/graphql"

    def __init__(self) -> None:
        self.headers = {"Content-Type": "application/json"}
        # Use a Session with retry strategy to avoid exhausting resources
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.headers)
        # Throttle interval between page requests to ease memory and resource usage
        self.sleep_interval = 0.1

    def fetch_markets(self, page_size: int = 100) -> List[Market]:
        """Fetch all (open) markets in pages of `page_size` questions each."""
        query = """
        query GetMarkets($first: Int!, $after: String) {
          questions(first: $first, after: $after, orderBy: FIRST_SEEN_DESC) {
            edges {
              node {
                id
                title
                url
                platform { id }
                options { name probability }
                firstSeenStr
              }
            }
            pageInfo { hasNextPage endCursor }
          }
        }
        """

        markets: List[Market] = []
        after: Optional[str] = None

        while True:
            variables = {"first": page_size, "after": after}
            # perform the POST via Session to reuse connections and apply retries
            resp = self.session.post(
                self.API_URL,
                json={"query": query, "variables": variables},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errors"):
                raise RuntimeError(f"GraphQL errors: {data['errors']}")

            conn = data["data"]["questions"]
            for edge in conn.get("edges", []):
                node = edge.get("node", {})
                fs_str = node.get("firstSeenStr")
                first_seen = datetime.fromisoformat(fs_str) if fs_str else None
                opts = [
                    {"name": o["name"], "probability": o["probability"]}
                    for o in node.get("options", [])
                ]
                markets.append(
                    Market(
                        id=node.get("id", ""),
                        title=node.get("title", ""),
                        url=node.get("url", ""),
                        platform=node.get("platform", {}).get("id", ""),
                        options=opts,
                        probability=opts[0]["probability"] if opts else None,
                        first_seen=first_seen,
                    )
                )

            page_info = conn.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
            # throttle to avoid resource exhaustion
            time.sleep(self.sleep_interval)

        return markets


def fetch_metaculus_times(question_id: str) -> Tuple[datetime, datetime]:
    """Fetch publish_time and close_time from Metaculus API for a given question id string like 'metaculus-1234'."""
    raw_id = question_id.split('-', 1)[-1]
    url = f"https://www.metaculus.com/api2/questions/{raw_id}/"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    pub = datetime.fromisoformat(data.get("publish_time"))
    close = datetime.fromisoformat(data.get("close_time"))
    return pub, close


def fetch_predictit_times(market_id: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Fetch close_time from PredictIt API for a market id string like 'predictit-123'."""
    raw_id = market_id.split('-', 1)[-1]
    resp = requests.get("https://www.predictit.org/api/marketdata/all/")
    resp.raise_for_status()
    data = resp.json().get("markets", [])
    for mkt in data:
        if str(mkt.get("id")) == raw_id:
            # PredictIt does not provide explicit open time via this endpoint
            close = None
            date_end = mkt.get("dateEnd")
            if date_end:
                try:
                    close = datetime.fromisoformat(date_end.replace("Z", "+00:00"))
                except Exception:
                    pass
            return None, close
    return None, None


def fetch_platform_times(market: Market) -> None:
    """Populate platform-specific open and close times for a Market object."""
    if market.platform == "metaculus":
        try:
            pub, close = fetch_metaculus_times(market.id)
            market.open_time = pub
            market.close_time = close
        except Exception:
            pass
    elif market.platform == "predictit":
        try:
            pub, close = fetch_predictit_times(market.id)
            market.open_time = pub
            market.close_time = close
        except Exception:
            pass
    # Add other platforms here, e.g. Polymarket, etc.
    else:
        # No platform-specific times available
        return


def main() -> None:
    client = MetaforecastClient()
    # Measure fetch time
    start_time = time.monotonic()
    markets = client.fetch_markets(page_size=1000)
    elapsed = time.monotonic() - start_time

    # Filter only open markets (probability strictly between 0 and 1)
    open_markets = [m for m in markets if m.probability is not None and 0 < m.probability < 1]

    # Populate open/close times and print each open market
    for m in open_markets:
        if m.platform not in  ["givewellopenphil", "xrisk", "wildeford", "smarkets"]:
            continue
        prob_str = f"{m.probability:.1%}" if m.probability is not None else "N/A"
        # fetch platform-specific times
        fetch_platform_times(m)
        opened = m.open_time.isoformat() if m.open_time else "N/A"
        closes = m.close_time.isoformat() if m.close_time else "N/A"
        print(
            f"{m.id} | {m.title} | {m.platform}"
            f" | opened: {opened} | closes: {closes} | prob: {prob_str} | {m.url}"
        )

    # Summarize counts per platform
    counts = Counter(m.platform for m in open_markets)
    print("\nSummary of open markets by platform:")
    for platform, count in counts.items():
        print(f"  {platform}: {count} market(s)")
    print(f"Total open markets: {len(open_markets)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()