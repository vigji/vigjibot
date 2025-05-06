from typing import List, Dict, Optional
import time
import requests
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
                    )
                )

            page_info = conn.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
            # throttle to avoid resource exhaustion
            time.sleep(self.sleep_interval)

        return markets


def main() -> None:
    client = MetaforecastClient()
    markets = client.fetch_markets(page_size=100)

    for m in markets:
        prob_str = f"{m.probability:.1%}" if m.probability is not None else "N/A"
        print(f"{m.id} | {m.title} | {m.platform} | {prob_str} | {m.url}")


if __name__ == "__main__":
    main()