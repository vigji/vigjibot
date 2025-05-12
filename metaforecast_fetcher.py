from typing import List, Dict, Optional
import requests
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Market:
    id: str
    title: str
    url: str
    platform: str
    options: List[Dict[str, float]]
    probability: Optional[float] = None
    resolve_time: Optional[datetime] = None


class MetaforecastClient:
    """Client for fetching markets from the Metaforecast GraphQL API using cursor-based pagination."""

    API_URL = "https://metaforecast.org/api/graphql"

    def __init__(self) -> None:
        self.headers = {"Content-Type": "application/json"}

    def fetch_markets(self, page_size: int = 100) -> List[Market]:
        """Fetch all open markets up to the given page size per request."""
        query = """
        query GetMarkets($first: Int!, $after: String) {
          questions(first: $first, after: $after, orderBy: FIRST_SEEN_DESC) {
            nodes {
              id
              title
              url
              platform { id }
              options { name probability type }
              extra { resolution_data { resolve_time } }
            }
            pageInfo { hasNextPage endCursor }
          }
        }
        """
        markets: List[Market] = []
        after: Optional[str] = None

        while True:
            variables = {"first": page_size, "after": after}
            response = requests.post(
                self.API_URL,
                json={"query": query, "variables": variables},
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")

            conn = data["data"]["questions"]
            nodes = conn.get("nodes", [])
            for node in nodes:
                opts = [
                    {"name": opt["name"], "probability": opt["probability"]}
                    for opt in node.get("options", [])
                ]
                res_data = node.get("extra", {}).get("resolution_data", {})
                rt_str = res_data.get("resolve_time")
                rt = datetime.fromisoformat(rt_str) if rt_str else None
                markets.append(
                    Market(
                        id=node["id"],
                        title=node.get("title", ""),
                        url=node.get("url", ""),
                        platform=node.get("platform", {}).get("id", ""),
                        options=opts,
                        probability=opts[0]["probability"] if opts else None,
                        resolve_time=rt,
                    )
                )

            page_info = conn.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")

        return markets


def main() -> None:
    client = MetaforecastClient()
    markets = client.fetch_markets(page_size=100)
    now = datetime.utcnow()
    for m in markets:
        status = "Open" if (m.resolve_time and m.resolve_time > now) else "Closed"
        print(f"{m.id} | {m.title} | {m.platform} | {status} | {m.url}")


if __name__ == "__main__":
    main()
