import requests
from bs4 import BeautifulSoup
from pprint import pprint
from pathlib import Path
import json
import re
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import List, Optional
import os
import pandas as pd
import time
from tqdm import tqdm

from common_markets import PooledMarket, BaseMarket, parse_datetime_flexible

BASE_URL = "https://www.gjopen.com"
QUESTIONS_URL = f"{BASE_URL}/questions"
LOGIN_URL = f"{BASE_URL}/users/sign_in"

@dataclass
class GJOpenAnswer:
    name: str
    probability: Optional[float] = None
    # id: Optional[int] = None # Uncomment if you plan to use it

@dataclass
class GJOpenMarket(BaseMarket):
    id: str # Changed from int
    question: str
    published_at: str
    predictors_count: int
    comments_count: int
    description: str
    binary: bool
    continuous_scored: bool
    outcomes: List[GJOpenAnswer]

    formatted_outcomes: str
    url: str
    q_type: str

    @classmethod
    def from_gjopen_question_data(cls, q_props: dict, question_url: str) -> Optional["GJOpenMarket"]: # Changed Market to GJOpenMarket
        if not q_props:
            return None

        outcomes_data = q_props.get("answers", [])
        outcomes_list = [
            GJOpenAnswer(
                name=a.get("name"),
                probability=a.get("probability"),
                # id=a.get("id")
            )
            for a in outcomes_data
        ]
        # Updated formatted_outcomes to handle None probabilities
        formatted_outcomes_str = "; ".join([
            f"{a.name.strip()}: {f'{a.probability*100:.1f}%' if a.probability is not None else 'N/A'}" 
            for a in outcomes_list
        ])
        formatted_outcomes_str = formatted_outcomes_str.replace("\n", "").replace("\r", "")
        
        return cls(
            id="gjopen_"+str(q_props.get("id")), # id is now string
            question=q_props.get("name", ""),
            published_at=q_props.get("published_at"),
            predictors_count=q_props.get("predictors_count"),
            comments_count=q_props.get("comments_count"),
            description=q_props.get("description", ""),
            binary=bool(q_props.get("binary?")), # Ensure bool conversion
            continuous_scored=bool(q_props.get("continuous_scored?")), # Ensure bool conversion
            outcomes=outcomes_list,
            url=question_url,
            q_type=q_props.get("type"),
            formatted_outcomes=formatted_outcomes_str
        )

    def to_pooled_market(self) -> PooledMarket:
        outcome_names = [ans.name for ans in self.outcomes]
        outcome_probs = [ans.probability for ans in self.outcomes]

        return PooledMarket(
            id=self.id,
            question=self.question,
            outcomes=outcome_names,
            outcome_probabilities=outcome_probs,
            formatted_outcomes=self.formatted_outcomes,
            url=self.url,
            published_at=parse_datetime_flexible(self.published_at),
            source_platform="GJOpen",
            volume=None,  # Not available directly from GJOpen API structure shown
            n_forecasters=self.predictors_count,
            comments_count=self.comments_count,
            original_market_type=self.q_type,
            is_resolved=None, # No direct field, q_type might hint but not a clear boolean
            raw_market_data=self
        )

class GoodJudgmentOpenScraper:
    """
    Scrapes market data from Good Judgment Open.
    Handles authentication, paginated question fetching, and data parsing.
    """
    BASE_URL = "https://www.gjopen.com"
    QUESTIONS_URL = f"{BASE_URL}/questions"
    LOGIN_URL = f"{BASE_URL}/users/sign_in"

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        """
        Initializes the scraper and logs in.
        Authentication credentials can be provided directly or loaded from
        environment variables (GJO_EMAIL, GJO_PASSWORD) or a JSON file
        (~/.gjopen_credentials.json).
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; PythonScraper/1.0)"
        })

        env_email = os.getenv("GJO_EMAIL")
        env_password = os.getenv("GJO_PASSWORD")

        if email and password:
            self.email = email
            self.password = password
        elif env_email and env_password:
            self.email = env_email
            self.password = env_password
        else:
            creds = self._load_credentials_from_file()
            self.email = creds["email"]
            self.password = creds["password"]
        
        self._login()

    def _load_credentials_from_file(self):
        creds_file = Path.home() / ".gjopen_credentials.json"
        if not creds_file.exists():
            raise FileNotFoundError(
                f"Credentials not provided and {creds_file} not found. "
                "Please provide email/password, set GJO_EMAIL/GJO_PASSWORD env vars, "
                "or create the JSON credentials file with format: "
                '{"email": "your@email.com", "password": "yourpassword"}'
            )
        with open(creds_file) as f:
            return json.load(f)

    def _login(self):
        """Logs into Good Judgment Open."""
        try:
            login_page = self.session.get(self.LOGIN_URL, timeout=10)
            login_page.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch login page: {e}")

        soup = BeautifulSoup(login_page.text, "html.parser")
        csrf_token_tag = soup.select_one('meta[name="csrf-token"]')
        if not csrf_token_tag or not csrf_token_tag.get("content"):
            raise ValueError("Could not find CSRF token on login page.")
        csrf_token = csrf_token_tag["content"]
        
        login_data = {
            "user[email]": self.email,
            "user[password]": self.password,
            "authenticity_token": csrf_token
        }
        try:
            resp = self.session.post(self.LOGIN_URL, data=login_data, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Login request failed: {e}")

        if "Invalid Email or password" in resp.text or "sign_in" in resp.url:
            raise ValueError("Login failed - please check credentials.")
        # print("Successfully logged in.")

    def _fetch_question_links_for_page(self, page: int) -> List[str]:
        """Fetches all question links from a given results page."""
        url = f"{self.QUESTIONS_URL}?sort=predictors_count&sort_dir=desc&page={page}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"    Warning: Failed to fetch page {page}: {e}")
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=re.compile(r"/questions/\d+"))
        return [urljoin(self.BASE_URL, link["href"]) for link in links]

    def _fetch_market_data_for_url(self, question_url: str) -> Optional[GJOpenMarket]:
        """Fetches and parses market data for a single question URL."""
        try:
            resp = self.session.get(question_url, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"    Warning: Failed to fetch market data for {question_url}: {e}")
            return None
            
        soup = BeautifulSoup(resp.text, "html.parser")
        react_div = soup.find(
            "div", {"data-react-class": "FOF.Forecast.PredictionInterfaces.OpinionPoolInterface"}
        )

        if react_div and react_div.has_attr("data-react-props"):
            try:
                props_str = react_div["data-react-props"]
                props = json.loads(props_str)
            except json.JSONDecodeError:
                # print(f"    Warning: Failed to decode JSON props for {question_url}")
                return None

            q_props = props.get("question", {})
            if not q_props:
                # print(f"    Warning: 'question' key missing in props for {question_url}")
                return None
              
            market_data = GJOpenMarket.from_gjopen_question_data(q_props, question_url)
            
            return market_data
        else:
            # This case might happen for questions without the specific react component (e.g. older ones, different types)
            # print(f"    No react props div found for {question_url}, cannot extract structured data.")
            return None

    def fetch_markets(self, max_pages: int = 15, only_open: bool = True) -> List[GJOpenMarket]:
        """
        Fetches markets from Good Judgment Open.

        Args:
            max_pages: The maximum number of pages to scrape. Defaults to 15.
            only_open: If True, attempts to fetch only open markets. 
                       (Note: GJOpen API for listing questions doesn't directly support
                        filtering by 'open' status, so this flag is a placeholder for
                        potential future post-filtering logic if resolution status becomes available.)
        
        Returns:
            A list of GJOpenMarket objects.
        """
        PAUSE_AFTER_PAGE = 0.6
        PAUSE_AFTER_MARKET = 0.7
        all_markets_data: List[GJOpenMarket] = []

        # The only_open flag is noted here, but GJOpen's question listing API 
        # doesn't have a direct filter for open/closed status. 
        # Filtering would typically happen after fetching if resolution status was available.
        if only_open:
            # print("Note: 'only_open' is set to True, but GJOpen API does not directly filter by open status when listing questions.")
            pass

        for page_num in tqdm(range(1, max_pages + 1), desc="Scraping GJOpen pages"):
            question_links = self._fetch_question_links_for_page(page_num)
            if not question_links:
                print(f"No more question links found on page {page_num}. Stopping.")
                break 
                
            market_objs_on_page: List[GJOpenMarket] = []
            for i, link in enumerate(question_links):
                try:
                    market_obj = self._fetch_market_data_for_url(link)
                    if market_obj:
                        market_objs_on_page.append(market_obj)
                except Exception as e:
                    print(f"    Failed to process {link}: {e}")
                finally:
                    if i < len(question_links) - 1:
                        time.sleep(PAUSE_AFTER_MARKET)
            
            if not market_objs_on_page and question_links: # Page had links but no markets parsed
                 print(f"No market objects successfully parsed from page {page_num}, though links were found. Stopping.")
                 break
            
            all_markets_data.extend(market_objs_on_page)
            
            if not market_objs_on_page and not question_links: # No links and no markets, definitely stop
                print(f"Stopping early on page {page_num} as no links or markets were found.")
                break

            if page_num < max_pages:
                time.sleep(PAUSE_AFTER_PAGE)

        return all_markets_data

if __name__ == "__main__":
    print("Starting GoodJudgmentOpenScraper example...")
    
    # Default: tries env vars, then file.
    # Ensure GJO_EMAIL and GJO_PASSWORD are set in your environment,
    # or you have a ~/.gjopen_credentials.json file.
    try:
        scraper = GoodJudgmentOpenScraper()
        print("Successfully initialized and logged into Good Judgment Open.")
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"Error initializing scraper: {e}")
        print("Please ensure credentials are set up via environment variables (GJO_EMAIL, GJO_PASSWORD) or ~/.gjopen_credentials.json")
        exit(1)

    num_pages_to_fetch = 2 
    print(f"Fetching the first {num_pages_to_fetch} page(s) of markets sorted by predictor count...")
    start_time = time.time()
    
    # Fetch markets (returns List[GJOpenMarket])
    # only_open=True is passed but note GJOpen's API limitations mentioned in fetch_markets docstring
    gjopen_markets_list = scraper.fetch_markets(max_pages=num_pages_to_fetch, only_open=True)
    end_time = time.time()

    print(f"Fetching took {end_time - start_time:.2f} seconds.")
    print(f"Fetched {len(gjopen_markets_list)} GJOpen markets.")

    if gjopen_markets_list:
        # Convert to PooledMarket format
        pooled_markets = [market.to_pooled_market() for market in gjopen_markets_list]
        print(f"Converted {len(pooled_markets)} markets to PooledMarket format.")

        if pooled_markets:
            print("Details of the first pooled market:")
            pprint(pooled_markets[0].__dict__)

            # Optional: Create a DataFrame from the pooled markets for analysis or CSV export
            df_pooled = pd.DataFrame([pm.__dict__ for pm in pooled_markets])
            # print(f"Created DataFrame with {len(df_pooled)} pooled markets. Columns: {df_pooled.columns.tolist()}")
            # print(df_pooled.head())
            # Example: Save to CSV
            # df_pooled.to_csv("gjopen_pooled_markets.csv", index=False)
            # print("Saved pooled markets to gjopen_pooled_markets.csv")
        else:
            print("No markets were successfully converted to PooledMarket format.")
    else:
        print("No markets were fetched from Good Judgment Open.")

