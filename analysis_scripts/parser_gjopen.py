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

BASE_URL = "https://www.gjopen.com"
QUESTIONS_URL = f"{BASE_URL}/questions"
LOGIN_URL = f"{BASE_URL}/users/sign_in"

@dataclass
class GJOpenAnswer:
    name: str
    probability: Optional[float] = None
    # id: Optional[int] = None # Uncomment if you plan to use it

@dataclass
class GJOpenMarket:
    id: int
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
    def from_gjopen_question_data(cls, q_props: dict, question_url: str) -> Optional["Market"]:
        if not q_props:
            return None

        outcomes_data = q_props.get("answers", [])
        outcomes = [
            GJOpenAnswer(
                name=a.get("name"),
                probability=a.get("probability"),
                # id=a.get("id")
            )
            for a in outcomes_data
        ]
        formatted_outcomes = "; ".join([f"{a.name.strip()}: {a.probability*100}%" for a in outcomes])
        formatted_outcomes = formatted_outcomes.replace("\n", "").replace("\r", "")
        return cls(
            id="gjopen_"+str(q_props.get("id")),
            question=q_props.get("name", ""),
            published_at=q_props.get("published_at"),
            predictors_count=q_props.get("predictors_count"),
            comments_count=q_props.get("comments_count"),
            description=q_props.get("description", ""),
            binary=q_props.get("binary?"),
            continuous_scored=q_props.get("continuous_scored?"),
            outcomes=outcomes,
            url=question_url,
            q_type=q_props.get("type"),
            formatted_outcomes=formatted_outcomes
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
                return None

            q_props = props.get("question", {})
            if not q_props:
                return None
              
            market_data = GJOpenMarket.from_gjopen_question_data(q_props, question_url)
            
            return market_data
        else:
            # This case might happen for questions without the specific react component (e.g. older ones, different types)
            # print(f"    No react props div found for {question_url}, cannot extract structured data.")
            return None


    def scrape_markets(self, num_pages_to_scrape: Optional[int] = None) -> pd.DataFrame:
        """
        Scrapes markets from Good Judgment Open.

        Args:
            num_pages_to_scrape: The number of pages to scrape. 
                                 Defaults to 10. If None, attempts to scrape all available pages.
        
        Returns:
            A pandas DataFrame containing the scraped market data.
        """
        PAUSE_AFTER_PAGE = 0.6
        PAUSE_AFTER_MARKET = 0.7
        all_markets_data: List[GJOpenMarket] = []

        if num_pages_to_scrape is None:
            num_pages_to_scrape = 15

        for page_num in tqdm(range(1, num_pages_to_scrape + 1), desc="Scraping pages"):
            # market_objs_on_page = _process_page(page_num)
            question_links = self._fetch_question_links_for_page(page_num)
            if not question_links:
                print(f"No question links found on page {page_num}.")
                return False
                
            market_objs_on_page = []
            for i, link in enumerate(question_links):
                # print(f"  Scraping {link}")
                try:
                    market_obj = self._fetch_market_data_for_url(link)
                    if market_obj:
                        market_objs_on_page.append(market_obj)

                except Exception as e:
                    print(f"    Failed to process {link}: {e}")
                finally:
                    if i < len(question_links) - 1: # Don't sleep after the last item
                        time.sleep(PAUSE_AFTER_MARKET)
            if not market_objs_on_page:
                print("Stopping early.")
                break
            all_markets_data.extend(market_objs_on_page)
            if page_num < num_pages_to_scrape: # Don't sleep after the last page
                time.sleep(PAUSE_AFTER_PAGE)

        if not all_markets_data:
            return pd.DataFrame()

        df = pd.DataFrame([market_obj.__dict__ for market_obj in all_markets_data])
        return df

if __name__ == "__main__":
    print("Starting GJOPEscraper example...")
    # To use credentials from environment variables:
    # Ensure GJO_EMAIL and GJO_PASSWORD are set.
    # scraper = GoodJudgmentOpenScraper()

    # To use credentials from a file (if env vars are not set):
    # Ensure ~/.gjopen_credentials.json exists and is correctly formatted.
    # scraper = GoodJudgmentOpenScraper()
    
    # To provide credentials directly (less secure, for testing):
    # scraper = GoodJudgmentOpenScraper(email="your@email.com", password="yourpassword")
    
    # Default: tries env vars, then file.
    scraper = GoodJudgmentOpenScraper()

    # Scrape a few pages (e.g., 2)
    num_pages = 5
    print(f"\nScraping the first {num_pages} page(s) of markets sorted by predictor count...")
    start_time = time.time()
    markets_df = scraper.scrape_markets(num_pages_to_scrape=num_pages)
    end_time = time.time()
    print(f"Scraping took {end_time - start_time:.2f} seconds")
    print(f"Scraped {len(markets_df)} markets")
    markets_df.to_csv("gjopen_markets.csv", index=False)

