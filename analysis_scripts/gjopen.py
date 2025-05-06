import requests
from bs4 import BeautifulSoup
from pprint import pprint
from pathlib import Path
import json
import re
from urllib.parse import urljoin
from dataclasses import dataclass, field
from typing import List, Optional, Any

BASE_URL = "https://www.gjopen.com"
QUESTIONS_URL = f"{BASE_URL}/questions"
LOGIN_URL = f"{BASE_URL}/users/sign_in"

@dataclass
class Answer:
    name: str
    probability: Optional[float] = None
    # id: Optional[int] = None # Uncomment if you plan to use it

@dataclass
class Market:
    id: int
    name: str
    published_at: str
    predictors_count: int
    comments_count: int
    description: str
    binary: bool
    continuous_scored: bool
    answers: List[Answer]
    url: str
    q_type: str

    @classmethod
    def from_gjopen_question_data(cls, q_props: dict, question_url: str, all_props: dict) -> Optional["Market"]:
        if not q_props:
            return None

        answers_data = q_props.get("answers", [])
        answers = [
            Answer(
                name=a.get("name"),
                probability=a.get("probability"),
                # id=a.get("id")
            )
            for a in answers_data
        ]

        return cls(
            id=q_props.get("id"),
            name=q_props.get("name", ""),
            published_at=q_props.get("published_at"),
            predictors_count=q_props.get("predictors_count"),
            comments_count=q_props.get("comments_count"),
            description=q_props.get("description", ""),
            binary=q_props.get("binary?"),
            continuous_scored=q_props.get("continuous_scored?"),
            answers=answers,
            url=question_url,
            q_type=q_props.get("type")
        )


class GJOpenScraper:
    def __init__(self, email, password):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })
        self.email = email
        self.password = password
        self._login()

    def _login(self):
        # First get the login page to get the CSRF token
        login_page = self.session.get(LOGIN_URL)
        soup = BeautifulSoup(login_page.text, "html.parser")
        csrf_token = soup.select_one('meta[name="csrf-token"]')["content"]
        
        # Login with credentials
        login_data = {
            "user[email]": self.email,
            "user[password]": self.password,
            "authenticity_token": csrf_token
        }
        resp = self.session.post(LOGIN_URL, data=login_data)
        if "Invalid Email or password" in resp.text:
            raise ValueError("Login failed - check credentials")

    def fetch_question_links(self, page=1):
        url = f"{QUESTIONS_URL}?sort=predictors_count&sort_dir=desc&page={page}"
        resp = self.session.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # find question links matching /questions/<id>
        links = soup.find_all("a", href=re.compile(r"/questions/\d+"))
        # build full URLs
        return [urljoin(BASE_URL, link["href"]) for link in links]

    def fetch_prediction_data(self, question_url: str) -> Optional[Market]:
        resp = self.session.get(question_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        react_div = soup.find(
            "div", {"data-react-class": "FOF.Forecast.PredictionInterfaces.OpinionPoolInterface"}
        )

        if react_div and react_div.has_attr("data-react-props"):
            props = json.loads(react_div["data-react-props"])
            q_props = props.get("question", {})
            print("=====================")
            print(f"Raw data:")
            pprint(q_props)
            print("=====================")
            market_data = Market.from_gjopen_question_data(q_props, question_url, props)

            if not market_data:
                print(f"    No question data found in props for {question_url}")
                return None
            
            # if market_data.continuous_scored:
            print("=====================")
            print(f"question encountered: {question_url}")
            # props)
            print("=====================")
            
            return market_data


def load_credentials():
    creds_file = Path.home() / ".gjopen_credentials.json"
    if not creds_file.exists():
        raise FileNotFoundError(
            f"Please create {creds_file} with your GJ Open credentials in format:"
            '{"email": "your@email.com", "password": "yourpassword"}'
        )
    with open(creds_file) as f:
        return json.load(f)

# Example usage
creds = load_credentials()
scraper = GJOpenScraper(creds["email"], creds["password"])

all_data = []
for page in range(2, 3):
    print(f"Fetching page {page}...")
    question_links = scraper.fetch_question_links(page)
    print(f"Found {len(question_links)} question links")
    for link in question_links:
        print(f"  Scraping {link}")
        try:
            market_obj = scraper.fetch_prediction_data(link)
            if market_obj:
                all_data.append(market_obj)
            else:
                print(f"    No data retrieved for {link}")
        except Exception as e:
            print(f"    Failed to scrape {link}: {e}")

# Example: Print titles of scraped markets
for item in all_data:
    if isinstance(item, Market): # Ensure it's a Market object
        pprint(item)
    # If you still have raw responses in all_data from previous versions, handle them
    # elif isinstance(item, requests.Response):
    #     print(f"Raw response for URL: {item.url}")

