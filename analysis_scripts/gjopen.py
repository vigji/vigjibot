import requests
from bs4 import BeautifulSoup
from pprint import pprint
from pathlib import Path
import json
import re
from urllib.parse import urljoin

BASE_URL = "https://www.gjopen.com"
QUESTIONS_URL = f"{BASE_URL}/questions"
LOGIN_URL = f"{BASE_URL}/users/sign_in"

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

    def fetch_prediction_data(self, question_url):
        # get page and parse for metadata
        resp = self.session.get(question_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # metadata from React props if available
        react_div = soup.find(
            "div", {"data-react-class": "FOF.Forecast.PredictionInterfaces.OpinionPoolInterface"}
        )
        if react_div and react_div.has_attr("data-react-props"):
            props = json.loads(react_div["data-react-props"])
            print("=====================")
            pprint(props)
            print("=====================")
            q = props.get("question", {})
            title = q.get("name", "")
            q_id = q.get("id")
            time_posted = q.get("published_at")
            num_forecasters = q.get("predictors_count")
            num_comments = q.get("comments_count")
            description = q.get("description", "")
            binary = q.get("binary?")
            continuous = q.get("continuous?")
        else:
            # fallback HTML title only
            elem = soup.select_one("h1.question-title")
            title = elem.text.strip() if elem else ""
            num_forecasters = None
            accuracy = None
        # always pull the crowd forecast table for probabilities
        # forecasts = self.fetch_crowd_forecast(question_url)
        # print(num_forecasters, accuracy, forecasts)
        resp = {
            "url": question_url,
            "title": title,
            "num_forecasters": num_forecasters,
            # "forecasts": forecasts,
        }
        # print(resp)
        return resp

    # def fetch_crowd_forecast(self, question_url):  # pragma: no cover
    #     """Fetch the crowd forecast table and return list of (answer, probability)."""
    #     url = question_url.rstrip("/") + "/crowd_forecast"
    #     resp = self.session.get(url)
    #     soup = BeautifulSoup(resp.text, "html.parser")
    #     table = soup.find("table", class_="consensus-table")
    #     forecasts = []
    #     if table:
    #         for row in table.select("tbody tr"):
    #             cells = row.find_all("td")
    #             if len(cells) >= 2:
    #                 label = cells[0].get_text(strip=True)
    #                 percent = cells[1].get_text(strip=True)
    #                 forecasts.append((label, percent))
    #     return forecasts

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
    for link in question_links[:10]:
        print(f"  Scraping {link}")
        try:
            data = scraper.fetch_prediction_data(link)
            all_data.append(data)
        except Exception as e:
            print(f"    Failed to scrape {link}: {e}")

