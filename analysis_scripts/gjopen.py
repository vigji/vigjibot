import requests
from bs4 import BeautifulSoup
from pprint import pprint
from pathlib import Path
import json

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
        url = f"{QUESTIONS_URL}?page={page}"
        resp = self.session.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select("a.card-question")
        return [BASE_URL + link["href"] for link in links]

    def fetch_prediction_data(self, question_url):
        resp = self.session.get(question_url)

        soup = BeautifulSoup(resp.text, "html.parser")
        pprint(soup)
        title = soup.select_one("h1.question-title").text.strip()
        forecast_blocks = soup.select(".forecast-answer")
        forecasts = []
        for block in forecast_blocks:
            label = block.select_one(".forecast-label").text.strip()
            percent = block.select_one(".forecast-value").text.strip()
            forecasts.append((label, percent))
        return {"url": question_url, "title": title, "forecasts": forecasts}

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
    for link in question_links:
        print(f"  Scraping {link}")
        try:
            data = scraper.fetch_prediction_data(link)
            all_data.append(data)
        except Exception as e:
            print(f"    Failed to scrape {link}: {e}")

# Print example output
pprint(all_data[:2])
