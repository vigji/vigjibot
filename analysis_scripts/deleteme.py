# %%
#scrape this page with beautiful soup:

from pprint import pprint
from bs4 import BeautifulSoup
from gjopen import GJOpenScraper, load_credentials

# target URL for crowd forecast
url = (
    "https://www.gjopen.com/questions/3801-"
    "will-nato-and-or-a-nato-member-state-publicly-announce-"
    "that-it-has-deployed-armed-forces-to-ukraine-before-1-july-2025"
    "/crowd_forecast"
)

# authenticate and fetch via scraper
creds = load_credentials()
scraper = GJOpenScraper(creds["email"], creds["password"])
response = scraper.session.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# %%
soup
# %%
