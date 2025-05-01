# %%
import os
import dotenv
from pathlib import Path
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
import requests
from pprint import pprint

# %%

# Set your private key as an environment variable or replace os.getenv("PK") with your key string
dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
private_key = os.getenv("PK")
host = "https://clob.polymarket.com"
chain_id = POLYGON  # Polygon Mainnet

# Initialize the client
client = ClobClient(host, key=private_key, chain_id=chain_id)

# Generate API credentials
api_creds = client.create_or_derive_api_creds()

# Access the credentials
print("API Key:", api_creds.api_key)
print("Secret:", api_creds.api_secret)
print("Passphrase:", api_creds.api_passphrase)


from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

# Assuming client is already initialized as shown above

# %% 
# %%
# Fetch open markets
open_markets = client.get_markets() # status="open", limit=100)
open_markets
# %%
# Display market information
for market in open_markets:
    print(f"Market ID: {market['id']}, Question: {market['question']}")

# %%
type(open_markets)
# %%
GET_MARKETS = "/markets"
GET_MARKET = "/markets/"
GET = "GET"
next_cursor="MA=="

def request(endpoint: str, method: str, headers=None, data=None):
    try:
        headers = overloadHeaders(method, headers)
        resp = requests.request(
            method=method, url=endpoint, headers=headers, json=data if data else None
        )
        if resp.status_code != 200:
            raise PolyApiException(resp)

        try:
            return resp.json()
        except requests.JSONDecodeError:
            return resp.text

    except requests.RequestException:
        raise PolyApiException(error_msg="Request exception!")


def get(endpoint, headers=None, data=None):
    return request(endpoint, GET, headers, data)

get("{}{}?next_cursor={}".format(client.host, GET_MARKETS, next_cursor))
# %%
