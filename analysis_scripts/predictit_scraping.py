import requests
from predictit_parser import parse_predictit_response, print_market_details


def get_predictit_markets():
    url = "https://www.predictit.org/api/marketdata/all/"
    response = requests.get(url)
    response.raise_for_status()
    return parse_predictit_response(response.json())


def main():
    markets = get_predictit_markets()
    print(f"Found {len(markets)} markets")
    
    # Print details for first 3 markets
    for market in markets[:10]:
        print_market_details(market)


if __name__ == "__main__":
    main()
