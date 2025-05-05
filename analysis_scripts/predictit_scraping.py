import requests

url = "https://www.predictit.org/api/marketdata/all/"

response = requests.get(url)
print(response.json())
