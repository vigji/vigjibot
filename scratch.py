# %%
from asknews_sdk import AsyncAskNewsSDK
import asyncio
import os

"""
More information available here:
https://docs.asknews.app/en/news
https://docs.asknews.app/en/deepnews

Installation:
pip install asknews
"""

or_key = os.getenv("OPENROUTER_API_KEY")
client_id = os.getenv("ASKNEWS_CLIENT_ID")
client_secret = os.getenv("ASKNEWS_SECRET")
print(or_key, client_id, client_secret)

ask = AsyncAskNewsSDK(
    client_id=client_id,
    client_secret=client_secret,
    scopes=["chat", "news", "stories", "analytics"],
)


# /news endpoint example
async def search_news(query):

  hot_response = await ask.news.search_news(
      query=query, # your natural language query
      n_articles=5, # control the number of articles to include in the context
      return_type="both",
      strategy="latest news" # enforces looking at the latest news only
  )

  print(hot_response.as_string)

  # get context from the "historical" database that contains a news archive going back to 2023
  historical_response = await ask.news.search_news(
      query=query,
      n_articles=10,
      return_type="both",
      strategy="news knowledge" # looks for relevant news within the past 60 days
  )

  print(historical_response.as_string)

# %%
# /deepnews endpoint example:
async def deep_research(
    query, sources, model, search_depth=2, max_depth=2
):

    # r = await ask.chat.list_chat_models()
    #print(r)

    response = await ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=search_depth,
        max_depth=max_depth,
        sources=sources,
        stream=False,
        return_sources=False,
        model=model,
        inline_citations="numbered"
    )

    print(response)


if __name__ == "__main__":
    query = """
    What is the TAM of the global market for electric vehicles in 2025? 

    Collect sources, and synthesize them in a final report avoiding overlapping information
    and making sure you mark news and opinion pieces with tags <news> ... </news> and <opinion> ... </opinion>
    """

    #     With your final report, please report the TAM in USD using the tags <TAM> ... </TAM>.

    sources = ["asknews"]
    model = "deepseek-basic"
    search_depth = 2
    max_depth = 2 

    # For now we do not have access here:
    asyncio.run(
        deep_research(
            query, sources, model, search_depth, max_depth
        )
    )

    # asyncio.run(search_news(query))