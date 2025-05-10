import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time

from common_markets import PooledMarket
from parser_gjopen import GoodJudgmentOpenScraper
from parser_manimarket import ManifoldScraper
from parser_polygamma import PolymarketGammaScraper
from parser_predictit import PredictItScraper

async def fetch_all_markets(only_open: bool = True) -> List[PooledMarket]:
    """
    Fetch markets from all available platforms.
    
    Args:
        only_open: If True, fetches only open markets.
    
    Returns:
        List of PooledMarket objects from all platforms.
    """
    scrapers = [
        GoodJudgmentOpenScraper(),
        ManifoldScraper(),
        PolymarketGammaScraper(),
        PredictItScraper(),
    ]
    
    all_markets: List[PooledMarket] = []
    
    for scraper in scrapers:
        try:
            platform_name = scraper.__class__.__name__.replace("Scraper", "")
            print(f"\nFetching markets from {platform_name}...")
            start_time = time.time()
            
            markets = await scraper.get_pooled_markets(only_open=only_open)
            
            end_time = time.time()
            print(f"Fetched {len(markets)} markets from {platform_name} in {end_time - start_time:.2f} seconds")
            all_markets.extend(markets)
            
        except Exception as e:
            print(f"Error fetching from {scraper.__class__.__name__}: {e}")
    
    return all_markets

def create_markets_dataframe(markets: List[PooledMarket]) -> pd.DataFrame:
    """
    Convert a list of PooledMarket objects to a pandas DataFrame.
    
    Args:
        markets: List of PooledMarket objects.
    
    Returns:
        DataFrame with all market data.
    """
    # Convert to list of dicts, excluding raw_market_data
    market_dicts = []
    for market in markets:
        market_dict = market.__dict__.copy()
        market_dict.pop('raw_market_data', None)  # Remove raw data to keep DataFrame clean
        market_dicts.append(market_dict)
    
    df = pd.DataFrame(market_dicts)
    
    # Sort by published_at if available
    if 'published_at' in df.columns:
        df = df.sort_values('published_at', ascending=False)
    
    return df

async def main():
    print("Starting market aggregation from all platforms...")
    start_time = time.time()
    
    # Fetch markets from all platforms
    all_markets = await fetch_all_markets(only_open=True)
    
    # Create DataFrame
    df = create_markets_dataframe(all_markets)
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    print(f"Total markets collected: {len(df)}")
    
    # Print summary by platform
    print("\nMarkets by platform:")
    print(df['source_platform'].value_counts())
    
    # Save to CSV
    output_path = Path("data/combined_markets.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved combined markets to {output_path}")
    
    # Print sample of the data
    print("\nSample of combined markets:")
    print(df[['id', 'question', 'source_platform', 'published_at', 'is_resolved']].head())

if __name__ == "__main__":
    asyncio.run(main())
