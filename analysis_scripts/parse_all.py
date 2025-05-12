import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time
import json
from datetime import datetime

from common_markets import PooledMarket
from parser_gjopen import GoodJudgmentOpenScraper
from parser_manimarket import ManifoldScraper
from parser_polygamma import PolymarketGammaScraper
from parser_predictit import PredictItScraper

def save_markets_to_cache(markets: List[PooledMarket], platform: str) -> Path:
    """Save markets to a cache file with timestamp."""
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_file = cache_dir / f"{platform}_{timestamp}.json"
    
    # Convert markets to dict, handling datetime serialization
    market_dicts = []
    for market in markets:
        market_dict = market.__dict__.copy()
        market_dict.pop('raw_market_data', None)
        
        # Convert datetime to string if present
        if 'published_at' in market_dict and market_dict['published_at'] is not None:
            market_dict['published_at'] = market_dict['published_at'].isoformat()
        
        market_dicts.append(market_dict)
    
    with open(cache_file, 'w') as f:
        json.dump(market_dicts, f, indent=2)
    
    return cache_file

async def fetch_platform_markets(scraper, only_open: bool) -> tuple[str, List[PooledMarket]]:
    """Fetch markets from a single platform."""
    platform_name = scraper.__class__.__name__.replace("Scraper", "")
    print(f"\nFetching markets from {platform_name}...")
    start_time = time.time()
    
    try:
        async with scraper:  # Use async context manager for proper session handling
            markets = await scraper.get_pooled_markets(only_open=only_open)
            
            # Save to cache
            cache_file = save_markets_to_cache(markets, platform_name)
            print(f"Saved {len(markets)} markets to cache: {cache_file}")
            
            end_time = time.time()
            print(f"Fetched {len(markets)} markets from {platform_name} in {end_time - start_time:.2f} seconds")
            return platform_name, markets
    except Exception as e:
        print(f"Error fetching from {platform_name}: {e}")
        return platform_name, []

async def fetch_all_markets(only_open: bool = True) -> List[PooledMarket]:
    """
    Fetch markets from all available platforms in parallel.
    
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
    
    # Fetch from all platforms in parallel
    results = await asyncio.gather(
        *[fetch_platform_markets(scraper, only_open) for scraper in scrapers],
        return_exceptions=True  # Handle exceptions gracefully
    )
    
    # Combine results, handling any exceptions
    all_markets: List[PooledMarket] = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error in parallel fetch: {result}")
            continue
        platform_name, markets = result
        all_markets.extend(markets)
    
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
    
    # Handle published_at column if it exists
    if 'published_at' in df.columns:
        try:
            # First convert all to datetime, handling timezone-aware and naive
            df['published_at'] = pd.to_datetime(df['published_at'])
            
            # Then convert all to timezone-naive
            df['published_at'] = df['published_at'].apply(
                lambda x: x.tz_localize(None) if x.tz is not None else x
            )
            
            df = df.sort_values('published_at', ascending=False)
        except Exception as e:
            print(f"Warning: Error processing published_at column: {e}")
            print("Continuing without sorting by published_at")
    
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
