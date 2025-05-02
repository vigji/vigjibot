import pandas as pd
from pathlib import Path
from polymarket_scraping import initialize_client, fetch_all_markets

def save_poly_questions():
    """Save Polymarket questions to a CSV file."""
    client = initialize_client()
    df = fetch_all_markets(client)
    active_df = df[df["active"] & ~df["closed"]].reset_index(drop=True)
    active_df.to_csv("poly_questions.csv")

if __name__ == "__main__":
    save_poly_questions() 