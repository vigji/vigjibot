# %%
%load_ext autoreload
%autoreload 2
import os
import dotenv
from pathlib import Path
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from regex import D
import requests
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import numpy as np
from openai import OpenAI

# %%

def initialize_client():
    """Initialize Polymarket CLOB client and print API credentials"""
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
    private_key = os.getenv("PK")
    host = "https://clob.polymarket.com"
    chain_id = POLYGON  # Polygon Mainnet

    client = ClobClient(host, key=private_key, chain_id=chain_id)
    api_creds = client.create_or_derive_api_creds()

    print("API Key:", api_creds.api_key)
    print("Secret:", api_creds.api_secret)
    print("Passphrase:", api_creds.api_passphrase)

    return client

def fetch_all_markets(client, max_pages=200, return_active=True):
    """Fetch all markets from Polymarket CLOB API"""
    markets = client.get_markets()
    all_markets = markets['data']
    
    page = 0
    for _ in tqdm(range(max_pages)):
        if not markets.get('next_cursor') or markets.get('next_cursor') == "LTE=":
            print(f"No more pages to fetch")
            break
        page += 1
        if page > max_pages:
            print(f"Reached max pages: {max_pages}")
            break
        next_cursor = markets['next_cursor']
        markets = client.get_markets(next_cursor=next_cursor)
        all_markets.extend(markets['data'])
    
    df = pd.DataFrame(all_markets)
    if not return_active:
        return df
    else:
        return df[df["active"] & ~df["closed"]].reset_index(drop=True)

# Initialize client and fetch markets
client = initialize_client()
df = fetch_all_markets(client)

active_df = df[df["active"] & ~df["closed"]].reset_index(drop=True)


# %%
meta_questions_df = pd.read_csv("questions_df.csv", index_col=0)

from embedding_utils import embed_questions_df

chunk_size = 200
poly_embeddings_df = embed_questions_df(active_df, chunk_size=chunk_size, question_column="question")
meta_embeddings_df = embed_questions_df(meta_questions_df, question_column="question_text")

poly_embeddings_df["question_text"] = active_df["question"]
meta_embeddings_df["question_text"] = meta_questions_df["question_text"]
poly_embeddings_df["source"] = "Polymarket"
meta_embeddings_df["source"] = "Metaculus"
combined_df = pd.concat([meta_embeddings_df, poly_embeddings_df], axis=0)
# %%
from sklearn.manifold import TSNE
import plotly.express as px


def reduce_dimensions(embeddings_df, n_components=2):
    """Reduce dimensionality of embeddings using UMAP."""
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings_df)
    return reduced_embeddings

def create_visualization(combined_df):
    """Create an interactive plotly visualization of the embeddings."""
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(combined_df.drop('source', axis=1))
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    viz_df['source'] = combined_df['source']
    
    # Add question text
    viz_df['question_text'] = ''

    # Create the plot
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='source',
        color_discrete_map={'Metaculus': 'red', 'Polymarket': 'gray'},
        hover_data=['question_text'],
        title='UMAP Visualization of Question Embeddings',
        labels={'x': 'UMAP Component 1', 'y': 'UMAP Component 2'}
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate="<br>".join([
            "Question: %{customdata[0]}",
            "<extra></extra>"
        ])
    )
    
    # Update layout
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        legend_title_text='Source'
    )
    
    return fig


# Create and show the visualization
fig = create_visualization(combined_df)
fig.show() 
# %%