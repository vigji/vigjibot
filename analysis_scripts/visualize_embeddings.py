import pandas as pd
import numpy as np
import plotly.express as px
from umap import UMAP
from pathlib import Path


def combine_embeddings(meta_embeddings_df, poly_embeddings_df):
    """Combine embeddings from both sources into a single DataFrame with source labels."""
    meta_df = meta_embeddings_df.copy()
    poly_df = poly_embeddings_df.copy()

    # Add source column
    meta_df["source"] = "Metaculus"
    poly_df["source"] = "Polymarket"

    # Combine the dataframes
    combined_df = pd.concat([meta_df, poly_df], axis=0)
    return combined_df


def reduce_dimensions(embeddings_df, n_components=2):
    """Reduce dimensionality of embeddings using UMAP."""
    umap = UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = umap.fit_transform(embeddings_df)
    return reduced_embeddings


def create_visualization(meta_embeddings_df, poly_embeddings_df, meta_questions_df):
    """Create an interactive plotly visualization of the embeddings."""
    # Combine embeddings
    combined_df = combine_embeddings(meta_embeddings_df, poly_embeddings_df)

    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(combined_df.drop("source", axis=1))

    # Create visualization DataFrame
    viz_df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    viz_df["source"] = combined_df["source"]

    # Add question text
    viz_df["question_text"] = ""
    viz_df.loc[viz_df["source"] == "Metaculus", "question_text"] = meta_questions_df[
        "question_text"
    ].values
    viz_df.loc[
        viz_df["source"] == "Polymarket", "question_text"
    ] = poly_embeddings_df.index.map(lambda x: x if isinstance(x, str) else "")

    # Create the plot
    fig = px.scatter(
        viz_df,
        x="x",
        y="y",
        color="source",
        color_discrete_map={"Metaculus": "red", "Polymarket": "gray"},
        hover_data=["question_text"],
        title="UMAP Visualization of Question Embeddings",
        labels={"x": "UMAP Component 1", "y": "UMAP Component 2"},
    )

    # Customize hover template
    fig.update_traces(
        hovertemplate="<br>".join(["Question: %{customdata[0]}", "<extra></extra>"])
    )

    # Update layout
    fig.update_layout(hovermode="closest", showlegend=True, legend_title_text="Source")

    return fig


if __name__ == "__main__":
    # Load the data
    meta_questions_df = pd.read_csv("questions_df.csv", index_col=0)
    meta_embeddings_df = pd.read_csv("meta_embeddings.csv", index_col=0)
    poly_embeddings_df = pd.read_csv("poly_embeddings.csv", index_col=0)

    # Create and show the visualization
    fig = create_visualization(
        meta_embeddings_df, poly_embeddings_df, meta_questions_df
    )
    fig.show()
