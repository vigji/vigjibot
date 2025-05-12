# %%

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from embedding_utils import embed_questions_df


df_file = Path("/Users/vigji/code/vigjibot/data/combined_markets.csv")
pooled_df = pd.read_csv(df_file)
pooled_df = pooled_df.drop_duplicates(subset=["question"])

# %%
# %%
pooled_df.head()
gj_df = pooled_df[pooled_df.source_platform == "GJOpen"]
len(gj_df), len(gj_df.question.unique())
# %%
# ================================================
# Embed questions
# ================================================

embedded_df = embed_questions_df(pooled_df, question_column="question")
embedded_df["question"] = pooled_df["question"]
embedded_df["source_platform"] = pooled_df["source_platform"]
embedded_df["formatted_outcomes"] = pooled_df["formatted_outcomes"]
embedded_df = embedded_df.reset_index(drop=True)
embedded_df.head()


distance_matrix = get_distance_matrix(embedded_df)

# Create and show the visualization
embedded_df["closest_questions"] = embedded_df.apply(
    lambda row: get_closest_questions(row, distance_matrix, embedded_df, n_closest=20),
    axis=1,
)
embedded_df["closest_questions_text"] = embedded_df["closest_questions"].apply(
    lambda x: "\n".join(
        [f"{q}  {a} ({source}; {distance})" for q, a, source, distance in x]
    )
)
embedded_df.head()

for i in [0, 2] + list(np.random.randint(0, len(embedded_df), 10)):
    example = embedded_df.iloc[i]
    print(example.question, example.source_platform)
    print(example.closest_questions_text)
    print("-" * 100)
# %%
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity


def reduce_dimensions(embeddings_data, n_components=2):
    """Reduce dimensionality of embeddings using UMAP."""
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings_data)
    return reduced_embeddings


def create_visualization(df_to_viz):
    """Create an interactive plotly visualization of the embeddings."""

    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(
        df_to_viz.drop(
            [
                "source_platform",
                "question",
                "closest_questions_text",
                "closest_questions",
                "formatted_outcomes",
            ],
            axis=1,
        )
    )

    # Create visualization DataFrame
    viz_df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    viz_df["source_platform"] = df_to_viz["source_platform"]

    # Add question text and closest questions
    viz_df["question"] = df_to_viz["question"]
    viz_df["closest_questions"] = df_to_viz["closest_questions"]
    viz_df["formatted_outcomes"] = df_to_viz["formatted_outcomes"]
    viz_df["closest_questions_formatted"] = viz_df["closest_questions"].apply(
        lambda x: "<br>".join([f"• {q}" for q in x])
    )

    # Create the plot
    fig = px.scatter(
        viz_df,
        x="x",
        y="y",
        color="source_platform",
        color_discrete_map={
            "Metaculus": "red",
            "Polymarket": "gray",
            "GJOpen": "blue",
            "PredictIt": "lightgreen",
            "Manifold": "orange",
        },
        hover_data=[
            "question",
            "source_platform",
            "formatted_outcomes",
            "closest_questions_formatted",
        ],
        title="UMAP Visualization of Question Embeddings",
        labels={"x": "TSNE Component 1", "y": "TSNE Component 2"},
    )

    # Customize hover template
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "Question: %{customdata[0]} (%{customdata[1]})",
                "Outcomes: %{customdata[2]}",
                "Closest Questions:<br>%{customdata[3]}<br>",
                "<extra></extra>",
            ]
        )
    )

    # Update layout
    fig.update_layout(
        hovermode="closest",
        showlegend=True,
        legend_title_text="Source",
        width=1000,
        height=1000,
    )

    return fig


# %%
create_visualization(embedded_df)
# %%
# %%
