# %%

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from parser_metaculus import MyMetaculusApi

df_file = Path("/Users/vigji/code/vigjibot/data/combined_markets.csv")
df = pd.read_csv(df_file)

from embedding_utils import embed_questions_df
# %%
meta_df = MyMetaculusApi.get_all_questions_df()
# %%
meta_df.head()
# %%
df.source_platform.value_counts()
# %%
df.head()
# %%
poly_df = df[df.source_platform == "Polymarket"]
poly_df[poly_df.volume > 10000].head()
sum(poly_df.volume > 20000), len(poly_df)
# Check that there are some english characters in the question
# poly_df[~poly_df.question.str.contains(r'[a-zA-Z]')].head()
plt.figure()
plt.hist(np.log(poly_df.volume), np.arange(0, 20, 0.5))
plt.show()
# %%
gjopen_df = df[df.source_platform == "GJOpen"]
# %%

# ================================================
# Embed questions
# ================================================

combined_df = pd.concat([meta_df, df])
embedded_df = embed_questions_df(df, question_column="question")
# %%
embedded_df.head()
embedded_df["question"] = df["question"]
embedded_df["source_platform"] = df["source_platform"]

# %%
from sklearn.metrics.pairwise import cosine_similarity

def get_distance_matrix(combined_df):
    """Create a distance matrix of the embeddings."""
    embeddings = combined_df.drop(['source_platform', 'question'], axis=1)
    cosine_similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - cosine_similarity_matrix

    return distance_matrix


def get_closest_questions(row, distance_matrix, n_closest=10):
    """Get the 10 closest questions from the distance matrix."""
    question_text = row['question']
    question_index = combined_df.index[combined_df['question'] == question_text].tolist()[0]
    distances = distance_matrix[question_index]
    
    # Keep getting more indices until we have enough Polymarket questions
    n_to_fetch = n_closest
    poly_questions = []
    while len(poly_questions) < n_closest:
        closest_indices = np.argsort(distances)[:n_to_fetch]
        closest_questions = combined_df.iloc[closest_indices]
        poly_questions = closest_questions[closest_questions['source_platform'] != 'Metaculus']['question'].tolist()
        n_to_fetch += n_closest
        
        if n_to_fetch > len(distances):  # Prevent infinite loop
            break
            
    return poly_questions[:n_closest]


distance_matrix = get_distance_matrix(embedded_df)

# Create and show the visualization
embedded_df["closest_questions"] = embedded_df.apply(lambda row: get_closest_questions(row, distance_matrix, n_closest=10), axis=1)
embedded_df["closest_questions_text"] = embedded_df["closest_questions"].apply(lambda x: "\n".join(x))
embedded_df = embedded_df.reset_index(drop=True)
embedded_df.head()

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
    reduced_embeddings = reduce_dimensions(df_to_viz.drop(['source_platform', 'question', 'closest_questions_text', 'closest_questions'], axis=1))
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    viz_df['source_platform'] = df_to_viz['source_platform']
    
    # Add question text and closest questions
    viz_df['question'] = df_to_viz['question']
    viz_df['closest_questions'] = df_to_viz['closest_questions']
    viz_df['closest_questions_formatted'] = viz_df['closest_questions'].apply(
        lambda x: "<br>".join([f"• {q}" for q in x])
    )

    # Create the plot
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='source_platform',
        color_discrete_map={'Metaculus': 'red', 'Polymarket': 'gray', 'GJOpen': 'blue', 'PredictIt': 'green', "Manifold": 'orange'},
        hover_data=['question', 'closest_questions_formatted'],
        title='UMAP Visualization of Question Embeddings',
        labels={'x': 'TSNE Component 1', 'y': 'TSNE Component 2'}
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate="<br>".join([
            "Question: %{customdata[0]}",
            "Closest Questions:<br>%{customdata[1]}<br>",
            "<extra></extra>"
        ])
    )
    
    # Update layout
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        legend_title_text='Source',
        width=1000,
        height=1000
    )
    
    return fig
# %%
create_visualization(embedded_df)
# %%
# %%