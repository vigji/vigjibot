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
plt.figure()
plt.hist(gjopen_df.n_forecasters, np.arange(0, 100, 10))
plt.show()
# %%
embedded_df = embed_questions_df(df, question_column="question")
# %%
embedded_df.head()

# %%

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
    reduced_embeddings = reduce_dimensions(combined_df.drop('source', axis=1))
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    viz_df['source'] = combined_df['source']
    
    # Add question text
    viz_df['question_text'] = ''
    viz_df.loc[viz_df['source'] == 'Metaculus', 'question_text'] = meta_questions_df['question_text'].values
    viz_df.loc[viz_df['source'] == 'Polymarket', 'question_text'] = poly_embeddings_df.index.map(lambda x: x if isinstance(x, str) else '')
    
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