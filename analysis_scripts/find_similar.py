

# %%
meta_questions_df = pd.read_csv("questions_df.csv", index_col=0)

from embedding_utils import embed_questions_df

def get_distance_matrix(combined_df):
    """Create a distance matrix of the embeddings."""
    embeddings = combined_df.drop(['source', 'question_text'], axis=1)
    cosine_similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - cosine_similarity_matrix

    return distance_matrix


def get_closest_questions(row, distance_matrix, n_closest=10):
    """Get the 10 closest questions from the distance matrix."""
    question_text = row['question_text']
    question_index = combined_df.index[combined_df['question_text'] == question_text].tolist()[0]
    distances = distance_matrix[question_index]
    
    # Keep getting more indices until we have enough Polymarket questions
    n_to_fetch = n_closest
    poly_questions = []
    while len(poly_questions) < n_closest:
        closest_indices = np.argsort(distances)[:n_to_fetch]
        closest_questions = combined_df.iloc[closest_indices]
        poly_questions = closest_questions[closest_questions['source'] == 'Polymarket']['question_text'].tolist()
        n_to_fetch += n_closest
        
        if n_to_fetch > len(distances):  # Prevent infinite loop
            break
            
    return poly_questions[:n_closest]


chunk_size = 200
poly_embeddings_df = embed_questions_df(active_df, chunk_size=chunk_size, question_column="question")
meta_embeddings_df = embed_questions_df(meta_questions_df, question_column="question_text")

poly_embeddings_df["question_text"] = active_df["question"]
meta_embeddings_df["question_text"] = meta_questions_df["question_text"]
poly_embeddings_df["source"] = "Polymarket"
meta_embeddings_df["source"] = "Metaculus"
combined_df = pd.concat([meta_embeddings_df, poly_embeddings_df], axis=0).reset_index(drop=True)

distance_matrix = get_distance_matrix(combined_df)

# Create and show the visualization
combined_df["closest_questions"] = combined_df.apply(lambda row: get_closest_questions(row, distance_matrix, n_closest=10), axis=1)
combined_df["closest_questions_text"] = combined_df["closest_questions"].apply(lambda x: "\n".join(x))


# %%
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity


def reduce_dimensions(embeddings_df, n_components=2):
    """Reduce dimensionality of embeddings using UMAP."""
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings_df)
    return reduced_embeddings

def create_visualization(combined_df):
    """Create an interactive plotly visualization of the embeddings."""
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(combined_df.drop(['source', 'question_text', 'closest_questions_text', 'closest_questions'], axis=1))
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    viz_df['source'] = combined_df['source']
    
    # Add question text and closest questions
    viz_df['question_text'] = combined_df['question_text']
    viz_df['closest_questions'] = combined_df['closest_questions']
    viz_df['closest_questions_formatted'] = viz_df['closest_questions'].apply(
        lambda x: "<br>".join([f"• {q}" for q in x])
    )

    # Create the plot
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='source',
        color_discrete_map={'Metaculus': 'red', 'Polymarket': 'gray'},
        hover_data=['question_text', 'closest_questions_formatted'],
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
fig = create_visualization(combined_df)
fig.show() 
# %%