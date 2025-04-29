# %%
%matplotlib inline
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
# import t-sne
from sklearn.manifold import TSNE
import plotly.express as px

# Get the absolute path to the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from openai import OpenAI
import os
from forecasting_tools.forecast_helpers.benchmark_displayer import get_json_files
from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
import sys
from pathlib import Path
from analysis_utils import get_all_runs_df

data_path = Path(__file__).parent.parent / "benchmarks"
assert data_path.exists()


# %%
all_runs = get_json_files(data_path)
print(len(all_runs))
print(all_runs[-1])

if Path("all_runs_df.csv").exists():
    all_runs_df = pd.read_csv("all_runs_df.csv", index_col=0)
else:
    all_runs_df = get_all_runs_df(all_runs, BenchmarkForBot)
    all_runs_df.to_csv("all_runs_df.csv")

# %%
# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_TOKEN"),
    base_url="https://api.deepinfra.com/v1/openai",
)

# input = ("The food was delicious and the waiter...",)  # or an array ["hello", "world"]


# %%
question_df = all_runs_df[["question_text", "question_id"]].drop_duplicates().reset_index(drop=True)

# %%
input_questions = question_df["question_text"].tolist()
len(question_df)
# %%

model = "BAAI/bge-m3"

cache_file = f"embeddings_cache_{model.replace('/', '.')}.csv"
if (Path(__file__).parent / cache_file).exists():
    embeddings_df = pd.read_csv(cache_file, index_col=0)
    embeddings_array = embeddings_df.to_numpy()
    
else:
    embeddings = openai.embeddings.create(
        model=model, input=input_questions, encoding_format="float"
    )


    embeddings_array = np.array([embedding.embedding for embedding in embeddings.data])
    embeddings_array.shape

    embeddings_df = pd.DataFrame(embeddings_array, index=question_df["question_id"])
    embeddings_df.to_csv(cache_file)
print(embeddings_array.shape)

# %%
# compute cosine distances as 1 - cosine similarity
cosine_distances = 1 - cosine_similarity(embeddings_array)

# K-means
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(cosine_distances)
question_df["cluster"] = kmeans.labels_
question_df.to_csv("questions_df.csv")
# %%
# compute t-sne
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_result = tsne.fit_transform(cosine_distances)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
tsne_df['question_text'] = question_df['question_text']
tsne_df['question_id'] = question_df['question_id']
tsne_df['cluster'] = question_df['cluster'].astype(str)

# %%
# Create interactive plot with Plotly
fig = px.scatter(
    tsne_df,
    x='x',
    y='y',
    color='cluster',
    hover_data={'question_text': True, 'question_id': True, 'cluster': True},
    title='t-SNE Visualization of Question Embeddings',
    labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
    color_discrete_sequence=px.colors.qualitative.Alphabet[:10]  # Using first 10 colors from Alphabet palette
)

# Customize hover template
fig.update_traces(
    hovertemplate="<br>".join([
        "Question ID: %{customdata[1]}",
        "Question: %{customdata[0]}",
        "Cluster: %{customdata[2]}",
        "<extra></extra>"
    ])
)

# Show the plot
fig.show()

# K-mean

# %%
tsne_df# %%

# %%
