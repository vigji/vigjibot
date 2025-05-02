
import os
from openai import OpenAI
import dotenv
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib


model = "BAAI/bge-m3"

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_TOKEN"),
    base_url="https://api.deepinfra.com/v1/openai",
)

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
os.getenv("DEEPINFRA_TOKEN")

def preprocess_questions(questions_list):
    return [q.strip() for q in questions_list]

def _embed_all_questions(questions_list):
    questions_list = preprocess_questions(questions_list)
    embeddings = openai.embeddings.create(
        model=model, input=questions_list, encoding_format="float"
    )
    return embeddings

def _embed_in_chunks(questions_list, chunk_size=200):
    questions_list = preprocess_questions(questions_list)
    embeddings_list = []
    for i in tqdm(range(0, len(questions_list), chunk_size)):
        chunk = questions_list[i:i+chunk_size]
        embeddings = openai.embeddings.create(
            model=model, input=chunk, encoding_format="float"
        )
        embeddings_list.extend(embeddings.data)


def embed_all_questions(questions_list, chunk_size=None):
    if chunk_size is None:
        embeddings = _embed_all_questions(questions_list)
    else:
        embeddings = _embed_in_chunks(questions_list, chunk_size)
    return np.array([embedding.embedding for embedding in embeddings.data])

def embed_with_cache(questions_list, cache_folder=None, chunk_size=None):
    """Embed only if the cache file does not exist. Use list hash to name the file."""
    if cache_folder is None:
        cache_folder = Path(__file__).parent / "embeddings_cache"
    cache_folder.mkdir(parents=True, exist_ok=True)

    cache_file = cache_folder / f"{hashlib.sha256(str(questions_list).encode()).hexdigest()}.npy"
    
    if cache_file.exists():
        print(f"Loading embeddings from cache file {cache_file}")
        return np.load(cache_file)
    else:
        embeddings_df = embed_all_questions(questions_list, chunk_size)
        np.save(cache_file, embeddings_df)
        return embeddings_df


def embed_questions_df(question_df, chunk_size=None):
    questions_list = question_df["question"].to_list()
    sanitized_questions = [q.strip() for q in questions_list]
    embeddings_array = embed_with_cache(sanitized_questions, chunk_size)
    return pd.DataFrame(embeddings_array, index=question_df["question_id"])


if __name__ == "__main__":
    test_questions = ["What is the capital of France?", "What is the capital of Germany?"]
    a = embed_with_cache(test_questions, Path(__file__).parent / "cache")
    print(a)
    b = embed_with_cache(test_questions, Path(__file__).parent / "cache")
