import pandas as pd
from pathlib import Path
from embedding_utils import embed_questions_df

def save_embeddings():
    """Save embeddings from both sources to CSV files."""
    # Load and process Metaculus questions
    meta_questions_df = pd.read_csv("questions_df.csv", index_col=0)
    meta_embeddings_df = embed_questions_df(meta_questions_df, question_column="question_text")
    meta_embeddings_df.to_csv("meta_embeddings.csv")
    
    # Load and process Polymarket questions
    poly_questions_df = pd.read_csv("poly_questions.csv", index_col=0)
    poly_embeddings_df = embed_questions_df(poly_questions_df, question_column="question")
    poly_embeddings_df.to_csv("poly_embeddings.csv")

if __name__ == "__main__":
    save_embeddings() 