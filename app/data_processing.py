import pandas as pd
from rag import embeddings
from langchain.vectorstores import FAISS
import re

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['cleaned_ingredients'] = df['ingredients'].apply(lambda x: re.sub(r'\d+\.?\d*\s*(oz|cl|ml|dash|tsp)?\s*', '', x).strip())
    df['text'] = df.apply(lambda row: f"Name: {row['name']}\nIngredients: {row['cleaned_ingredients']}\nInstructions: {row['instructions']}", axis=1)
    return df

def create_vector_store(df):
    texts = df['text'].tolist()
    metadatas = [{"name": row['name'], "ingredients": row['cleaned_ingredients']} for _, row in df.iterrows()]
    store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    store.save_local("cocktails_faiss_index")