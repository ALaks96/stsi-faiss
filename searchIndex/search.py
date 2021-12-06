import time
import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


@st.cache
def read_data(file="data/product_kaggle.csv"):
    """Read the data"""
    df=pd.read_csv(file, encoding = "ISO-8859-1" )
    return df


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="./models/product_index_kaggle"):
    """Load and deserialize the Faiss index."""
    return faiss.read_index(path_to_faiss)

# def vector_search(query, model, index, num_results=10):
#     """Tranforms query to vector using a pretrained, sentence-level 
#     DistilBERT model and finds similar vectors using FAISS.
#     Args:
#         query (str): User query that should be more than a sentence long.
#         model (sentence_transformers.SentenceTransformer.SentenceTransformer)
#         index (`numpy.ndarray`): FAISS index that needs to be deserialized.
#         num_results (int): Number of results to return.
#     Returns:
#         D (:obj:`numpy.array` of `float`): Distance between results and query.
#         I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
#     """
#     print(query)
#     print(index)
#     vector = model.encode(list(query))
#     D, I = index.search(np.array(vector).astype("float32"), k=num_results)
#     return D, I

def search(query, model, index):
   t=time.time()
   query_vector = model.encode([query])
   k = 5
   top_k = index.search(query_vector, k)
   Time = 'totaltime:' +str(time.time()-t)
   return top_k[1][0], Time

def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx][column]) for idx in I[0]]


def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Vector-based searches with Sentence Transformers and Faiss")

    # User search
    user_input = st.text_area("Search box")
    print(user_input)

    # Filters
    st.sidebar.markdown("**Filters**")
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

    # Fetch results
    if user_input:
        results, Time = search(user_input, model, faiss_index)
        st.write(f"""**Elapsed time**: {Time}""")
        for result in results:
            print(result)
            productInfo = (data[data["_unit_id"]==result][["product_title","url"]])
        # Get individual results
            st.write(
                f"""**{productInfo}** 
            """
            )
        


if __name__ == "__main__":
    main()