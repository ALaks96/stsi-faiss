
import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Load product database
df=pd.read_csv("data/product_kaggle.csv", encoding = "ISO-8859-1" )

# get rid of products without a description
df['product_description'] = str(df['product_title']) + str(df['product_description']) 
df = df[df['product_description'].notna()] 
df['id'] = df.index
# Create a product description object
data=df.product_description.to_list()

# Load pre-trained sentence transformer fine-tuned to STS
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Check if CUDA is available ans switch to GPU
if torch.cuda.is_available():
   model = model.to(torch.device("cuda"))
print(model.device)

# Create embeddings
embeddings = model.encode(data)

# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, df._unit_id.values)

faiss.write_index(index, './models/product_index_kaggle')