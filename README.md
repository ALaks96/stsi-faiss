# Vector based semantic search index

This project is a simple pilot to test out a semantic search index based on sentence transformers and FAISS. To allow better interaction with the index, streamlit hosts a query functionality and exposes results in a very basic user interface. The search index was tested on an open ecommerce dataset from Kaggle. To get the dataset visit the following url and save it in searchIndex/data : https://data.world/crowdflower/ecommerce-search-relevance

You can either build a docker image to create this simple test project, or create your own virtual env with the necessary requirements to build the search index.

# Docker

Simply navigate to the searchIndex folder and launch the build as follows:

'''
cd searchIndex && docker build -t stsi-faiss:latest .
'''

To run the image after build:

'''
docker run -p 8501:8501 stsi-faiss:latest
'''

Navigate to localhost:8051 to see the search page.

# Venv local

(Recommended) create a venv and install all the requirements (in requirements.txt)

To create the index yourserlf, launch the indexing script in /searchIndex/indexing.py after downloading the dataset in searchIndex/data as follows:

'''
cd searchIndex && python indexing.py
'''

This will create the index containing product description embeddings, saving it in the searchIndex/models folder.

To launch the search page (user interface) after creating the index, launch the following command:

'''
cd searchIndex && streamlit run search.py
'''

Check out localhost:8501 for the search page