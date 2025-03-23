

import os
import time
# from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse, parse_qs
from langchain_pinecone import PineconeVectorStore      # not compatible with the retriever expected by langchain
# from langchain.vectorstores import Pinecone           # deprecated
from langchain_community.vectorstores import Pinecone
# from langchain.embeddings import HuggingFaceEmbeddings    # deprecated
# from langchain_community.embeddings import HuggingFaceEmbeddings      # deprecated
from langchain_huggingface import HuggingFaceEmbeddings



# load data from the pdf file
def load_pdf(data_directory):
    loader = DirectoryLoader(data_directory,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


# build a function to split text
def text_splitter(documents):
    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=20)
    chunks = rec_splitter.split_documents(documents)

    return chunks


def download_huggingface_embedding(model_name):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    return embedding


def cc_pinecone_index(index_name: str, pc, spec):
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    # check whether the index already exists
    if index_name not in existing_indexes:
        # create index
        # dimenstion is set based the embedding model and the dimension of its vectors
        pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)

        while not pc.describe_index(index_name).status["ready"]:
            # wait until index is ready
            time.sleep(1)
        
        print(f"Index {index_name} has been successfully created.")
    else:
        print(f"Index {index_name} already exists!")

    # connect to the index
    index = pc.Index(index_name)

    return index



def load_vectorstore(index, index_name, embedding, text_chunks=None):
    index_stats = index.describe_index_stats()
    existing_vector_count = index_stats["total_vector_count"]

    if existing_vector_count == 0:
        print("Index is empty. Uploading new documents...")
        
        vectorstore_from_chunks = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=index_name,
        embedding=embedding
        )

        print("Upload complete.")

        return vectorstore_from_chunks
    else:
        print(f"Index already contains {existing_vector_count} vectors. Skipping upload.")
        # load the existing vector store whether or not new data was added
        # vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)
        print(f"Existing vectorstore is now loaded.")
    
        return vectorstore