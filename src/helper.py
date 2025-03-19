

import os
import time
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse, parse_qs



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