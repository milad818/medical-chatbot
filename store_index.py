
import pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf, text_splitter, cc_pinecone_index
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain.embeddings import HuggingFaceEmbeddings
import os




load_dotenv()

# model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin"


repo_id = "TheBloke/Llama-2-7B-Chat-GGML"  # Example repo
model_name = "llama-2-7b-chat.ggmlv3.q2_K.bin"
model_path = os.path.join("./model", model_name)
embedding_name = "sentence-transformers/all-MiniLM-L6-v2"

# Download model
if not os.path.exists(model_path):
    hf_hub_download(repo_id=repo_id, filename=model_name, local_dir="./model")
    print(f"Model {model_name} downloaded successfully.")
else:
    print(f"Model {model_name} is already downloaded and ready to use.")

# Download embedding
embedding = HuggingFaceEmbeddings(model_name=embedding_name)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)

data_directory = "./data"
documents = load_pdf(data_directory)

text_chunks = text_splitter(documents)


# pinecone vector database specifications (vectorstore specs.)
index_name = "medical-chatbot"
cloud = os.getenv("PINECONE_CLOUD", "aws")
region = os.getenv("PINECONE_REGION", "us-east-1")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
spec = pinecone.ServerlessSpec(cloud=cloud, region=region)

index = cc_pinecone_index(index_name, pc, spec)

# check if the index already exists
index_stats = index.describe_index_stats()
tot_num_vectors = index_stats["total_vector_count"]
# print(tot_num_vectors)

if tot_num_vectors == 0:
    print("Index is empty. Inserting data...")
    vectorstore_from_chunks = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=index_name,
        embedding=embedding
    )
    print("Data has successfully been inserted.")
else:
    print(f"Index already contains {tot_num_vectors} vectors. Skipping insertion.")