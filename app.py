from flask import Flask, render_template, jsonify, request
from src.helper import *
# from langchain.vectorstores import Pinecone       # deprecated
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers            # deprecated
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *



load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")


app = Flask(__name__)

# set variables required to retrieve/initialize a pinecone vector index and vectorstore
cloud = os.getenv("PINECONE_CLOUD", "aws")
region = os.getenv("PINECONE_REGION", "us-east-1")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)    # pinecone client
spec = pinecone.ServerlessSpec(cloud=cloud, region=region)
index_name = "medical-chatbot"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# create and connect a pinecone index
index = cc_pinecone_index(index_name, pc, spec)

# download huggingface embedding
embedding = download_huggingface_embedding(model_name=model_name)

# load vectorstore
vectorstore = load_vectorstore(index, index_name, embedding=embedding)
print(isinstance(vectorstore, PineconeVectorStore))

# define prompt template
MEDICAL_PROMPT = PromptTemplate(template=prompt_template,
                                input_variables=["question", "context"])


# store prompt in chain_type_kwargs to be used in the retrieval chain
chain_type_kwargs={"prompt": MEDICAL_PROMPT}

# initialize the LLaMA-2 model (via CTransformers)
llm = CTransformers(model="model\llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type="llama",
                    config={"max_new_tokens": 512,  # limits response length
                            "temperature": 0.8})

# convert the vectorstore into a retriever that returns the top 2 most relevant documents (k=2)
# it is key for context-aware responses in Retrieval-Augmented Generation (RAG).
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# creat the Retrieval-Augmented QA pipeline
qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",     # concatenates retrieved documents into one context before passing to LLaMA
                                        # ALTERNATIVES: "map_reduce" OR "refine"
                retriever=retriever,
                return_source_documents=True,   # returns source docs used for the answer
                chain_type_kwargs=chain_type_kwargs)


# this decorator tells Flask to run the index() function when a user accesses the root URL ("/")
@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)


