# 🚀 **End-to-End Medical Chatbot** 🏥💬  

Your AI-powered medical assistant is here! This project brings **Llama 2** to life, enabling an intelligent chatbot for medical queries given a particular data resource (`.pdf` or `.txt` file). Follow the steps below to get it up and running!  

---

## 🎯 **How to Get Started?**  

### 🛠 **Step 1: Clone the Repository**  
First, grab a copy of the project:  
```bash
git clone https://github.com/milad818/medical-chatbot.git
cd medical-chatbot
```

### 🏗 **Step 2: Set Up Your Virtual Environment**
Let's create an isolated environment to keep dependencies in check:
```
conda create -n medchatbot python=3.8 -y
conda activate medchatbot
```

### 📦 **Step 3: Install Dependencies**
Once inside your environment, install all required packages:
```
pip install -r requirements.txt
```

### 🔑 **Step 4: Set Up Your Pinecone Credentials**
Create a .env file in the root directory and add the following:
```
PINECONE_API_KEY = "your_pinecone_api_key_here"
```

### 📥 **Step 5: Download the Model**
Download the quantized Llama 2 model from [TheBloke](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) and place it in the model directory.

✅ Model Name: `llama-2-7b-chat.ggmlv3.q4_0.bin`

### 📌 **Step 6: Store Index**
Before running the chatbot, make sure to store the vector index:
```
python store_index.py
```

### 🚀 **Step 7: Launch the Chatbot**
Finally, start the application:
```
python app.py
```
Once it's running, open localhost (you can find it in your terminal once `app.py` is run) in your browser to chat with your AI-powered medical assistant! 🩺🤖

🏗 Tech Stack Used:

- ✨ Python – The backbone of the chatbot.
- 🧠 LangChain – For powerful LLM integrations.
- 🔥 Flask – Web framework for the chatbot API.
- 🚀 Meta Llama 2 – The brain behind medical responses.
- 📡 Pinecone – Vector database for fast retrieval.

