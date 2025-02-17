import os
from dotenv import load_dotenv
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS

CAMINHO = "./files"
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

""" Verify if key was load """
if not GROQ_API_KEY:
    raise EnvironmentError("Api key not found")


""" Show Free Models to use """
clients = Groq(api_key=GROQ_API_KEY)
print(clients.models.list())

""" Load and Split files pages  """
def load_files():
    chunks = []
    file = os.listdir(CAMINHO)    
    for i in file:
        if i.endswith(".pdf"):
            load = PyPDFLoader(os.path.join(CAMINHO, i))
            pages = load.load_and_split()
            chunks.extend(pages)
    return chunks

""" Create vector database with text of the file pages"""
def embeddings(files):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(files, embedding)
    return db

""" Configuration and Creation of chat connect with vector database """
def conversation(chat):
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=chat.as_retriever())


if __name__ == "__main__":
    
    beginning = load_files()    
    chat = embeddings(beginning)
    qa = conversation(chat)
    
    while True:
        query = input("Enter your questions about the docs here: ")

        if query.lower() == "exit":
            print("Thank you for using your chatbot")
            break
        
        result = qa.invoke({'query': query})
        print(f"Here is the answer: {result['result']} \n")
    