import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths and API key
CHROMA_PATH = "kb_index"
DATA_PATH = "data"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is set in the environment for Google Generative AI SDK
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ------------------- STEP 1: CREATE VECTOR STORE ------------------- #
def create_vector_store():
    print("📥 Loading PDF documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")

    print("🧠 Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    print("🔎 Generating embeddings using GoogleGenerativeAI...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print("📦 Saving embeddings to FAISS store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(CHROMA_PATH)
    print("✅ Vector store created.")


# ------------------- STEP 2: LOAD VECTOR STORE ------------------- #
def load_existing_vector_store():
    print("🔁 Loading vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(CHROMA_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# ------------------- STEP 3: GENERATE ANSWER ------------------- #
def generate_gemini_answer(question, vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are AskMike, an expert aviation security AI assistant.
Answer based only on the given context. If unsure or context is lacking, say: "I'm not sure based on the current data."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=5),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return chain.run(question)


# ------------------- OPTIONAL TESTING ------------------- #
if __name__ == "__main__":
    # Uncomment below to create vector store for the first time
    # create_vector_store()

    db = load_existing_vector_store()
    query = "What are the key steps in airport baggage screening?"
    response = generate_gemini_answer(query, db)
    print("🤖 Answer:", response)
