import os
import torch
import gc
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

# Document processing imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFacePipeline

# FastAPI App initialization
app = FastAPI(
    title="Real Estate Assistant",
    description="A RAG-based Real Estate Assistant to help with property matching and transaction support.",
    version="1.0.0"
)

# Global variables for the assistant components
qa_chain = None

# Configuration
access_token = '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'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# File path: Adjust this if necessary.
TXT_FILE = "./alpha.txt"  # Make sure that this file exists and is accessible

class QueryRequest(BaseModel):
    query: str

def initialize_vector_store(chunks: List[Dict], embedding_model, persist_directory: str = "./chroma_db_instance") -> Chroma:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print(f"Stored {len(chunks)} chunks in the vector database at '{persist_directory}'.")
    return vector_store

def load_or_initialize_vector_store(chunks: List[Dict], embedding_model, persist_directory: str = "./chroma_db_instance") -> Chroma:
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading vector store from persisted directory.")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    return initialize_vector_store(chunks, embedding_model, persist_directory)

def load_documents_and_build_chain() -> any:
    # Step 1: Load and Split Data
    if not os.path.exists(TXT_FILE):
        raise FileNotFoundError(f"Text file '{TXT_FILE}' not found.")
    documents = TextLoader(TXT_FILE).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    # Step 2: Vector Store Setup
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        huggingfacehub_api_token='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'
    )
    vector_store = load_or_initialize_vector_store(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Step 3: LLM Setup with 4-bit Quantization
    model_id = "tiiuae/falcon-7b-instruct"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=access_token
    )

    # Optimized pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        top_k=30,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype=torch.float16
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RAG Chain Setup
    prompt_template = ChatPromptTemplate.from_template("""
    [REAL ESTATE ASSISTANT PROTOCOL]
    Role: Expert property advisor for Italy & USA
    Database: Comprehensive property listings (apartments, villas, commercial)

    TASKS:
    1. PROPERTY MATCHING:
    - Request: Location, type, price range, features
    - Output: Top 3-5 listings with key details

    2. TRANSACTION SUPPORT:
    - Buying: Step-by-step guidance, agency contacts
    - Selling: Market analysis, pricing strategy

    3. GENERAL QUERIES:
    - Market trends, area insights, legal aspects

    RESPONSE FORMAT:
    - Clear, structured bullet points
    - Comparative analysis when relevant
    - Highlight unique property features

    <CONTEXT>
    {context}
    </CONTEXT>

    Query: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain_local = create_retrieval_chain(retriever, document_chain)
    print("RAG Chain has been initialized.")
    return qa_chain_local

@app.on_event("startup")
async def startup_event():
    global qa_chain
    try:
        # This might take some time, so inform the logs accordingly.
        print("Initializing documents, vector store, and RAG chain. Please wait...")
        qa_chain = load_documents_and_build_chain()
        print("Application startup complete.")
    except Exception as e:
        print("Failed to initialize the RAG chain:", e)
        raise e

@app.post("/query")
async def query_assistant(req: QueryRequest):
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is not yet initialized. Try again later.")

    if not req.query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        # Memory management to clear up unused GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Invoke the chain with the user's query
        response = qa_chain.invoke({"input": req.query})

        # Again, clear caches to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return the result
        return {"response": response.get('result', 'No answer found')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optionally, you can define a simple health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
