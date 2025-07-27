from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import platform

load_dotenv()

def get_local_llm_model():
    """Get a high-performance local LLM model using llama.cpp"""
    
    # Determine model path - adjust as needed for your model storage location
    model_path = os.path.expanduser("~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    
    # Use different settings for Mac vs other platforms
    is_mac = platform.system() == "Darwin"
    n_gpu_layers = 1 if is_mac and platform.processor() == "arm" else -1
    n_threads = 8  # Adjust based on your CPU
    
    return LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        n_ctx=2048,  # Context window
        top_p=0.95,
        n_gpu_layers=n_gpu_layers,  # Enable GPU acceleration if available
        n_threads=n_threads,
        stop=["Human:", "USER:"],  # Stop tokens
        verbose=False,  # Set to True for debugging
        streaming=True,  # Better UX with streaming
    )

def get_local_embedding_model():
    """Get higher quality embeddings than the current MiniLM-L6"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # Better than MiniLM for retrieval
        model_kwargs={"device": "cpu"}
    )