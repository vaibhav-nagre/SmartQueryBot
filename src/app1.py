import streamlit as st
import time
import threading

# Must be the first Streamlit command
st.set_page_config(page_title="SmartQuery Bot", page_icon="🤖")

# Helper functions for auto-dismissing messages
def auto_dismiss_message(container, duration=3):
    """Make a message disappear after specified duration"""
    time.sleep(duration)
    container.empty()

def success_auto_dismiss(message, duration=3):
    """Show success message that disappears automatically"""
    container = st.empty()
    container.success(message)
    # Use threading to avoid blocking the main thread
    threading.Thread(target=auto_dismiss_message, args=(container, duration), daemon=True).start()

def info_auto_dismiss(message, duration=3):
    """Show info message that disappears automatically"""
    container = st.empty()
    container.info(message)
    threading.Thread(target=auto_dismiss_message, args=(container, duration), daemon=True).start()

def warning_auto_dismiss(message, duration=4):
    """Show warning message that disappears automatically"""
    container = st.empty()
    container.warning(message)
    threading.Thread(target=auto_dismiss_message, args=(container, duration), daemon=True).start()

def error_auto_dismiss(message, duration=5):
    """Show error message that disappears automatically"""
    container = st.empty()
    container.error(message)
    threading.Thread(target=auto_dismiss_message, args=(container, duration), daemon=True).start()

    
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb
import os
import torch
import time
import re
import hashlib
import functools
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor

# Force CPU usage for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS memory limit

# Import Hugging Face modules
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline

load_dotenv()

# Performance optimization: Use memoization for expensive operations
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Default embedding model - cached
EMBEDDING_MODEL = get_embedding_model()

# Performance optimization: Cache the LLM model with better network error handling
@st.cache_resource
def get_llm_model():
    try:
        # Import required libraries
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        # First, test if HuggingFace is reachable
        try:
            import requests
            response = requests.head("https://huggingface.co", timeout=5)
            if response.status_code >= 400:
                raise ConnectionError("Hugging Face servers returned error status")
            huggingface_available = True
        except Exception as e:
            huggingface_available = False
            # Store model info in session state (silent, no message)
            if "model_info" not in st.session_state:
                st.session_state.model_info = "Limited Offline Mode"
        
        # If HuggingFace is available, try to load models from there
        if huggingface_available:
            try:
                # Try loading a smaller but good quality model with lower bandwidth requirements
                tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/opt-350m",  # Smaller model that downloads faster
                    use_fast=True,
                    local_files_only=False
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    "facebook/opt-350m",
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    local_files_only=False
                )
                
                hf_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    do_sample=True
                )
                
                # Store model info in session state (silent, no message)
                if "model_info" not in st.session_state:
                    st.session_state.model_info = "OPT-350M"
                
                return HuggingFacePipeline(pipeline=hf_pipeline)
                
            except Exception as e:
                # Continue silently
                pass
        
        # Fallback to local distilgpt2 (should be available in cache)
        try:
            # Try to load model from cache only
            tokenizer = AutoTokenizer.from_pretrained(
                "distilgpt2",
                local_files_only=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                local_files_only=True
            )
            
            hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                temperature=0.7
            )
            
            # Store model info in session state (silent, no message)
            if "model_info" not in st.session_state:
                st.session_state.model_info = "DistilGPT2"
                
            return HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            # Final fallback - minimal text generation
            from langchain.llms import FakeListLLM
            
            # Create a very simple fake responder as absolute fallback
            responses = [
                "I found some information about that in the website content.",
                "The website mentions this topic but doesn't provide detailed information.",
                "According to the website, there are several aspects to consider.",
                "The website content suggests this is an important topic.",
                "Based on the website content, I can provide this information."
            ]
            
            # Store model info in session state (silent, no message)
            if "model_info" not in st.session_state:
                st.session_state.model_info = "Basic Response Generator"
                
            return FakeListLLM(responses=responses)
            
    except Exception as e:
        # Emergency fallback
        from langchain.llms import FakeListLLM
        
        # Store model info in session state
        if "model_info" not in st.session_state:
            st.session_state.model_info = "Emergency Fallback Mode"
            
        return FakeListLLM(responses=["I'm having trouble accessing language models right now. Please try again later."])

# Wrap the pipeline in a LangChain compatible format
LLM_MODEL = get_llm_model()

# Create a cache for website data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_website_content(url):
    loader = WebBaseLoader(url)
    return loader.load()

# Performance optimization: Cache vectorstores by URL hash
@st.cache_data(ttl=3600)
def create_vectorstore(document_chunks, url):
    # Use a unique collection name based on URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    collection_name = f"chroma_{url_hash}"
    
    vector_store = Chroma.from_documents(
        document_chunks,
        EMBEDDING_MODEL,
        collection_name=collection_name,
        persist_directory=None
    )
    return vector_store

def get_vectorstore_from_url(url):
    # Fetch content with caching
    document = fetch_website_content(url)
    
    # Better chunking strategy with more overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    document_chunks = text_splitter.split_documents(document)
    
    # Create vector store with caching
    return create_vectorstore(document_chunks, url)

# Performance optimization: Smaller k values for faster retrieval
def get_context_retriever_chain(vector_store):
    # Use MMR with optimized parameters
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,  # Reduced from 5 for better performance
            "fetch_k": 5,  # Reduced from 8 for better performance
            "lambda_mult": 0.7
        }
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise search query generator. Extract key terms from the question."),
        ("user", "{input}"),
        ("user", "Generate a search query with the exact keywords that would find relevant information about this specific question.")
    ])
    retriever_chain = create_history_aware_retriever(LLM_MODEL, retriever, prompt)
    return retriever_chain

# Cache the RAG chain creation
@functools.lru_cache(maxsize=5)
def get_conversational_rag_chain(retriever_chain): 
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question directly and specifically using ONLY the following context: {context}""")
    ])
    stuff_documents_chain = create_stuff_documents_chain(LLM_MODEL, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Add this to your global variables section
_GLOBAL_RAG_CHAIN = None

# Then modify the function to use the global cache
def get_conversational_rag_chain(retriever_chain):
    global _GLOBAL_RAG_CHAIN
    # Only create a new chain if one doesn't exist
    if _GLOBAL_RAG_CHAIN is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question directly and specifically using ONLY the following context: {context}""")
        ])
        stuff_documents_chain = create_stuff_documents_chain(LLM_MODEL, prompt)
        _GLOBAL_RAG_CHAIN = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    return _GLOBAL_RAG_CHAIN

# Replace the get_response function with this implementation
def get_response(user_input):
    try:
        # Get retriever chain and conversation chain
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        
        # Truncate input if needed
        if len(user_input) > 200:
            user_input = user_input[:200] + "..."
        
        # Use ThreadPoolExecutor instead of signal for timeout handling
        with ThreadPoolExecutor() as executor:
            # Submit the task
            future = executor.submit(
                conversation_rag_chain.invoke,
                {"chat_history": [], "input": user_input}
            )
            
            try:
                # Wait for the result with a timeout
                response = future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                return "I'm sorry, but the response is taking too long to generate. Please try a simpler question or a different website."
        
        # Handle empty responses gracefully
        if not response.get('answer') or len(response.get('answer', '').strip()) < 5:
            return (
                "I couldn't find a specific answer to your question in the website content.\n\n"
                "Could you try:\n"
                "- Rephrasing your question\n"
                "- Being more specific\n"
                "- Asking about another topic from the website"
            )

        # Rest of your existing processing code
        answer = response['answer'].strip()

        # Remove citations like [123], [326], etc.
        answer = re.sub(r'\[\d+\]|\[\d+\]\[\d+\]', '', answer)

        # Remove lines starting with ^ or containing "Guidelines:" or "System:" or context instructions
        answer_lines = [
            line for line in answer.split('\n')
            if line.strip() and
               not line.strip().startswith('^') and
               "Guidelines:" not in line and
               not line.strip().lower().startswith('system:') and
               not line.strip().lower().startswith('answer the question using only the following context:')
        ]

        # Remove any lines that look like citations or URLs
        answer_lines = [
            line for line in answer_lines
            if not re.match(r'^\s*(Archived|Retrieved|http|www\.|See also)', line, re.I)
        ]

        # Remove duplicate/empty lines and strip whitespace
        cleaned = []
        seen = set()
        for line in answer_lines:
            line = line.strip()
            if line and line not in seen:
                cleaned.append(line)
                seen.add(line)

        # If the answer is a comma-separated list, format as bullets
        if len(cleaned) == 1 and ',' in cleaned[0] and len(cleaned[0].split(',')) > 2:
            items = [item.strip().capitalize() for item in cleaned[0].split(',') if item.strip()]
            cleaned = [f"- {item}" for item in items]
        else:
            # Always format each line as a bullet point for clarity
            cleaned = [f"- {line}" for line in cleaned]

        # Bold important numbers (like "36 La Liga titles")
        cleaned = [
            re.sub(r'(\d+\s+(?:titles?|championships?|wins?|records?|years?|percent|percentage|people|users|customers|million|billion))', r'**\1**', line, flags=re.I)
            for line in cleaned
        ]

        # Prettify: capitalize first letter of each bullet and ensure proper punctuation
        prettified = []
        for line in cleaned:
            content = line[2:].strip()
            if content and not content.endswith('.'):
                content += '.'
            prettified.append(f"- {content[0].upper() + content[1:]}" if content else line)

        answer = '\n'.join(prettified).strip()

        # Add this section to remove irrelevant/generic responses
        if any(generic in answer.lower() for generic in ["i don't know", "i cannot", "not in the context", "can't find"]):
            return (
                "I couldn't find specific information about that in the website content.\n\n"
                "Could you try:\n"
                "- Rephrasing your question\n"
                "- Being more specific\n"
                "- Asking about another topic from the website"
            )

        # Final fallback if answer is still empty
        if not answer:
            return (
                "I couldn't find a specific answer to your question in the website content.\n\n"
                "Could you try:\n"
                "- Rephrasing your question\n"
                "- Being more specific\n"
                "- Asking about another topic from the website"
            )

        return answer

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

# For displaying AI messages with better formatting and elaborate button
def display_ai_message(content, message_idx):
    with st.chat_message("AI"):
        st.markdown(content, unsafe_allow_html=True)
        
        # Only show the "More" button if this is NOT the greeting message
        # Check if it's not the first message OR if it's not the greeting message
        is_greeting = "Hello, I am a Vaibhav's bot. How can I help you?" in content
        is_first_message = message_idx == 0
        
        # Only show the button for non-greeting substantive responses
        if not (is_greeting and is_first_message):
            # Create columns for better layout - message takes most space, button takes less
            cols = st.columns([0.85, 0.15])
            
            # Add the elaborate button with custom styling
            button_pressed = cols[1].button("🔍 More", 
                             key=f"elaborate_{message_idx}",
                             use_container_width=True)
        else:
            # Create a placeholder for consistent UI layout
            button_pressed = False
    
    # Initialize elaboration counters if not present
    if "elaboration_counters" not in st.session_state:
        st.session_state.elaboration_counters = {}
    if message_idx not in st.session_state.elaboration_counters:
        st.session_state.elaboration_counters[message_idx] = 0
        
    # Initialize elaboration storage
    if "elaborations" not in st.session_state:
        st.session_state.elaborations = {}
    if message_idx not in st.session_state.elaborations:
        st.session_state.elaborations[message_idx] = []
    
    # Move elaboration logic OUTSIDE the chat_message context
    if button_pressed:
        # Show spinner while generating elaborated response
        with st.spinner("Generating more details..."):
            # Get the original question from chat history
            original_question = ""
            for i, msg in enumerate(st.session_state.chat_history):
                if isinstance(msg, AIMessage) and i == message_idx:
                    if i > 0 and isinstance(st.session_state.chat_history[i-1], HumanMessage):
                        original_question = st.session_state.chat_history[i-1].content
            
            # Generate more elaborate response
            if original_question:
                # Increment the elaboration counter
                st.session_state.elaboration_counters[message_idx] += 1
                counter = st.session_state.elaboration_counters[message_idx]
                
                # Modify the prompt to ask for more or different details based on counter
                if counter == 1:
                    elaborate_prompt = f"{original_question} (Please provide more details and examples)"
                elif counter == 2:
                    elaborate_prompt = f"{original_question} (Please provide additional facts and information not mentioned previously)"
                elif counter == 3:
                    elaborate_prompt = f"{original_question} (Please provide more technical details or advanced information)"
                else:
                    elaborate_prompt = f"{original_question} (Please provide any remaining interesting facts or alternative perspectives)"
                
                # Get elaborated response
                elaborate_response = get_elaborate_response(elaborate_prompt)
                
                # Store this elaboration
                st.session_state.elaborations[message_idx].append(elaborate_response)
                
                # Display all elaborations for this message
                for idx, elab in enumerate(st.session_state.elaborations[message_idx]):
                    with st.container():
                        st.markdown(f"<div class='elaborate-container level-{idx+1}'>", unsafe_allow_html=True)
                        if idx == 0:
                            st.markdown("🔍 **Additional Details:**", unsafe_allow_html=True)
                        else:
                            st.markdown(f"🔍 **More Details ({idx+1}):**", unsafe_allow_html=True)
                        st.markdown(elab, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

# Function to generate more elaborate responses - improved to find different information
def get_elaborate_response(user_input):
    try:
        # Use retrieval chain with more documents for additional details
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        
        # Get more context by increasing retrieval count and encouraging novelty
        response = conversation_rag_chain.invoke({
            "chat_history": [],
            "input": user_input
        })
        
        # Process response similar to get_response but with more details preserved
        answer = response['answer'].strip()
        answer = re.sub(r'\[\d+\]|\[\d+\]\[\d+\]', '', answer)

        # Remove system instructions but keep more content
        answer_lines = [
            line for line in answer.split('\n')
            if line.strip() and
               not line.strip().startswith('^') and
               not line.strip().lower().startswith('system:') and
               not line.strip().lower().startswith('answer the question using only the following context:')
        ]
        
        # Remove citations and URLs
        answer_lines = [
            line for line in answer_lines
            if not re.match(r'^\s*(Archived|Retrieved|http|www\.|See also)', line, re.I)
        ]
        
        # Format as bullet points with more detail preserved
        cleaned = []
        for line in answer_lines:
            line = line.strip()
            if line:
                if not line.startswith("- "):
                    line = f"- {line}"
                cleaned.append(line)
                
        # Bold important information
        cleaned = [
            re.sub(r'(\d+\s+(?:titles?|championships?|wins?|records?))', r'**\1**', line, flags=re.I)
            for line in cleaned
        ]
        
        # Add punctuation and capitalize
        prettified = []
        for line in cleaned:
            if line.startswith("- "):
                content = line[2:].trip() 
                if content and not content.endswith(('.', '!', '?')):
                    content += '.'
                prettified.append(f"- {content[0].upper() + content[1:]}" if content else line)
            else:
                prettified.append(line)
                
        return '\n'.join(prettified)
        
    except Exception as e:
        return f"I couldn't generate additional details due to: {str(e)}"

# --- Streamlit UI ---
#st.set_page_config(page_title="SmartQuery Bot", page_icon="🤖")
st.title("SmartQuery Bot")

with st.sidebar:
    website_url = st.text_input("Website URL")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Vaibhav's bot. How can I help you?"),
        ]
        st.session_state.vector_store = None

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Vaibhav's bot. How can I help you?"),
        ]
    
    # Initialize loading state if it doesn't exist
    if "loading" not in st.session_state:
        st.session_state.loading = False
        
    # Replace the existing website loading section with this optimized version:

    if (
        "vector_store" not in st.session_state
        or "last_url" not in st.session_state
        or st.session_state.last_url != website_url
    ):
        # Reset the global RAG chain when URL changes
        _GLOBAL_RAG_CHAIN = None
        
        # Initialize processing state tracking
        if "processing_stage" not in st.session_state:
            st.session_state.processing_stage = "start"
            st.session_state.progress_value = 0
        
        # Create animation container that persists across reruns
        loading_container = st.container()
        with loading_container:
            # Create modern UI layout
            st.markdown("<div class='animation-wrapper'><div class='pulse-beacon'></div></div>", unsafe_allow_html=True)
            st.markdown("<h3 class='loading-title glowing-text'>Analyzing Website Content</h3>", unsafe_allow_html=True)
            
            # Native Streamlit progress bar (more reliable than JS)
            progress_bar = st.progress(st.session_state.progress_value)
            
            # Show current progress as text
            progress_text = st.empty()
            progress_text.markdown(f"<div class='progress-text-container'>{int(st.session_state.progress_value * 100)}%</div>", unsafe_allow_html=True)
            
            # Create step indicators
            steps_container = st.container()
            with steps_container:
                col1, col2, col3, col4 = st.columns(4)
                
                # Update step indicators based on current stage
                if st.session_state.processing_stage in ["start", "fetching"]:
                    with col1:
                        st.markdown("<div class='step-indicator active'><div class='step-number'>1</div><div class='step-label'>Fetching</div></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='step-indicator'><div class='step-number'>2</div><div class='step-label'>Processing</div></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='step-indicator'><div class='step-number'>3</div><div class='step-label'>Embedding</div></div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown("<div class='step-indicator'><div class='step-number'>4</div><div class='step-label'>Finalizing</div></div>", unsafe_allow_html=True)
                elif st.session_state.processing_stage == "processing":
                    with col1:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Fetching</div></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='step-indicator active'><div class='step-number'>2</div><div class='step-label'>Processing</div></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='step-indicator'><div class='step-number'>3</div><div class='step-label'>Embedding</div></div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown("<div class='step-indicator'><div class='step-number'>4</div><div class='step-label'>Finalizing</div></div>", unsafe_allow_html=True)
                elif st.session_state.processing_stage == "embedding":
                    with col1:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Fetching</div></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Processing</div></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='step-indicator active'><div class='step-number'>3</div><div class='step-label'>Embedding</div></div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown("<div class='step-indicator'><div class='step-number'>4</div><div class='step-label'>Finalizing</div></div>", unsafe_allow_html=True)
                elif st.session_state.processing_stage == "finalizing":
                    with col1:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Fetching</div></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Processing</div></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Embedding</div></div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown("<div class='step-indicator active'><div class='step-number'>4</div><div class='step-label'>Finalizing</div></div>", unsafe_allow_html=True)
                elif st.session_state.processing_stage == "complete":
                    with col1:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Fetching</div></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Processing</div></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Embedding</div></div>", unsafe_allow_html=True)
                    with col4:
                        st.markdown("<div class='step-indicator completed'><div class='step-number'>✓</div><div class='step-label'>Finalizing</div></div>", unsafe_allow_html=True)
            
            # Show status message according to current stage
            status_container = st.empty()
            
            if st.session_state.processing_stage == "start":
                status_container.markdown("<div class='status-container'><div class='status-icon'></div><div class='status-message typing-effect'>Initiating website analysis...</div></div>", unsafe_allow_html=True)
                
                # Set stage to fetching and update progress
                st.session_state.processing_stage = "fetching"
                st.session_state.progress_value = 0.05
                time.sleep(0.5)  # Short delay for visual feedback
                st.rerun()
                
            elif st.session_state.processing_stage == "fetching":
                try:
                    # Performance optimization - use ThreadPoolExecutor for faster loading
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        # Show connecting status
                        status_container.markdown("<div class='status-container'><div class='status-icon connecting'></div><div class='status-message'>Connecting to website...</div></div>", unsafe_allow_html=True)
                        
                        # First check if the website is reachable
                        try:
                            import requests
                            response = requests.head(website_url, timeout=5)
                            if response.status_code >= 400:
                                raise ConnectionError(f"Website returned error status: {response.status_code}")
                        except Exception as e:
                            raise ConnectionError(f"Cannot connect to website: {str(e)}")
                        
                        # Create the loader only if website is reachable
                        loader = WebBaseLoader(website_url)
                        
                        # Start loading in background thread with timeout protection
                        try:
                            future = executor.submit(lambda: loader.load())
                            
                            # Show progress while waiting for response
                            for i in range(5, 25):
                                progress_bar.progress(i/100)
                                progress_text.markdown(f"<div class='progress-text-container'>{i}%</div>", unsafe_allow_html=True)
                                time.sleep(0.05)
                            
                            # Get document with timeout
                            import concurrent.futures
                            document = future.result(timeout=30)  # 30 second timeout
                            
                            # Save document to session state immediately
                            st.session_state.document = document
                            
                        except concurrent.futures.TimeoutError:
                            raise TimeoutError("Website loading timed out. Please try a different URL.")
                    
                    # Show downloading status
                    status_container.markdown("<div class='status-container'><div class='status-icon downloading'></div><div class='status-message'>Downloading website content...</div></div>", unsafe_allow_html=True)
                    
                    # Faster loading with optimized thread usage
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        future = executor.submit(lambda: loader.load())
                        
                        st.session_state.document = future.result()  # Store in session state
                        
                        # Show progress while loading content
                        for i in range(25, 35):
                            progress_bar.progress(i/100)
                            progress_text.markdown(f"<div class='progress-text-container'>{i}%</div>", unsafe_allow_html=True)
                            time.sleep(0.05)
                        
                        # Get loaded document and store in session state
                        st.session_state.document = future.result()
                    
                    # Move to processing stage
                    st.session_state.processing_stage = "processing"
                    st.session_state.progress_value = 0.35
                    st.rerun()
                    
                except Exception as e:
                    status_container.markdown(f"<div class='status-container error'><div class='status-icon error'></div><div class='status-message error-message'>Error: {str(e)}</div></div>", unsafe_allow_html=True)
                    st.error(f"Failed to process website: {str(e)}")
                    time.sleep(2)
                    
                    # Reset states
                    st.session_state.processing_stage = "start"
                    st.session_state.progress_value = 0
                    st.session_state.loading = False
                    st.rerun()
                    
            elif st.session_state.processing_stage == "processing":
                status_container.markdown("<div class='status-container'><div class='status-icon processing'></div><div class='status-message'>Processing content structure...</div></div>", unsafe_allow_html=True)
                
                # Performance optimization - more aggressive chunking for faster processing
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,  # Larger chunks for faster processing
                    chunk_overlap=50,  # Reduced overlap
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                # Show chunking progress
                for i in range(35, 50):
                    progress_bar.progress(i/100)
                    progress_text.markdown(f"<div class='progress-text-container'>{i}%</div>", unsafe_allow_html=True)
                    time.sleep(0.03)  # Even shorter delay
                
                # Process document in background
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Use document from session state
                    future = executor.submit(text_splitter.split_documents, st.session_state.document)
                    
                    # Update status while processing
                    status_container.markdown("<div class='status-container'><div class='status-icon chunking'></div><div class='status-message'>Chunking content into semantic units...</div></div>", unsafe_allow_html=True)
                    
                    # Show progress while chunking
                    for i in range(50, 60):
                        progress_bar.progress(i/100)
                        progress_text.markdown(f"<div class='progress-text-container'>{i}%</div>", unsafe_allow_html=True)
                        time.sleep(0.03)
                    
                    # Get result
                    document_chunks = future.result()
                
                # Move to embedding stage
                st.session_state.document_chunks = document_chunks  # Store for next stage
                st.session_state.processing_stage = "embedding"
                st.session_state.progress_value = 0.6
                st.rerun()
                
            elif st.session_state.processing_stage == "embedding":
                status_container.markdown("<div class='status-container'><div class='status-icon embedding'></div><div class='status-message'>Creating vector embeddings...</div></div>", unsafe_allow_html=True)
                
                # Retrieve document chunks from session state
                document_chunks = st.session_state.document_chunks
                
                # Performance optimization for vector store creation
                collection_name = f"chroma_{abs(hash(website_url))}"
                
                # Create vector store with progress updates
                for i in range(60, 85):
                    progress_bar.progress(i/100)
                    progress_text.markdown(f"<div class='progress-text-container'>{i}%</div>", unsafe_allow_html=True)
                    
                    if i == 70:
                        status_container.markdown("<div class='status-container'><div class='status-icon vectors'></div><div class='status-message'>Generating semantic embeddings...</div></div>", unsafe_allow_html=True)
                    elif i == 80:
                        status_container.markdown("<div class='status-container'><div class='status-icon optimizing'></div><div class='status-message'>Optimizing vector representations...</div></div>", unsafe_allow_html=True)
                    
                    time.sleep(0.03)
                
                # Generate embeddings faster with optimized settings
                vector_store = Chroma.from_documents(
                    document_chunks,
                    EMBEDDING_MODEL,
                    collection_name=collection_name,
                    persist_directory=None
                )
                
                # Move to finalizing stage
                st.session_state.vector_store = vector_store  # Store in session state
                st.session_state.processing_stage = "finalizing"
                st.session_state.progress_value = 0.85
                st.rerun()
                
            elif st.session_state.processing_stage == "finalizing":
                status_container.markdown("<div class='status-container'><div class='status-icon finalizing'></div><div class='status-message'>Finalizing knowledge database...</div></div>", unsafe_allow_html=True)
                
                # Final progress updates
                for i in range(85, 101):
                    progress_bar.progress(i/100)
                    progress_text.markdown(f"<div class='progress-text-container'>{i}%</div>", unsafe_allow_html=True)
                    
                    if i == 90:
                        status_container.markdown("<div class='status-container'><div class='status-icon connecting-db'></div><div class='status-message'>Connecting to retrieval system...</div></div>", unsafe_allow_html=True)
                    elif i == 95:
                        status_container.markdown("<div class='status-container'><div class='status-icon completing'></div><div class='status-message'>Finalizing analysis...</div></div>", unsafe_allow_html=True)
                    
                    time.sleep(0.02)  # Fast animation
                
                # Save URL to session state
                st.session_state.last_url = website_url
                
                # Show success message
                status_container.markdown("<div class='status-container success'><div class='status-icon success'></div><div class='status-message success-message'>Website analysis complete!</div></div>", unsafe_allow_html=True)
                
                # Short celebration animation
                st.markdown("<div class='celebration'></div>", unsafe_allow_html=True)
                time.sleep(0.8)  # Brief delay for success message
                
                # Complete the process
                st.session_state.processing_stage = "complete"
                st.session_state.loading = False
                st.rerun()
                
            elif st.session_state.processing_stage == "complete":
                # Clean up session state and move on
                del st.session_state.processing_stage
                del st.session_state.progress_value
                if "document_chunks" in st.session_state:
                    del st.session_state.document_chunks

    # Update the user interaction section
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # Check if vector store exists before processing the query
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
            st.error("Please load a website first by entering a URL in the sidebar.")
        else:
            # First, add and display the human message immediately
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            # Display all messages including the new human message
            for i, message in enumerate(st.session_state.chat_history):
                if isinstance(message, AIMessage):
                    display_ai_message(message.content, i)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)
            
            # Now show spinner and generate response
            with st.spinner('Generating response...'):
                response = get_response(user_query)
                # Add the AI response to chat history
                st.session_state.chat_history.append(AIMessage(content=response))
            
            # Rerun to refresh the UI with the new AI message
            st.rerun()
    else:
        # Display existing messages when no new input
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, AIMessage):
                display_ai_message(message.content, i)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

# Function to check network status
def check_network_status():
    try:
        import requests
        response = requests.head("https://www.google.com", timeout=3)
        if response.status_code < 400:
            return True
        return False
    except:
        return False

# Use this before attempting to load models
network_available = check_network_status()
if not network_available:
    warning_auto_dismiss("Network connectivity issues detected. SmartQueryBot will operate in limited offline mode.")
# Update CSS for better formatting
st.markdown("""
<style>
    /* Improve chat container styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    /* Human message styling */
    [data-testid="stChatMessageContent"] {
        border-radius: 15px;
        padding: 0.8rem;
    }
    
    /* AI message styling */
    .stChatMessage [data-testid="stChatMessageContent"] p {
        line-height: 1.5;
        margin-bottom: 0.7rem;
    }
    
    /* Elaborate button styling */
    .stButton button {
        background-color: #6c757d;
        color: white;
        border-radius: 20px;
        border: none;
        transition: all 0.3s ease;
        font-weight: bold;
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        background-color: #5a6268;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    /* Style for additional details */
    .elaborate-container {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Style for additional details with different levels */
    .elaborate-container.level-1 {
        background: #f8f9fa;
        border-left: 4px solid #6c757d;
    }
    
    .elaborate-container.level-2 {
        background: #f0f4f8;
        border-left: 4px solid #5b6d9a;
    }
    
    .elaborate-container.level-3 {
        background: #e8eff7;
        border-left: 4px solid #4682b4;
    }
    
    .elaborate-container.level-4 {
        background: #e0ebf5;
        border-left: 4px solid #385d8a;
    }
    
    /* Fix for overflow text in messages */
    [data-testid="stChatMessageContent"] p {
        word-break: break-word;
        white-space: pre-wrap;
    }
    
    /* Loading animation styles */
    .loading-title {
        text-align: center;
        color: #4682b4;
        margin-bottom: 1.5rem;
        animation: pulse 2s infinite ease-in-out;
    }
    
    .step-indicator {
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        background: #f8f9fa;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .step-indicator.active {
        background: linear-gradient(135deg, #4682b4, #6eb4ff);
        box-shadow: 0 4px 10px rgba(70, 130, 180, 0.3);
        transform: translateY(-5px);
        animation: active-pulse 1.5s infinite alternate;
    }
    
    .step-indicator.completed {
        background: linear-gradient(135deg, #28a745, #5dd879);
        box-shadow: 0 4px 10px rgba(40, 167, 69, 0.3);
        transform: translateY(-2px);
    }
    
    .step-number {
        width: 30px;
        height: 30px;
        line-height: 30px;
        border-radius: 50%;
        background: white;
        margin: 0 auto 8px auto;
        font-weight: bold;
        color: #495057;
        transition: all 0.3s ease;
    }
    
    .step-indicator.active .step-number {
        color: #4682b4;
        box-shadow: 0 0 8px rgba(70, 130, 180, 0.5);
    }
    
    .step-indicator.completed .step-number {
        color: #28a745;
        animation: complete-check 0.5s ease-out;
    }
    
    .step-label {
        font-size: 0.8rem;
        color: #495057;
        transition: color 0.3s ease;
    }
    
    .step-indicator.active .step-label,
    .step-indicator.completed .step-label {
        color: white;
    }
    
    .status-text {
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 1rem;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #4682b4;
        transition: all 0.3s ease;
    }
    
    .status-text.success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
        font-weight: bold;
        animation: success-pulse 1s ease;
    }
    
    .status-text.error {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
        font-weight: bold;
    }
    
    /* Smooth progress bar transitions */
    .stProgress > div > div {
        background-color: #4682b4;
        background-image: linear-gradient(to right, #4682b4, #6eb4ff);
        transition: width 0.3s ease-in-out;
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    
    @keyframes active-pulse {
        0% { box-shadow: 0 4px 10px rgba(70, 130, 180, 0.3); }
        100% { box-shadow: 0 4px 15px rgba(70, 130, 180, 0.5); }
    }
    
    @keyframes complete-check {
        0% { transform: scale(0.5); opacity: 0.5; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes success-pulse {
        0% { transform: scale(0.98); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* New animations for smoother transitions */
    .animate-fade-in {
        animation: fade-in 0.5s ease forwards;
    }
    
    .animate-slide-in {
        animation: slide-in 0.4s ease forwards;
        opacity: 0;
        transform: translateY(10px);
    }
    
    .animate-activate {
        animation: activate 0.5s ease forwards;
    }
    
    .animate-complete {
        animation: complete 0.5s ease forwards;
    }
    
    .animate-success {
        animation: success-appear 0.7s ease forwards;
    }
    
    @keyframes fade-in {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    @keyframes slide-in {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes activate {
        0% { transform: scale(0.95); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1) translateY(-5px); }
    }
    
    @keyframes complete {
        0% { background: linear-gradient(135deg, #4682b4, #6eb4ff); }
        100% { background: linear-gradient(135deg, #28a745, #5dd879); }
    }
    
    @keyframes success-appear {
        0% { opacity: 0; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.03); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Circular Progress Ring */
    .progress-ring-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    
    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: #f0f4f8;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1), 
                    inset 0 2px 5px rgba(255, 255, 255, 0.9),
                    inset 0 -2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .progress-circle {
        position: absolute;
        top: 10px;
        left: 10px;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: conic-gradient(#4682b4 0%, #6eb4ff 30%, transparent 30%);
        mask: radial-gradient(transparent 45%, black 46%);
        -webkit-mask: radial-gradient(transparent 45%, black 46%);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .progress-text {
        font-size: 24px;
        font-weight: 700;
        color: #4682b4;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Steps Timeline */
    .steps-timeline {
        display: flex;
        justify-content: space-between;
        width: 100%;
        padding: 20px 10px;
        position: relative;
        margin: 30px 0;
    }
    
    .steps-timeline::before {
        content: '';
        position: absolute;
        top: 40px;
        left: 0;
        width: 100%;
        height: 4px;
        background: #e9ecef;
        z-index: 0;
    }
    
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 25%;
        position: relative;
        z-index: 1;
    }
    
    .step-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #f8f9fa;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 18px;
        color: #adb5bd;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .step-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    
    .step-title {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .step-progress {
        width: 90%;
        height: 5px;
        background: #e9ecef;
        border-radius: 3px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #4682b4, #6eb4ff);
        transition: width 0.3s ease-in-out;
        border-radius: 3px;
    }
    
    /* Step states */
    .step.active .step-icon {
        background: linear-gradient(135deg, #4682b4, #6eb4ff);
        color: white;
        transform: scale(1.2);
        box-shadow: 0 4px 10px rgba(70, 130, 180, 0.3);
        animation: pulse-step 1.5s infinite alternate;
    }
    
    @keyframes pulse-step {
        0% {
            box-shadow: 0 4px 10px rgba(70, 130, 180, 0.3);
        }
        100% {
            box-shadow: 0 4px 15px rgba(70, 130, 180, 0.7);
        }
    }
    
    .step.active .step-title {
        color: #4682b4;
        font-weight: 600;
    }
    
    .step.completed .step-icon {
        background: linear-gradient(135deg, #28a745, #5dd879);
        color: white;
    }
    
    .step.completed .step-title {
        color: #28a745;
    }
    
    .step.completed .progress-bar {
        width: 100% !important;
        background: linear-gradient(90deg, #28a745, #5dd879);
    }
    
    /* Status Container */
    .status-container {
        display: flex;
        align-items: center;
        padding: 15px;
        border-radius: 10px;
        background: #f8f9fa;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-left: 4px solid #4682b4;
    }
    
    .status-icon {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 15px;
        position: relative;
        background: #4682b4;
    }
    
    .status-icon::before, 
    .status-icon::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        animation: status-pulse 1s ease-in-out infinite alternate;
    }
    
    @keyframes status-pulse {
        0% {
            transform: scale(1);
            opacity: 0.5;
        }
        100% {
            transform: scale(1.3);
            opacity: 0;
        }
    }
    
    .status-message {
        flex: 1;
        font-size: 16px;
        color: #495057;
    }
    
    /* Status icons with specific animations */
    .status-icon.connecting::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        border: 2px solid #4682b4;
        border-radius: 50%;
        animation: ripple 1.5s linear infinite;
    }
    
    @keyframes ripple {
        0% {
            transform: scale(1);
            opacity: 0.8;
        }
        100% {
            transform: scale(1.5);
            opacity: 0;
        }
    }
    
    .status-icon.downloading {
        background: none;
        border: 2px solid #4682b4;
    }
    
    .status-icon.downloading::before {
        content: '↓';
        position: absolute;
        top: 0;
        left: 5px;
        font-size: 12px;
        color: #4682b4;
        animation: downloading 1s infinite;
    }
    
    @keyframes downloading {
        0%, 100% {
            transform: translateY(-2px);
        }
        50% {
            transform: translateY(2px);
        }
    }
    
    .status-icon.processing {
        background: none;
        border: 2px solid #4682b4;
        overflow: hidden;
    }
    
    .status-icon.processing::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 200%;
        height: 100%;
        background: linear-gradient(90deg, transparent, #4682b4, transparent);
        animation: processing 1.5s infinite;
    }
    
    @keyframes processing {
        0% {
            left: -100%;
        }
        100% {
            left: 100%;
        }
    }
    
    .status-icon.embedding,
    .status-icon.analyzing,
    .status-icon.chunking,
    .status-icon.vectors,
    .status-icon.optimizing {
        background: conic-gradient(#4682b4, #6eb4ff, #4682b4);
        animation: rotate 2s linear infinite;
    }
    
    @keyframes rotate {
        100% {
            transform: rotate(360deg);
        }
    }
    
    .status-icon.finalizing,
    .status-icon.connecting-db,
    .status-icon.completing {
        background: #4682b4;
        clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
        animation: pulse-diamond 1.5s infinite alternate;
    }
    
    @keyframes pulse-diamond {
        0% {
            transform: scale(1) rotate(0deg);
        }
        100% {
            transform: scale(1.2) rotate(45deg);
        }
    }
    
    /* Success state */
    .status-container.success {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .status-icon.success {
        background: #28a745;
    }
    
    .status-icon.success::before {
        content: '✓';
        position: absolute;
        top: -5px;
        left: 5px;
        color: white;
        font-size: 16px;
        font-weight: bold;
    }
    
    .status-message.success-message {
        color: #155724;
        font-weight: 600;
        animation: success-appear 0.5s ease-in-out;
    }
    
    @keyframes success-appear {
        0% {
            opacity: 0;
            transform: translateY(10px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Error state */
    .animation-wrapper.error .pulse-beacon {
        background: rgba(220, 53, 69, 0.2);
        box-shadow: 0 0 0 rgba(220, 53, 69, 0.4);
        animation: pulse-ring-error 2s cubic-bezier(0.25, 0.8, 0.25, 1) infinite;
    }
    
    @keyframes pulse-ring-error {
        0% {
            transform: scale(0.33);
            opacity: 0.6;
        }
        80% {
            transform: scale(1.8);
            opacity: 0;
        }
        100% {
            transform: scale(0.33);
            opacity: 0;
        }
    }
    
    .status-container.error {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    .status-icon.error {
        background: #dc3545;
    }
    
    .status-icon.error::before,
    .status-icon.error::after {
        content: '';
        position: absolute;
        width: 12px;
        height: 2px;
        background: white;
        top: 9px;
        left: 4px;
    }
    
    .status-icon.error::before {
        transform: rotate(45deg);
    }
    
    .status-icon.error::after {
        transform: rotate(-45deg);
    }
    
    .status-message.error-message {
        color: #721c24;
        font-weight: 500;
    }
    
    /* Typing effect animation */
    .typing-effect {
        border-right: 2px solid #4682b4;
        white-space: nowrap;
        overflow: hidden;
        animation: typing 2s steps(40, end), blink-caret 0.75s step-end infinite;
    }
    
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: #4682b4; }
    }
    
    /* Celebration animation */
    .celebration {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(70, 130, 180, 0.2) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(40, 167, 69, 0.2) 0%, transparent 50%);
        animation: celebration-bg 3s ease;
    }
    
    .celebration::before,
    .celebration::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        background-repeat: no-repeat;
        background-size: cover;
        mix-blend-mode: screen;
        opacity: 0;
        animation: confetti 2s ease-out forwards;
    }
    
    @keyframes celebration-bg {
        0% {
            opacity: 0;
        }
        20% {
            opacity: 1;
        }
        80% {
            opacity: 1;
        }
        100% {
            opacity: 0;
        }
    }
    
    @keyframes confetti {
        0% {
            background-position: 0% 0%;
            opacity: 0;
        }
        25% {
            opacity: 1;
        }
        50% {
            background-position: 100% 100%;
        }
        75% {
            opacity: 1;
        }
        100% {
            background-position: 0% 100%;
            opacity: 0;
        }
    }
    .progress-text-container {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4682b4;
    margin: 10px 0;
}

/* Make animations more efficient */
@keyframes pulse {
    0% { opacity: 0.8; }
    100% { opacity: 1; }
}

/* Optimize animation performance */
.step-indicator.active {
    animation: pulse-step 1.5s ease-in-out infinite alternate;
    will-change: transform, box-shadow;
}

.status-icon {
    will-change: transform;
}

/* Hardware acceleration for smoother animations */
.progress-ring {
    transform: translateZ(0);
    backface-visibility: hidden;
}

/* Faster transitions for better perception of speed */
.step-indicator {
    transition: all 0.2s ease-out;
}

.status-container {
    transition: all 0.2s ease-out;
}
</style>
""", unsafe_allow_html=True)
