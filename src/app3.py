import streamlit as st
import time
import threading
import json
import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from functools import lru_cache
from streamlit.components.v1 import html

# MUST BE THE FIRST STREAMLIT COMMAND - MOVE TO TOP
st.set_page_config(
    page_title="SmartQuery Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import LangChain components
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Initialize models
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
LLM_MODEL = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
EMBEDDING_MODEL = OpenAIEmbeddings()

# Global RAG chain for conversation
_GLOBAL_RAG_CHAIN = None

# Utility functions for notifications
def success_auto_dismiss(message, timeout=3):
    """Show success message that auto-dismisses after timeout seconds"""
    with st.empty():
        st.success(message)
        time.sleep(timeout)
        st.empty()

def info_auto_dismiss(message, timeout=3):
    """Show info message that auto-dismisses after timeout seconds"""
    with st.empty():
        st.info(message)
        time.sleep(timeout)
        st.empty()

# Web content fetching
@st.cache_data(ttl=3600)
def fetch_website_content_cached(url):
    """Cached version of website content fetching"""
    try:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to fetch website content: {str(e)}")
        raise e

def fetch_website_content(url):
    """Fetch content from website URL"""
    return fetch_website_content_cached(url)

# Conversation management functions
def save_conversation(title=None):
    """Save current conversation to file"""
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        return False
    
    os.makedirs("conversation_history", exist_ok=True)
    
    if not title:
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                title = msg.content[:30] + "..." if len(msg.content) > 30 else msg.content
                break
    
    if not title:
        title = "Conversation " + datetime.now().strftime("%Y-%m-%d %H:%M")
    
    conversation_data = {
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "website_url": st.session_state.get("last_url", ""),
        "messages": []
    }
    
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            conversation_data["messages"].append({
                "role": "AI",
                "content": msg.content
            })
        elif isinstance(msg, HumanMessage):
            conversation_data["messages"].append({
                "role": "Human",
                "content": msg.content
            })
    
    safe_title = "".join(c for c in title[:20] if c.isalnum() or c in " -_").strip()
    safe_title = safe_title.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_title}.json"
    filepath = os.path.join("conversation_history", filename)
    
    with open(filepath, "w") as f:
        json.dump(conversation_data, f, indent=2)
    
    load_conversation_list()
    return True

def load_conversation_list():
    """Load list of saved conversations"""
    conversations = []
    
    if os.path.exists("conversation_history"):
        for filename in os.listdir("conversation_history"):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join("conversation_history", filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        conversations.append({
                            "id": filename,
                            "title": data.get("title", "Untitled"),
                            "timestamp": data.get("timestamp", ""),
                            "website_url": data.get("website_url", ""),
                            "message_count": len(data.get("messages", [])),
                            "path": filepath
                        })
                except:
                    pass
    
    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    st.session_state.saved_conversations = conversations
    return conversations

def load_conversation(filepath):
    """Load a specific conversation"""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            
            messages = []
            for msg in data.get("messages", []):
                if msg["role"] == "AI":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "Human":
                    messages.append(HumanMessage(content=msg["content"]))
            
            if "website_url" in data and data["website_url"]:
                st.session_state.last_url = data["website_url"]
                
            return messages
    except Exception as e:
        st.error(f"Error loading conversation: {str(e)}")
        return None

def delete_conversation(filepath):
    """Delete a saved conversation"""
    try:
        os.remove(filepath)
        load_conversation_list()
        return True
    except:
        return False

def display_ai_message(content, message_idx):
    with st.chat_message("AI"):
        st.markdown(content)
        
        cols = st.columns([0.85, 0.15])
        
        # Don't show elaborate button for initial messages or analysis messages
        if (not "Hello! I'm your AI assistant" in content and 
            not "I've analyzed the content" in content and
            not "Hello! I am Vaibhav's AI assistant" in content and
            message_idx > 0):  # Also check message index to avoid initial message
            if cols[1].button("Elaborate", key=f"elaborate_{message_idx}", help="Get a more detailed explanation"):
                if "response_contexts" in st.session_state and message_idx in st.session_state.response_contexts:
                    context = st.session_state.response_contexts[message_idx]
                    
                    user_question = ""
                    if message_idx > 0 and len(st.session_state.chat_history) >= message_idx:
                        if isinstance(st.session_state.chat_history[message_idx-1], HumanMessage):
                            user_question = st.session_state.chat_history[message_idx-1].content
                    
                    elaborate_animation = st.empty()
                    
                    with elaborate_animation.container():
                        st.markdown("""
                        <div class="generating-response">
                            <div class="response-dots">
                                <div class="response-dot"></div>
                                <div class="response-dot"></div>
                                <div class="response-dot"></div>
                            </div>
                            <span class="response-text">Generating detailed explanation...</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    detailed_response = get_elaborate_response(user_question, context, message_idx)
                    elaborate_animation.empty()
                    
                    if "elaborated_responses" not in st.session_state:
                        st.session_state.elaborated_responses = {}
                    st.session_state.elaborated_responses[message_idx] = detailed_response
                else:
                    st.warning("Sorry, I don't have enough context to elaborate on this response.")
    
    if "elaborated_responses" in st.session_state and message_idx in st.session_state.elaborated_responses:
        with st.expander("**Detailed Explanation:**"):
            st.markdown(st.session_state.elaborated_responses[message_idx], unsafe_allow_html=True)

def get_context_retriever_chain(vector_store):
    """Get retriever chain from vector store"""
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    embeddings_filter = EmbeddingsFilter(
        embeddings=EMBEDDING_MODEL,
        similarity_threshold=0.7
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )
    
    return compression_retriever

def get_conversational_rag_chain(retriever_chain):
    """Get RAG chain that maintains conversation context"""
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions about web content. 
    Use only the following context to answer the user's question. 
    
    If the context doesn't contain relevant information, simply say 
    "I don't have enough information about that in this page content."
    
    Make your answers clear and easy for anyone to understand, even if they're not familiar with the topic.
    When explaining technical terms, include a brief definition.
    
    Context:
    {context}
    
    Question:
    {input}
    """)
    
    document_chain = create_stuff_documents_chain(LLM_MODEL, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    return retrieval_chain

def get_response(user_input):
    try:
        global _GLOBAL_RAG_CHAIN
        if _GLOBAL_RAG_CHAIN is None:
            retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
            _GLOBAL_RAG_CHAIN = get_conversational_rag_chain(retriever_chain)
            
        if len(user_input) > 200:
            user_input = user_input[:200] + "..."
            
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                _GLOBAL_RAG_CHAIN.invoke,
                {"chat_history": [], "input": user_input}
            )
            try:
                response = future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                return "I'm sorry, but the response is taking too long to generate. Please try a simpler question or a different website."
                
        answer = response.get('answer', '').strip()
        context = []
        if 'context' in response:
            context = [doc.page_content for doc in response.get('context', [])]
        context_text = "\n".join(context)
        
        if not answer or len(answer) < 5:
            return "No specific answer found in the website content."

        answer_lines = [
            line for line in answer.split('\n')
            if line.strip()
            and not re.match(r'^\s*(system:|context:|answer the question|based on the context|according to the context)', line.strip(), re.I)
            and not re.match(r'^\s*(Archived|Retrieved|See also|http|www\.)', line, re.I)
        ]

        answer_lines = [re.sub(r'(http|www\.)[^\s]+', '', line) for line in answer_lines]
        answer_lines = [re.sub(r'\[\d+\]|\[\d+\]\[\d+\]', '', line) for line in answer_lines]

        is_listing = False
        if any(re.match(r'^\s*(\d+\.|-|\*|•)', line) for line in answer_lines) or len(answer_lines) > 2:
            is_listing = True
        
        formatted_response = ""
        
        if not is_listing and len("".join(answer_lines)) < 500:
            paragraph = " ".join([line.strip() for line in answer_lines if line.strip()])
            paragraph = re.sub(r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-ZaZ]+){1,3})\b', r'**\1**', paragraph)
            paragraph = re.sub(r'(\d+(?:\.\d+)?(?:\s*%)?)', r'**\1**', paragraph)
            formatted_response = paragraph
        else:
            formatted_points = []
            for i, line in enumerate(answer_lines):
                if len(line.strip()) < 3:
                    continue
                    
                line = re.sub(r'(\d+(?:\.\d+)?(?:\s*%)?)', r'**\1**', line)
                line = re.sub(r'\b([A-Z][A-ZaZ]+(?:\s+[A-Z][A-ZaZ]+){1,3})\b', r'**\1**', line)
                
                if not line.endswith(('.', '!', '?')):
                    line += '.'
                
                if not re.match(r'^\s*(\d+\.|-|\*|•)', line):
                    formatted_points.append(f"- {line[0].upper() + line[1:]}")
                else:
                    formatted_points.append(line)
            
            formatted_response = '\n'.join(formatted_points)
            
        if "response_contexts" not in st.session_state:
            st.session_state.response_contexts = {}
        
        response_id = len(st.session_state.chat_history)
        st.session_state.response_contexts[response_id] = context_text
        
        return formatted_response if formatted_response.strip() else "No specific answer found in the website content."
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Error generating response."

def get_elaborate_response(user_input, context, message_idx):
    try:
        cache_key = f"{user_input[:50]}_{message_idx}"
        if "elaborate_cache" not in st.session_state:
            st.session_state.elaborate_cache = {}
            
        if cache_key in st.session_state.elaborate_cache:
            return st.session_state.elaborate_cache[cache_key]
            
        prompt = f"""
Please provide a comprehensive, well-structured explanation about the following topic based ONLY on the given context.
Format your response in a clear, educational way with these sections:
1. Main concept/overview (1-2 paragraphs)
2. Key details (using bullet points for lists, paragraphs for explanations)
3. Additional context or examples if relevant
4. A brief conclusion

- Use markdown formatting for readability
- Bold important terms and concepts
- Use paragraphs for explanations and bullet points for lists
- Include definitions for technical terms
- If the context doesn't contain enough information, acknowledge the limitations

QUESTION: {user_input}
CONTEXT: {context}
"""
        
        if hasattr(LLM_MODEL, "invoke"):
            detailed_response = LLM_MODEL.invoke(prompt)
            if hasattr(detailed_response, "content"):
                detailed_response = detailed_response.content
        else:
            detailed_response = LLM_MODEL.generate([prompt]).generations[0][0].text
            
        response_lines = detailed_response.strip().split("\n")
        response_lines = [
            line for line in response_lines 
            if not re.match(r'^\s*(system:|context:|question:|answer:|based on the context|according to the context)', line.strip(), re.I)
        ]
        
        formatted_lines = []
        for line in response_lines:
            # Format headers properly
            if re.match(r'^\d+\.?\s+', line):
                # Convert "1. Title" to "### Title"
                formatted_header = re.sub(r'^\d+\.?\s+(.+)$', r'### \1', line)
                formatted_lines.append(formatted_header)
            else:
                # Keep regular lines as they are
                formatted_lines.append(line)
        
        formatted_text = "\n".join(formatted_lines)
        
        important_term_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b(?!\*\*)')
        number_pattern = re.compile(r'(\d+(?:\.\d+)?(?:\s*%)?)')
        
        formatted_text = important_term_pattern.sub(r'**\1**', formatted_text)
        formatted_text = number_pattern.sub(r'**\1**', formatted_text)
        
        st.session_state.elaborate_cache[cache_key] = formatted_text
        
        return formatted_text
    
    except Exception as e:
        return f"Failed to generate a detailed explanation: {str(e)}"

# Initialize session states
if "loaded_conversation" not in st.session_state:
    st.session_state.loaded_conversation = None

if "thinking" not in st.session_state:
    st.session_state.thinking = False

if "generating_response" not in st.session_state:
    st.session_state.generating_response = False

if "show_homepage" not in st.session_state:
    st.session_state.show_homepage = True

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your AI assistant. Enter a website URL in the sidebar to begin analyzing content, or ask me anything!")
    ]

# Apply professional Neumorphism CSS styling - ChatGPT inspired
st.markdown("""
<style>
    /* Professional Neumorphic UI - ChatGPT Style Dark Mode */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        /* Modern Dark Theme Variables */
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --bg-card: #1c2128;
        --bg-hover: #2d333b;
        
        --shadow-dark: rgba(0, 0, 0, 0.8);
        --shadow-light: rgba(48, 54, 61, 0.3);
        --shadow-glow: rgba(88, 166, 255, 0.15);
        
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #656d76;
        --text-inverse: #0d1117;
        
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-purple: #a5a5ff;
        --accent-orange: #ff7b72;
        --accent-yellow: #f2cc60;
        
        --border-primary: #30363d;
        --border-secondary: #21262d;
        --border-focus: #58a6ff;
        
        --gradient-primary: linear-gradient(135deg, #58a6ff, #a5a5ff);
        --gradient-secondary: linear-gradient(135deg, #3fb950, #58a6ff);
        --gradient-accent: linear-gradient(135deg, #a5a5ff, #ff7b72);
    }

    /* Global Reset & Base Styles */
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    /* Remove all browser default outlines and focus rings */
    input, textarea, select, button {
        outline: none !important;
    }

    input:focus, textarea:focus, select:focus, button:focus {
        outline: none !important;
    }

    input:active, textarea:active, select:active, button:active {
        outline: none !important;
    }

    /* Remove Streamlit specific input outlines */
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox select:focus {
        outline: none !important;
        border-color: transparent !important;
    }

    body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Hide Streamlit Branding & Default Elements */
    #MainMenu, .stDeployButton, footer, header {
        visibility: hidden !important;
        height: 0 !important;
    }

    .stApp > header {
        background-color: transparent !important;
    }

    /* Remove Empty Containers - Advanced Targeting */
    .element-container:empty,
    [data-testid="element-container"]:empty,
    [data-testid="stVerticalBlock"]:empty,
    div:empty:not([class*="chat"]):not([class*="search"]):not([class*="signature"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }

    /* Professional Neumorphic Components */
    .neu-card {
        background: var(--bg-card);
        border-radius: 16px;
        border: 1px solid var(--border-primary);
        box-shadow: 
            8px 8px 16px var(--shadow-dark),
            -8px -8px 16px var(--shadow-light),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .neu-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            inset 0 1px 0 rgba(255, 255, 255, 0.05),
            0 0 20px var(--shadow-glow);
        border-color: var(--border-focus);
    }

    .neu-inset {
        background: var(--bg-secondary);
        border-radius: 12px;
        border: 1px solid var(--border-secondary);
        box-shadow: 
            inset 4px 4px 8px var(--shadow-dark),
            inset -4px -4px 8px var(--shadow-light);
    }

    .neu-button {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 
            4px 4px 8px var(--shadow-dark),
            -4px -4px 8px var(--shadow-light);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .neu-button:hover {
        transform: translateY(-1px);
        box-shadow: 
            6px 6px 12px var(--shadow-dark),
            -6px -6px 12px var(--shadow-light);
        border-color: var(--border-focus);
    }

    .neu-button:active {
        transform: translateY(0);
        box-shadow: 
            inset 2px 2px 4px var(--shadow-dark),
            inset -2px -2px 4px var(--shadow-light);
    }

    .neu-button.primary {
        background: var(--gradient-primary);
        border-color: var(--accent-blue);
        color: white;
    }

    .neu-button.success {
        background: var(--gradient-secondary);
        border-color: var(--accent-green);
        color: white;
    }



    /* Custom Resize Handle for Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border-right: 2px solid transparent !important;
        background-image: 
            linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%),
            linear-gradient(180deg, var(--accent-blue), var(--accent-purple)) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        min-width: 380px !important;
        max-width: 600px !important;
        width: 420px !important;
        resize: horizontal !important;
        overflow: auto !important;
        position: relative !important;
        box-shadow: 
            20px 0 40px rgba(0, 0, 0, 0.5),
            inset -1px 0 0 rgba(88, 166, 255, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* Custom resize handle styling */
    [data-testid="stSidebar"]::after {
        content: '';
        position: absolute;
        top: 50%;
        right: -8px;
        transform: translateY(-50%);
        width: 16px;
        height: 60px;
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        border-radius: 8px;
        cursor: ew-resize;
        opacity: 0;
        transition: all 0.3s ease;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
    }

    [data-testid="stSidebar"]:hover::after {
        opacity: 0.7;
    }

    [data-testid="stSidebar"]::after:hover {
        opacity: 1 !important;
        transform: translateY(-50%) scale(1.1);
        box-shadow: 0 6px 16px rgba(88, 166, 255, 0.4);
    }

    /* Resize grip indicator */
    [data-testid="stSidebar"] {
        position: relative;
    }

    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: -2px;
        width: 2px;
        height: 100%;
        background: linear-gradient(
            180deg, 
            var(--accent-blue) 0%, 
            var(--accent-purple) 25%, 
            var(--accent-green) 50%, 
            var(--accent-orange) 75%, 
            var(--accent-blue) 100%
        );
        animation: sidebar-glow 4s ease-in-out infinite;
        z-index: 1;
    }

    @keyframes sidebar-glow {
        0%, 100% { 
            opacity: 0.6;
            filter: blur(0px);
        }
        50% { 
            opacity: 1;
            filter: blur(1px);
        }
    }

    [data-testid="stSidebar"]:hover {
        box-shadow: 
            25px 0 50px rgba(0, 0, 0, 0.6),
            inset -1px 0 0 rgba(88, 166, 255, 0.4),
            0 0 30px rgba(88, 166, 255, 0.1) !important;
    }

    /* Enhanced conversation item styling */
    .conversation-item {
        background: rgba(22, 27, 34, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(88, 166, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin-bottom: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .conversation-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(88, 166, 255, 0.05), transparent);
        transition: all 0.5s ease;
    }

    .conversation-item:hover::before {
        left: 100%;
    }

    .conversation-item:hover {
        transform: translateX(4px) !important;
        border-color: rgba(88, 166, 255, 0.3) !important;
        box-shadow: 
            -4px 4px 20px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(88, 166, 255, 0.1) !important;
        background: rgba(28, 33, 40, 0.9) !important;
    }

    /* Loading animations */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .loading-shimmer {
        background: linear-gradient(
            90deg,
            rgba(28, 33, 40, 0.8) 0%,
            rgba(88, 166, 255, 0.1) 50%,
            rgba(28, 33, 40, 0.8) 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }

    /* Floating action buttons */
    .fab {
        position: fixed !important;
        width: 56px !important;
        height: 56px !important;
        border-radius: 50% !important;
        background: var(--gradient-primary) !important;
        border: none !important;
        color: white !important;
        font-size: 24px !important;
        cursor: pointer !important;
        box-shadow: 
            0 8px 24px rgba(88, 166, 255, 0.4),
            0 0 20px rgba(88, 166, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        z-index: 1000 !important;
    }

    .fab:hover {
        transform: scale(1.1) rotate(10deg) !important;
        box-shadow: 
            0 12px 32px rgba(88, 166, 255, 0.5),
            0 0 30px rgba(88, 166, 255, 0.3) !important;
        background: var(--gradient-secondary) !important;
    }

    /* Status indicators */
    .status-indicator {
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        display: inline-block !important;
        margin-right: 8px !important;
        animation: pulse 2s ease-in-out infinite !important;
    }

    .status-online {
        background: var(--accent-green) !important;
        box-shadow: 0 0 8px rgba(63, 185, 80, 0.5) !important;
    }

    .status-loading {
        background: var(--accent-orange) !important;
        box-shadow: 0 0 8px rgba(255, 123, 114, 0.5) !important;
    }

    .status-offline {
        background: var(--text-muted) !important;
        box-shadow: 0 0 8px rgba(101, 109, 118, 0.5) !important;
    }

    @keyframes pulse {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.5;
            transform: scale(1.2);
        }
    }

    /* Professional Header Text Animation */
    .sidebar-header h2 {
        font-size: 19px;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.015em;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: subtle-fade 4s ease-in-out infinite alternate;
    }

    .sidebar-header h2:hover {
        color: rgba(88, 166, 255, 0.9);
        text-shadow: 0 0 8px rgba(88, 166, 255, 0.2);
    }

    @keyframes subtle-fade {
        0% {
            opacity: 0.9;
        }
        100% {
            opacity: 1;
            text-shadow: 0 0 4px rgba(88, 166, 255, 0.1);
        }
    }

    /* Simplified Response Generation Animation */
    .generating-response {
        display: flex;
        align-items: center;
        padding: 20px 24px !important;
        background: linear-gradient(135deg, #1e3a2e, #0f2419) !important;
        border: 2px solid var(--accent-green) !important;
        border-radius: 22px 22px 22px 6px !important;
        margin: 20px 20px 24px 20px !important;
        margin-right: 100px !important;
        color: #ffffff !important;
        box-shadow: 
            6px 6px 15px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(63, 185, 80, 0.2) !important;
        animation: responseSlide 0.5s ease-out !important;
    }

    @keyframes responseSlide {
        0% {
            opacity: 0;
            transform: translateX(-20px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .response-dots {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .response-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        animation: dotPulse 1.5s ease-in-out infinite;
    }

    .response-dot:nth-child(2) {
        animation-delay: 0.2s;
        background: var(--accent-blue);
    }

    .response-dot:nth-child(3) {
        animation-delay: 0.4s;
        background: var(--accent-purple);
    }

    @keyframes dotPulse {
        0%, 80%, 100% { 
            transform: scale(1); 
            opacity: 0.7; 
        }
        40% { 
            transform: scale(1.3); 
            opacity: 1; 
        }
    }

    .response-text {
        margin-left: 16px;
        color: #ffffff !important;
        font-style: italic;
        font-size: 14px;
        font-weight: 500;
    }


    /* Enhanced Homepage Design */
    .hero-section {
        text-align: center;
        padding: 80px 40px;
        background: radial-gradient(ellipse at center, rgba(88, 166, 255, 0.1) 0%, transparent 70%);
        border-radius: 32px;
        margin: 40px 0;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(88, 166, 255, 0.05), transparent);
        animation: hero-shimmer 8s ease-in-out infinite;
        pointer-events: none;
    }

    @keyframes hero-shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }

    .hero-logo {
        width: 120px;
        height: 120px;
        margin: 0 auto 32px;
        background: var(--gradient-primary);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        font-weight: 900;
        color: white;
        box-shadow: 
            0 20px 40px rgba(88, 166, 255, 0.3),
            0 10px 20px rgba(0, 0, 0, 0.2);
        animation: logo-float 6s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    @keyframes logo-float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 16px;
        letter-spacing: -0.05em;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        margin: 0 0 32px;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    .capabilities-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 24px;
        margin: 60px 0;
    }

    .capability-card {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            8px 8px 16px var(--shadow-dark),
            -8px -8px 16px var(--shadow-light);
        position: relative;
        overflow: hidden;
    }

    .capability-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(88, 166, 255, 0.1), transparent);
        transition: all 0.5s ease;
    }

    .capability-card:hover::before {
        left: 100%;
    }

    .capability-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: var(--border-focus);
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            0 0 30px var(--shadow-glow);
    }

    .capability-icon {
        font-size: 3rem;
        margin-bottom: 20px;
        display: block;
        position: relative;
        z-index: 1;
    }

    .capability-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 12px;
        position: relative;
        z-index: 1;
    }

    .capability-description {
        color: var(--text-secondary);
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    .stats-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin: 60px 0;
        flex-wrap: wrap;
    }

    .stat-item {
        text-align: center;
        padding: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        min-width: 120px;
        box-shadow: 
            4px 4px 8px var(--shadow-dark),
            -4px -4px 8px var(--shadow-light);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 900;
        color: var(--accent-blue);
        display: block;
        margin-bottom: 8px;
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
    }

    [data-testid="stSidebarUserContent"] {
        padding: 20px 16px !important;
        background: transparent !important;
        position: relative !important;
        overflow-y: auto !important;
        height: 100vh !important;
    }

    /* Animated background particles */
    [data-testid="stSidebarUserContent"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(88, 166, 255, 0.3), transparent),
            radial-gradient(2px 2px at 40% 70%, rgba(165, 165, 255, 0.3), transparent),
            radial-gradient(1px 1px at 90% 40%, rgba(63, 185, 80, 0.3), transparent),
            radial-gradient(1px 1px at 60% 10%, rgba(255, 123, 114, 0.3), transparent);
        background-size: 200% 200%;
        animation: particles-float 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes particles-float {
        0%, 100% { 
            background-position: 0% 0%, 100% 100%, 0% 100%, 100% 0%;
        }
        50% { 
            background-position: 100% 100%, 0% 0%, 100% 0%, 0% 100%;
        }
    }

    /* Hide sidebar collapse button but make it cooler */
    [data-testid="stSidebar"] button[aria-label="Close sidebar"] {
        display: none !important;
    }

    /* Professional Sidebar Header */
    .sidebar-header {
        text-align: center !important;
        margin-bottom: 28px !important;
        padding: 20px 18px !important;
        background: rgba(24, 28, 34, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease !important;
    }

    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(
            90deg, 
            transparent, 
            rgba(255, 255, 255, 0.12), 
            transparent
        );
        opacity: 0.6;
    }

    .sidebar-header:hover {
        transform: translateY(-1px) !important;
        border-color: rgba(88, 166, 255, 0.15) !important;
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        background: rgba(24, 28, 34, 0.8) !important;
    }

    .sidebar-header .logo {
        width: 56px !important;
        height: 56px !important;
        margin: 0 auto 14px !important;
        background: linear-gradient(135deg, #4a90e2, #357abd) !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: white !important;
        position: relative !important;
        box-shadow: 
            0 4px 12px rgba(88, 166, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        transition: all 0.3s ease !important;
    }

    .sidebar-header:hover .logo {
        transform: scale(1.02) !important;
        box-shadow: 
            0 5px 15px rgba(88, 166, 255, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    .sidebar-header h2 {
        font-size: 19px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 0 6px !important;
        letter-spacing: -0.015em !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        position: relative !important;
    }

    .sidebar-header h2:hover {
        color: rgba(88, 166, 255, 0.9) !important;
        text-shadow: 0 0 8px rgba(88, 166, 255, 0.2) !important;
    }

    .sidebar-header p {
        font-size: 13px !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
        font-weight: 400 !important;
        opacity: 0.8 !important;
        transition: all 0.3s ease !important;
    }

    .sidebar-header:hover p {
        opacity: 0.9 !important;
        color: rgba(255, 255, 255, 0.7) !important;
    }

    /* WOW Sidebar Sections */
    .sidebar-section {
        background: rgba(28, 33, 40, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid transparent !important;
        background-image: 
            linear-gradient(rgba(28, 33, 40, 0.8), rgba(28, 33, 40, 0.8)),
            linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-green)) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        border-radius: 20px !important;
        padding: 24px 20px !important;
        margin-bottom: 24px !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        z-index: 1 !important;
    }

    .sidebar-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg, 
            transparent, 
            rgba(88, 166, 255, 0.1), 
            transparent
        );
        transition: all 0.6s ease !important;
        z-index: -1 !important;
    }

    .sidebar-section:hover::before {
        left: 100% !important;
    }

    .sidebar-section:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(88, 166, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        background-image: 
            linear-gradient(rgba(28, 33, 40, 0.9), rgba(28, 33, 40, 0.9)),
            linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-green)) !important;
    }

    /* Animated Section Titles */
    .section-title {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 0 20px !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        position: relative !important;
        padding-bottom: 12px !important;
        transition: all 0.3s ease !important;
    }

    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0%;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        transition: width 0.4s ease;
        border-radius: 1px;
    }

    .sidebar-section:hover .section-title::after {
        width: 100%;
    }

    .section-title .icon {
        font-size: 20px !important;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: icon-pulse 2s ease-in-out infinite !important;
        filter: drop-shadow(0 0 8px rgba(88, 166, 255, 0.3)) !important;
    }

    @keyframes icon-pulse {
        0%, 100% { 
            transform: scale(1);
            filter: drop-shadow(0 0 8px rgba(88, 166, 255, 0.3));
        }
        50% { 
            transform: scale(1.1);
            filter: drop-shadow(0 0 12px rgba(88, 166, 255, 0.5));
        }
    }

    /* Enhanced Input Fields */
    .stTextInput > div > div > input {
        background: rgba(13, 17, 23, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid transparent !important;
        background-image: 
            linear-gradient(rgba(13, 17, 23, 0.8), rgba(13, 17, 23, 0.8)),
            linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 16px 20px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        outline: none !important;
    }

    .stTextInput > div > div > input:focus {
        transform: translateY(-2px) !important;
        box-shadow: 
            0 8px 32px rgba(88, 166, 255, 0.3),
            0 0 20px rgba(88, 166, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        outline: none !important;
        border: 2px solid transparent !important;
        background-image: 
            linear-gradient(rgba(13, 17, 23, 0.9), rgba(13, 17, 23, 0.9)),
            linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    }

    /* Remove any default browser outlines */
    .stTextInput > div > div > input:active,
    .stTextInput > div > div > input:focus-visible {
        outline: none !important;
        border: 2px solid transparent !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
        font-style: italic !important;
        font-weight: 400 !important;
    }

    /* WOW Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--bg-card), var(--bg-hover)) !important;
        border: 2px solid transparent !important;
        background-image: 
            linear-gradient(135deg, var(--bg-card), var(--bg-hover)),
            linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 14px 20px !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent
        );
        transition: all 0.5s ease;
    }

    .stButton button:hover::before {
        left: 100%;
    }

    .stButton button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 8px 32px rgba(88, 166, 255, 0.3),
            0 0 20px rgba(88, 166, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        background-image: 
            linear-gradient(135deg, var(--bg-hover), var(--bg-card)),
            linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    }

    .stButton button:active {
        transform: translateY(-1px) scale(0.98) !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.4),
            inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    /* Special Primary Buttons */
    .stButton button[kind="primary"] {
        background: var(--gradient-primary) !important;
        border: 2px solid var(--accent-blue) !important;
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        box-shadow: 
            0 6px 24px rgba(88, 166, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    .stButton button[kind="primary"]:hover {
        box-shadow: 
            0 10px 40px rgba(88, 166, 255, 0.5),
            0 0 30px rgba(88, 166, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        background: var(--gradient-secondary) !important;
    }

    /* Enhanced Metrics */
    .stMetric {
        background: rgba(28, 33, 40, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(88, 166, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 16px 12px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .stMetric:hover::before {
        transform: scaleX(1);
    }

    .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(88, 166, 255, 0.1) !important;
        border-color: rgba(88, 166, 255, 0.4) !important;
    }

    /* Conversation List Enhancements */
    .stContainer {
        position: relative !important;
    }

    /* Status Messages */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid transparent !important;
        animation: slide-in 0.3s ease-out !important;
    }

    @keyframes slide-in {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Caption styling */
    .stCaption {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        opacity: 0.8 !important;
        transition: opacity 0.3s ease !important;
    }

    .sidebar-section:hover .stCaption {
        opacity: 1 !important;
    }

    /* Scrollbar styling for sidebar */
    [data-testid="stSidebarUserContent"]::-webkit-scrollbar {
        width: 6px;
    }

    [data-testid="stSidebarUserContent"]::-webkit-scrollbar-track {
        background: rgba(48, 54, 61, 0.3);
        border-radius: 3px;
    }

    [data-testid="stSidebarUserContent"]::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        border-radius: 3px;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebarUserContent"]::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-purple), var(--accent-green));
    }

    /* Main Content Area - ChatGPT Style */
    .main .block-container {
        max-width: 1100px !important;
        margin: 0 auto !important;
        padding: 40px 20px 140px !important;
        background: var(--bg-primary);
    }

    /* Chat Messages - Enhanced ChatGPT Style Design */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        margin: 0 0 32px 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
        opacity: 1 !important;
        animation: messageSlideIn 0.6s ease-out !important;
    }

    @keyframes messageSlideIn {
        0% {
            opacity: 0;
            transform: translateY(20px) scale(0.95);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    [data-testid="stChatMessageContent"] {
        background: var(--bg-card) !important;
        border: 2px solid var(--border-primary) !important;
        border-radius: 20px !important;
        padding: 20px 24px !important;
        margin: 0 !important;
        box-shadow: 
            6px 6px 12px var(--shadow-dark),
            -6px -6px 12px var(--shadow-light) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        position: relative !important;
        overflow: hidden !important;
    }

    /* Human Messages - Right aligned with blue theme */
    [data-testid="stChatMessage"] [data-testid="user"] ~ [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #2c5aa0, #1e3d72) !important;
        border: 2px solid var(--accent-blue) !important;
        border-radius: 22px 22px 6px 22px !important;
        margin-left: 100px !important;
        margin-right: 20px !important;
        color: #ffffff !important;
        position: relative !important;
        box-shadow: 
            8px 8px 20px rgba(0, 0, 0, 0.4),
            0 0 25px rgba(88, 166, 255, 0.2) !important;
    }

    /* AI Messages - Left aligned with green theme */
    [data-testid="stChatMessage"]:not(:has([data-testid="user"])) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #1e3a2e, #0f2419) !important;
        border: 2px solid var(--accent-green) !important;
        border-radius: 22px 22px 22px 6px !important;
        margin-left: 20px !important;
        margin-right: 100px !important;
        color: #ffffff !important;
        position: relative !important;
        box-shadow: 
            8px 8px 20px rgba(0, 0, 0, 0.4),
            0 0 25px rgba(63, 185, 80, 0.2) !important;
    }

    /* Enhanced Message Hover Effects */
    [data-testid="stChatMessageContent"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 
            8px 8px 20px rgba(0, 0, 0, 0.5),
            0 0 25px rgba(88, 166, 255, 0.15) !important;
    }

    /* Message Text Styling */
    [data-testid="stChatMessageContent"] p {
        color: #ffffff !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        margin: 0 0 8px 0 !important;
        font-weight: 400 !important;
    }

    [data-testid="stChatMessageContent"] strong {
        color: #ffd700 !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }

    [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessageContent"] ol {
        margin: 12px 0 !important;
        padding-left: 20px !important;
    }

    [data-testid="stChatMessageContent"] li {
        color: #ffffff !important;
        margin: 6px 0 !important;
    }

    /* Avatar Styling */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        background: var(--gradient-primary) !important;
        border: 2px solid var(--border-primary) !important;
        box-shadow: 
            4px 4px 8px var(--shadow-dark),
            -4px -4px 8px var(--shadow-light) !important;
    }

    /* Enhanced Elaborate Button */
    button[key*="elaborate_"] {
        background: linear-gradient(135deg, #1e3a2e, #0f2419) !important;
        border: 2px solid var(--accent-green) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        margin-top: 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            4px 4px 10px rgba(0, 0, 0, 0.4),
            0 0 15px rgba(63, 185, 80, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    button[key*="elaborate_"]::before {
        content: '✨';
        margin-right: 6px;
        font-size: 12px;
        animation: sparkle 2s ease-in-out infinite;
    }

    @keyframes sparkle {
        0%, 100% { 
            opacity: 0.6;
            transform: scale(1) rotate(0deg);
        }
        50% { 
            opacity: 1;
            transform: scale(1.1) rotate(180deg);
        }
    }

    button[key*="elaborate_"]::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        transition: all 0.6s ease;
        pointer-events: none;
    }

    button[key*="elaborate_"]:hover::after {
        left: 100%;
    }

    button[key*="elaborate_"]:hover {
        background: linear-gradient(135deg, #2d5a47, #1e3a2e) !important;
        border-color: #ffffff !important;
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 
            6px 6px 15px rgba(0, 0, 0, 0.5),
            0 0 25px rgba(63, 185, 80, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.5) !important;
    }

    button[key*="elaborate_"]:active {
        transform: translateY(0) scale(1.02) !important;
        box-shadow: 
            2px 2px 6px rgba(0, 0, 0, 0.4),
            0 0 15px rgba(63, 185, 80, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }

    /* Enhanced Expander for Detailed Explanations */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1e3a2e, #0f2419) !important;
        border: 2px solid var(--accent-green) !important;
        border-radius: 16px 16px 0 0 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 16px 20px !important;
        margin-top: 20px !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 
            0 6px 15px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(63, 185, 80, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .streamlit-expanderHeader::before {
        content: '🔍';
        margin-right: 10px;
        font-size: 16px;
        animation: searchPulse 2s ease-in-out infinite;
    }

    @keyframes searchPulse {
        0%, 100% { 
            opacity: 0.7;
            transform: scale(1);
        }
        50% { 
            opacity: 1;
            transform: scale(1.1);
        }
    }

    .streamlit-expanderHeader::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        transition: all 0.8s ease;
        pointer-events: none;
    }

    .streamlit-expanderHeader:hover::after {
        left: 100%;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #2d5a47, #1e3a2e) !important;
        border-color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.5),
            0 0 30px rgba(63, 185, 80, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3) !important;
    }

    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(30, 58, 46, 0.95), rgba(15, 36, 25, 0.95)) !important;
        border: 2px solid var(--accent-green) !important;
        border-top: none !important;
        border-radius: 0 0 16px 16px !important;
        padding: 25px !important;
        color: #ffffff !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 
            0 10px 25px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(63, 185, 80, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .streamlit-expanderContent::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue), var(--accent-purple), var(--accent-green));
        background-size: 200% 100%;
        animation: expanderGlow 3s ease-in-out infinite;
    }

    @keyframes expanderGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Enhanced notification styles */
    .stSuccess {
        background: linear-gradient(135deg, rgba(63, 185, 80, 0.15), rgba(63, 185, 80, 0.05)) !important;
        border: 1px solid var(--accent-green) !important;
        border-left: 4px solid var(--accent-green) !important;
        border-radius: 12px !important;
        color: var(--accent-green) !important;
        font-weight: 600 !important;
        box-shadow: 
            0 4px 12px rgba(63, 185, 80, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        animation: notificationSlide 0.5s ease-out !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(255, 123, 114, 0.15), rgba(255, 123, 114, 0.05)) !important;
        border: 1px solid var(--accent-orange) !important;
        border-left: 4px solid var(--accent-orange) !important;
        border-radius: 12px !important;
        color: var(--accent-orange) !important;
        font-weight: 600 !important;
        box-shadow: 
            0 4px 12px rgba(255, 123, 114, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        animation: notificationSlide 0.5s ease-out !important;
    }

    .stInfo {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.15), rgba(88, 166, 255, 0.05)) !important;
        border: 1px solid var(--accent-blue) !important;
        border-left: 4px solid var(--accent-blue) !important;
        border-radius: 12px !important;
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        box-shadow: 
            0 4px 12px rgba(88, 166, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        animation: notificationSlide 0.5s ease-out !important;
    }

    @keyframes notificationSlide {
        0% {
            opacity: 0;
            transform: translateX(-20px) scale(0.95);
        }
        100% {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }

    /* Cool Background Effects */
    .main .block-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: 
            radial-gradient(circle at 25% 25%, rgba(88, 166, 255, 0.03) 0%, transparent 25%),
            radial-gradient(circle at 75% 75%, rgba(63, 185, 80, 0.03) 0%, transparent 25%),
            radial-gradient(circle at 75% 25%, rgba(165, 165, 255, 0.03) 0%, transparent 25%),
            radial-gradient(circle at 25% 75%, rgba(255, 123, 114, 0.03) 0%, transparent 25%);
        animation: backgroundFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes backgroundFloat {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
            opacity: 0.5;
        }
        33% {
            transform: translateY(-10px) rotate(1deg);
            opacity: 0.7;
        }
        66% {
            transform: translateY(5px) rotate(-1deg);
            opacity: 0.6;
        }
    }

    /* Scrollbar Enhancement */
    ::-webkit-scrollbar {
        width: 8px !important;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary) !important;
        border-radius: 4px !important;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple)) !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-green), var(--accent-blue)) !important;
        transform: scaleY(1.1) !important;
    }

    /* Message Entrance Animations with Staggered Delays */
    [data-testid="stChatMessage"]:nth-child(1) {
        animation-delay: 0.1s !important;
    }

    [data-testid="stChatMessage"]:nth-child(2) {
        animation-delay: 0.2s !important;
    }

    [data-testid="stChatMessage"]:nth-child(3) {
        animation-delay: 0.3s !important;
    }

    [data-testid="stChatMessage"]:nth-child(4) {
        animation-delay: 0.4s !important;
    }

    [data-testid="stChatMessage"]:nth-child(5) {
        animation-delay: 0.5s !important;
    }

    [data-testid="stChatMessage"]:nth-child(n+6) {
        animation-delay: 0.6s !important;
    }

    /* Floating Action Button for Scroll to Top */
    .scroll-to-top {
        position: fixed !important;
        bottom: 180px !important;
        right: 30px !important;
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        background: var(--gradient-primary) !important;
        border: none !important;
        color: white !important;
        font-size: 20px !important;
        cursor: pointer !important;
        box-shadow: 
            0 8px 20px rgba(88, 166, 255, 0.4),
            0 0 20px rgba(88, 166, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        z-index: 999 !important;
        opacity: 0 !important;
        transform: scale(0.8) !important;
        animation: fabBounce 2s ease-in-out infinite !important;
    }

    .scroll-to-top.visible {
        opacity: 1 !important;
        transform: scale(1) !important;
    }

    .scroll-to-top:hover {
        transform: scale(1.1) translateY(-3px) !important;
        box-shadow: 
            0 12px 30px rgba(88, 166, 255, 0.5),
            0 0 30px rgba(88, 166, 255, 0.3) !important;
        background: var(--gradient-secondary) !important;
    }

    @keyframes fabBounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-3px);
        }
    }

    /* Loading State Enhancements */
    .thinking-bubble {
        background: linear-gradient(135deg, #1e3a2e, #0f2419) !important;
        border: 2px solid var(--accent-green) !important;
        border-radius: 22px 22px 22px 6px !important;
        padding: 20px 24px !important;
        margin: 20px 20px 24px 20px !important;
        margin-right: 120px !important;
        color: #ffffff !important;
        box-shadow: 
            8px 8px 20px rgba(0, 0, 0, 0.4),
            0 0 25px rgba(63, 185, 80, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        position: relative;
        overflow: hidden;
        animation: thinkingPulse 2s ease-in-out infinite !important;
    }

    @keyframes thinkingPulse {
        0%, 100% {
            box-shadow: 
                8px 8px 20px rgba(0, 0, 0, 0.4),
                0 0 25px rgba(63, 185, 80, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        50% {
            box-shadow: 
                8px 8px 20px rgba(0, 0, 0, 0.4),
                0 0 35px rgba(63, 185, 80, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }
    }

    /* Message Count Badge */
    .message-count {
        position: fixed !important;
        bottom: 180px !important;
        left: 30px !important;
        background: var(--gradient-accent) !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        box-shadow: 
            0 6px 15px rgba(165, 165, 255, 0.4),
            0 0 20px rgba(165, 165, 255, 0.2) !important;
        z-index: 999 !important;
        opacity: 0.9 !important;
        transition: all 0.3s ease !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
    }

    .message-count:hover {
        opacity: 1 !important;
        transform: scale(1.05) !important;
        box-shadow: 
            0 8px 20px rgba(165, 165, 255, 0.5),
            0 0 25px rgba(165, 165, 255, 0.3) !important;
    }
        margin: 0 4px;
        border-radius: 50%;
        background: var(--accent-blue);
        animation: thinking-pulse 1.4s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.5);
    }

    .ai-thinking-orb:nth-child(2) {
        animation-delay: 0.2s;
        background: var(--accent-green);
        box-shadow: 0 0 8px rgba(63, 185, 80, 0.5);
    }

    .ai-thinking-orb:nth-child(3) {
        animation-delay: 0.4s;
        background: var(--accent-purple);
        box-shadow: 0 0 8px rgba(165, 165, 255, 0.5);
    }

    @keyframes thinking-pulse {
        0%, 80%, 100% { 
            transform: scale(1); 
            opacity: 0.7; 
        }
        40% { 
            transform: scale(1.3); 
            opacity: 1; 
        }
    }

    .ai-thinking span {
        margin-left: 12px;
        color: var(--text-secondary);
        font-style: italic;
        font-size: 14px;
    }

    /* Independent Chat Search Bar - ChatGPT Style */
    .chat-search-container {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 999999 !important;
        background: linear-gradient(180deg, rgba(13, 17, 23, 0) 0%, rgba(13, 17, 23, 0.8) 20%, var(--bg-primary) 50%) !important;
        backdrop-filter: blur(20px) !important;
        border-top: 1px solid var(--border-primary) !important;
        padding: 20px 20px 30px 20px !important;
        box-shadow: 
            0 -20px 40px rgba(0, 0, 0, 0.8),
            0 -8px 16px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }

    .chat-search-wrapper {
        max-width: 900px !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 24px !important;
        padding: 8px 8px 8px 20px !important;
        box-shadow: 
            8px 8px 16px var(--shadow-dark),
            -8px -8px 16px var(--shadow-light),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .chat-search-wrapper:hover {
        border-color: var(--border-focus) !important;
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            0 0 20px var(--shadow-glow),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }

    .chat-search-wrapper:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            0 0 30px rgba(88, 166, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }

    .chat-search-input {
        flex: 1 !important;
        background: transparent !important;
        border: none !important;
        outline: none !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        padding: 12px 0 !important;
        width: 100% !important;
        resize: none !important;
        overflow: hidden !important;
        min-height: 24px !important;
        max-height: 120px !important;
        line-height: 1.5 !important;
    }

    .chat-search-input::placeholder {
        color: var(--text-muted) !important;
        font-style: italic !important;
    }

    .chat-search-input:focus {
        outline: none !important;
    }

    .chat-send-button {
        width: 44px !important;
        height: 44px !important;
        border-radius: 16px !important;
        border: none !important;
        background: var(--gradient-primary) !important;
        color: white !important;
        font-size: 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            4px 4px 8px var(--shadow-dark),
            -4px -4px 8px var(--shadow-light),
            0 0 15px rgba(88, 166, 255, 0.2) !important;
    }

    .chat-send-button:hover {
        background: var(--gradient-secondary) !important;
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 
            6px 6px 12px var(--shadow-dark),
            -6px -6px 12px var(--shadow-light),
            0 0 25px rgba(88, 166, 255, 0.4) !important;
    }

    .chat-send-button:active {
        transform: translateY(0) scale(1) !important;
        box-shadow: 
            inset 2px 2px 4px var(--shadow-dark),
            inset -2px -2px 4px var(--shadow-light) !important;
    }

    .chat-send-button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        background: var(--bg-secondary) !important;
        color: var(--text-muted) !important;
        transform: none !important;
        box-shadow: 
            2px 2px 4px var(--shadow-dark),
            -2px -2px 4px var(--shadow-light) !important;
    }

    /* Professional Creator Signature - Top Right */
    .creator-signature {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 16px !important;
        padding: 12px 16px !important;
        box-shadow: 
            8px 8px 16px var(--shadow-dark),
            -8px -8px 16px var(--shadow-light),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        z-index: 999999 !important;
        transition: all 0.3s ease !important;
        opacity: 0.9 !important;
    }

    .creator-signature:hover {
        opacity: 1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            0 0 20px var(--shadow-glow),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        border-color: var(--border-focus) !important;
    }

    .creator-avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        background: var(--gradient-primary) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3) !important;
    }

    .creator-info {
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
    }

    .creator-name {
        color: var(--text-primary) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
    }

    .creator-title {
        color: var(--text-secondary) !important;
        font-size: 11px !important;
        font-weight: 400 !important;
        line-height: 1 !important;
    }
    
    /* Ensure chat content doesn't overlap with fixed search bar */
    .main-content {
        padding-bottom: 140px !important;
    }
    
    /* Override Streamlit's main content area */
    .main .block-container {
        padding-bottom: 140px !important;
    }

    /* Additional insurance for proper spacing */
    @media (max-width: 768px) {
        .main-content, .main .block-container {
            padding-bottom: 120px !important;
        }
        
        [data-testid="stChatInput"] {
            width: 95% !important;
            bottom: 15px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Professional Design
with st.sidebar:
    # Professional Header with Clickable Title
    st.markdown("""
    <div class="sidebar-header" style="cursor: pointer;" onclick="window.location.reload();">
        <div class="logo">SQ</div>
        <h2>SmartQuery Bot</h2>
        <p>AI-Powered Web Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add JavaScript to handle homepage navigation
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const header = document.querySelector('.sidebar-header');
        if (header) {
            header.addEventListener('click', function() {
                // Use Streamlit's rerun functionality
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    data: {key: 'homepage_clicked', value: true}
                }, '*');
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Check if homepage was clicked
    if st.session_state.get('homepage_clicked', False):
        st.session_state.show_homepage = True
        st.session_state.vector_store = None
        st.session_state.last_url = ""
        st.session_state.loaded_conversation = None
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your AI assistant. Enter a website URL in the sidebar to begin analyzing content, or ask me anything!")
        ]
        globals()['_GLOBAL_RAG_CHAIN'] = None
        st.session_state.homepage_clicked = False
        st.rerun()
    
    # Website URL Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">🌐</span>Website Analysis</div>', unsafe_allow_html=True)
    website_url = st.text_input(
        "Enter Website URL:",
        placeholder="https://example.com",
        label_visibility="collapsed",
        help="Enter any website URL to start analyzing its content"
    )
    if website_url:
        st.caption(f"📍 Analyzing: {website_url[:50]}{'...' if len(website_url) > 50 else ''}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat Management Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">🧹</span>Chat Management</div>', unsafe_allow_html=True)
    
    if "clear_chat_pressed" not in st.session_state:
        st.session_state.clear_chat_pressed = False

    col1, col2 = st.columns(2)
    with col1:
        clear_pressed = st.button("🗑️ Clear", use_container_width=True, help="Clear current conversation")
    with col2:
        new_chat_pressed = st.button("✨ New", use_container_width=True, help="Start a new conversation")
    
    if clear_pressed or new_chat_pressed or st.session_state.clear_chat_pressed:
        st.session_state.clear_chat_pressed = True
        
        st.warning("⚠️ This will clear your current conversation.")
        col_confirm, col_cancel = st.columns(2)
        
        with col_confirm:
            confirm = st.button("✅ Confirm", use_container_width=True)
        with col_cancel:
            cancel = st.button("❌ Cancel", use_container_width=True)
        
        if confirm:
            if "chat_history" in st.session_state and len(st.session_state.chat_history) > 1:  
                save_conversation("Auto-saved before clearing")
                st.success("💾 Previous conversation auto-saved!")
                
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm your AI assistant. Enter a website URL to begin analyzing content, or ask me anything!")
            ]
            st.session_state.vector_store = None
            st.session_state.last_url = ""
            st.session_state.loaded_conversation = None
            
            # Reset global RAG chain
            globals()['_GLOBAL_RAG_CHAIN'] = None
            
            st.session_state.clear_chat_pressed = False
            st.rerun()
        
        if cancel:
            st.session_state.clear_chat_pressed = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Conversation History Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">💬</span>Conversation History</div>', unsafe_allow_html=True)
    
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("� Save", use_container_width=True, help="Save current conversation"):
            if save_conversation():
                st.success("💾 Conversation saved!", icon="✅")
                load_conversation_list()
            else:
                st.warning("Nothing to save.", icon="⚠️")
    
    with col_load:
        refresh_pressed = st.button("🔄 Refresh", use_container_width=True, help="Refresh conversation list")
        if refresh_pressed:
            load_conversation_list()
            st.success("🔄 List refreshed!", icon="✅")
    
    if "saved_conversations" not in st.session_state:
        load_conversation_list()
        
    if not st.session_state.saved_conversations:
        st.info("📝 No saved conversations yet.", icon="ℹ️")
    else:
        st.markdown("#### 📚 Saved Conversations")
        
        # Display conversations in a more compact way
        for i, convo in enumerate(st.session_state.saved_conversations[:5]):  # Show only last 5
            with st.container():
                col_load, col_delete = st.columns([0.8, 0.2])
                
                with col_load:
                    if st.button(f"📄 {convo['title'][:25]}{'...' if len(convo['title']) > 25 else ''}", 
                               key=f"load_{i}", use_container_width=True,
                               help=f"Load conversation: {convo['title']}"):
                        messages = load_conversation(convo["path"])
                        if messages:
                            st.session_state.chat_history = messages
                            st.session_state.loaded_conversation = convo["title"]
                            st.success(f"✅ Loaded: {convo['title'][:20]}{'...' if len(convo['title']) > 20 else ''}")
                            st.rerun()
                
                with col_delete:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete this conversation",
                               use_container_width=True):
                        if delete_conversation(convo["path"]):
                            st.success("🗑️ Deleted!")
                            st.rerun()
                
                # Show metadata
                st.caption(f"💬 {convo['message_count']} msgs • 📅 {convo['timestamp']}")
                if convo["website_url"]:
                    st.caption(f"🔗 {convo['website_url'][:40]}{'...' if len(convo['website_url']) > 40 else ''}")
                
                if i < len(st.session_state.saved_conversations) - 1:
                    st.markdown("---")
        
        if len(st.session_state.saved_conversations) > 5:
            st.caption(f"📁 Showing 5 of {len(st.session_state.saved_conversations)} conversations")

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick Stats Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">📊</span>Quick Stats</div>', unsafe_allow_html=True)
    
    total_conversations = len(st.session_state.get("saved_conversations", []))
    current_messages = len(st.session_state.get("chat_history", [])) - 1  # Exclude welcome message
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conversations", total_conversations, delta=None)
    with col2:
        st.metric("Messages", current_messages, delta=None)
    
    if st.session_state.get("vector_store"):
        st.success("🌐 Website loaded", icon="✅")
    else:
        st.info("🌐 No website loaded", icon="ℹ️")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
# Main content area
if website_url and website_url != st.session_state.get("last_url", ""):
    st.session_state.last_url = website_url
    st.session_state.thinking = True
    st.session_state.show_homepage = False
    with st.spinner("Analyzing website content..."):
        try:
            docs = fetch_website_content(website_url)
            if docs and len(docs) > 0:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)
                st.session_state.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=EMBEDDING_MODEL
                )
                _GLOBAL_RAG_CHAIN = None
                # Remove the automatic analysis message
                st.success("✅ Website analyzed successfully! Ask me anything about the content.")
            else:
                st.error("Could not extract content from the website.")
        except Exception as e:
            st.error(f"Error processing website: {str(e)}")
    st.session_state.thinking = False
    st.rerun()

# Display homepage, welcome screen, or chat interface
if st.session_state.show_homepage or not st.session_state.get("vector_store"):
    # Enhanced Homepage Design
    homepage_html = """
    <div class="hero-section">
        <div class="hero-logo">SQ</div>
        <p class="hero-title">SmartQuery Bot</p>
        <p class="hero-subtitle">Enter a website URL in the sidebar to begin</p>
        <p class="hero-subtitle">Transform any website into an intelligent conversation</p>
    </div>
    

    
    <div class="stats-container">
        <div class="stat-item">
            <span class="stat-number">∞</span>
            <span class="stat-label">Websites</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">AI</span>
            <span class="stat-label">Powered</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">24/7</span>
            <span class="stat-label">Available</span>
        </div>
    </div>
    """
    
    # Render the homepage HTML
    try:
        st.markdown(homepage_html, unsafe_allow_html=True)
    except Exception as e:
        # Fallback to html component if markdown fails
        html(homepage_html, height=800)
else:
    # Main content wrapper to handle fixed search bar spacing
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Chat interface - CHATGPT STYLE CONTAINER
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    if st.session_state.loaded_conversation:
        st.info(f"Loaded conversation: {st.session_state.loaded_conversation}")

    # Display messages without creating empty containers
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            display_ai_message(message.content, i)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    if "thinking" in st.session_state and st.session_state.thinking:
        with st.chat_message("AI"):
            st.markdown("""
            <div class="generating-response">
                <div class="response-dots">
                    <div class="response-dot"></div>
                    <div class="response-dot"></div>
                    <div class="response-dot"></div>
                </div>
                <span class="response-text">🔍 Analyzing website content... Almost ready! ✨</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Close the chat container
    st.markdown("</div>", unsafe_allow_html=True)

    # Close main content wrapper
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add floating UI enhancements
    message_count = len(st.session_state.chat_history)
    st.markdown(f"""
    <div class="message-count">
        💬 {message_count} messages
    </div>
    
    <button class="scroll-to-top" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}});" title="Scroll to top">
        ⬆️
    </button>
    """, unsafe_allow_html=True)
    
    # Add JavaScript functionality separately to avoid f-string conflicts
    st.markdown("""
    <script>
    // Show/hide scroll to top button based on scroll position
    window.addEventListener('scroll', function() {
        const scrollBtn = document.querySelector('.scroll-to-top');
        if (scrollBtn) {
            if (window.pageYOffset > 300) {
                scrollBtn.classList.add('visible');
            } else {
                scrollBtn.classList.remove('visible');
            }
        }
    });
    
    // Auto-scroll to bottom when new messages are added
    function scrollToBottom() {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    // Add entrance animations to new messages and auto-scroll
    const observer = new MutationObserver(function(mutations) {
        let hasNewMessages = false;
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1 && node.hasAttribute('data-testid') && 
                    node.getAttribute('data-testid') === 'stChatMessage') {
                    node.style.opacity = '0';
                    node.style.transform = 'translateY(20px) scale(0.95)';
                    setTimeout(() => {
                        node.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                        node.style.opacity = '1';
                        node.style.transform = 'translateY(0) scale(1)';
                    }, 100);
                    hasNewMessages = true;
                }
            });
        });
        
        // Auto-scroll to bottom when new messages are added
        if (hasNewMessages) {
            setTimeout(scrollToBottom, 200);
        }
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Simple chat input using Streamlit's native component with enhanced styling
    user_query = st.chat_input("✨ Ask anything about this website content... Let's explore together! 🚀", key="chat_input")
    if user_query is not None and user_query != "":
        # Add human message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        # Set flag to show we're generating a response
        st.session_state.generating_response = True
        st.rerun()
    
    # Check if we need to generate a response
    if st.session_state.get('generating_response', False):
        # Get the last user message
        last_user_message = None
        for message in reversed(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                last_user_message = message.content
                break
        
        if last_user_message:
            # Show generating response animation
            response_placeholder = st.empty()
            with response_placeholder.container():
                with st.chat_message("AI"):
                    st.markdown("""
                    <div class="generating-response">
                        <div class="response-dots">
                            <div class="response-dot"></div>
                            <div class="response-dot"></div>
                            <div class="response-dot"></div>
                        </div>
                        <span class="response-text">🧠 Generating intelligent response... Hold tight! 🎯</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            response = get_response(last_user_message)
            response_placeholder.empty()
            st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.generating_response = False
            st.rerun()
    
# Professional Creator Signature
st.markdown("""
<div class="creator-signature">
    <div class="creator-avatar">V</div>
    <div class="creator-info">
        <div class="creator-name">Vaibhav Nagre</div>
        <div class="creator-title">AI Developer</div>
    </div>
</div>
""", unsafe_allow_html=True)