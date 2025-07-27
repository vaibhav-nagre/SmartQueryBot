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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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

# Initialize models with error handling
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    LLM_MODEL = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    EMBEDDING_MODEL = OpenAIEmbeddings()
    AI_AVAILABLE = True
except Exception as e:
    st.warning("⚠️ OpenAI API key not configured. Running in demo mode.")
    LLM_MODEL = None
    EMBEDDING_MODEL = None
    AI_AVAILABLE = False

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
    """Display AI message with enhanced neuomorphic styling and minimal animations"""
    
    # Custom AI message container with neuomorphic design
    st.markdown(f"""
    <div class="chat-message ai-message" data-message-id="{message_idx}">
        <div class="message-avatar ai-avatar">
            <div class="avatar-inner">🤖</div>
        </div>
        <div class="message-content ai-content">
            <div class="message-text">{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Elaborate button - only for non-initial messages
    if (not "Hello! I'm your AI assistant" in content and 
        not "I've analyzed the content" in content and
        not "Hello! I am Vaibhav's AI assistant" in content and
        message_idx > 0):
        
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            if st.button("📖 Get Detailed Explanation", 
                        key=f"elaborate_{message_idx}", 
                        help="Get a comprehensive explanation",
                        use_container_width=True):
                # Show elaborate animation
                elaborate_placeholder = st.empty()
                with elaborate_placeholder.container():
                    st.markdown("""
                    <div class="elaborate-animation">
                        <div class="elaborate-spinner">
                            <div class="spinner-dot"></div>
                            <div class="spinner-dot"></div>
                            <div class="spinner-dot"></div>
                            <div class="spinner-dot"></div>
                        </div>
                        <div class="elaborate-text">🧠 Generating comprehensive explanation...</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Get context and elaborate
                if "response_contexts" in st.session_state and message_idx in st.session_state.response_contexts:
                    context = st.session_state.response_contexts[message_idx]
                    
                    user_question = ""
                    if message_idx > 0 and len(st.session_state.chat_history) >= message_idx:
                        if isinstance(st.session_state.chat_history[message_idx-1], HumanMessage):
                            user_question = st.session_state.chat_history[message_idx-1].content
                    
                    detailed_response = get_elaborate_response(user_question, context, message_idx)
                    elaborate_placeholder.empty()
                    
                    if "elaborated_responses" not in st.session_state:
                        st.session_state.elaborated_responses = {}
                    st.session_state.elaborated_responses[message_idx] = detailed_response
                else:
                    elaborate_placeholder.empty()
                    st.warning("⚠️ No additional context available for elaboration.")
    
    # Show elaborated response if available
    if "elaborated_responses" in st.session_state and message_idx in st.session_state.elaborated_responses:
        with st.expander("📚 **Detailed Explanation**", expanded=False):
            st.markdown(f"""
            <div class="detailed-explanation">
                {st.session_state.elaborated_responses[message_idx]}
            </div>
            """, unsafe_allow_html=True)

def display_human_message(content, message_idx):
    """Display human message with enhanced neuomorphic styling"""
    
    st.markdown(f"""
    <div class="chat-message human-message" data-message-id="{message_idx}">
        <div class="message-content human-content">
            <div class="message-text">{content}</div>
        </div>
        <div class="message-avatar human-avatar">
            <div class="avatar-inner">👤</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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

    /* =========================== */
    /* ANIMATION STATES - THINKING, GENERATING, ELABORATING */
    /* =========================== */
    
    /* Thinking Animation */
    .thinking-animation {
        display: flex;
        align-items: flex-start;
        margin: 16px 20px;
        gap: 12px;
        animation: messageSlideIn 0.4s ease-out;
    }
    
    .thinking-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--accent-orange), var(--accent-yellow));
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.3),
            -2px -2px 6px rgba(48, 54, 61, 0.3);
        animation: thinkingPulse 2s ease-in-out infinite;
    }
    
    @keyframes thinkingPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 
                4px 4px 8px rgba(0, 0, 0, 0.3),
                -2px -2px 6px rgba(48, 54, 61, 0.3);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 
                6px 6px 12px rgba(0, 0, 0, 0.4),
                -3px -3px 8px rgba(48, 54, 61, 0.4),
                0 0 15px rgba(255, 123, 114, 0.3);
        }
    }
    
    .thinking-content {
        background: rgba(255, 123, 114, 0.1);
        border: 1px solid rgba(255, 123, 114, 0.3);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        margin-right: 60px;
        backdrop-filter: blur(10px);
    }
    
    .thinking-dots {
        display: flex;
        gap: 6px;
        margin-bottom: 8px;
    }
    
    .thinking-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-orange);
        animation: thinkingDotPulse 1.5s ease-in-out infinite;
    }
    
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    .thinking-dot:nth-child(4) { animation-delay: 0.6s; }
    
    @keyframes thinkingDotPulse {
        0%, 80%, 100% { 
            transform: scale(1) translateY(0); 
        }
        40% { 
            transform: scale(1.3) translateY(-10px); 
            opacity: 1;
        }
    }
    
    .thinking-text {
        color: var(--text-primary);
        font-size: 14px;
        font-weight: 500;
        font-style: italic;
    }
    
    /* Response Generation Animation */
    .response-generation {
        display: flex;
        align-items: flex-start;
        margin: 16px 20px;
        gap: 12px;
        animation: messageSlideIn 0.4s ease-out;
    }
    
    .response-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.3),
            -2px -2px 6px rgba(48, 54, 61, 0.3);
        animation: generatingPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes generatingPulse {
        0%, 100% { 
            transform: scale(1) rotate(0deg);
            box-shadow: 
                4px 4px 8px rgba(0, 0, 0, 0.3),
                -2px -2px 6px rgba(48, 54, 61, 0.3);
        }
        50% { 
            transform: scale(1.1) rotate(180deg);
            box-shadow: 
                6px 6px 12px rgba(0, 0, 0, 0.4),
                -3px -3px 8px rgba(48, 54, 61, 0.4),
                0 0 15px rgba(88, 166, 255, 0.4);
        }
    }
    
    .response-content-generation {
        background: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        margin-right: 60px;
        backdrop-filter: blur(10px);
    }
    
    .generation-spinner {
        display: flex;
        gap: 4px;
        margin-bottom: 8px;
        align-items: center;
        justify-content: flex-start;
    }
    
    .spinner-ring {
        width: 12px;
        height: 12px;
        border: 2px solid transparent;
        border-top: 2px solid var(--accent-blue);
        border-radius: 50%;
        animation: spinnerRotate 1s linear infinite;
    }
    
    .spinner-ring:nth-child(2) { 
        animation-delay: 0.1s; 
        border-top-color: var(--accent-purple);
    }
    
    .spinner-ring:nth-child(3) { 
        animation-delay: 0.2s; 
        border-top-color: var(--accent-green);
    }
    
    @keyframes spinnerRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .generation-text {
        color: var(--text-primary);
        font-size: 14px;
        font-weight: 500;
        font-style: italic;
    }
    
    /* Elaborate Animation */
    .elaborate-animation {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(165, 165, 255, 0.1);
        border: 1px solid rgba(165, 165, 255, 0.3);
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .elaborate-spinner {
        display: flex;
        gap: 6px;
        margin-right: 16px;
    }
    
    .spinner-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--accent-purple);
        animation: elaborateDotBounce 1.4s ease-in-out infinite;
    }
    
    .spinner-dot:nth-child(2) { 
        animation-delay: 0.2s; 
        background: var(--accent-blue);
    }
    
    .spinner-dot:nth-child(3) { 
        animation-delay: 0.4s; 
        background: var(--accent-green);
    }
    
    .spinner-dot:nth-child(4) { 
        animation-delay: 0.6s; 
        background: var(--accent-orange);
    }
    
    @keyframes elaborateDotBounce {
        0%, 80%, 100% { 
            transform: scale(1) translateY(0); 
        }
        40% { 
            transform: scale(1.2) translateY(-10px); 
        }
    }
    
    .elaborate-text {
        color: var(--text-primary);
        font-size: 14px;
        font-weight: 500;
        font-style: italic;
    }
    
    /* Detailed Explanation Styling */
    .detailed-explanation {
        background: rgba(28, 33, 40, 0.6);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 
            inset 2px 2px 4px rgba(0, 0, 0, 0.2),
            inset -2px -2px 4px rgba(48, 54, 61, 0.1);
    }
    
    .detailed-explanation h3 {
        color: var(--accent-blue);
        margin: 16px 0 8px 0;
        font-weight: 600;
    }
    
    .detailed-explanation p {
        color: var(--text-primary);
        margin: 8px 0;
        line-height: 1.6;
    }
    
    .detailed-explanation ul, .detailed-explanation ol {
        color: var(--text-primary);
        margin: 8px 0 8px 20px;
    }
    
    .detailed-explanation li {
        margin: 4px 0;
        line-height: 1.5;
    }
    
    /* Initialize custom scrollbar for sidebar */
    [data-testid="stSidebarUserContent"]::-webkit-scrollbar {
        width: 8px;
        background: rgba(13, 17, 23, 0.8);
    }

    [data-testid="stSidebarUserContent"]::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        border-radius: 4px;
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

    .streamlit-expanderHeader:hover {
        transform: translateY(-1px) !important;
        border-color: rgba(88, 166, 255, 0.15) !important;
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        background: rgba(24, 28, 34, 0.8) !important;
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

    /* MODERN NEUOMORPHIC CHAT INTERFACE - FIXED POSITIONING */
    
    /* Modern Chat Container - Start from top */
    .modern-chat-container {
        background: var(--bg-primary);
        border-radius: 0;
        padding: 20px 0 0 0 !important; /* Start from top */
        max-width: 1000px;
        margin: 0 auto;
        min-height: calc(100vh - 200px);
        position: relative;
    }
    
    /* Main Content Area Fixes */
    .main .block-container {
        padding-top: 1rem !important; /* Reduce top padding */
        padding-bottom: 120px !important; /* Account for fixed chat input */
        max-width: 1100px !important;
        margin: 0 auto !important;
    }
    
    /* Loaded Conversation Indicator */
    .loaded-conversation-indicator {
        background: rgba(58, 166, 255, 0.1);
        border: 1px solid rgba(58, 166, 255, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 0 20px 20px 20px;
        color: var(--accent-blue);
        font-size: 14px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* Chat Message Base */
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin: 16px 20px;
        gap: 12px;
        animation: messageSlideIn 0.4s ease-out;
        opacity: 0;
        animation-fill-mode: forwards;
    }
    
    @keyframes messageSlideIn {
        0% {
            opacity: 0;
            transform: translateY(10px) scale(0.98);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    /* Message Avatars */
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.3),
            -2px -2px 6px rgba(48, 54, 61, 0.3);
        transition: all 0.3s ease;
    }
    
    .ai-avatar {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    }
    
    .human-avatar {
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
        order: 2;
    }
    
    .avatar-inner {
        font-size: 18px;
        color: white;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Message Content */
    .message-content {
        flex: 1;
        position: relative;
    }
    
    .ai-content {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        margin-right: 60px;
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.3),
            -3px -3px 8px rgba(48, 54, 61, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .human-content {
        background: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: 18px 18px 4px 18px;
        padding: 16px 20px;
        margin-left: 60px;
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.3),
            -3px -3px 8px rgba(48, 54, 61, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        order: 1;
        transition: all 0.3s ease;
    }
    
    .message-text {
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 15px;
        font-weight: 400;
        word-wrap: break-word;
    }
    
    /* Fixed Message Count - Top Right */
    .message-count {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 20px !important;
        padding: 8px 16px !important;
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        z-index: 999 !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.3),
            -2px -2px 4px rgba(48, 54, 61, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .message-count:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.4),
            -3px -3px 6px rgba(48, 54, 61, 0.3),
            0 0 15px rgba(88, 166, 255, 0.2) !important;
        border-color: var(--border-focus) !important;
    }
    
    /* Elegant Creator Signature - Bottom Right */
    .creator-signature {
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        background: linear-gradient(145deg, var(--bg-card), var(--bg-tertiary)) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 50px !important;
        padding: 12px 20px 12px 12px !important;
        color: var(--text-primary) !important;
        z-index: 999 !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.4),
            -4px -4px 8px rgba(48, 54, 61, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        max-width: 220px !important;
        cursor: pointer !important;
        opacity: 0.9 !important;
    }
    
    .creator-signature:hover {
        transform: translateY(-3px) scale(1.02) !important;
        opacity: 1 !important;
        box-shadow: 
            12px 12px 24px rgba(0, 0, 0, 0.5),
            -6px -6px 12px rgba(48, 54, 61, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 20px rgba(88, 166, 255, 0.3) !important;
        border-color: var(--accent-blue) !important;
    }
    
    .creator-avatar {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple)) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        flex-shrink: 0 !important;
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.4),
            -2px -2px 4px rgba(88, 166, 255, 0.1) !important;
        position: relative !important;
        animation: avatarGlow 3s ease-in-out infinite !important;
    }
    
    @keyframes avatarGlow {
        0%, 100% { 
            box-shadow: 
                4px 4px 8px rgba(0, 0, 0, 0.4),
                -2px -2px 4px rgba(88, 166, 255, 0.1);
        }
        50% { 
            box-shadow: 
                4px 4px 8px rgba(0, 0, 0, 0.4),
                -2px -2px 4px rgba(88, 166, 255, 0.1),
                0 0 15px rgba(88, 166, 255, 0.4);
        }
    
    .creator-info {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
        min-width: 0 !important;
    }
    
    .creator-name {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        letter-spacing: -0.01em !important;
    }
    
    .creator-title {
        font-size: 12px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    
    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .creator-signature {
            right: 16px !important;
            bottom: 90px !important;
            padding: 8px 16px 8px 8px !important;
            max-width: 180px !important;
        }
        
        .creator-avatar {
            width: 32px !important;
            height: 32px !important;
            font-size: 14px !important;
        }
        
        .creator-name {
            font-size: 12px !important;
        }
    
        
        .message-count {
            top: 10px !important;
            right: 10px !important;
            padding: 6px 12px !important;
            font-size: 12px !important;
        }
    
    /* Hide default Streamlit chat elements */
    [data-testid="stChatMessage"] {
        display: none !important;
    }
    
    /* =========================== */
    /* REDESIGNED SIDEBAR STYLES */
    /* =========================== */
    
    /* Modern Sidebar Header */
    .modern-sidebar-header {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-tertiary));
        border-radius: 20px;
        padding: 24px 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border-primary);
        box-shadow: 
            8px 8px 16px var(--shadow-dark),
            -8px -8px 16px var(--shadow-light),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .modern-sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-green));
        background-size: 200% 100%;
        animation: headerGlow 4s ease-in-out infinite;
    }
    
    @keyframes headerGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
    }
    
    .brand-logo {
        position: relative;
        width: 48px;
        height: 48px;
    }
    
    .logo-inner {
        width: 100%;
        height: 100%;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.5),
            -2px -2px 6px rgba(88, 166, 255, 0.1);
        position: relative;
        animation: logoPulse 3s ease-in-out infinite;
    }
    
    @keyframes logoPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .logo-icon {
        font-size: 20px;
        z-index: 2;
    }
    
    .logo-pulse {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple));
        opacity: 0.3;
        animation: pulseRing 2s ease-out infinite;
    }
    
    @keyframes pulseRing {
        0% {
            transform: scale(1);
            opacity: 0.3;
        }
        100% {
            transform: scale(1.3);
            opacity: 0;
        }
    }
    
    .brand-content {
        flex: 1;
    }
    
    .brand-title {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .brand-subtitle {
        font-size: 12px !important;
        color: var(--text-secondary) !important;
        margin: 4px 0 8px 0 !important;
        font-weight: 500 !important;
    }
    
    .status-bar {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: statusPulse 2s ease-in-out infinite;
    }
    
    .status-dot.online {
        background: var(--accent-green);
        box-shadow: 0 0 8px rgba(63, 185, 80, 0.5);
    }
    
    @keyframes statusPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }
    
    .status-text {
        font-size: 11px;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    .home-button {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 12px;
        color: var(--accent-blue);
        font-weight: 500;
    }
    
    .home-button:hover {
        background: rgba(88, 166, 255, 0.2);
        border-color: var(--accent-blue);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
    }
    
    .home-icon {
        font-size: 14px;
    }
    
    /* Elegant Dividers */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent var(--border-primary), transparent);
        margin: 20px 0;
        position: relative;
    }
    
    .sidebar-divider::before {
        content: '';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 4px;
        height: 4px;
        background: var(--accent-blue);
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.5);
    }
    
    /* Section Headers */
    .analysis-section,
    .controls-section,
    .history-section,
    .stats-section {
        margin-bottom: 20px;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding: 12px 16px;
        background: linear-gradient(145deg, rgba(88, 166, 255, 0.05), rgba(165, 165, 255, 0.05));
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .section-icon {
        font-size: 18px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple));
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3);
    }
    
    .section-info h3 {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
    }
    
    .section-info p {
        font-size: 11px !important;
        color: var(--text-secondary) !important;
        margin: 2px 0 0 0 !important;
    }
    
    /* URL Preview */
    .url-preview {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: linear-gradient(145deg, rgba(63, 185, 80, 0.05), rgba(88, 166, 255, 0.05));
        border: 1px solid rgba(63, 185, 80, 0.3);
        border-radius: 12px;
        margin-top: 12px;
        backdrop-filter: blur(10px);
    }
    
    .preview-icon {
        font-size: 14px;
        color: var(--accent-green);
    }
    
    .preview-text {
        flex: 1;
    }
    
    .preview-label {
        font-size: 10px;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .preview-url {
        font-size: 11px;
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 2px;
        word-break: break-all;
    }
    
    /* Clear Confirmation */
    .clear-confirmation {
        padding: 16px;
        background: linear-gradient(145deg, rgba(255, 123, 114, 0.05), rgba(255, 193, 120, 0.05));
        border: 1px solid rgba(255, 123, 114, 0.3);
        border-radius: 12px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }
    
    .confirmation-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    
    .warning-icon {
        font-size: 16px;
        color: var(--accent-orange);
    }
    
    .confirmation-header h4 {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
    }
    
    .clear-confirmation p {
        font-size: 11px !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }
    
    /* Conversation Cards */
    .conversation-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-tertiary));
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .conversation-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(88, 166, 255, 0.2);
    }
    
    .conversation-header {
        margin-bottom: 8px;
    }
    
    .conversation-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
        line-height: 1.3;
    }
    
    .conversation-meta {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .conversation-meta span {
        font-size: 10px;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    .conversation-url {
        font-size: 10px;
        color: var(--text-secondary);
        background: rgba(88, 166, 255, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
        margin-top: 8px;
        word-break: break-all;
    }
    
    .conversation-separator {
        height: 1px;
        background: var(--border-secondary);
        margin: 12px 0;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 24px 16px;
        background: linear-gradient(145deg, rgba(88, 166, 255, 0.03), rgba(165, 165, 255, 0.03));
        border: 1px dashed var(--border-primary);
        border-radius: 12px;
        margin: 16px 0;
    }
    
    .empty-icon {
        font-size: 32px;
        margin-bottom: 12px;
        opacity: 0.6;
    }
    
    .empty-text h4 {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 0 4px 0 !important;
    }
    
    .empty-text p {
        font-size: 11px !important;
        color: var(--text-muted) !important;
        margin: 0 !important;
    }
    
    /* More Conversations */
    .more-conversations {
        text-align: center;
        font-size: 10px;
        color: var(--text-muted);
        padding: 8px;
        background: rgba(88, 166, 255, 0.05);
        border-radius: 8px;
        margin-top: 12px;
    }
    
    /* Stats Container */
    .stats-container {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        margin-top: 16px;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: linear-gradient(145deg, var(--bg-card), var(--bg-tertiary));
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        border-color: var(--accent-blue);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
    }
    
    .stat-icon {
        font-size: 16px;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple));
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(88, 166, 255, 0.3);
    }
    
    .stat-content {
        flex: 1;
    }
    
    .stat-number {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .stat-label {
        font-size: 10px;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 2px;
    }
    
    /* =========================== */
    /* HOMEPAGE STYLES */
    /* =========================== */
    
    .homepage-container {
        min-height: 80vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        background: 
            radial-gradient(circle at 20% 20%, rgba(88, 166, 255, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 80% 80%, rgba(165, 165, 255, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(63, 185, 80, 0.03) 0%, transparent 50%);
        position: relative;
        overflow: hidden;
    }
    
    .homepage-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            linear-gradient(45deg, transparent 49%, rgba(88, 166, 255, 0.03) 50%, transparent 51%),
            linear-gradient(-45deg, transparent 49%, rgba(165, 165, 255, 0.03) 50%, transparent 51%);
        background-size: 60px 60px;
        animation: backgroundMove 20s linear infinite;
        opacity: 0.3;
    }
    
    @keyframes backgroundMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(60px, 60px); }
    }
    
    .hero-section {
        text-align: center;
        max-width: 800px;
        z-index: 2;
        position: relative;
    }
    
    .hero-logo {
        width: 120px;
        height: 120px;
        margin: 0 auto 32px;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            0 0 40px rgba(88, 166, 255, 0.3);
        animation: heroLogoPulse 4s ease-in-out infinite;
        position: relative;
    }
    
    @keyframes heroLogoPulse {
        0%, 100% { 
            transform: scale(1) rotate(0deg);
            box-shadow: 
                12px 12px 24px var(--shadow-dark),
                -12px -12px 24px var(--shadow-light),
                0 0 40px rgba(88, 166, 255, 0.3);
        }
        50% { 
            transform: scale(1.05) rotate(5deg);
            box-shadow: 
                16px 16px 32px var(--shadow-dark),
                -16px -16px 32px var(--shadow-light),
                0 0 60px rgba(88, 166, 255, 0.5);
        }
    }
    
    .hero-title {
        font-size: 48px !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        margin: 0 0 16px 0 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleGradient 3s ease-in-out infinite;
    }
    
    @keyframes titleGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-size: 20px !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        margin: 0 0 32px 0 !important;
        line-height: 1.5 !important;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 24px;
        margin: 48px 0;
        max-width: 1000px;
        width: 100%;
    }
    
    .feature-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-tertiary));
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 
            8px 8px 16px var(--shadow-dark),
            -8px -8px 16px var(--shadow-light);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-green));
        background-size: 200% 100%;
        animation: cardGlow 3s ease-in-out infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    @keyframes cardGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: var(--accent-blue);
        box-shadow: 
            12px 12px 24px var(--shadow-dark),
            -12px -12px 24px var(--shadow-light),
            0 0 30px rgba(88, 166, 255, 0.3);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 20px;
        display: block;
        animation: featureIconFloat 3s ease-in-out infinite;
    }
    
    @keyframes featureIconFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .feature-title {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 0 12px 0 !important;
    }
    
    .feature-description {
        font-size: 14px !important;
        color: var(--text-secondary) !important;
        line-height: 1.6 !important;
        margin: 0 !important;
    }
    
    .cta-section {
        margin: 48px 0 24px 0;
        text-align: center;
    }
    
    .cta-text {
        font-size: 18px !important;
        color: var(--text-secondary) !important;
        margin: 0 0 24px 0 !important;
        font-weight: 500 !important;
    }
    
    .cta-button {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 16px 32px;
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        border-radius: 16px;
        font-size: 16px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.3),
            -2px -2px 8px rgba(88, 166, 255, 0.1);
        cursor: pointer;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 
            12px 12px 24px rgba(0, 0, 0, 0.4),
            -4px -4px 12px rgba(88, 166, 255, 0.2),
            0 0 20px rgba(88, 166, 255, 0.4);
        background: linear-gradient(145deg, var(--accent-purple), var(--accent-blue));
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 36px !important;
        }
        
        .hero-subtitle {
            font-size: 16px !important;
        }
        
        .hero-logo {
            width: 80px;
            height: 80px;
            font-size: 32px;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }
        
        .feature-card {
            padding: 24px 20px;
        }
    }

    /* =========================== */
    /* ENHANCED INPUT STYLES */
    /* =========================== */
    
    .stTextInput > div > div > input {
        background: linear-gradient(145deg, var(--bg-secondary), var(--bg-tertiary)) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 13px !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 
            inset 2px 2px 6px var(--shadow-dark),
            inset -2px -2px 6px var(--shadow-light) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 
            inset 2px 2px 6px var(--shadow-dark),
            inset -2px -2px 6px var(--shadow-light),
            0 0 12px rgba(88, 166, 255, 0.3) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
        font-size: 12px !important;
    }
    
    /* Enhanced Button Styles */
    .stButton > button {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-tertiary)) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 
            4px 4px 8px var(--shadow-dark),
            -4px -4px 8px var(--shadow-light) !important;
    }
    
    .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        transform: translateY(-1px) !important;
        box-shadow: 
            6px 6px 12px var(--shadow-dark),
            -6px -6px 12px var(--shadow-light),
            0 0 16px rgba(88, 166, 255, 0.3) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(145deg, var(--accent-blue), var(--accent-purple)) !important;
        color: white !important;
        border-color: var(--accent-blue) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(145deg, var(--accent-purple), var(--accent-blue)) !important;
        box-shadow: 
            6px 6px 12px var(--shadow-dark),
            -6px -6px 12px var(--shadow-light),
            0 0 20px rgba(88, 166, 255, 0.5) !important;
    }

    /* =========================== */
    /* EXISTING STYLES CONTINUE */
    /* =========================== */
</style>
""", unsafe_allow_html=True)

# Initialize session state for homepage
if "show_homepage" not in st.session_state:
    st.session_state.show_homepage = True
if "homepage_initialized" not in st.session_state:
    st.session_state.homepage_initialized = False

# Redesigned Sidebar - Ultra Professional & Clean
with st.sidebar:
    # Modern Header with Elegant Brand Identity - Enhanced CSS
    st.markdown("""
    <style>
    /* Enhanced Sidebar Styling */
    .modern-sidebar-header {
        background: linear-gradient(145deg, #161b22, #21262d);
        border-radius: 20px;
        padding: 24px 20px;
        margin-bottom: 20px;
        border: 1px solid #30363d;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.4),
            -8px -8px 16px rgba(48, 54, 61, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .modern-sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #58a6ff, #a5a5ff, #3fb950);
        background-size: 200% 100%;
        animation: headerGlow 4s ease-in-out infinite;
    }
    
    @keyframes headerGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
    }
    
    .brand-logo {
        position: relative;
        width: 48px;
        height: 48px;
    }
    
    .logo-inner {
        width: 100%;
        height: 100%;
        background: linear-gradient(145deg, #58a6ff, #a5a5ff);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.5),
            -2px -2px 6px rgba(88, 166, 255, 0.1);
        position: relative;
        animation: logoPulse 3s ease-in-out infinite;
    }
    
    @keyframes logoPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .logo-icon {
        font-size: 20px;
        z-index: 2;
        color: white;
    }
    
    .logo-pulse {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: linear-gradient(145deg, #58a6ff, #a5a5ff);
        opacity: 0.3;
        animation: pulseRing 2s ease-out infinite;
    }
    
    @keyframes pulseRing {
        0% {
            transform: scale(1);
            opacity: 0.3;
        }
        100% {
            transform: scale(1.3);
            opacity: 0;
        }
    }
    
    .brand-content {
        flex: 1;
    }
    
    .brand-title {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(90deg, #58a6ff, #a5a5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .brand-subtitle {
        font-size: 12px !important;
        color: #8b949e !important;
        margin: 4px 0 8px 0 !important;
        font-weight: 500 !important;
    }
    
    .status-bar {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: statusPulse 2s ease-in-out infinite;
    }
    
    .status-dot.online {
        background: #3fb950;
        box-shadow: 0 0 8px rgba(63, 185, 80, 0.5);
    }
    
    @keyframes statusPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }
    
    .status-text {
        font-size: 11px;
        color: #8b949e;
        font-weight: 500;
    }
    
    .home-button {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 12px;
        color: #58a6ff;
        font-weight: 500;
    }
    
    .home-button:hover {
        background: rgba(88, 166, 255, 0.2);
        border-color: #58a6ff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
    }
    
    .home-icon {
        font-size: 14px;
    }
    
    /* Enhanced Section Styles */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #30363d, transparent);
        margin: 20px 0;
        position: relative;
    }
    
    .sidebar-divider::before {
        content: '';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 4px;
        height: 4px;
        background: #58a6ff;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.5);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding: 12px 16px;
        background: linear-gradient(145deg, rgba(88, 166, 255, 0.05), rgba(165, 165, 255, 0.05));
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .section-icon {
        font-size: 18px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #58a6ff, #a5a5ff);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3);
        color: white;
    }
    
    .section-info h3 {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        margin: 0 !important;
    }
    
    .section-info p {
        font-size: 11px !important;
        color: #8b949e !important;
        margin: 2px 0 0 0 !important;
    }
    </style>
    
    <div class="modern-sidebar-header">
        <div class="brand-container">
            <div class="brand-logo">
                <div class="logo-inner">
                    <div class="logo-icon">🧠</div>
                    <div class="logo-pulse"></div>
                </div>
            </div>
            <div class="brand-content">
                <h1 class="brand-title">SmartQuery</h1>
                <p class="brand-subtitle">AI-Powered Analysis</p>
                <div class="status-bar">
                    <div class="status-dot online"></div>
                    <span class="status-text">Ready to analyze</span>
                </div>
            </div>
        </div>
        <div class="home-button" onclick="goToHomepage()">
            <div class="home-icon">🏠</div>
            <span class="home-text">Home</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript for homepage navigation
    st.markdown("""
    <script>
    function goToHomepage() {
        // Simple page reload to go to homepage
        window.location.reload();
    }
    
    // Add click handler for home button
    document.addEventListener('DOMContentLoaded', function() {
        const homeButton = document.querySelector('.home-button');
        if (homeButton) {
            homeButton.addEventListener('click', function() {
                // Add visual feedback
                this.style.transform = 'translateY(0) scale(0.95)';
                setTimeout(() => {
                    this.style.transform = 'translateY(-1px) scale(1)';
                    // Reload page after animation
                    window.location.reload();
                }, 150);
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Elegant Divider
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Website Analysis Section - Redesigned
    st.markdown("""
    <div class="analysis-section">
        <div class="section-header">
            <div class="section-icon">🌐</div>
            <div class="section-info">
                <h3>Website Analysis</h3>
                <p>Enter URL to start analyzing</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern URL Input
    website_url = st.text_input(
        "Website URL",
        placeholder="🔗 https://example.com",
        label_visibility="collapsed",
        help="Enter any website URL to start analyzing its content",
        key="website_url_input"
    )
    
    if website_url:
        st.markdown(f"""
        <div class="url-preview">
            <div class="preview-icon">📍</div>
            <div class="preview-text">
                <div class="preview-label">Analyzing:</div>
                <div class="preview-url">{website_url[:45]}{'...' if len(website_url) > 45 else ''}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Elegant Divider
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Chat Controls Section - Redesigned
    st.markdown("""
    <div class="controls-section">
        <div class="section-header">
            <div class="section-icon">💬</div>
            <div class="section-info">
                <h3>Chat Controls</h3>
                <p>Manage your conversation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern Action Buttons
    if "clear_chat_pressed" not in st.session_state:
        st.session_state.clear_chat_pressed = False

    col1, col2 = st.columns(2)
    with col1:
        clear_pressed = st.button("🗑️ Clear Chat", use_container_width=True, 
                                help="Clear current conversation")
    with col2:
        new_chat_pressed = st.button("✨ New Chat", use_container_width=True, 
                                   help="Start a new conversation")
    
    if clear_pressed or new_chat_pressed or st.session_state.clear_chat_pressed:
        st.session_state.clear_chat_pressed = True
        
        st.markdown("""
        <div class="clear-confirmation">
            <div class="confirmation-header">
                <div class="warning-icon">⚠️</div>
                <h4>Clear Conversation?</h4>
            </div>
            <p>This will save your current conversation automatically before clearing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            confirm = st.button("✅ Confirm", use_container_width=True)
        with col_cancel:
            cancel = st.button("❌ Cancel", use_container_width=True)
        
        if confirm:
            if "chat_history" in st.session_state and len(st.session_state.chat_history) > 1:  
                save_conversation("Auto-saved before clearing")
                st.success("💾 Conversation auto-saved!", icon="✅")
                
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm your AI assistant. Enter a website URL in the sidebar to begin analyzing content, or ask me anything!")
            ]
            st.session_state.vector_store = None
            st.session_state.last_url = ""
            st.session_state.loaded_conversation = None
            st.session_state.show_homepage = False
            
            # Reset global RAG chain
            globals()['_GLOBAL_RAG_CHAIN'] = None
            
            st.session_state.clear_chat_pressed = False
            st.rerun()
        
        if cancel:
            st.session_state.clear_chat_pressed = False
            st.rerun()
    
    # Elegant Divider
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Conversation History Section - Redesigned
    st.markdown("""
    <div class="history-section">
        <div class="section-header">
            <div class="section-icon">📚</div>
            <div class="section-info">
                <h3>Conversation History</h3>
                <p>Manage saved conversations</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Save & Refresh Controls
    col_save, col_refresh = st.columns(2)
    with col_save:
        if st.button("💾 Save Current", use_container_width=True, help="Save current conversation"):
            if save_conversation():
                st.success("💾 Saved successfully!", icon="✅")
                load_conversation_list()
            else:
                st.warning("Nothing to save.", icon="⚠️")
    
    with col_refresh:
        refresh_pressed = st.button("🔄 Refresh List", use_container_width=True, help="Refresh conversation list")
        if refresh_pressed:
            load_conversation_list()
            st.success("🔄 List updated!", icon="✅")
    
    # Load conversation list if not in session state
    if "saved_conversations" not in st.session_state:
        load_conversation_list()
        
    # Display conversations with modern design
    if not st.session_state.saved_conversations:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📝</div>
            <div class="empty-text">
                <h4>No saved conversations</h4>
                <p>Your conversations will appear here</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show conversations with modern cards
        for i, convo in enumerate(st.session_state.saved_conversations[:5]):  # Show only last 5
            st.markdown(f"""
            <div class="conversation-card">
                <div class="conversation-header">
                    <div class="conversation-title">{convo['title'][:30]}{'...' if len(convo['title']) > 30 else ''}</div>
                    <div class="conversation-meta">
                        <span class="message-count">💬 {convo['message_count']}</span>
                        <span class="timestamp">📅 {convo['timestamp'][-8:]}</span>
                    </div>
                </div>
                {f'<div class="conversation-url">� {convo["website_url"][:35]}{"..." if len(convo["website_url"]) > 35 else ""}</div>' if convo["website_url"] else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons for each conversation
            col_load, col_delete = st.columns([0.75, 0.25])
            
            with col_load:
                if st.button(f"📂 Load", key=f"load_{i}", use_container_width=True,
                           help=f"Load: {convo['title']}"):
                    messages = load_conversation(convo["path"])
                    if messages:
                        st.session_state.chat_history = messages
                        st.session_state.loaded_conversation = convo["title"]
                        st.session_state.show_homepage = False
                        st.success(f"✅ Loaded successfully!")
                        st.rerun()
            
            with col_delete:
                if st.button("🗑️", key=f"delete_{i}", help="Delete this conversation",
                           use_container_width=True):
                    if delete_conversation(convo["path"]):
                        st.success("🗑️ Deleted!")
                        st.rerun()
            
            if i < min(4, len(st.session_state.saved_conversations) - 1):
                st.markdown('<div class="conversation-separator"></div>', unsafe_allow_html=True)
        
        if len(st.session_state.saved_conversations) > 5:
            st.markdown(f"""
            <div class="more-conversations">
                📁 Showing 5 of {len(st.session_state.saved_conversations)} conversations
            </div>
            """, unsafe_allow_html=True)

    # Elegant Divider
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Statistics Section - Redesigned
    st.markdown("""
    <div class="stats-section">
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div class="section-info">
                <h3>Session Stats</h3>
                <p>Your activity overview</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate stats
    total_conversations = len(st.session_state.get("saved_conversations", []))
    current_messages = max(0, len(st.session_state.get("chat_history", [])) - 1)  # Exclude welcome message
    has_website = 1 if st.session_state.get('last_url') or st.session_state.get('vector_store') else 0
    
    # Clean metrics display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Messages", current_messages)
        st.metric("📚 Saved", total_conversations)
    with col2:
        st.metric("🌐 Website", "Active" if has_website else "None")
        if st.session_state.get("vector_store"):
            st.success("✅ Ready to chat", icon="🤖")
        else:
            st.info("ℹ️ Enter URL to start", icon="🌐")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
# Main content area
if website_url and website_url != st.session_state.get("last_url", ""):
    st.session_state.last_url = website_url
    st.session_state.thinking = True
    st.session_state.show_homepage = False
    st.session_state.analysis_complete = False  # Reset analysis flag
    st.session_state.analysis_ready_to_complete = False
    st.session_state.demo_processing_started = False
    st.rerun()  # Show the animation first

# Display homepage, welcome screen, chat interface, or loading animation
if st.session_state.get("thinking", False):
    # Website Analysis Loading Animation
    st.markdown("""
    <style>
    .analysis-container {
        min-height: 70vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 40px;
        text-align: center;
        background: 
            radial-gradient(circle at 30% 30%, rgba(88, 166, 255, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 70% 70%, rgba(165, 165, 255, 0.06) 0%, transparent 50%);
    }
    
    .analysis-logo {
        width: 160px;
        height: 160px;
        margin: 0 auto 40px;
        background: linear-gradient(145deg, #58a6ff, #a5a5ff);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 72px;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            0 0 60px rgba(88, 166, 255, 0.3);
        animation: analysisRotate 2s linear infinite;
        position: relative;
        overflow: hidden;
    }
    
    .analysis-logo::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: sweep 2s linear infinite;
    }
    
    @keyframes analysisRotate {
        0% { 
            transform: rotate(0deg) scale(1);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.3),
                0 0 60px rgba(88, 166, 255, 0.3);
        }
        25% {
            transform: rotate(90deg) scale(1.05);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.4),
                0 0 80px rgba(88, 166, 255, 0.5);
        }
        50% { 
            transform: rotate(180deg) scale(1);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.3),
                0 0 60px rgba(165, 165, 255, 0.4);
        }
        75% {
            transform: rotate(270deg) scale(1.05);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.4),
                0 0 80px rgba(165, 165, 255, 0.5);
        }
        100% { 
            transform: rotate(360deg) scale(1);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.3),
                0 0 60px rgba(88, 166, 255, 0.3);
        }
    }
    
    @keyframes sweep {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .analysis-title {
        font-size: 48px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 0 20px 0 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, #58a6ff, #a5a5ff, #3fb950);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: analysisGradient 3s ease-in-out infinite;
    }
    
    @keyframes analysisGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .analysis-subtitle {
        font-size: 24px !important;
        font-weight: 500 !important;
        color: #c9d1d9 !important;
        margin: 0 0 30px 0 !important;
        line-height: 1.4 !important;
        animation: analysisPulse 2s ease-in-out infinite;
    }
    
    @keyframes analysisPulse {
        0%, 100% { opacity: 0.8; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.02); }
    }
    
    .analysis-progress {
        width: 100%;
        max-width: 400px;
        margin: 30px auto;
        position: relative;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: rgba(139, 148, 158, 0.2);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #58a6ff, #a5a5ff, #3fb950);
        background-size: 200% 100%;
        border-radius: 4px;
        animation: progressAnimation 3s ease-in-out infinite;
        position: relative;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 30%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: progressShine 2s linear infinite;
    }
    
    @keyframes progressAnimation {
        0% { 
            width: 0%;
            background-position: 0% 50%;
        }
        50% { 
            width: 70%;
            background-position: 100% 50%;
        }
        100% { 
            width: 85%;
            background-position: 0% 50%;
        }
    }
    
    @keyframes progressShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(400%); }
    }
    
    .analysis-steps {
        margin-top: 40px;
        display: flex;
        flex-direction: column;
        gap: 15px;
        align-items: center;
    }
    
    .analysis-step {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px 25px;
        background: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        animation: stepFadeIn 0.6s ease-out forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    
    .analysis-step:nth-child(1) { animation-delay: 0.2s; }
    .analysis-step:nth-child(2) { animation-delay: 0.6s; }
    .analysis-step:nth-child(3) { animation-delay: 1.0s; }
    .analysis-step:nth-child(4) { animation-delay: 1.4s; }
    
    @keyframes stepFadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .step-icon {
        font-size: 24px;
        animation: stepIconSpin 2s linear infinite;
    }
    
    @keyframes stepIconSpin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .step-text {
        font-size: 16px;
        color: #c9d1d9;
        font-weight: 500;
    }
    
    .analysis-dots {
        display: flex;
        gap: 8px;
        margin-top: 30px;
        justify-content: center;
    }
    
    .analysis-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(45deg, #58a6ff, #a5a5ff);
        animation: dotPulse 1.5s ease-in-out infinite;
    }
    
    .analysis-dot:nth-child(1) { animation-delay: 0s; }
    .analysis-dot:nth-child(2) { animation-delay: 0.3s; }
    .analysis-dot:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes dotPulse {
        0%, 60%, 100% {
            transform: scale(1);
            opacity: 0.7;
        }
        30% {
            transform: scale(1.4);
            opacity: 1;
        }
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .analysis-container {
            padding: 40px 20px;
        }
        
        .analysis-title {
            font-size: 36px !important;
        }
        
        .analysis-subtitle {
            font-size: 18px !important;
        }
        
        .analysis-logo {
            width: 120px;
            height: 120px;
            font-size: 56px;
        }
        
        .analysis-steps {
            margin-top: 30px;
        }
        
        .analysis-step {
            padding: 12px 20px;
            flex-direction: column;
            text-align: center;
            gap: 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Analysis Animation HTML
    st.markdown("""
    <div class="analysis-container">
        <div class="analysis-logo">🧠</div>
        <p class="analysis-title">Analyzing Website</p>
        <p class="analysis-subtitle">Processing content with advanced AI...</p>
        <div class="analysis-progress">
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
        <div class="analysis-steps">
            <div class="analysis-step">
                <div class="step-icon">🌐</div>
                <div class="step-text">Fetching website content</div>
            </div>
            <div class="analysis-step">
                <div class="step-icon">📄</div>
                <div class="step-text">Parsing and cleaning text</div>
            </div>
            <div class="analysis-step">
                <div class="step-icon">🔍</div>
                <div class="step-text">Creating knowledge vectors</div>
            </div>
            <div class="analysis-step">
                <div class="step-icon">🤖</div>
                <div class="step-text">Preparing AI responses</div>
            </div>
        </div>
        <div class="analysis-dots">
            <div class="analysis-dot"></div>
            <div class="analysis-dot"></div>
            <div class="analysis-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add interactive JavaScript for analysis animation
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        console.log('Animation loaded!');
        
        // Enhanced analysis logo interaction
        const analysisLogo = document.querySelector('.analysis-logo');
        if (analysisLogo) {
            analysisLogo.addEventListener('click', function() {
                this.style.animation = 'none';
                this.style.transform = 'scale(1.2) rotate(720deg)';
                setTimeout(() => {
                    this.style.animation = 'analysisRotate 2s linear infinite';
                    this.style.transform = '';
                }, 600);
            });
        }
        
        // Simulate analysis progress steps
        const steps = document.querySelectorAll('.analysis-step');
        let currentStep = 0;
        
        function updateStepProgress() {
            if (currentStep < steps.length) {
                const step = steps[currentStep];
                const stepText = step.querySelector('.step-text');
                const stepIcon = step.querySelector('.step-icon');
                
                // Highlight current step
                step.style.background = 'rgba(88, 166, 255, 0.2)';
                step.style.borderColor = 'rgba(88, 166, 255, 0.4)';
                step.style.transform = 'scale(1.05)';
                
                // Add completion effect after delay
                setTimeout(() => {
                    stepIcon.textContent = '✅';
                    stepIcon.style.animation = 'none';
                    step.style.background = 'rgba(63, 185, 80, 0.1)';
                    step.style.borderColor = 'rgba(63, 185, 80, 0.3)';
                    step.style.transform = 'scale(1)';
                    
                    currentStep++;
                    if (currentStep < steps.length) {
                        setTimeout(updateStepProgress, 500);
                    }
                }, 1000 + Math.random() * 1000);
            }
        }
        
        // Start step progression after initial delay
        setTimeout(updateStepProgress, 1000);
        
        // Add floating particles effect
        function createParticle() {
            const particle = document.createElement('div');
            particle.style.position = 'fixed';
            particle.style.width = '4px';
            particle.style.height = '4px';
            particle.style.background = 'rgba(88, 166, 255, 0.6)';
            particle.style.borderRadius = '50%';
            particle.style.pointerEvents = 'none';
            particle.style.zIndex = '1000';
            
            const startX = Math.random() * window.innerWidth;
            const startY = window.innerHeight + 10;
            
            particle.style.left = startX + 'px';
            particle.style.top = startY + 'px';
            
            document.body.appendChild(particle);
            
            let y = startY;
            let x = startX;
            let opacity = 0.6;
            
            const animate = () => {
                y -= 2;
                x += Math.sin(y * 0.01) * 0.5;
                opacity -= 0.002;
                
                particle.style.top = y + 'px';
                particle.style.left = x + 'px';
                particle.style.opacity = opacity;
                
                if (y > -10 && opacity > 0) {
                    requestAnimationFrame(animate);
                } else {
                    document.body.removeChild(particle);
                }
            };
            
            requestAnimationFrame(animate);
        }
        
        // Create particles periodically
        const particleInterval = setInterval(() => {
            if (Math.random() < 0.3) {
                createParticle();
            }
        }, 200);
        
        // Clean up particles when page changes
        setTimeout(() => {
            clearInterval(particleInterval);
        }, 10000);
        
        // Add subtle glow effect to progress bar
        const progressBar = document.querySelector('.progress-fill');
        if (progressBar) {
            setInterval(() => {
                const intensity = 0.5 + Math.sin(Date.now() * 0.003) * 0.3;
                progressBar.style.boxShadow = `0 0 20px rgba(88, 166, 255, ${intensity})`;
            }, 100);
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Process website analysis in the background after showing animation
    if not st.session_state.get("analysis_complete", False):
        if not AI_AVAILABLE:
            # Demo mode - simulate analysis with proper timing
            if not st.session_state.get("demo_processing_started", False):
                st.session_state.demo_processing_started = True
                st.session_state.demo_start_time = time.time()
                st.rerun()
            
            # Check if enough time has passed (minimum 4 seconds for animation)
            elapsed = time.time() - st.session_state.get("demo_start_time", 0)
            if elapsed >= 4.0:  # 4 seconds minimum for animation
                st.session_state.analysis_complete = True
                st.session_state.vector_store = "demo_mode"  # Set dummy vector store
                st.session_state.thinking = False
                st.session_state.demo_processing_started = False
                st.success("✅ Website analyzed successfully! Ask me anything about the content.")
                st.rerun()
        else:
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
                    st.session_state.analysis_complete = True
                    _GLOBAL_RAG_CHAIN = None
                    
                    # Complete the analysis
                    st.session_state.thinking = False
                    st.success("✅ Website analyzed successfully! Ask me anything about the content.")
                    st.rerun()
                else:
                    st.session_state.thinking = False
                    st.session_state.analysis_complete = False
                    st.error("Could not extract content from the website.")
            except Exception as e:
                st.session_state.thinking = False
                st.session_state.analysis_complete = False
                st.error(f"Error processing website: {str(e)}")

elif st.session_state.show_homepage or not st.session_state.get("vector_store"):
    # Clear any potential HTML output issues
    st.empty()
    
    # Simple and Clean Homepage Design
    st.markdown("""
    <style>
    .homepage-container {
        min-height: 70vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 40px;
        text-align: center;
        background: 
            radial-gradient(circle at 30% 30%, rgba(88, 166, 255, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 70% 70%, rgba(165, 165, 255, 0.06) 0%, transparent 50%);
    }
    
    .hero-logo {
        width: 140px;
        height: 140px;
        margin: 0 auto 40px;
        background: linear-gradient(145deg, #58a6ff, #a5a5ff);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            0 0 60px rgba(88, 166, 255, 0.2);
        animation: heroLogoPulse 4s ease-in-out infinite;
        cursor: pointer;
    }
    
    @keyframes heroLogoPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.3),
                0 0 60px rgba(88, 166, 255, 0.2);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.4),
                0 0 80px rgba(88, 166, 255, 0.4);
        }
    }
    
    .hero-title {
        font-size: 56px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 0 20px 0 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, #58a6ff, #a5a5ff, #3fb950);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleGradient 4s ease-in-out infinite;
    }
    
    @keyframes titleGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-size: 24px !important;
        font-weight: 400 !important;
        color: #c9d1d9 !important;
        margin: 0 0 40px 0 !important;
        line-height: 1.4 !important;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .hero-description {
        font-size: 18px !important;
        color: #8b949e !important;
        margin: 0 0 50px 0 !important;
        line-height: 1.6 !important;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .start-prompt {
        margin-top: 60px;
        text-align: center;
        opacity: 0.9;
        animation: fadeInUp 2s ease-out 0.5s both;
        position: relative;
    }
    
    .prompt-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        padding: 25px 35px;
        background: linear-gradient(145deg, rgba(88, 166, 255, 0.1), rgba(165, 165, 255, 0.08));
        border: 2px solid rgba(88, 166, 255, 0.3);
        border-radius: 25px;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .prompt-container:hover {
        transform: translateY(-3px) scale(1.02);
        background: linear-gradient(145deg, rgba(88, 166, 255, 0.15), rgba(165, 165, 255, 0.12));
        border-color: rgba(88, 166, 255, 0.5);
        box-shadow: 
            0 15px 35px rgba(88, 166, 255, 0.2),
            0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .prompt-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .prompt-container:hover::before {
        left: 100%;
    }
    
    .prompt-glow {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 120%;
        height: 120%;
        background: radial-gradient(circle, rgba(88, 166, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulse 3s ease-in-out infinite;
        z-index: -1;
    }
    
    .arrow-pointer {
        font-size: 32px;
        color: #58a6ff;
        margin: 0;
        animation: bounce 2.5s ease-in-out infinite;
        text-shadow: 0 0 20px rgba(88, 166, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .prompt-container:hover .arrow-pointer {
        transform: scale(1.2) rotate(15deg);
        color: #a5a5ff;
    }
    
    .prompt-content {
        text-align: left;
    }
    
    .start-text {
        font-size: 18px !important;
        color: #ffffff !important;
        margin: 0 0 5px 0 !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .start-subtext {
        font-size: 14px !important;
        color: #8b949e !important;
        margin: 0 !important;
        font-weight: 400;
        letter-spacing: 0.3px;
        opacity: 0.8;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 0.3;
            transform: translate(-50%, -50%) scale(1);
        }
        50% {
            opacity: 0.1;
            transform: translate(-50%, -50%) scale(1.1);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 0.8;
            transform: translateY(0);
        }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    .cta-section {
        margin: 20px 0;
    }
    
    .cta-button {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 18px 36px;
        background: linear-gradient(145deg, #58a6ff, #a5a5ff);
        color: white;
        border: none;
        border-radius: 50px;
        font-size: 18px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            0 0 32px rgba(88, 166, 255, 0.2);
        cursor: pointer;
    }
    
    .cta-button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.4),
            0 0 48px rgba(88, 166, 255, 0.4);
        background: linear-gradient(145deg, #a5a5ff, #58a6ff);
    }
    
    .features-list {
        margin-top: 60px;
        display: flex;
        justify-content: center;
        gap: 40px;
        flex-wrap: wrap;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 16px;
        color: #c9d1d9;
        font-weight: 500;
    }
    
    .feature-icon {
        font-size: 24px;
        animation: featureFloat 3s ease-in-out infinite;
    }
    
    @keyframes featureFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-4px); }
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .homepage-container {
            padding: 40px 20px;
        }
        
        .hero-title {
            font-size: 42px !important;
        }
        
        .hero-subtitle {
            font-size: 20px !important;
        }
        
        .hero-logo {
            width: 100px;
            height: 100px;
            font-size: 48px;
        }
        
        .start-prompt {
            margin-top: 40px;
        }
        
        .prompt-container {
            flex-direction: column;
            gap: 15px;
            padding: 20px 25px;
        }
        
        .prompt-content {
            text-align: center;
        }
        
        .arrow-pointer {
            font-size: 28px;
        }
        
        .start-text {
            font-size: 16px !important;
        }
        
        .start-subtext {
            font-size: 12px !important;
        }
        
        .features-list {
            flex-direction: column;
            gap: 20px;
            align-items: center;
        }
        
        .cta-button {
            padding: 16px 32px;
            font-size: 16px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Homepage HTML content
    st.markdown("""<div class="homepage-container">
        <div class="hero-logo">🧠</div>
        <p class="hero-title">SmartQuery AI</p>
        <p class="hero-subtitle">Transform any website into an intelligent conversation partner</p>
        <p class="hero-description">Analyze web content with advanced AI, ask questions, and get comprehensive answers in real-time.</p>
        <div class="start-prompt">
            <div class="prompt-container">
                <div class="arrow-pointer">↖</div>
                <div class="prompt-content">
                    <p class="start-text">🚀 Start by entering a URL above</p>
                </div>
            </div>
            <div class="prompt-glow"></div>
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Add interactive JavaScript
    st.markdown("""
    <script>
    // Function to focus on URL input
    function focusUrlInput() {
        // Add click animation to button
        const ctaButton = document.querySelector('.cta-button');
        if (ctaButton) {
            ctaButton.style.transform = 'translateY(0) scale(0.95)';
            setTimeout(() => {
                ctaButton.style.transform = 'translateY(-3px) scale(1)';
            }, 150);
        }
        
        // Focus on sidebar URL input
        setTimeout(() => {
            const urlInput = document.querySelector('[data-testid="stSidebar"] input[type="text"]') || 
                           document.querySelector('[data-testid="stSidebar"] input') ||
                           document.querySelector('.stTextInput input');
            
            if (urlInput) {
                urlInput.focus();
                urlInput.style.boxShadow = '0 0 0 2px #58a6ff';
                urlInput.placeholder = 'Enter website URL here to get started...';
                setTimeout(() => {
                    urlInput.style.boxShadow = '';
                }, 2000);
            } else {
                // Fallback: scroll to sidebar
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                if (sidebar) {
                    sidebar.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        }, 200);
    }
    
    // Simple homepage interactions
    document.addEventListener('DOMContentLoaded', function() {
        // Logo click animation
        const heroLogo = document.querySelector('.hero-logo');
        if (heroLogo) {
            heroLogo.addEventListener('click', function() {
                this.style.transform = 'scale(1.1) rotate(10deg)';
                setTimeout(() => {
                    this.style.transform = 'scale(1) rotate(0deg)';
                }, 300);
            });
        }
        
        // Enhanced prompt container interaction
        const promptContainer = document.querySelector('.prompt-container');
        if (promptContainer) {
            promptContainer.addEventListener('click', function() {
                // Add ripple effect
                const ripple = document.createElement('div');
                ripple.style.position = 'absolute';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(88, 166, 255, 0.3)';
                ripple.style.transform = 'scale(0)';
                ripple.style.animation = 'ripple 0.6s linear';
                ripple.style.left = '50%';
                ripple.style.top = '50%';
                ripple.style.width = '20px';
                ripple.style.height = '20px';
                ripple.style.marginLeft = '-10px';
                ripple.style.marginTop = '-10px';
                
                this.appendChild(ripple);
                
                // Focus on URL input
                focusUrlInput();
                
                setTimeout(() => {
                    this.removeChild(ripple);
                }, 600);
            });
            
            // Add ripple animation keyframes
            const style = document.createElement('style');
            style.textContent = `
                @keyframes ripple {
                    to {
                        transform: scale(4);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Feature items stagger animation
        const featureItems = document.querySelectorAll('.feature-item');
        featureItems.forEach((item, index) => {
            item.style.opacity = '0';
            item.style.transform = 'translateY(20px)';
            setTimeout(() => {
                item.style.transition = 'all 0.6s ease-out';
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            }, 1000 + (index * 200));
        });
    });
    
    // Global function for homepage navigation
    function goToHomepage() {
        window.location.reload();
    }
    </script>
    """, unsafe_allow_html=True)

else:
    # Main content wrapper to handle fixed search bar spacing
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Chat interface - Modern Neuomorphic Container
    st.markdown("<div class='modern-chat-container'>", unsafe_allow_html=True)

    if st.session_state.loaded_conversation:
        st.markdown(f"""
        <div class="loaded-conversation-indicator">
            📂 <strong>Loaded:</strong> {st.session_state.loaded_conversation}
        </div>
        """, unsafe_allow_html=True)

    # Display messages with enhanced styling
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            display_ai_message(message.content, i)
        elif isinstance(message, HumanMessage):
            display_human_message(message.content, i)

    # Enhanced thinking/analyzing animation
    if "thinking" in st.session_state and st.session_state.thinking:
        st.markdown("""
        <div class="thinking-animation">
            <div class="thinking-avatar">
                <div class="avatar-inner thinking">🤖</div>
            </div>
            <div class="thinking-content">
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
                <div class="thinking-text">🔍 Analyzing website content... Almost ready! ✨</div>
            </div>
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
    
    """, unsafe_allow_html=True)
    
    # Add JavaScript functionality separately to avoid f-string conflicts
    st.markdown("""
    <script>
 
    
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
            # Show enhanced generating response animation
            response_placeholder = st.empty()
            with response_placeholder.container():
                st.markdown("""
                <div class="response-generation">
                    <div class="response-avatar">
                        <div class="avatar-inner generating">🤖</div>
                    </div>
                    <div class="response-content-generation">
                        <div class="generation-spinner">
                            <div class="spinner-ring"></div>
                            <div class="spinner-ring"></div>
                            <div class="spinner-ring"></div>
                        </div>
                        <div class="generation-text">🧠 Generating intelligent response... Hold tight! 🎯</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            response = get_response(last_user_message)
            response_placeholder.empty()
            st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.generating_response = False
            st.rerun()
    
# Elegant Creator Signature - Always visible at bottom right
st.markdown("""
<style>
.creator-title {
    font-size: 10px !important;
    color: rgba(255, 255, 255, 0.7) !important;
}
</style>

<a href="https://www.linkedin.com/in/nagre/" target="_blank" style="text-decoration: none;">
    <div class="creator-signature" title="Click to visit LinkedIn profile">
        <div class="creator-avatar">V</div>
        <div class="creator-info">
            <div class="creator-name">Vaibhav Nagre</div>
            <div class="creator-title">AI Developer</div>
        </div>
    </div>
</a>

<script>
// Add smooth interactions to creator signature
document.addEventListener('DOMContentLoaded', function() {
    const signature = document.querySelector('.creator-signature');
    if (signature) {
        signature.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.02)';
        });
        
        signature.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    }
});
</script>
""", unsafe_allow_html=True)