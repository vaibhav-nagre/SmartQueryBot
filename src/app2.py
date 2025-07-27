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
        
        if not "Hello! I am Vaibhav's AI assistant" in content and not "I've analyzed the content" in content:
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
                        <div class="ai-thinking">
                            <div class="ai-thinking-orb"></div>
                            <div class="ai-thinking-orb"></div>
                            <div class="ai-thinking-orb"></div>
                            <span style="margin-left: 12px; color: var(--text-secondary);">Generating detailed explanation...</span>
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am Vaibhav's AI assistant. How can I help you today?")
    ]

# Apply CSS styling - DARK MODE ONLY, NO HAMBURGER
st.markdown("""
<style>
    /* Enhanced Neumorphic UI with Dark Mode Only - FIXED CHAT EMPTY BOX AND SEARCH BAR */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    :root {
        /* Dark Theme Variables Only */
        --bg: #1a1f33;
        --shadow-dark: rgba(0, 0, 0, 0.5);
        --shadow-light: rgba(67, 76, 108, 0.4);
        --text: #e6e7ee;
        --text-secondary: #b8b9cf;
        --text-muted: #8a8eaf;
        --primary: #5e72e4;
        --primary-hover: #324cdd;
        --accent: #2dce89;
        --error: #f5365c;
        --warning: #fb6340;
        --border: #2e3650;
        --card: #252f4a;
        --input: #2a2f4c;
    }

    /* Global Styling */
    body, .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg);
        color: var(--text);
        transition: all 0.3s ease;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, div, li, a {
        font-family: 'Poppins', sans-serif;
        color: var(--text);
    }

    /* AGGRESSIVE EMPTY BOX REMOVAL - ENHANCED */
    .element-container:empty,
    [data-testid="element-container"]:empty,
    .stMarkdown:empty,
    [data-testid="stMarkdown"]:empty,
    [data-testid="stVerticalBlock"]:empty,
    [data-testid="stHorizontalBlock"]:empty,
    [data-testid="stVerticalBlockContainer"]:empty,
    [data-testid="stHorizontalBlockContainer"]:empty,
    .main [data-testid="stVerticalBlock"] > div:empty,
    .main [data-testid="stHorizontalBlock"] > div:empty,
    .main [data-testid="stVerticalBlockContainer"] > div:empty,
    .main [data-testid="stHorizontalBlockContainer"] > div:empty,
    div:empty:not([class*="chat"]):not([class*="signature"]):not([class*="stChatInput"]) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }

    /* REMOVE SIDEBAR EMPTY SPACES */
    [data-testid="stSidebar"] .element-container:empty,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:empty,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:empty,
    [data-testid="stSidebar"] [data-testid="stVerticalBlockContainer"]:empty,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlockContainer"]:empty {
        display: none !important;
        visibility: hidden !important;
    }

    /* HIDE MAIN CONTENT EMPTY BOXES */
    .main [data-testid="stVerticalBlock"]:empty,
    .main [data-testid="stHorizontalBlock"]:empty,
    .main [data-testid="stVerticalBlockContainer"]:empty,
    .main [data-testid="stHorizontalBlockContainer"]:empty,
    .main .element-container:empty {
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }



    /* Message styling */
    [data-testid="stChatMessage"] {
        margin-bottom: 1.2rem;
        border-radius: 15px;
        background-color: var(--card);
        box-shadow: 3px 3px 8px var(--shadow-dark), 
                  -3px -3px 8px var(--shadow-light);
        transition: all 0.3s ease;
    }

    /* Human message styling */
    [data-testid="stChatMessageContent"].human {
        background-color: var(--card);
        border-radius: 15px;
        box-shadow: inset 3px 3px 5px var(--shadow-dark), 
                  inset -3px -3px 5px var(--shadow-light);
        border-left: 3px solid var(--primary);
        transition: all 0.3s ease;
    }

    /* AI message styling */
    [data-testid="stChatMessageContent"].ai {
        background-color: var(--card);
        border-radius: 15px;
        box-shadow: inset 3px 3px 5px var(--shadow-dark), 
                  inset -3px -3px 5px var(--shadow-light);
        border-left: 3px solid var(--accent);
        transition: all 0.3s ease;
    }

    /* REVOLUTIONARY SEARCH BAR - PROPERLY MOUNTED TO BOTTOM */
    .smart-search-container {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 2000 !important;
        font-family: 'Poppins', sans-serif !important;
        background: var(--card) !important;
        border-top: 1px solid var(--border) !important;
        box-shadow: 0 -20px 40px rgba(0, 0, 0, 0.3), 
                    0 -8px 16px rgba(0, 0, 0, 0.2),
                    inset 0 8px 16px var(--shadow-light), 
                    inset 0 -8px 16px var(--shadow-dark) !important;
        backdrop-filter: blur(15px) !important;
        padding: 15px 20px 20px 20px !important;
        transition: all 0.3s ease !important;
    }

    .smart-search-container:hover {
        box-shadow: 0 -25px 50px rgba(0, 0, 0, 0.4), 
                    0 -12px 24px rgba(0, 0, 0, 0.3),
                    inset 0 10px 20px var(--shadow-light), 
                    inset 0 -10px 20px var(--shadow-dark) !important;
        border-top-color: var(--primary) !important;
    }

    /* Inner wrapper for centering content */
    .smart-search-wrapper {
        max-width: 700px !important;
        margin: 0 auto !important;
        background: var(--bg) !important;
        border-radius: 25px !important;
        padding: 8px !important;
        box-shadow: 8px 8px 16px var(--shadow-dark), 
                    -8px -8px 16px var(--shadow-light) !important;
        border: 1px solid rgba(94, 114, 228, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }

    .smart-search-wrapper:hover {
        transform: translateY(-2px) !important;
        box-shadow: 10px 10px 20px var(--shadow-dark), 
                    -10px -10px 20px var(--shadow-light) !important;
        border-color: var(--primary) !important;
    }

    /* Override Streamlit's column styling for search - TIGHTER SPACING */
    .smart-search-container [data-testid="column"] {
        padding: 0 2px !important;
        margin: 0 !important;
        gap: 0 !important;
    }

    /* Reduce gap specifically for action buttons columns */
    .smart-search-container [data-testid="column"]:nth-child(3),
    .smart-search-container [data-testid="column"]:nth-child(4) {
        padding: 0 1px !important;
    }

    /* Style the custom search input */
    .smart-search-container .stTextInput {
        flex: 1 !important;
        margin: 0 !important;
    }

    .smart-search-container .stTextInput > div {
        border: none !important;
        background: transparent !important;
    }

    .smart-search-container .stTextInput > div > div {
        border: none !important;
        background: transparent !important;
    }

    .smart-search-container .stTextInput > div > div > input {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: var(--text) !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 12px 20px !important;
        margin: 0 !important;
        line-height: 1.5 !important;
        border-radius: 20px !important;
        width: 100% !important;
    }

    .smart-search-container .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
        font-style: italic !important;
        transition: all 0.3s ease !important;
    }

    .smart-search-container .stTextInput > div > div > input:focus::placeholder {
        opacity: 0.5 !important;
        transform: translateX(10px) !important;
    }

    /* Style the action buttons - TIGHTER SPACING */
    .smart-search-container .stButton {
        margin: 0 !important;
        width: 100% !important;
    }

    .smart-search-container .stButton button {
        width: 40px !important;
        height: 40px !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 3px 3px 6px var(--shadow-dark), 
                    -3px -3px 6px var(--shadow-light) !important;
        background-color: var(--card) !important;
        color: var(--primary) !important;
        min-width: 40px !important;
        padding: 0 !important;
        margin: 0 auto !important;
    }

    .smart-search-container .stButton button:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 8px 20px rgba(94, 114, 228, 0.3) !important;
        background-color: var(--primary) !important;
        color: white !important;
    }

    .smart-search-container .stButton button:active {
        transform: translateY(0) !important;
        box-shadow: inset 2px 2px 4px var(--shadow-dark) !important;
    }

    /* Special styling for send button */
    .smart-search-container .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
        color: white !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .smart-search-container .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.1) !important;
        box-shadow: 0 10px 25px rgba(94, 114, 228, 0.5) !important;
    }

    /* Add animated search icon */
    .smart-search-container .stTextInput::before {
        content: '🔍' !important;
        position: absolute !important;
        left: 20px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        font-size: 16px !important;
        z-index: 10 !important;
        pointer-events: none !important;
        animation: searchPulse 2s ease-in-out infinite !important;
    }

    @keyframes searchPulse {
        0%, 100% { opacity: 0.7; transform: translateY(-50%) scale(1); }
        50% { opacity: 1; transform: translateY(-50%) scale(1.1); }
    }

    /* Add bottom padding to main content to prevent overlap */
    .main .block-container {
        padding-bottom: 120px !important;
    }

    /* Style suggestion buttons */
    .smart-search-container + div .stButton button {
        background: var(--card) !important;
        border-radius: 12px !important;
        padding: 10px 15px !important;
        width: 100% !important;
        height: auto !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        margin-bottom: 5px !important;
        box-shadow: 3px 3px 8px var(--shadow-dark), 
                    -3px -3px 8px var(--shadow-light) !important;
    }

    .smart-search-container + div .stButton button:hover {
        background: var(--primary) !important;
        color: white !important;
        transform: translateX(8px) !important;
        box-shadow: 5px 5px 15px rgba(94, 114, 228, 0.3) !important;
    }

    /* Hide default Streamlit chat input and form styling */
    [data-testid="stChatInput"],
    [data-testid="stChatFloatingInputContainer"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Responsive design for mobile */
    @media (max-width: 768px) {
        .smart-search-container {
            width: 95% !important;
            bottom: 15px !important;
        }
        
        .smart-search-wrapper {
            border-radius: 20px !important;
        }
        
        .smart-search-input-container {
            padding: 10px 15px !important;
        }
        
        .smart-search-input {
            font-size: 14px !important;
        }
        
        .search-action-btn {
            width: 32px !important;
            height: 32px !important;
            font-size: 12px !important;
        }
        
        .send-btn {
            width: 38px !important;
            height: 38px !important;
        }
        
        .search-suggestions {
            top: -100px !important;
        }
    }

    /* Button styling */
    .stButton button {
        background-color: var(--card);
        color: var(--text);
        border: none;
        border-radius: 15px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 5px 5px 10px var(--shadow-dark), 
                  -5px -5px 10px var(--shadow-light);
    }

    .stButton button:hover {
        box-shadow: 3px 3px 8px var(--shadow-dark), 
                  -3px -3px 8px var(--shadow.light);
        transform: translateY(-2px);
    }

    .stButton button:active {
        box-shadow: inset 3px 3px 5px var(--shadow-dark), 
                  inset -3px -3px 5px var(--shadow.light);
        transform: translateY(0);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg);
        border-right: 1px solid var(--border);
        transition: all 0.3s ease;
    }

    [data-testid="stSidebarUserContent"] {
        padding: 1.5rem 1rem;
    }

    /* Sidebar section styling */
    .sidebar-section {
        background-color: var(--card);
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 5px 5px 10px var(--shadow-dark), 
                  -5px -5px 10px var(--shadow-light);
        transition: all 0.3s ease;
    }
    
    .sidebar-section h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Section headings */
    .section-heading {
        margin: 0 0 1rem 0;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text);
    }

    /* Text input fields */
    .stTextInput > div > div > input {
        background-color: var(--input);
        color: var(--text);
        border-radius: 15px;
        border: none;
        padding: 0.75rem 1rem;
        box-shadow: inset 3px 3px 5px var(--shadow-dark), 
                  inset -3px -3px 5px var(--shadow-light);
        transition: all 0.3s ease;
    }

    /* Elaborate button */
    button[key*="elaborate_"] {
        background-color: var(--card);
        color: var(--primary);
        border-radius: 12px;
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 3px 3px 8px var(--shadow-dark), 
                  -3px -3px 8px var(--shadow-light);
    }

    button[key*="elaborate_"]:hover {
        box-shadow: 2px 2px 5px var(--shadow-dark), 
                  -2px -2px 5px var(--shadow-light);
        transform: translateY(-2px);
    }

    button[key*="elaborate_"]:active {
        box-shadow: inset 2px 2px 5px var(--shadow-dark), 
                  inset -2px -2px 5px var(--shadow-light);
        transform: translateY(0);
    }

    /* AI thinking animation */
    .ai-thinking {
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: var(--card);
        border-radius: 15px;
        box-shadow: inset 3px 3px 5px var(--shadow-dark), 
                  inset -3px -3px 5px var(--shadow-light);
        transition: all 0.3s ease;
    }

    .ai-thinking-orb {
        width: 10px;
        height: 10px;
        margin: 0 4px;
        border-radius: 50%;
        background: var(--primary);
        animation: float 1.5s ease-in-out infinite;
        box-shadow: 2px 2px 5px var(--shadow-dark), 
                  -2px -2px 5px var(--shadow-light);
    }

    .ai-thinking-orb:nth-child(2) {
        animation-delay: 0.2s;
        background: var(--accent);
    }

    .ai-thinking-orb:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }

    /* Creator signature - MOVED TO AVOID SEARCH BAR CONFLICT */
    .creator-signature {
        position: fixed;
        bottom: 100px;
        left: 20px;
        font-size: 0.9rem;
        padding: 8px 16px;
        border-radius: 20px;
        background: var(--card);
        box-shadow: 3px 3px 8px var(--shadow-dark), 
                  -3px -3px 8px var(--shadow-light);
        z-index: 1500;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .creator-signature:hover {
        transform: translateY(-3px);
        box-shadow: 4px 4px 10px var(--shadow-dark), 
                  -4px -4px 10px var(--shadow-light);
    }

    .creator-signature .avatar-mini {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 14px;
    }

    .creator-signature span {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Remove spacing from main content - OPTIMIZED FOR CHATGPT STYLE */
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0 !important;
    }

    [data-testid="stVerticalBlock"] > div:last-child {
        margin-bottom: 0 !important;
    }

    /* Hide specific empty containers that cause the box */
    .main > div:empty,
    .main [data-testid="stVerticalBlock"] > [style*="flex"]:empty,
    .main [data-testid="stHorizontalBlock"] > [style*="flex"]:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Force hide any remaining empty spaces in main area */
    .main [data-testid="stVerticalBlock"] > div[style*="width"]:empty {
        display: none !important;
    }

    /* Welcome screen - ChatGPT style */
    .welcome-container {
        max-width: 100% !important;
        margin: 0 auto !important;
        text-align: center !important;
        padding: 2rem 1rem !important;
    }

    /* CHATGPT-STYLE NARROW CENTERED LAYOUT */
    .main .block-container {
        max-width: 1300px !important;
        margin: 0 auto !important;
        padding: 2rem 1rem 140px 1rem !important;
    }

    /* Chat container styling */
    .chat-container {
        max-width: 100% !important;
        margin: 0 auto !important;
        padding: 0 !important;
    }

    /* Message styling - ChatGPT style */
    [data-testid="stChatMessage"] {
        margin-bottom: 1.5rem !important;
        border-radius: 15px !important;
        background-color: var(--card) !important;
        box-shadow: 3px 3px 8px var(--shadow-dark), 
                  -3px -3px 8px var(--shadow-light) !important;
        transition: all 0.3s ease !important;
        max-width: 100% !important;
    }

    /* Remove top margin from first message */
    [data-testid="stChatMessage"]:first-child {
        margin-top: 0 !important;
    }

    /* Add spacing between messages */
    [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
        margin-top: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - NO HAMBURGER
with st.sidebar:
    # Simple header - NO EMPTY CONTAINERS
    st.markdown("<h2 style='font-weight: 700; margin: 0 0 1.5rem 0;'>SmartQuery Bot</h2>", unsafe_allow_html=True)
    
    # Website URL input - CLEAN CONTAINER
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-heading">📌 Enter Website</h3>', unsafe_allow_html=True)
    website_url = st.text_input(
        "Website URL:",
        placeholder="https://example.com",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Simple separator
    st.markdown("---")
    
    # Clear chat button - CLEAN CONTAINER
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-heading">🧹 Chat Management</h3>', unsafe_allow_html=True)
    
    if "clear_chat_pressed" not in st.session_state:
        st.session_state.clear_chat_pressed = False

    if st.button("Clear Chat History", use_container_width=True) or st.session_state.clear_chat_pressed:
        st.session_state.clear_chat_pressed = True
        
        confirm = st.checkbox("Confirm clearing chat?")
        cancel = st.button("Cancel")
        
        if confirm:
            if "chat_history" in st.session_state and len(st.session_state.chat_history) > 1:  
                save_conversation("Auto-saved before clearing")
                
            st.session_state.chat_history = [
                AIMessage(content=f"Hello! I am Vaibhav Nagre's AI assistant. How can I help you?")
            ]
            st.session_state.vector_store = None
            st.session_state.last_url = ""
            st.session_state.loaded_conversation = None
            _GLOBAL_RAG_CHAIN = None
            
            st.session_state.clear_chat_pressed = False
            st.rerun()
        
        if cancel:
            st.session_state.clear_chat_pressed = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Conversation history - CLEAN CONTAINER
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-heading">💬 Conversation History</h3>', unsafe_allow_html=True)
    
    if st.button("📥 Save Current Conversation", use_container_width=True):
        if save_conversation():
            st.success("Conversation saved!")
            load_conversation_list()
        else:
            st.warning("Nothing to save.")
    
    if "saved_conversations" not in st.session_state:
        load_conversation_list()
        
    if not st.session_state.saved_conversations:
        st.info("No saved conversations.")
    else:
        st.markdown("#### Saved Conversations")
        
        if "delete_success" in st.session_state and st.session_state.delete_success:
            st.session_state.delete_success = False
        
        for i, convo in enumerate(st.session_state.saved_conversations):
            cols = st.columns([0.7, 0.3])
            
            with cols[0]:
                if st.button(f"{convo['title']}", key=f"load_{i}", use_container_width=True):
                    messages = load_conversation(convo["path"])
                    if messages:
                        st.session_state.chat_history = messages
                        st.session_state.loaded_conversation = convo["title"]
                        st.rerun()
                st.caption(f"{convo['message_count']} messages • {convo['timestamp']}")
            
            with cols[1]:
                if st.button("🗑️", key=f"delete_{i}", help="Delete this conversation"):
                    if delete_conversation(convo["path"]):
                        st.session_state.delete_success = True
                        st.rerun()
            
            if convo["website_url"]:
                st.caption(f"URL: {convo['website_url']}")

    st.markdown("</div>", unsafe_allow_html=True)
    
# Main content area
if website_url and website_url != st.session_state.get("last_url", ""):
    st.session_state.last_url = website_url
    st.session_state.thinking = True
    
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
                
                st.session_state.chat_history.append(
                    AIMessage(content="I've analyzed the website content. You can now ask me questions about it!")
                )
            else:
                st.error("Could not extract content from the website.")
        except Exception as e:
            st.error(f"Error processing website: {str(e)}")
    
    st.session_state.thinking = False
    st.rerun()

# Display welcome or chat interface - CHATGPT STYLE LAYOUT
if not st.session_state.get("vector_store"):
    # Welcome screen with ChatGPT-style layout
    st.markdown("""
    <div class="welcome-container">
        <div style="width: 80px; height: 80px; border-radius: 50%; background: var(--card); display: flex; align-items: center; justify-content: center; box-shadow: 6px 6px 12px var(--shadow-dark), -6px -6px 12px var(--shadow-light); margin: 0 auto 1rem;">
            <div style="font-size: 2rem; font-weight: 600; background: linear-gradient(135deg, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">SQ</div>
        </div>
        <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem; color: var(--text);">SmartQuery Bot</h1>
        <p style="color: var(--text-secondary); margin-bottom: 2rem; font-size: 1rem;">Enter a website URL in the sidebar to begin analyzing content</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
            <div style="background-color: var(--card); border-radius: 12px; padding: 1.2rem; box-shadow: 4px 4px 12px var(--shadow-dark), -4px -4px 12px var(--shadow-light);">
                <h4 style="color: var(--primary); margin: 0 0 0.5rem 0;">Website Analysis</h4>
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Explore web content with AI assistance</p>
            </div>
            <div style="background-color: var(--card); border-radius: 12px; padding: 1.2rem; box-shadow: 4px 4px 12px var(--shadow-dark), -4px -4px 12px var(--shadow-light);">
                <h4 style="color: var(--accent); margin: 0 0 0.5rem 0;">Smart Summaries</h4>
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Get concise overviews of any content</p>
            </div>
            <div style="background-color: var(--card); border-radius: 12px; padding: 1.2rem; box-shadow: 4px 4px 12px var(--shadow-dark), -4px -4px 12px var(--shadow-light);">
                <h4 style="color: var(--warning); margin: 0 0 0.5rem 0;">Deep Explanations</h4>
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Click "Elaborate" for detailed insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
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
            <div class="ai-thinking">
                <div class="ai-thinking-orb"></div>
                <div class="ai-thinking-orb"></div>
                <div class="ai-thinking-orb"></div>
                <span style="margin-left: 12px; color: var(--text-secondary);">Analyzing content...</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Close the chat container first
    st.markdown("</div>", unsafe_allow_html=True)
    
    # REVOLUTIONARY CUSTOM SEARCH BAR - STREAMLIT COMPATIBLE - MOVED OUTSIDE CHAT CONTAINER
    if st.session_state.get("vector_store"):
        # Initialize search state
        if "show_suggestions" not in st.session_state:
            st.session_state.show_suggestions = False
        if "current_query" not in st.session_state:
            st.session_state.current_query = ""
        if "selected_suggestion" not in st.session_state:
            st.session_state.selected_suggestion = ""
            
        # Search suggestions
        search_suggestions = [
            "What is the main topic of this website?",
            "Summarize the key points from this page",
            "What are the most important details?",
            "Explain this content in simple terms",
            "What specific information is available here?",
            "How does this relate to my interests?"
        ]
        
        # Create the floating search bar container - properly fixed at bottom
        st.markdown('<div class="smart-search-container"><div class="smart-search-wrapper">', unsafe_allow_html=True)
        
        # Create columns for the search interface with tighter spacing for action buttons
        search_col1, search_col2, search_col3, search_col4 = st.columns([0.08, 0.78, 0.07, 0.07])
        
        with search_col1:
            suggestions_clicked = st.button("💡", help="Show suggestions", key="show_suggestions_btn")
            if suggestions_clicked:
                st.session_state.show_suggestions = not st.session_state.show_suggestions
        
        with search_col2:
            # Use selected suggestion as default value if available
            default_value = st.session_state.get("selected_suggestion", "")
            if default_value and "main_search_input" not in st.session_state:
                st.session_state.main_search_input = default_value
                st.session_state.selected_suggestion = ""  # Clear after using
            
            user_query = st.text_input(
                "Search",
                placeholder="Ask anything about this website content...",
                label_visibility="collapsed",
                key="main_search_input",
                help="Type your question and press Enter or click Send"
            )
        
        with search_col3:
            clear_clicked = st.button("🗑️", help="Clear input", key="clear_btn")
            if clear_clicked:
                # Clear by rerunning without the value
                if "main_search_input" in st.session_state:
                    del st.session_state["main_search_input"]
                st.rerun()
        
        with search_col4:
            send_clicked = st.button("✈️", help="Send message", key="send_btn", type="primary")
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Show suggestions if enabled
        if st.session_state.show_suggestions:
            st.markdown("### 💡 Try asking:")
            suggestion_cols = st.columns(2)
            
            for i, suggestion in enumerate(search_suggestions):
                with suggestion_cols[i % 2]:
                    suggestion_clicked = st.button(f"💡 {suggestion}", key=f"suggestion_{i}", use_container_width=True)
                    if suggestion_clicked:
                        # Store suggestion in a separate state variable
                        st.session_state.selected_suggestion = suggestion
                        st.session_state.show_suggestions = False
                        st.rerun()
        
        # Process the query
        if user_query and (send_clicked or user_query != st.session_state.current_query):
            st.session_state.current_query = user_query
            st.session_state.show_suggestions = False
            
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            with st.chat_message("AI"):
                with st.spinner("Thinking..."):
                    response = get_response(user_query)
            
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()
    
    # Fallback message if no vector store
    else:
        st.markdown("""
        <div class="smart-search-container" style="opacity: 0.5; pointer-events: none;">
            <div class="smart-search-wrapper">
                <div style="display: flex; align-items: center; gap: 15px; padding: 15px 25px;">
                    <div style="width: 24px; height: 24px; background: var(--text-muted); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px;">🔍</div>
                    <div style="flex: 1; color: var(--text-muted); font-style: italic;">
                        Enter a website URL first to start chatting...
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Add creator signature
st.markdown("""
<div class="creator-signature">
    <div class="avatar-mini">V</div>
    <span>Crafted by Vaibhav Nagre</span>
</div>
""", unsafe_allow_html=True)