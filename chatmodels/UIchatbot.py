import streamlit as st
from dotenv import load_dotenv
import time
from datetime import datetime

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Personality Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .stChatMessage p {
        color: #ffffff;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* User message - right aligned */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Chat Header */
    .chat-header {
        text-align: center;
        padding: 30px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .chat-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.5em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .chat-header p {
        color: #f0f0f0;
        font-size: 1.1em;
        margin-top: 10px;
    }
    
    /* Chat Input */
    .stChatInput {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------
@st.cache_resource
def get_model(temp):
    return ChatMistralAI(model="mistral-small-2506", temperature=temp)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: center;'>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Mode Selection
    st.markdown("<h3 style='color: white;'>🎭 AI Personality</h3>", unsafe_allow_html=True)
    mode_choice = st.selectbox(
        "Select Mode:",
        ["😡 Angry", "😂 Funny", "😢 Sad", "🧠 Professional", "🌟 Motivational", "🤓 Nerdy"],
        label_visibility="collapsed"
    )
    
    # Temperature Control
    st.markdown("<h3 style='color: white;'>🌡️ Creativity Level</h3>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.9, 0.1, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Stats
    st.markdown("<h3 style='color: white;'>📊 Chat Stats</h3>", unsafe_allow_html=True)
    if "messages" in st.session_state:
        msg_count = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
        st.metric("Messages Sent", msg_count)
    
    st.markdown("---")
    
    # Export Chat
    if st.button("💾 Export Chat"):
        if "messages" in st.session_state:
            chat_text = f"Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            for msg in st.session_state.messages:
                if isinstance(msg, HumanMessage):
                    chat_text += f"You: {msg.content}\n\n"
                elif isinstance(msg, AIMessage):
                    chat_text += f"Bot: {msg.content}\n\n"
            st.download_button("📥 Download", chat_text, "chat_history.txt", "text/plain")
    
    st.markdown("---")
    st.markdown("<p style='color: white; text-align: center; font-size: 12px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)


# ---------------- MODE MAPPING ----------------
mode_map = {
    "😡 Angry": "You are an angry AI agent. You respond aggressively and impatiently.",
    "😂 Funny": "You are a very funny AI agent. You respond with humor and jokes.",
    "😢 Sad": "You are a very sad AI agent. You respond in a depressed and emotional tone.",
    "🧠 Professional": "You are a professional AI assistant. You respond formally and provide detailed, well-structured answers.",
    "🌟 Motivational": "You are a motivational coach. You inspire and encourage with positive energy.",
    "🤓 Nerdy": "You are a nerdy AI who loves tech, science, and references. You use technical jargon and pop culture references."
}
mode = mode_map[mode_choice]
model = get_model(temperature)


# ---------------- MAIN CONTENT ----------------
st.markdown(f"""
<div class='chat-header'>
    <h1>🤖 AI Personality Chatbot</h1>
    <p>Currently in <strong>{mode_choice}</strong> mode | Experience AI with dynamic personalities</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SESSION MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=mode)]
    st.session_state.current_mode = mode
elif st.session_state.get("current_mode") != mode:
    # Update only the system message, keep chat history
    st.session_state.messages[0] = SystemMessage(content=mode)
    st.session_state.current_mode = mode


# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)


# ---------------- USER INPUT ----------------
user_input = st.chat_input("💬 Type your message here...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Get AI response with typing effect
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
            st.write(response.content)
    
    # Add AI message
    st.session_state.messages.append(AIMessage(content=response.content))
    st.rerun()


# ---------------- FOOTER ----------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=mode)]
        st.rerun()