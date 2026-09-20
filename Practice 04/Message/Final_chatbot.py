import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are a helpful, professional, and reliable AI assistant.

Your responsibilities:
- Answer the user's questions clearly and accurately.
- Keep explanations easy to understand.
- Give step-by-step explanations when appropriate.
- Use examples when they improve understanding.
- If you are unsure about something, say so instead of inventing information.
- Be concise for simple questions and detailed for complex questions.
- Maintain a polite and professional tone.
"""

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --------------------------------------------------
# Initialize Model
# --------------------------------------------------

try:
    model = ChatGroq(
        model=MODEL_NAME,
        temperature=0.7
    )

except Exception as e:
    st.error("Unable to initialize the AI model.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Initialize Chat History
# --------------------------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI Chatbot")
st.caption("Powered by Groq • LangChain • Streamlit")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Chat Settings")

    st.write("### Model")
    st.code(MODEL_NAME)

    st.write("### Conversation")

    # Count only user/AI messages
    visible_messages = [
        msg
        for msg in st.session_state.chat_history
        if isinstance(msg, (HumanMessage, AIMessage))
    ]

    user_messages = [
        msg
        for msg in visible_messages
        if isinstance(msg, HumanMessage)
    ]

    ai_messages = [
        msg
        for msg in visible_messages
        if isinstance(msg, AIMessage)
    ]

    st.metric(
        "Messages",
        len(visible_messages)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("You", len(user_messages))

    with col2:
        st.metric("AI", len(ai_messages))

    st.divider()

    # Clear conversation
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

        st.rerun()

    st.divider()

    st.write("### 💡 About")

    st.info(
        "This chatbot uses a Groq-hosted "
        "GPT-OSS model through LangChain."
    )

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.chat_history:

    # System message is only for the model
    if isinstance(message, SystemMessage):
        continue

    if isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "assistant"

    with st.chat_message(role):

        st.markdown(message.content)

# --------------------------------------------------
# User Input
# --------------------------------------------------

user_input = st.chat_input(
    "💬 Type your message here..."
)

# --------------------------------------------------
# Process User Message
# --------------------------------------------------

if user_input:

    user_input = user_input.strip()

    # Ignore empty messages
    if not user_input:
        st.warning("Please enter a message.")
        st.stop()

    # --------------------------------------------------
    # Add User Message
    # --------------------------------------------------

    user_message = HumanMessage(
        content=user_input
    )

    st.session_state.chat_history.append(
        user_message
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------
    # Generate AI Response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = model.invoke(
                    st.session_state.chat_history
                )

                ai_response = result.content

                # Store AI response
                st.session_state.chat_history.append(
                    AIMessage(content=ai_response)
                )

                # Display AI response
                st.markdown(ai_response)

            except Exception as e:

                st.error(
                    "Sorry, I couldn't generate a response."
                )

                st.exception(e)

                # Remove the last user message if
                # the model failed
                if (
                    st.session_state.chat_history
                    and isinstance(
                        st.session_state.chat_history[-1],
                        HumanMessage
                    )
                ):
                    st.session_state.chat_history.pop()