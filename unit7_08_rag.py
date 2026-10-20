from dotenv import load_dotenv
import os
from anthropic import Anthropic
import streamlit as st

load_dotenv()

# CONFIGURATION
ROUTER_MODEL = "claude-haiku-4-5"
ANSWER_MODEL = "claude-haiku-4-5"

KNOWLEDGE_BASE_DIR = "data/knowledge_base"

KNOWN_FILES = ["harvard.txt", "cornell.txt", "duke.txt"]

NO_FILE = "none.txt"

BASE_SYSTEM_PROMPT = "You are a helpful assistant. Keep answers clear and concise."

GREETING = (
    "Hello! I have knowledge about Harvard, Cornell, and Duke universities. "
    "Ask me anything about these schools!"
)



# PAGE SETUP
st.set_page_config(page_title="Claude RAG Chatbot", page_icon="🔍", layout="wide")
st.title("🔍 Claude RAG Chatbot")
st.caption("Retrieve → Augment → Generate")
st.markdown("---")



# API CLIENT
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not provided. Add ANTHROPIC_API_KEY to your .env file.")
    st.stop()

client = Anthropic(api_key=api_key)


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

if "last_file" not in st.session_state:
    st.session_state.last_file = NO_FILE

def history_for_api(messages):
    """
    Return a copy of the conversation that is safe to send to the Anthropic API.

    Why this exists: we seed the chat with an assistant greeting so the UI is
    not empty on load. The Anthropic API requires the first message to be from
    the user. Sending the raw history would fail with a 400 error on the very
    first question.

    So we drop any assistant messages sitting at the front of the list. The rest
    of the history is passed through untouched, which is what gives the bot
    memory of earlier turns ("what about its acceptance rate?" still works).
    """
    trimmed = list(messages)
    while trimmed and trimmed[0]["role"] == "assistant":
        trimmed.pop(0)
    return trimmed

# STEP 1: RETRIEVE

ROUTER_SYSTEM_PROMPT = f"""You are a retrieval router for a question-answering system.

Your only job is to decide which single knowledge base file, if any, contains the information needed to answer the user's question.

Available files:
- harvard.txt : facts about Harvard University (history, campus, admissions, academics, student life)
- cornell.txt : facts about Cornell University (history, campus, admissions, academics, student life)
- duke.txt : facts about Duke University (history, campus, admissions, academics, student life)
- {NO_FILE} : the question is not about Harvard, Cornell, or Duke, or it is small talk or a general question

Rules:
- If the question is about exactly one of those three universities, return that university's file.
- If the question compares several of those universities, return the file for the one that is the main subject of the question.
- If the question is about any other school, any other topic, or is general conversation, return {NO_FILE}.
- Respond with the filename and nothing else. No explanation, no punctuation, no quotes, no markdown."""


def get_knowledge_file(prompt: str) -> str:
    """
    THE RETRIEVAL STEP.

    Ask a cheap model which knowledge base file is relevant to this question,
    then validate its answer before trusting it.

    Args:
        prompt: the raw text the user just typed

    Returns:
        One of KNOWN_FILES, or NO_FILE if nothing relevant was found.
    """
    try:
        response = client.messages.create(
            model=ROUTER_MODEL,

            max_tokens=16,
           
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip().lower()

    except Exception as err:

        st.warning(f"Retrieval step failed, answering without documents. ({err})")
        return NO_FILE


    for filename in KNOWN_FILES:
        if filename in raw:
            return filename

    return NO_FILE


def load_document(filename: str) -> str:
    """
    Read the retrieved knowledge base file off disk.

    Kept as its own function so the RAG flow below reads cleanly, and so the
    day you swap flat files for a database you only rewrite this one function.
    """
    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# STEP 2: AUGMENT

def build_grounded_system_prompt(filename: str, content: str) -> str:
    """
    THE AUGMENT STEP.

    Combine the retrieved document with instructions on how to use it, and
    return the finished system prompt.

    Args:
        filename: which file was retrieved (used so the bot can cite it)
        content: the full text of that file

    Returns:
        The system prompt string to hand to the generation call.
    """
    return f"""You are a helpful assistant answering questions about universities. Keep answers clear and concise.

Answer the user's question using ONLY the reference document provided below. Treat this document as the authoritative source, even if it conflicts with what you already believe.

If the answer is not contained in the document, say plainly that the document does not cover it, then make clear that anything further you add is general knowledge rather than sourced from the document. Never invent facts, numbers, dates, or quotes that do not appear in the document.

When a specific figure or fact comes from the document, you may mention that it comes from {filename}.

=== BEGIN REFERENCE DOCUMENT: {filename} ===
{content}
=== END REFERENCE DOCUMENT ===

Everything after this point is the conversation with the user."""


# SIDEBAR
with st.sidebar:
    st.header("📚 Knowledge Base Info")

    sidebar_file_info = st.empty()

    if st.session_state.last_file != NO_FILE:
        sidebar_file_info.success(f"**Last used:** {st.session_state.last_file}")
    else:
        sidebar_file_info.info("**No knowledge base file retrieved for this API call**")

    st.caption("The chatbot retrieves a relevant file before answering.")
    st.markdown("---")

    st.subheader("Available Knowledge:")
    for filename in KNOWN_FILES:
        exists = os.path.exists(os.path.join(KNOWLEDGE_BASE_DIR, filename))
        label = filename if exists else f"{filename} (missing)"
        st.markdown(f"- {'🎓' if exists else '⚠️'} **{label}**")

    st.markdown("---")
    st.caption(f"Router: {ROUTER_MODEL}")
    st.caption(f"Answer: {ANSWER_MODEL}")
    st.markdown("---")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.session_state.last_file = NO_FILE
        st.rerun()



# REPLAY CONVERSATION HISTORY

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


# MAIN RAG WORKFLOW

if prompt := st.chat_input("Ask a question about Harvard, Cornell, or Duke..."):

    # Record and display the user's turn immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Retrieving relevant knowledge..."):
        selected_file = get_knowledge_file(prompt)


    document_content = None

    if selected_file in KNOWN_FILES:
        try:
            document_content = load_document(selected_file)
            st.session_state.last_file = selected_file
            sidebar_file_info.success(f"**Last used:** {selected_file}")

        except FileNotFoundError:

            st.warning(
                f"Knowledge base file '{selected_file}' not found in "
                f"{KNOWLEDGE_BASE_DIR}/. Answering without it."
            )
            selected_file = NO_FILE
            st.session_state.last_file = NO_FILE
            sidebar_file_info.info("**No knowledge base file retrieved for this API call**")

    else:
        st.session_state.last_file = NO_FILE
        sidebar_file_info.info("**No knowledge base file retrieved for this API call**")


    if document_content is not None:
        system_prompt = build_grounded_system_prompt(selected_file, document_content)
    else:
        system_prompt = BASE_SYSTEM_PROMPT

    with st.chat_message("assistant"):
        try:
            with client.messages.stream(
                model=ANSWER_MODEL,
                max_tokens=1024,

                system=system_prompt,

                messages=history_for_api(st.session_state.messages),
            ) as stream:
                response_text = st.write_stream(stream.text_stream)

        except Exception as err:
            st.error(f"API call failed: {err}")

            st.session_state.messages.pop()
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response_text})

