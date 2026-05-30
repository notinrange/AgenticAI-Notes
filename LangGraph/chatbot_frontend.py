import streamlit as st
from chatbot_backend import chatbot,get_threads, ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, ToolMessage,AIMessage,AIMessageChunk
import uuid

# ***************************************** UTILITY FUNCTIONS ************************************
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

# ***************************************** SESSION SETUP ******************************


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']  = get_threads()

if 'ingested_docs' not in st.session_state:
    st.session_state["ingested_docs"] = {}


add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# ***************************************** SIDEBAR UI ******************************
st.sidebar.title('LangGraph PDF ChatBot')
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")


if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type = "pdf")
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processes for this chat.")
    else:
        with st.sidebar.status("Indexing PDF", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label=" PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# st.sidebar.header('My Conversations')


# for thread_id in st.session_state['chat_threads'][::-1]:
#     if st.sidebar.button(str(thread_id)):
#         st.session_state['thread_id'] = thread_id
#         messages = load_conversation(thread_id)

#         temp_messages = []

#         for message in messages:
#             if isinstance(message, HumanMessage):
#                 role = 'user'
#             elif isinstance(message, (AIMessage, AIMessageChunk)):
#                 role = 'assistant'
#             else:
#                 continue

#             temp_messages.append({'role': role, 'content' : message.content})
        
#         st.session_state['message_history'] = temp_messages

# ***************************************** MAIN UI *************************************

st.title("Multi Utility Chatbot")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Type here')


if user_input:
    # add message to message_history
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # response = chatbot.invoke({'messages': [HumanMessage(content = user_input)]}, config = CONFIG)
    # ai_message = response['messages'][-1].content

    # # add message to message_history
    # st.session_state['message_history'].append({'role':'assistance', 'content':ai_message})
    # with st.chat_message('assistance'):
    #     st.text(ai_message)
    CONFIG = {
        'configurable' : {'thread_id' : st.session_state['thread_id']},
        'metadata' : {'thread_id' : st.session_state['thread_id']},
        'run_name' : 'char_turn'
        }



    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if metadata.get("langgraph_node")=="tools":
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                    if message_chunk.content:
                        yield str(message_chunk.content)

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed : {doc_meta.get('filename')}"
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()


if selected_thread:
    st.session_state['thread_id'] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []

    for msg in messages:
        role = 'user' if isinstance(msg,HumanMessage) else "assistant"
        temp_messages.append({'role':role, 'content' : msg.content})
    st.session_state['message_history'] = temp_messages
    st.session_state['ingested_docs'].setdefault(str(selected_thread),{})
    st.rerun()



    
