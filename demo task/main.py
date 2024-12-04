import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from get_data import load_csv_data,load_text_files,load_pdf_pages,load_excel_data,load_docs_files,chat_with_youtube_videos,chat_with_api_data,load_json_file_data
from create_rag import get_rag_chain_with_vector_store,get_rag_chain_for_api_data,get_rag_chain_for_json_data,get_rag_chain_for_youtube_data
import pandas as pd
import os

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "file_data" not in st.session_state:
    st.session_state.file_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  
if "chain" not in st.session_state:
    st.session_state.chain = None

st.title("Upload a file and ask anything about it")

# Sidebar for file upload
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file to start", type=["csv", "xlsx", "txt", "pdf","json"]
)
st.sidebar.write(st.session_state.file_data)
url = st.sidebar.text_input(label='Enter your url here')

user_query = st.chat_input("Ask your query")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    chain = st.session_state.chain
    response=chain.invoke(user_query)
    st.session_state.chat_history.append(AIMessage(content=response))
    print(response)
if url!="":
    if "youtube" in url:
        chunks = chat_with_youtube_videos(url)
        chain = get_rag_chain_for_youtube_data(chunks)
        st.session_state.chain = chain
    else:
        chunks = chat_with_api_data(url)
        chain = get_rag_chain_for_api_data(chunks)   
        st.session_state.chain = chain

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# Handle uploaded file
if uploaded_file:
    # Only process the file if it hasn't been processed or a new file is uploaded
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        file_name = uploaded_file.name
        save_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        print(f"File saved to {save_path}")

        chunks = []
        try:
            if uploaded_file.name.endswith(".csv"):
                text_content = pd.read_csv(uploaded_file)
                st.session_state.file_data = text_content
                st.sidebar.write(st.session_state.file_data)
                data = load_csv_data(save_path)
                chunks.extend(data)

            elif uploaded_file.name.endswith(".xlsx"):
                text_content = pd.read_excel(uploaded_file)
                st.session_state.file_data = text_content
                st.sidebar.write(text_content)
                data=load_excel_data(save_path)
                chunks.extend(data)

            elif uploaded_file.name.endswith(".txt"):
                text_content = uploaded_file.read().decode("utf-8")
                st.session_state.file_data = text_content
                st.sidebar.write(text_content)
                data = load_text_files(save_path)
                chunks.extend(data)

            elif uploaded_file.name.endswith(".docx"):
                print("hello")
                text_content = uploaded_file.read()
                # print(text_content)
                st.session_state.file_data = text_content
                st.sidebar.write(text_content)
                data = load_docs_files(save_path)
                chunks.extend(data)

            elif uploaded_file.name.endswith(".pdf"):
                pdf_content = uploaded_file.read()  
                st.session_state.file_data = pdf_content
                st.sidebar.write(pdf_content)
                data = load_pdf_pages(save_path)
                chunks.extend(data)

            elif uploaded_file.name.endswith(".json"):
                json_data = uploaded_file.read()
                st.session_state.file_data = json_data
                st.sidebar.write(json_data)
                json_data_chunks = load_json_file_data(save_path)
                chain = get_rag_chain_for_json_data(json_data_chunks)   
                st.session_state.chain = chain

            else:
                st.warning("Unsupported file type.")

            if chunks:
                chain = get_rag_chain_with_vector_store(chunks)
                st.session_state.chain = chain  # Save chain in session state
            st.session_state.uploaded_file_name = file_name

        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    if st.session_state.get("file_data") is None and url == "":
        st.info("Please upload a file or URL in the sidebar to get started.")

