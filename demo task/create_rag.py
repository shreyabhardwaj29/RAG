from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY is not set in the .env file.")
os.environ["NVIDIA_API_KEY"] = api_key


def get_rag_chain_with_vector_store(text_chunks):
    """
    Create a chain that takes a question, retrieves relevant documents, 
    constructs a prompt, passes that to a model, and retrieves from a vector store.
    """
    # Step 1: Create vector store from document chunks
    embeddings = NVIDIAEmbeddings()
    db = FAISS.from_documents(text_chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # Step 2: Define the RAG chain for question answering
    llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", temperature=0)
    template = """SYSTEM: You are a question answer bot. 
                 Be on point in your response and do not include what is not asked.
                 Respond to the following question: {question} only from 
                 the below context :{context}. 
                 If you don't know the answer, just say that you don't know.
               """
    rag_prompt_custom = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    print("RAG chain Created")
    
    return rag_chain

def get_rag_chain_for_api_data(docs):
    print(docs,"docs")
    # vectorstore = Chroma.from_documents(documents=docs, collection_name='abc', embedding = NVIDIAEmbeddings())
    vectorstore = FAISS.from_documents(docs, NVIDIAEmbeddings())
    print(vectorstore,"vector")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print(retriever)
    llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct", temperature=0)
    template= """SYSTEM: You are a question answer bot. 
                 Be on point to the answer and don't add this 'According to the provided context' text in response.
                 Understand the context provided and answer accordingly.
                 Respond to the following question: {question} only from 
                 the below context :{context}. 
                 If you don't know the answer, just say that you don't know.
               """
    rag_prompt_custom = PromptTemplate.from_template(template)
    rag_chain =(
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
        )
    return rag_chain
def get_rag_chain_for_json_data(json_chunks):
    vectorstore = FAISS.from_documents(json_chunks, NVIDIAEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct", temperature=0)
    template= """SYSTEM: You are a question answer bot. 
                 Be on point with your response.
                 The context has data of employees such as job title, first name, last name, 
                 phone number, email, region, user id, company, description  etc.
                 If asked to brief about the company where an employee works, give the description about the company.
                 Respond to the following question: {question} only from 
                 the below context :{context}. 
                 If you don't know the answer, just say that you don't know the answer and the question is out of context that is provided and be specific.Also, do not include any other information from the provided context.
               """
    rag_prompt_custom = PromptTemplate.from_template(template)
    rag_chain =(
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | StrOutputParser()
    )
    return rag_chain

def get_rag_chain_for_youtube_data(json_chunks):
    vectorstore = FAISS.from_documents(json_chunks, NVIDIAEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct", temperature=0)
    template= """SYSTEM: You are a question answer bot. 
                 Be on point with your response.
                 With the context, answer for what is being asked if the answer is present in the given context.
                 Respond to the following question: {question} only from 
                 the below context :{context}. 
                 If you don't know the answer, just say that you don't know the answer and the question is out of context that is provided and be specific.Also, do not include any other information from the provided context.
               """
    rag_prompt_custom = PromptTemplate.from_template(template)
    rag_chain =(
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | StrOutputParser()
    )
    return rag_chain