from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader,JSONLoader,UnstructuredExcelLoader,UnstructuredFileLoader,YoutubeLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter,RecursiveJsonSplitter,CharacterTextSplitter
from langchain.schema import Document
import requests


# from langchain_unstructured import UnstructuredLoader


def get_text_chunks(text):
    """
    Split document into smaller sized chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=50)
    chunks = text_splitter.split_documents(text) 
    return chunks 


def load_pdf_pages(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load() 
    chunks=get_text_chunks(data)
    return chunks
    
def load_text_files(file_path):
    loader = TextLoader(file_path)
    text=loader.load()
    chunks=get_text_chunks(text)
    return chunks

def load_docs_files(file_path):
    loader = UnstructuredFileLoader(file_path)
    text=loader.load()
    chunks=get_text_chunks(text)
    return chunks

def load_csv_data(file_path):
    loader = CSVLoader(file_path)
    text = loader.load()
    chunks=get_text_chunks(text)
    return chunks
def load_excel_data(file_path):
    loader = UnstructuredExcelLoader(file_path)
    text = loader.load()
    chunks=get_text_chunks(text)
    return chunks

def chat_with_youtube_videos(url):
    loader = YoutubeLoader.from_youtube_url(
    url, add_video_info=False
    )
    loaded_data = loader.load()
    chunks = get_text_chunks(loaded_data)
    return chunks

def chat_with_api_data(url):
    response = requests.get(url).json()
    data_dict = {item["id"]: item for item in response}
    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    json_chunks = splitter.split_json(json_data=data_dict)
    docs = []
    for idx, chunk in enumerate(json_chunks):
        document = Document(
            metadata={
                "source": "C:\\Users\\shreyab\\Desktop\\Python RAG Practice\\employees.json",
                "seq_num": idx + 1,
            },
            page_content=str(chunk)
        )
        docs.append(document)
    return docs
def load_json_file_data(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".Employees[]",
        text_content=False,
    )
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = splitter.split_documents(docs)
    return texts