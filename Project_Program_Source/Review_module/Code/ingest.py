
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


DATA_PATH="/Users/shilpabai/Downloads/DATA298A/Data/visio/data"
DB_FAISS_PATH="/Users/shilpabai/Downloads/DATA298A/Data/visio/vectorstores/db_faiss"
csv_file_path = "/Users/shilpabai/Downloads/DATA298A/Data/visio/data/sephora_reviews.csv"


def create_vector_db():
    #loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    # Load CSV data
    loader = CSVLoader(file_path=csv_file_path, encoding="utf-8", csv_args={'delimiter': ',', 'quotechar': '"'})
    documents =loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()