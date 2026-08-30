from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import yaml
class PDFVectorStore:
    def __init__(self,logger):
        """
        Initialize the pdf vector store using config.yaml.

        - data_folder: Path to the folder containing PDF files.
        - save_path: Directory where the FAISS vector database will be saved.
        - chunk_size: Size of each document chunk in characters.
        - chunk_overlap: Number of overlapping characters between chunks.
        - model_name: HuggingFace model name used for generating embeddings.
        """
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)["vector_db"]
        
        self.data_folder = config["data_folder"]
        self.save_path = config["save_path"]
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.model_name = config["model_name"]

        os.makedirs(self.save_path, exist_ok=True)
        self.logger = logger
        self.logger.info(f"Loading Embeding Model {self.model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
    
    def create_db(self):
        """
        Create a FAISS vector database from all PDF files in the data folder.

        Actions:
            - Reads and loads text content from all PDFs.
            - Splits the content into chunks using specified chunk size and overlap.
            - Converts the chunks into embeddings using the embedding model.
            - Saves the resulting FAISS vector store to disk at the configured save path.
        """
        documents = []
        self.logger.info(f"Reading all the pdfs from {self.data_folder}")
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.data_folder, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        self.logger.info(f"Dividing text into chunk of size {self.chunk_size} with overlap of {self.chunk_overlap}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        
        self.logger.info(f"Storing into DB {self.model_name}")
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        vectorstore.save_local(self.save_path)
    
    def read_db(self):
        """
        Load an existing FAISS vector database from disk.
        """
        vectorstore = FAISS.load_local(
        self.save_path,
        self.embedding_model,
        allow_dangerous_deserialization=True)
        return vectorstore

if __name__ == "__main__":
    from log_setup import setup_logger
    logger = setup_logger(__name__, '')
    logger.info(f"-----------------------STARTED LOGGING---------------------------")
    db_class = PDFVectorStore(logger)
    db_class.create_db()
    
