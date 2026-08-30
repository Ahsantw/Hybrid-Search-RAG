from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
import os
import yaml


class TopKRetriever(BaseRetriever):
    """Wraps a retriever and trims its results to its top k documents.

    EnsembleRetriever fuses each sub-retriever's top-k list via reciprocal
    rank fusion but returns the full union (up to k per sub-retriever), so
    this re-applies k after fusion.
    """

    retriever: BaseRetriever
    k: int

    def _get_relevant_documents(self, query, *, run_manager: CallbackManagerForRetrieverRun):
        docs = self.retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        return docs[: self.k]


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

    def get_hybrid_retriever(self, db, k, bm25_weight=0.5, dense_weight=0.5, fetch_k=None):
        """
        Build a hybrid retriever that combines keyword search (BM25) with
        embedding search (FAISS) over the same set of chunks, so exact terms
        (e.g. case names, citations) and semantically related passages both
        surface.

        - db: FAISS vectorstore returned by read_db().
        - k: Final number of chunks returned after fusion.
        - bm25_weight / dense_weight: Relative weight of each retriever in
          the ranked fusion.
        - fetch_k: Number of candidates each sub-retriever pulls *before*
          fusion (defaults to max(3*k, 8)). This must be wider than k: if
          each sub-retriever only contributes k candidates, a single
          lexically-close-but-wrong-document BM25 hit can bump a correct
          dense chunk out of the final top-k just by being scarce
          competition. Fetching more candidates first gives reciprocal rank
          fusion a large enough shared pool that one bad pick from either
          side can't dominate purely by scarcity.
        """
        if fetch_k is None:
            fetch_k = max(3 * k, 8)

        dense_retriever = db.as_retriever(search_kwargs={"k": fetch_k})

        all_docs = list(db.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(all_docs, k=fetch_k)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[bm25_weight, dense_weight],
        )
        return TopKRetriever(retriever=ensemble_retriever, k=k)

if __name__ == "__main__":
    from log_setup import setup_logger
    logger = setup_logger(__name__, '')
    logger.info(f"-----------------------STARTED LOGGING---------------------------")
    db_class = PDFVectorStore(logger)
    db_class.create_db()
    
