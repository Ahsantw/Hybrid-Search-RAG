from src.convert_llama_to_open import OpenVINOLLMLoader
from src.log_setup import setup_logger
from src.vector_db import PDFVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose
import yaml
import re
import time
set_verbose(False)
# Configure logging
logger = setup_logger(__name__, '')
logger.info(f"-----------------------STARTED LOGGING---------------------------")

def main(logger):
    """
    1. Loads an OpenVINO quantized LLM (loads from disk or converts from HF).
    2. Creates a FAISS vector store from all PDF files in the configured folder.
    3. Loads the vector store for retrieval.
    4. Defines a custom prompt template using special tokens.
    5. Enters a loop to:
        - Accept user questions.
        - Retrieve relevant document chunks.
        - Get a response from the LLM.
        - Print and log the answer.
    """
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)["vector_db"]

    loader = OpenVINOLLMLoader(logger)
    llm_model = loader.load_openvino_llm()


    db_class = PDFVectorStore(logger)
    db = db_class.read_db()


    prompt_template = PromptTemplate.from_template(
            """<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are an expert assistant. Use the following extracted parts of a document to answer the question accurately and concisely. 
            If the answer is not found, say you don't know. Don't try to make up an answer.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Context:
            {context}

            Question: {question}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
            )

    while True:
        question = input("Question ('exit'): ")
        logger.info(f'Question {question}')
        if question.strip().lower() == "exit":
            break
        st = time.time()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": config["k"]}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True,
            verbose=False,
        )

        result = qa_chain.invoke(question)
        answer = result['result']
        for doc in result['source_documents']:
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "N/A")
            print(f" - Page {page}, File: {source}")
            logger.info(f" - Page {page}, File: {source}")

        logger.info(f"Answer {answer}")
        print(f"Answer: {answer}")
        et = time.time()
        print('Response Time:',(et-st))


if __name__ == "__main__":
    try:
        main(logger)
    except Exception as e:
        logger.error(f"Failed to Run the pipeline : {e}")
