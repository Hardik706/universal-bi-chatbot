import os
import sys
import logging
from typing import List, Dict, Any

# Ensure parent directory is in the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings, get_llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Manages loading, indexing, and querying unstructured documents (PDFs, text)
    using PyMuPDF, RecursiveCharacterTextSplitter, sentence-transformers, and FAISS.
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store_path = settings.VECTOR_DB_DIR
        self._vector_store = None

    def _get_vector_store(self) -> FAISS | None:
        """
        Loads the existing FAISS index from disk, or returns None if not indexed.
        """
        if self._vector_store is not None:
            return self._vector_store
            
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                self._vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for loading local FAISS pickling
                )
                logger.info("Loaded existing FAISS vector store index.")
                return self._vector_store
            except Exception as e:
                logger.error(f"Error loading FAISS vector store index: {e}")
                
        return None

    def load_pdf_with_pymupdf(self, file_path: str) -> List[Document]:
        """
        Uses PyMuPDF (fitz) to extract text page-by-page from a PDF file.
        Wraps pages into LangChain Document objects with metadata.
        """
        import fitz  # PyMuPDF
        
        documents = []
        filename = os.path.basename(file_path)
        
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    metadata = {
                        "source": filename,
                        "file_path": file_path,
                        "page": page_num + 1,  # 1-indexed for reader display
                        "total_pages": len(doc)
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
            doc.close()
            logger.info(f"Loaded {len(documents)} pages from PDF: {filename}")
        except Exception as e:
            logger.exception(f"Error reading PDF {file_path} with PyMuPDF")
            
        return documents

    def ingest_document(self, file_path: str) -> bool:
        """
        Ingests a single document (PDF or TXT), splits it, adds to FAISS vector index,
        and saves the updated index to disk.
        """
        documents = []
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            documents = self.load_pdf_with_pymupdf(file_path)
        elif ext in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    metadata = {
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "page": 1,
                        "total_pages": 1
                    }
                    documents = [Document(page_content=content, metadata=metadata)]
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return False
        else:
            logger.warning(f"Unsupported file format for ingestion: {ext}")
            return False

        if not documents:
            logger.warning("No text extracted from document.")
            return False

        # Split documents using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(split_docs)} chunks.")

        # Load or initialize FAISS vector index
        vector_store = self._get_vector_store()
        if vector_store is None:
            # Create a new index
            self._vector_store = FAISS.from_documents(split_docs, self.embeddings)
            logger.info("Created new FAISS index.")
        else:
            # Add to existing index
            self._vector_store.add_documents(split_docs)
            logger.info("Added documents to existing FAISS index.")

        # Save to disk
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            self._vector_store.save_local(self.vector_store_path)
            logger.info(f"FAISS index successfully saved to: {self.vector_store_path}")
            return True
        except Exception as e:
            logger.exception("Failed to save FAISS vector store index")
            return False

    def query(self, query_text: str, k: int = 4) -> Dict[str, Any]:
        """
        Queries the vector store and synthesises a clean response using the configured LLM.
        """
        vector_store = self._get_vector_store()
        if vector_store is None:
            return {
                "answer": "No documents have been indexed yet. Please upload files in the UI to build the RAG knowledge base.",
                "source_documents": []
            }

        try:
            # Retrieve relevant chunks
            docs = vector_store.similarity_search(query_text, k=k)
            
            # Form context
            context_pieces = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                context_pieces.append(
                    f"--- Chunk {i+1} (Source: {source}, Page: {page}) ---\n{doc.page_content}"
                )
            context_str = "\n\n".join(context_pieces)

            # Invoke LLM
            llm = get_llm(temperature=0.0)
            
            system_prompt = (
                "You are an expert Enterprise BI Assistant.\n"
                "Answer the user's question using ONLY the provided documentation chunks. "
                "Be detailed, factual, and copy-paste key specifications if helpful.\n"
                "If the answer cannot be found in the provided context, clearly state that you do "
                "not have that information and cite what is missing.\n\n"
                f"Context Chunks:\n{context_str}"
            )
            
            messages = [
                ("system", system_prompt),
                ("user", query_text)
            ]
            
            response = llm.invoke(messages)
            
            # Format return dict
            source_docs_formatted = []
            for doc in docs:
                source_docs_formatted.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1),
                    "content": doc.page_content
                })

            return {
                "answer": response.content,
                "source_documents": source_docs_formatted
            }
        except Exception as e:
            logger.exception("RAG querying failed")
            return {
                "answer": f"An error occurred while retrieving information: {str(e)}",
                "source_documents": []
            }
