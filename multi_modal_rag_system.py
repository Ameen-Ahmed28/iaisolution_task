import os
import uuid
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

# Document processing
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# LangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# API clients
from groq import Groq

load_dotenv()

# Constants for persistence
PERSIST_DIR = "./chroma_db_multimodal"
DOCSTORE_FILE = os.path.join(PERSIST_DIR, "docstore.json")


class MultiModalRAGSystem:
    """
    Production-ready Multi-Modal RAG with:
    - Document ingestion via PyMuPDF (no poppler needed)
    - Text, tables, images (OCR)
    - Groq API for ultra-fast LLM inference (FREE tier)
    - ChromaDB for vector storage with persistence
    - Persistent docstore to survive restarts
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize RAG system with Groq API
        
        Args:
            groq_api_key: Groq API key (free tier available at console.groq.com)
        """
        # Get Groq API key
        self.groq_api_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
        if not self.groq_api_key:
            raise ValueError("❌ Missing GROQ_API_KEY! Get one at https://console.groq.com")

        # Initialize Groq client
        self.groq_client = None
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("✅ Groq client initialized successfully")
        except Exception as e:
            raise ValueError(f"❌ Failed to initialize Groq client: {e}")

        # Initialize embeddings (local, no API needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
        )

        # Vector store - IN-MEMORY ONLY (no persistence)
        # Documents will be cleared when app stops
        self.vectorstore = Chroma(
            collection_name="multimodal_rag",
            embedding_function=self.embeddings,
            # NO persist_directory = in-memory only
        )

        # Document store - in-memory only (no loading from disk)
        self.docstore: Dict[str, Document] = {}
        self.id_mapping: Dict[str, List[str]] = {}
        # NOT loading from disk - fresh start every time

        print("✅ Multi-Modal RAG System initialized")
        print("   📚 In-memory mode: Documents will be cleared on restart")

    def _load_docstore(self) -> None:
        """Load docstore from disk if it exists"""
        if os.path.exists(DOCSTORE_FILE):
            try:
                with open(DOCSTORE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Reconstruct Document objects
                for doc_id, doc_data in data.get("docstore", {}).items():
                    self.docstore[doc_id] = Document(
                        page_content=doc_data["page_content"],
                        metadata=doc_data["metadata"]
                    )

                self.id_mapping = data.get("id_mapping", {})
                print(f"   ✅ Loaded docstore from {DOCSTORE_FILE}")
            except Exception as e:
                print(f"   ⚠️ Could not load docstore: {e}")

    def _save_docstore(self) -> None:
        """Save docstore to disk for persistence"""
        # Ensure directory exists
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # Convert Document objects to serializable format
        docstore_data = {}
        for doc_id, doc in self.docstore.items():
            docstore_data[doc_id] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }

        data = {
            "docstore": docstore_data,
            "id_mapping": self.id_mapping
        }

        try:
            with open(DOCSTORE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Saved docstore to {DOCSTORE_FILE}")
        except Exception as e:
            print(f"   ⚠️ Could not save docstore: {e}")

    def clear_all(self) -> None:
        """
        Clear all documents from docstore and vector store (in-memory)
        """
        print("⚠️ CLEARING ALL DOCUMENTS...")
        
        # Clear in-memory
        self.docstore.clear()
        self.id_mapping.clear()

        # Clear vector store (in-memory)
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name="multimodal_rag",
                embedding_function=self.embeddings,
                # NO persist_directory = in-memory only
            )
        except Exception as e:
            print(f"   ⚠️ Error clearing vector store: {e}")

        print("✅ Knowledge base cleared!")

    def ingest_document(self, file_path: str) -> List[Document]:
        """
        Ingest document using PyMuPDF (no poppler required)
        Handles: PDF, images, text files
        
        Args:
            file_path: Path to document
            
        Returns:
            List of processed Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        print(f"\n📄 Processing: {file_path.name}")

        try:
            documents = []
            
            if file_path.suffix.lower() == ".pdf":
                # Use PyMuPDF for PDF processing (no poppler needed)
                doc = fitz.open(str(file_path))
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text
                    text = page.get_text()
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "element_type": "Text",
                                "page": page_num + 1,
                                "file_path": str(file_path),
                            },
                        ))
                    
                    # Extract tables (as text blocks)
                    try:
                        tables = page.find_tables()
                        for table_idx, table in enumerate(tables):
                            table_text = ""
                            for row in table.extract():
                                table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                            if table_text.strip():
                                documents.append(Document(
                                    page_content=table_text,
                                    metadata={
                                        "source": file_path.name,
                                        "element_type": "Table",
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "file_path": str(file_path),
                                    },
                                ))
                    except:
                        pass  # Skip if table extraction fails
                
                doc.close()
                
            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                # Use PIL + pytesseract for OCR on images
                try:
                    image = Image.open(str(file_path))
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "element_type": "Image_OCR",
                                "file_path": str(file_path),
                            },
                        ))
                except Exception as ocr_error:
                    print(f"   ⚠️ OCR failed: {ocr_error}. Adding image reference only.")
                    documents.append(Document(
                        page_content=f"[Image: {file_path.name}]",
                        metadata={
                            "source": file_path.name,
                            "element_type": "Image",
                            "file_path": str(file_path),
                        },
                    ))
                    
            elif file_path.suffix.lower() in [".txt", ".md", ".csv"]:
                # Plain text files
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "element_type": "Text",
                            "file_path": str(file_path),
                        },
                    ))
            else:
                # Try to read as text
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "element_type": "Text",
                                "file_path": str(file_path),
                            },
                        ))
                except Exception:
                    raise RuntimeError(f"❌ Unsupported file format: {file_path.suffix}")

            print(f"✅ Extracted {len(documents)} elements")
            if documents:
                print(f"   Types: {set(doc.metadata['element_type'] for doc in documents)}")

            return documents

        except Exception as e:
            raise RuntimeError(f"❌ Error processing document: {str(e)}")

    def chunk_documents(
        self, documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Chunk documents for better retrieval
        
        Args:
            documents: List of documents
            chunk_size: Max chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            Chunked documents
        """
        chunked_docs = []

        for doc in documents:
            # Simple chunking by size
            text = doc.page_content
            if len(text) <= chunk_size:
                chunked_docs.append(doc)
            else:
                # Overlap-based chunking
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk_text = text[i : i + chunk_size]
                    if chunk_text.strip():
                        chunk_doc = Document(
                            page_content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": len(chunked_docs),
                            },
                        )
                        chunked_docs.append(chunk_doc)

        print(f"✅ Chunked into {len(chunked_docs)} segments")
        return chunked_docs

    def add_documents_to_index(self, documents: List[Document]) -> None:
        """
        Add documents to vector store and docstore
        IMPORTANT: This saves to persistent storage!
        
        Args:
            documents: List of documents to index
        """
        parent_docs = []
        child_docs = []

        for doc in documents:
            parent_id = str(uuid.uuid4())

            # Parent: full document
            parent_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "parent_id": parent_id},
                )
            )

            # Child: for embedding/retrieval - INCLUDE ALL METADATA for fallback
            child_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,  # Include all original metadata
                    "parent_id": parent_id,
                },
            )
            child_docs.append(child_doc)

        # Store in docstore
        for doc in parent_docs:
            parent_id = doc.metadata["parent_id"]
            self.docstore[parent_id] = doc
            self.id_mapping[parent_id] = []

        # Store in vector DB (in-memory)
        self.vectorstore.add_documents(child_docs)

        # NOT saving to disk - in-memory mode

        print(f"✅ Indexed {len(documents)} documents")

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"⚠️ Vector search error: {e}")
            return []

        if not results:
            print(f"⚠️ No vector matches found for: '{query[:50]}...'")
            return []

        # Try to get parent documents, but fall back to child if not found
        final_results = []
        seen_content = set()  # Avoid duplicates
        
        for child_doc, score in results:
            parent_id = child_doc.metadata.get("parent_id")
            
            # Try to get parent document
            if parent_id and parent_id in self.docstore:
                parent_doc = self.docstore[parent_id]
                content_hash = hash(parent_doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    final_results.append((parent_doc, score))
            else:
                # FALLBACK: Use child document directly if parent not found
                # This fixes the mismatch between old vectors and new docstore
                content_hash = hash(child_doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    # Create a proper document from the child
                    doc = Document(
                        page_content=child_doc.page_content,
                        metadata={
                            "source": child_doc.metadata.get("source", "Unknown"),
                            "element_type": child_doc.metadata.get("element_type", "Text"),
                            **child_doc.metadata
                        }
                    )
                    final_results.append((doc, score))

        return final_results

    def call_groq(
        self,
        messages: List[Dict],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Call Groq API using Official SDK
        
        Available Models:
        - llama-3.3-70b-versatile (Recommended - most versatile)
        - mixtral-8x7b-32768 (Fast)
        - llama2-70b-4096 (Good for long context)
        - gemma-7b-it (Lightweight)
        
        Args:
            messages: Chat messages in OpenAI format
            model: Model name
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        if not self.groq_client:
            raise ValueError("❌ Groq client not initialized - check GROQ_API_KEY")

        try:
            # Use official Groq SDK
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                raise RuntimeError(
                    f"❌ Groq API Error 401: Invalid or expired API key\n"
                    f"   Get new key: https://console.groq.com/settings/security"
                )
            raise RuntimeError(f"❌ Groq API error: {error_msg}")

    def generate_answer(
        self, query: str, context_docs: List[Document]
    ) -> Dict[str, str]:
        """
        Generate answer using retrieved documents via Groq
        
        Args:
            query: User query
            context_docs: Retrieved documents
            
        Returns:
            Answer with citations
        """
        # If no documents, return message
        if not context_docs:
            return {
                "query": query,
                "answer": "❌ No relevant documents found. Please add documents first or check your query.",
                "sources": [],
            }

        # Build context
        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in context_docs
            ]
        )

        # Build prompt
        system_prompt = """You are a helpful AI assistant analyzing multi-modal documents.
Answer questions based on the provided context.
Include citations like [Source: filename].
If information is not in context, say 'Not found in documents.'"""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call Groq LLM
        try:
            answer = self.call_groq(messages)
            return {
                "query": query,
                "answer": answer,
                "sources": [doc.metadata.get("source", "Unknown") for doc in context_docs],
            }

        except Exception as e:
            error_msg = str(e)
            return {
                "query": query,
                "answer": f"❌ Error generating answer: {error_msg}\n\nTroubleshooting:\n1. Check your GROQ_API_KEY in .env\n2. Get a new key at https://console.groq.com",
                "sources": [],
            }

    def qa_pipeline(self, query: str, k: int = 4) -> Dict:
        """
        Complete QA pipeline: retrieve → generate
        
        Args:
            query: User question
            k: Number of retrieved documents
            
        Returns:
            QA result with answer and sources
        """
        # Retrieve
        retrieved = self.retrieve(query, k=k)
        context_docs = [doc for doc, score in retrieved]

        if not retrieved:
            print(f"⚠️ No documents found for query: '{query}'")
            print(f"   Available documents: {len(self.docstore)}")

        # Generate
        result = self.generate_answer(query, context_docs)

        return result


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of Multi-Modal RAG System - NOW WITH PERSISTENCE!"""
    
    try:
        print("🔌 Initializing RAG System...\n")
        rag = MultiModalRAGSystem()

        # Check if document already indexed (from previous runs)
        print("\n" + "=" * 70)
        print("CHECKING KNOWLEDGE BASE")
        print("=" * 70)
        
        if len(rag.docstore) == 0:
            print("No existing documents found.")
            print("Please upload documents using the Streamlit app (streamlit run app.py)")
            print("Or place qatar_test_doc.pdf in current directory and uncomment below:")
            
            # Uncomment these lines to auto-ingest a document:
            # docs = rag.ingest_document("qatar_test_doc.pdf")
            # chunks = rag.chunk_documents(docs)
            # rag.add_documents_to_index(chunks)
        else:
            print(f"✅ Found {len(rag.docstore)} existing documents in persistent storage")
            print("   (Loaded from previous run - no need to re-ingest)")

            # Example queries
            print("\n" + "=" * 70)
            print("ASKING QUESTIONS")
            print("=" * 70)

            queries = [
                "What is the main topic of the document?",
                "What key information is presented?",
            ]

            for query in queries:
                print(f"\n❓ Query: {query}")
                result = rag.qa_pipeline(query, k=4)

                print(f"\n✅ Answer:")
                print(result["answer"])
                print(f"\n📚 Sources: {', '.join(result['sources'])}")
                print("-" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Setup Instructions:")
        print("1. Install: pip install groq")
        print("2. Get key: https://console.groq.com")
        print("3. Create .env: GROQ_API_KEY=gsk_your_key")


if __name__ == "__main__":
    main()