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
import requests
from groq import Groq

load_dotenv()


class MultiModalRAGSystem:
    """
    Production-ready Multi-Modal RAG with:
    - Document ingestion via PyMuPDF (no poppler needed)
    - Text, tables, images (OCR)
    - Groq API for fast inference (FREE tier)
    - OpenRouter API for diverse model access (FREE tier)
    - ChromaDB for vector storage
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        use_groq: bool = True,
    ):
        """
        Initialize RAG system with API keys from environment
        
        Args:
            groq_api_key: Groq API key (free tier available)
            openrouter_api_key: OpenRouter API key (free tier available)
            use_groq: Use Groq for LLM inference (recommended - fastest)
        """
        # BUG FIX #1: Strip whitespace from API keys
        self.groq_api_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.openrouter_api_key = (openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.use_groq = use_groq

        if not self.groq_api_key and not self.openrouter_api_key:
            raise ValueError(
                "❌ Missing API keys! Set GROQ_API_KEY or OPENROUTER_API_KEY in .env"
            )

        # Initialize Groq client with error handling
        self.groq_client = None
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("✅ Groq client initialized successfully")
            except Exception as e:
                print(f"⚠️ Groq client initialization failed: {e}")
                self.groq_client = None

        # Initialize embeddings (local, no API needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
        )

        # Vector store
        self.vectorstore = Chroma(
            collection_name="multimodal_rag",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db_multimodal",
        )

        # Document store
        self.docstore: Dict[str, Document] = {}
        self.id_mapping: Dict[str, List[str]] = {}

        print("✅ Multi-Modal RAG System initialized")
        print(f"   Groq API: {'✓' if self.groq_client else '✗'}")
        print(f"   OpenRouter API: {'✓' if self.openrouter_api_key else '✗'}")

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
                    print(f"⚠️ OCR failed: {ocr_error}. Adding image reference only.")
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

            # Child: for embedding/retrieval
            child_doc = Document(
                page_content=doc.page_content,
                metadata={"parent_id": parent_id},
            )
            child_docs.append(child_doc)

        # Store in docstore
        for doc in parent_docs:
            parent_id = doc.metadata["parent_id"]
            self.docstore[parent_id] = doc
            self.id_mapping[parent_id] = []

        # Store in vector DB
        self.vectorstore.add_documents(child_docs)

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
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Get unique parents
        parent_results = {}
        for child_doc, score in results:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id in self.docstore:
                if parent_id not in parent_results:
                    parent_results[parent_id] = (self.docstore[parent_id], score)

        return list(parent_results.values())

    def call_groq(
        self, 
        messages: List[Dict], 
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Call Groq API using Official SDK (RECOMMENDED - Cleaner & More Reliable)
        
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
            # Use official Groq SDK (matches their docs exactly)
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

    def call_openrouter(
        self, 
        messages: List[Dict], 
        model: str = "meta-llama/llama-2-7b-chat",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Call OpenRouter API - FIXED VERSION
        
        Free Models Available:
        - meta-llama/llama-2-7b-chat (Reliable free model)
        - mistralai/mistral-7b-instruct
        - gryphe/mychoice-13b
        
        Args:
            messages: Chat messages in OpenAI format
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        if not self.openrouter_api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not set in environment")

        try:
            # Use requests as per OpenRouter docs - FIXED
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "MultiModal-RAG",  # BUG FIX: Add title
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,  # BUG FIX: Use max_tokens for OpenRouter (not max_completion_tokens)
                    "top_p": 1.0,  # BUG FIX: Add top_p parameter
                },
                timeout=30,
            )

            if response.status_code == 401:
                raise RuntimeError(
                    f"❌ OpenRouter API Error 401: Invalid or expired API key\n"
                    f"   Get new key: https://openrouter.ai/account/api-keys"
                )
            elif response.status_code == 400:
                error_detail = response.text
                print(f"⚠️ OpenRouter Response: {error_detail}")
                raise RuntimeError(f"❌ OpenRouter Bad Request: {error_detail}")
            
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            raise RuntimeError(f"❌ OpenRouter API error: {error_msg}")

    def generate_answer(
        self, query: str, context_docs: List[Document], use_groq: Optional[bool] = None
    ) -> Dict[str, str]:
        """
        Generate answer using retrieved documents
        
        Args:
            query: User query
            context_docs: Retrieved documents
            use_groq: Use Groq (True) or OpenRouter (False)
            
        Returns:
            Answer with citations
        """
        use_groq = use_groq if use_groq is not None else self.use_groq

        # If no documents, return message
        if not context_docs:
            return {
                "query": query,
                "answer": "❌ No relevant documents found. Please add documents first.",
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

        # Call LLM with better error handling
        try:
            if use_groq:
                answer = self.call_groq(messages)
            else:
                answer = self.call_openrouter(messages)

            return {
                "query": query,
                "answer": answer,
                "sources": [doc.metadata.get("source", "Unknown") for doc in context_docs],
            }

        except Exception as e:
            error_msg = str(e)
            return {
                "query": query,
                "answer": f"❌ Error generating answer: {error_msg}\n\nTroubleshooting:\n1. Check your API key in .env\n2. Verify API key is valid\n3. Try switching providers (use_groq=False)",
                "sources": [],
            }

    def qa_pipeline(self, query: str, k: int = 4, use_groq: Optional[bool] = None) -> Dict:
        """
        Complete QA pipeline: retrieve → generate
        
        Args:
            query: User question
            k: Number of retrieved documents
            use_groq: Use Groq or OpenRouter
            
        Returns:
            QA result with answer and sources
        """
        # Retrieve
        retrieved = self.retrieve(query, k=k)
        context_docs = [doc for doc, score in retrieved]

        # Generate
        result = self.generate_answer(query, context_docs, use_groq)

        return result


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of Multi-Modal RAG System"""

    try:
        # Initialize
        print("🔌 Initializing RAG System...\n")
        rag = MultiModalRAGSystem(use_groq=True)

        # Example 1: Create sample documents
        sample_docs = [
            Document(
                page_content="Diabetic retinopathy is a serious complication affecting blood vessels in the retina. Early detection through OCR and imaging analysis is crucial for preventing vision loss.",
                metadata={
                    "source": "medical_report.pdf",
                    "element_type": "Text",
                    "page": 1,
                },
            ),
            Document(
                page_content="Table: Retinopathy Classification\nStage | Severity | Treatment\nMild | Microaneurysms | Monitoring\nModerate | Bleeding | Medication\nSevere | Macular edema | Laser/Injection",
                metadata={
                    "source": "medical_report.pdf",
                    "element_type": "Table",
                    "page": 2,
                },
            ),
            Document(
                page_content="Transfer learning leverages pre-trained models to solve new tasks faster. In medical imaging, pre-trained CNNs (ResNet, DenseNet) significantly improve accuracy on limited datasets.",
                metadata={
                    "source": "ml_guide.pdf",
                    "element_type": "Text",
                    "page": 1,
                },
            ),
        ]

        # Add to index
        rag.add_documents_to_index(sample_docs)

        # Example 2: QA
        print("\n" + "=" * 70)
        print("MULTI-MODAL RAG QA DEMO")
        print("=" * 70)

        queries = [
            "What is diabetic retinopathy and how is it treated?",
            "How does transfer learning help in medical imaging?",
            "What are the stages of retinopathy?",
        ]

        for query in queries:
            print(f"\n❓ Query: {query}")
            result = rag.qa_pipeline(query, k=3, use_groq=True)

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