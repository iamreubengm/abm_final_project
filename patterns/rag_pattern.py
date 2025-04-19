import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

class RAGPattern:
    """
    Implementation of the Retrieval Augmented Generation (RAG) pattern.
    
    This pattern enhances LLM responses by retrieving relevant information from a knowledge base
    and incorporating it into the context provided to the LLM.
    """
    
    def __init__(self, knowledge_base_dir: str = "data/knowledge_base"):
        """
        Initialize the RAG pattern with a knowledge base directory.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base documents
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.documents = []
        self.document_embeddings = None
        self.index = None
        self.embedding_model = None
        
        # Create the knowledge base directory if it doesn't exist
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        logger.info(f"Initialized RAG pattern with knowledge base directory: {knowledge_base_dir}")
    
    def load_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def load_knowledge_base(self):
        """
        Load documents from the knowledge base directory and create embeddings.
        """
        if self.embedding_model is None:
            self.load_model()
        
        self.documents = []
        
        # Load all JSON and text files in the knowledge base directory
        for file_path in self.knowledge_base_dir.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".json", ".txt", ".md"]:
                try:
                    if file_path.suffix == ".json":
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                            # Handle different document formats
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and "content" in item:
                                        self.documents.append({
                                            "content": item["content"],
                                            "source": str(file_path),
                                            "metadata": {k: v for k, v in item.items() if k != "content"}
                                        })
                            elif isinstance(data, dict) and "documents" in data:
                                for doc in data["documents"]:
                                    if isinstance(doc, dict) and "content" in doc:
                                        self.documents.append({
                                            "content": doc["content"],
                                            "source": str(file_path),
                                            "metadata": {k: v for k, v in doc.items() if k != "content"}
                                        })
                            elif isinstance(data, dict) and "content" in data:
                                self.documents.append({
                                    "content": data["content"],
                                    "source": str(file_path),
                                    "metadata": {k: v for k, v in data.items() if k != "content"}
                                })
                    else:
                        # For text and markdown files, read the content directly
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            self.documents.append({
                                "content": content,
                                "source": str(file_path),
                                "metadata": {"format": file_path.suffix[1:]}
                            })
                except Exception as e:
                    logger.error(f"Error loading document {file_path}: {str(e)}")
        
        if not self.documents:
            logger.warning(f"No documents found in knowledge base directory: {self.knowledge_base_dir}")
            return
        
        logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
        
        # Create embeddings for all documents
        try:
            # Split documents into chunks if they're too long
            chunked_documents = []
            for doc in self.documents:
                content = doc["content"]
                # Simple chunking by paragraphs
                paragraphs = content.split("\n\n")
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Skip empty paragraphs
                        chunked_documents.append({
                            "content": paragraph,
                            "source": doc["source"],
                            "metadata": {**doc["metadata"], "chunk_id": i}
                        })
            
            self.documents = chunked_documents
            logger.info(f"Split documents into {len(self.documents)} chunks")
            
            # Create embeddings
            contents = [doc["content"] for doc in self.documents]
            self.document_embeddings = self.embedding_model.encode(contents)
            
            # Create FAISS index for fast similarity search
            embedding_dim = self.document_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.index.add(np.array(self.document_embeddings).astype('float32'))
            
            logger.info(f"Created document embeddings and search index")
        except Exception as e:
            logger.error(f"Error creating document embeddings: {str(e)}")
            raise
    
    def add_document(self, content: str, source: str = "user_input", metadata: Dict[str, Any] = None):
        """
        Add a new document to the knowledge base.
        
        Args:
            content: Document content text
            source: Source identifier for the document
            metadata: Additional metadata for the document
        """
        if self.embedding_model is None:
            self.load_model()
        
        if metadata is None:
            metadata = {}
        
        # Add the document to the collection
        self.documents.append({
            "content": content,
            "source": source,
            "metadata": metadata
        })
        
        # Update embeddings and index
        embedding = self.embedding_model.encode([content])[0].astype('float32').reshape(1, -1)
        
        if self.document_embeddings is None:
            self.document_embeddings = embedding
            embedding_dim = embedding.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embedding])
        
        self.index.add(embedding)
        
        # Optionally save to disk
        file_path = self.knowledge_base_dir / f"{source.replace('/', '_')}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "content": content,
                    "source": source,
                    "metadata": metadata
                }, f, indent=2)
            logger.info(f"Saved new document to {file_path}")
        except Exception as e:
            logger.error(f"Error saving document to {file_path}: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with relevance scores
        """
        if self.embedding_model is None or self.index is None:
            self.load_knowledge_base()
        
        if not self.documents:
            logger.warning("No documents in knowledge base to retrieve from")
            return []
        
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0].astype('float32').reshape(1, -1)
        
        # Search for similar documents
        k = min(top_k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Ensure valid index
                doc = self.documents[idx]
                results.append({
                    "content": doc["content"],
                    "source": doc["source"],
                    "metadata": doc["metadata"],
                    "relevance": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to relevance score
                })
        
        return results
    
    def generate_context(self, query: str, top_k: int = 3) -> str:
        """
        Generate a context string from relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of top documents to include
            
        Returns:
            Context string with relevant information
        """
        retrieved_docs = self.retrieve(query, top_k)
        
        if not retrieved_docs:
            return ""
        
        # Format the context string
        context_parts = ["Here is some relevant information that might help:"]
        
        for i, doc in enumerate(retrieved_docs):
            source = doc["source"].split("/")[-1] if "/" in doc["source"] else doc["source"]
            relevance = doc["relevance"]
            content = doc["content"].strip()
            
            # Add the document content with source information
            context_parts.append(f"\nSource {i+1} ({source}, relevance: {relevance:.2f}):")
            context_parts.append(content)
        
        return "\n".join(context_parts)
    
    def enhance_prompt(self, query: str, original_prompt: str, top_k: int = 3) -> str:
        """
        Enhance an original prompt with retrieved information.
        
        Args:
            query: Query text to retrieve relevant information
            original_prompt: Original prompt to be enhanced
            top_k: Number of top documents to include
            
        Returns:
            Enhanced prompt with retrieved information
        """
        context = self.generate_context(query, top_k)
        
        if not context:
            return original_prompt
        
        # Add the context to the original prompt
        enhanced_prompt = (
            f"{original_prompt}\n\n"
            f"{context}\n\n"
            f"Please use the above information to provide a more accurate and informed response. "
            f"If the information doesn't contain what you need, rely on your general knowledge."
        )
        
        return enhanced_prompt
    
    def save_knowledge_base_metadata(self):
        """
        Save metadata about the knowledge base.
        """
        metadata = {
            "document_count": len(self.documents),
            "sources": list(set(doc["source"] for doc in self.documents)),
            "embedding_model": self.embedding_model.__class__.__name__ if self.embedding_model else None,
            "last_updated": datetime.now().isoformat()
        }
        
        metadata_path = self.knowledge_base_dir / "metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved knowledge base metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge base metadata: {str(e)}")