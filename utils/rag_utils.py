# utils/rag_utils.py
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import numpy as np
from datetime import datetime
import glob
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic

from config import FINANCIAL_KB_PATH, DEFAULT_MODEL, EMBEDDING_MODEL, VECTOR_DB_PATH

class FinancialRAG:
    """
    Implements Retrieval Augmented Generation (RAG) for financial information.
    
    This class provides utilities for creating, querying, and managing a knowledge base
    of financial information to enhance AI responses with domain-specific context.
    """
    
    def __init__(self, client: Optional[Anthropic] = None):
        """
        Initialize the FinancialRAG with an optional Anthropic client.
        
        Args:
            client: Optional Anthropic API client. If None, only retrieval will be available.
        """
        self.client = client
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Ensure directories exist
        os.makedirs(FINANCIAL_KB_PATH, exist_ok=True)
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # Initialize vector index and document store
        self.index = None
        self.documents = []
        self.document_embeddings = []
        
        # Load existing index if available
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing vector index or create a new one."""
        index_path = os.path.join(VECTOR_DB_PATH, "financial_kb.index")
        documents_path = os.path.join(VECTOR_DB_PATH, "financial_kb_documents.json")
        
        if os.path.exists(index_path) and os.path.exists(documents_path):
            # Load existing index
            self.index = faiss.read_index(index_path)
            
            # Load documents
            with open(documents_path, "r") as f:
                self.documents = json.load(f)
            
            print(f"Loaded existing index with {len(self.documents)} documents")
        else:
            # Create new index and documents from KB files
            self._create_index_from_kb()
    
    def _create_index_from_kb(self):
        """Create vector index from knowledge base documents."""
        # Get all text files in the KB directory
        kb_files = glob.glob(os.path.join(FINANCIAL_KB_PATH, "*.txt"))
        
        if not kb_files:
            print("No knowledge base files found. Creating empty index.")
            # Create empty index
            dimension = 384  # Dimension of the embedding model
            self.index = faiss.IndexFlatL2(dimension)
            self.documents = []
            return
        
        all_chunks = []
        documents = []
        
        # Process each file
        for file_path in kb_files:
            file_name = os.path.basename(file_path)
            
            try:
                # Read file
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Chunk the content into paragraphs
                chunks = self._chunk_text(content)
                
                # Add each chunk to documents with metadata
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    documents.append({
                        "source": file_name,
                        "chunk_id": i,
                        "content": chunk,
                        "created_at": datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        
        # Create embeddings for all chunks
        embeddings = self._get_embeddings(all_chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = documents
        
        # Save index and documents
        self._save_index()
        
        print(f"Created index with {len(documents)} chunks from {len(kb_files)} files")
    
    def _save_index(self):
        """Save the current index and documents to disk."""
        index_path = os.path.join(VECTOR_DB_PATH, "financial_kb.index")
        documents_path = os.path.join(VECTOR_DB_PATH, "financial_kb_documents.json")
        
        # Save index
        faiss.write_index(self.index, index_path)
        
        # Save documents
        with open(documents_path, "w") as f:
            json.dump(self.documents, f)
        
        print(f"Saved index with {len(self.documents)} documents")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Chunk text into smaller pieces for processing.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        # First try to split by paragraphs
        paragraphs = text.split("\n\n")
        
        # If paragraphs are too big, split them further
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too big, split it
            if len(paragraph) > chunk_size:
                # Add any current chunk to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split paragraph into sentences
                sentences = paragraph.split(". ")
                
                # Build chunks from sentences
                temp_chunk = ""
                for sentence in sentences:
                    if not sentence.endswith("."):
                        sentence += "."
                    
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += " " + sentence
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                
                # Add any remaining temp_chunk
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                # If adding paragraph would exceed chunk size, start a new chunk
                if len(current_chunk) + len(paragraph) > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Otherwise add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
        
        # Add final chunk if any
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Create overlapping chunks if needed
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    # Get the end of the previous chunk
                    prev_end = chunks[i-1][-overlap:]
                    # Combine with the current chunk
                    overlapped_chunks.append(prev_end + chunks[i])
            
            return overlapped_chunks
        else:
            return chunks
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.embedding_model.encode(texts, convert_to_numpy=True)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Numpy array of embedding
        """
        return self.embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)
    
    def add_document(self, content: str, source: str) -> bool:
        """
        Add a new document to the knowledge base.
        
        Args:
            content: Document content
            source: Document source or identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            # Chunk the content
            chunks = self._chunk_text(content)
            
            # Get embeddings
            embeddings = self._get_embeddings(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Add to documents
            doc_start_idx = len(self.documents)
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    "source": source,
                    "chunk_id": doc_start_idx + i,
                    "content": chunk,
                    "created_at": datetime.now().isoformat()
                })
            
            # Save updated index
            self._save_index()
            
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def query(self, query: str, n_results: int = 3) -> str:
        """
        Query the knowledge base for relevant context.
        
        Args:
            query: The query string
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            Relevant context as string
        """
        if not self.index or self.index.ntotal == 0:
            return ""
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Search for similar embeddings
            distances, indices = self.index.search(query_embedding, n_results)
            
            # Retrieve relevant documents
            relevant_docs = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    relevant_docs.append(self.documents[idx])
            
            # Format as context
            context = self._format_context(relevant_docs, query)
            
            return context
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return ""
    
    def _format_context(self, documents: List[Dict], query: str) -> str:
        """
        Format documents as context for the AI.
        
        Args:
            documents: List of relevant document dictionaries
            query: The original query
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context = "RELEVANT FINANCIAL INFORMATION:\n\n"
        
        for doc in documents:
            source = doc.get("source", "Unknown")
            content = doc.get("content", "")
            
            context += f"Source: {source}\n\n{content}\n\n---\n\n"
        
        return context
    
    def rag_response(self, user_query: str, system_prompt: str = "") -> str:
        """
        Generate a response using RAG.
        
        Args:
            user_query: The user's query
            system_prompt: Optional system prompt to use
            
        Returns:
            AI-generated response with relevant context
        """
        if not self.client:
            raise ValueError("No Anthropic client provided for generation")
        
        # Get relevant context
        context = self.query(user_query)
        
        # Create prompt
        if context:
            prompt = f"""
            I'll help you with your question about personal finance.
            
            {context}
            
            Based on the information above and my knowledge of personal finance, here's my response to your question:
            
            USER QUERY: {user_query}
            """
        else:
            prompt = f"""
            I'll help you with your question about personal finance.
            
            USER QUERY: {user_query}
            """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=system_prompt or "You are a helpful, accurate financial advisor assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def build_knowledge_base(self, clear_existing: bool = False) -> bool:
        """
        Build or rebuild the knowledge base from documents.
        
        Args:
            clear_existing: Whether to clear existing index
            
        Returns:
            Boolean indicating success
        """
        try:
            if clear_existing:
                # Reset index and documents
                self.index = None
                self.documents = []
            
            # Create new index
            self._create_index_from_kb()
            return True
        except Exception as e:
            print(f"Error building knowledge base: {e}")
            return False
    
    def search_financial_terms(self, term: str, n_results: int = 3) -> List[Dict]:
        """
        Search for financial terms in the knowledge base.
        
        Args:
            term: Financial term to search for
            n_results: Number of results to return
            
        Returns:
            List of matching documents with definitions
        """
        if not self.index or self.index.ntotal == 0:
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(f"define {term} finance")
            
            # Search for similar embeddings
            distances, indices = self.index.search(query_embedding, n_results)
            
            # Retrieve and format relevant documents
            results = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "term": term,
                        "definition": doc.get("content", ""),
                        "source": doc.get("source", "Unknown")
                    })
            
            return results
        except Exception as e:
            print(f"Error searching financial terms: {e}")
            return []
    
    def get_related_financial_concepts(self, concept: str, n_results: int = 5) -> List[str]:
        """
        Find related financial concepts based on semantic similarity.
        
        Args:
            concept: Financial concept to find related concepts for
            n_results: Number of related concepts to return
            
        Returns:
            List of related financial concepts
        """
        # Define some common financial concepts for fallback
        common_concepts = [
            "compound interest", "dollar cost averaging", "diversification",
            "asset allocation", "risk tolerance", "emergency fund",
            "retirement planning", "tax optimization", "debt snowball",
            "debt avalanche", "portfolio rebalancing", "credit score",
            "liquidity", "net worth", "cash flow", "inflation",
            "time value of money", "opportunity cost", "sinking fund",
            "amortization", "depreciation", "capital gains"
        ]
        
        if not self.index or self.index.ntotal == 0:
            # Return random selection of common concepts if no index
            import random
            return random.sample(common_concepts, min(n_results, len(common_concepts)))
        
        try:
            # Get concept embedding
            concept_embedding = self._get_embedding(concept)
            
            # Search for similar embeddings
            distances, indices = self.index.search(concept_embedding, n_results + 5)
            
            # Extract potential concepts from content
            related_concepts = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    content = self.documents[idx].get("content", "")
                    
                    # Extract potential concepts (simple approach)
                    lines = content.split("\n")
                    for line in lines:
                        if ":" in line and len(line) < 100:
                            potential_concept = line.split(":")[0].strip()
                            if (potential_concept.lower() != concept.lower() and
                                potential_concept not in related_concepts and
                                len(potential_concept.split()) <= 5):
                                related_concepts.append(potential_concept)
                                break
            
            # If we couldn't find enough related concepts, add from common concepts
            if len(related_concepts) < n_results:
                for c in common_concepts:
                    if c.lower() != concept.lower() and c not in related_concepts:
                        related_concepts.append(c)
                        if len(related_concepts) >= n_results:
                            break
            
            return related_concepts[:n_results]
        except Exception as e:
            print(f"Error finding related concepts: {e}")
            # Return random selection of common concepts as fallback
            import random
            return random.sample(common_concepts, min(n_results, len(common_concepts)))