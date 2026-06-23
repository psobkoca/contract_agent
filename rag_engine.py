import os
import csv
import re
import math
import json
import chromadb
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import config

class BM25:
    """In-memory BM25 ranker for sparse keyword search."""
    def __init__(self, corpus: List[str], chunk_ids: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids = chunk_ids
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(self.tokenize(d)) for d in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_lens = [len(self.tokenize(d)) for d in corpus]
        self.doc_term_freqs = [Counter(self.tokenize(d)) for d in corpus]
        
        self.df = Counter()
        for tf in self.doc_term_freqs:
            for term in tf.keys():
                self.df[term] += 1

    def tokenize(self, text: str) -> List[str]:
        """Normalizes and tokenizes text into alphabetic words."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    def get_scores(self, query: str) -> Dict[str, float]:
        """Computes BM25 scores for all corpus documents against the query."""
        query_terms = self.tokenize(query)
        scores = {}
        
        for idx, chunk_id in enumerate(self.chunk_ids):
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf = self.doc_term_freqs[idx]
            
            for term in query_terms:
                if term not in self.df:
                    continue
                df_t = self.df[term]
                idf = math.log((self.corpus_size - df_t + 0.5) / (df_t + 0.5) + 1.0)
                
                tf_t = tf.get(term, 0)
                numerator = tf_t * (self.k1 + 1.0)
                denominator = tf_t + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len)) if self.avg_doc_len > 0 else 1.0
                score += idf * (numerator / denominator)
                
            scores[chunk_id] = score
            
        return scores

class RAGEngine:
    """Legal Retrieval-Augmented Generation (RAG) Engine using ChromaDB + Hybrid BM25."""
    
    def __init__(self, vector_dir: str = "./vector_store"):
        self.vector_dir = vector_dir
        self.collection_name = "legal_precedents"
        self.chroma_client = chromadb.PersistentClient(path=self.vector_dir)
        self.collection = None
        
        # Dense Embedding Model (FR-10)
        logger.info("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Reranker Model (FR-12)
        logger.info("Initializing CrossEncoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # In-memory BM25 index & chunk data
        self.bm25 = None
        self.chunk_map = {} # Maps chunk_id to doc details
        
    def load_precedent_metadata(self) -> Dict[str, dict]:
        """Loads precedent_index.csv metadata mapped by file name."""
        meta_map = {}
        if not os.path.exists("precedent_index.csv"):
            logger.warning("precedent_index.csv not found!")
            return meta_map
            
        with open("precedent_index.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_map[row["file_name"]] = row
        return meta_map

    def build_or_load_vector_store(self, force_rebuild: bool = False) -> None:
        """Loads existing vector store or builds it from legal_precedents/ (FR-09)."""
        # Check if database has collection already and we are not forcing rebuild
        collection_names = [col.name for col in self.chroma_client.list_collections()]
        
        if self.collection_name in collection_names and not force_rebuild:
            logger.info(f"Loading existing ChromaDB collection: {self.collection_name}")
            self.collection = self.chroma_client.get_collection(self.collection_name)
            self._initialize_bm25()
            return
            
        # Rebuild required
        logger.info(f"Building/Rebuilding vector database in {self.vector_dir}...")
        if self.collection_name in collection_names:
            self.chroma_client.delete_collection(self.collection_name)
            
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Use Cosine Distance for vector query
        )
        
        # 1. Document Loading using LlamaIndex
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
        
        logger.info("Loading precedent documents from legal_precedents/...")
        reader = SimpleDirectoryReader(input_dir="legal_precedents")
        documents = reader.load_data()
        
        # 2. Chunking (FR-10)
        splitter = SentenceSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info(f"Chunked {len(documents)} documents into {len(nodes)} text passages.")
        
        # Load registry metadata
        meta_registry = self.load_precedent_metadata()
        
        ids, embeddings, documents_text, metadatas = [], [], [], []
        
        for idx, node in enumerate(nodes):
            chunk_id = f"pass_{idx:04d}"
            text = node.text
            file_name = node.metadata.get("file_name")
            
            # Lookup metadata registry from CSV
            meta = meta_registry.get(file_name, {})
            
            # Construct chunk metadata (FR-10)
            chunk_meta = {
                "source_file": file_name or "Unknown",
                "clause_type": meta.get("clause_type") or meta.get("category") or "Unknown",
                "jurisdiction": meta.get("jurisdiction") or "Unknown",
                "document_date": meta.get("document_date") or "Unknown",
                "title": meta.get("title") or "Unknown",
                "document_type": meta.get("document_type") or "Unknown"
            }
            
            # Generate dense embedding (FR-10)
            embedding = self.embedding_model.encode(text).tolist()
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents_text.append(text)
            metadatas.append(chunk_meta)
            
        # Add to ChromaDB
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas
            )
            logger.success(f"Successfully indexed {len(ids)} passages into ChromaDB.")
            
        self._initialize_bm25()

    def _initialize_bm25(self) -> None:
        """Builds in-memory BM25 index from active database records."""
        logger.info("Initializing BM25 index from stored passages...")
        data = self.collection.get()
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        self.chunk_map = {}
        for idx, chunk_id in enumerate(ids):
            self.chunk_map[chunk_id] = {
                "text": documents[idx],
                "metadata": metadatas[idx]
            }
            
        self.bm25 = BM25(corpus=documents, chunk_ids=ids)
        logger.success(f"BM25 index built with {len(documents)} passages.")

    def hybrid_search(self, query_text: str, top_k: Optional[int] = None) -> List[dict]:
        """Performs hybrid search: dense cosine similarity + sparse BM25 merged via RRF (FR-11)."""
        if top_k is None:
            top_k = config.rag.top_k

        if self.collection is None or self.bm25 is None:
            raise RuntimeError("RAGEngine is not initialized. Call build_or_load_vector_store() first.")

        # 1. Dense Search (ChromaDB Cosine similarity query)
        query_emb = self.embedding_model.encode(query_text).tolist()
        # Retrieve n_results=corpus_size to get a full ranking for RRF
        corpus_size = len(self.chunk_map)
        dense_results = self.collection.query(
            query_embeddings=[query_emb], 
            n_results=corpus_size
        )
        dense_ids = dense_results.get("ids", [[]])[0]
        dense_ranks = {chunk_id: rank + 1 for rank, chunk_id in enumerate(dense_ids)}

        # 2. Sparse Search (BM25)
        bm25_scores = self.bm25.get_scores(query_text)
        # Sort all chunks by BM25 score descending
        sorted_bm25_ids = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)
        # Rank only documents that have a positive score
        sparse_ranks = {}
        rank_idx = 1
        for cid in sorted_bm25_ids:
            if bm25_scores[cid] > 0.0:
                sparse_ranks[cid] = rank_idx
                rank_idx += 1

        # 3. Reciprocal Rank Fusion (RRF) Merge
        rrf_scores = {}
        for chunk_id in self.chunk_map.keys():
            d_rank = dense_ranks.get(chunk_id, 99999)
            s_rank = sparse_ranks.get(chunk_id, 99999)
            
            # Standard RRF formula: 1 / (60 + rank)
            rrf_score = 1.0 / (60.0 + d_rank) + 1.0 / (60.0 + s_rank)
            rrf_scores[chunk_id] = rrf_score

        # Sort chunk IDs by RRF score descending
        top_chunk_ids = sorted(self.chunk_map.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # Construct final results list
        results = []
        for cid in top_chunk_ids:
            results.append({
                "chunk_id": cid,
                "text": self.chunk_map[cid]["text"],
                "metadata": self.chunk_map[cid]["metadata"],
                "rrf_score": rrf_scores[cid],
                "dense_rank": dense_ranks.get(cid, -1),
                "sparse_rank": sparse_ranks.get(cid, -1)
            })

        logger.info(f"Hybrid search returned top-{len(results)} candidate passages.")
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Legal RAG Knowledge Base Builder")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the vector knowledge base.")
    args = parser.parse_args()
    
    rag = RAGEngine()
    rag.build_or_load_vector_store(force_rebuild=args.rebuild)

if __name__ == "__main__":
    main()
