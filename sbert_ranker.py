import gc
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import  pandas as pd
from .base_ranker import BaseRanker
import logging


class SBERTRanker(BaseRanker):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.faiss_index = None
        self.candidate_embeddings = None
        self.logger = logging.getLogger(__name__)

    def fit(self, job_texts, candidate_texts):
        """Initialize SBERT model and create candidate embeddings"""
        sbert_config = self.config['models']['sbert']

        self.model = SentenceTransformer(sbert_config['model_name'])

        self.logger.info("Creating candidate embeddings...")
        self.candidate_embeddings = self.model.encode(
            list(candidate_texts),
            batch_size=sbert_config['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(self.candidate_embeddings)

        # Build FAISS index
        self._build_faiss_index()
        self.is_fitted = True

    def _build_faiss_index(self):
        """Build FAISS index for fast search"""
        dimension = self.candidate_embeddings.shape[1]
        n_candidates = self.candidate_embeddings.shape[0]

        if n_candidates > 10000:
            # Use IVF for large datasets
            nlist = min(1000, n_candidates // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(self.candidate_embeddings)
            self.logger.info(f"Built IVF index with {nlist} clusters")
        else:
            # Use flat index for smaller datasets
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.logger.info("Built flat index")

        self.faiss_index.add(self.candidate_embeddings)
        self.logger.info(f"Added {self.faiss_index.ntotal} candidates to index")

    def rank(self, job_texts, candidate_texts, top_k=10):
        """Rank candidates using SBERT with fallback"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before ranking")

        try:
            return self._safe_faiss_rank(job_texts, top_k)
        except Exception as e:
            self.logger.warning(f"FAISS ranking failed: {e}. Using numpy fallback.")
            return self._numpy_fallback_rank(job_texts, top_k)

    def _safe_faiss_rank(self, job_texts, top_k):
        """Safe FAISS ranking with smaller batches"""
        sbert_config = self.config['models']['sbert']

        # Process in very small batches
        batch_size = 4  # Much smaller
        job_list = list(job_texts)
        all_results = []

        for i in range(0, len(job_list), batch_size):
            batch_jobs = job_list[i:i + batch_size]

            job_embeddings = self.model.encode(
                batch_jobs,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            faiss.normalize_L2(job_embeddings)

            # Search in smaller top_k to reduce memory
            search_k = min(top_k, 50)
            similarities, indices = self.faiss_index.search(job_embeddings, search_k)

            # Extract results for this batch
            for batch_idx, (job_sims, job_indices) in enumerate(zip(similarities, indices)):
                job_idx = i + batch_idx
                for rank, (sim, cand_idx) in enumerate(zip(job_sims[:top_k], job_indices[:top_k])):
                    all_results.append({
                        'job_id': job_idx,
                        'cand_id': int(cand_idx),
                        'sbert_score': float(sim),
                        'rank': rank + 1
                    })

            # Clean memory after each batch
            del job_embeddings
            gc.collect()

        return pd.DataFrame(all_results)

    def _numpy_fallback_rank(self, job_texts, top_k):
        """Fallback ranking using pure NumPy"""
        self.logger.info("Using NumPy fallback for ranking...")

        job_embeddings = self.model.encode(
            list(job_texts),
            batch_size=8,
            convert_to_numpy=True
        )

        # Normalize
        job_embeddings = job_embeddings / np.linalg.norm(job_embeddings, axis=1, keepdims=True)

        # Calculate similarities
        similarities = np.dot(job_embeddings, self.candidate_embeddings.T)

        # Extract top-k
        results = []
        for job_idx, job_sims in enumerate(similarities):
            top_indices = np.argsort(job_sims)[::-1][:top_k]
            for rank, cand_idx in enumerate(top_indices):
                results.append({
                    'job_id': job_idx,
                    'cand_id': int(cand_idx),
                    'sbert_score': float(job_sims[cand_idx]),
                    'rank': rank + 1
                })

        return pd.DataFrame(results)

