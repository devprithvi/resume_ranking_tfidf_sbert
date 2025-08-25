from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_ranker import BaseRanker
import logging


class TFIDFRanker(BaseRanker):
    def __init__(self, config):
        super().__init__(config)
        self.vectorizer = None
        self.logger = logging.getLogger(__name__)

    def fit(self, job_texts, candidate_texts):
        """Fit TF-IDF vectorizer"""
        # Combine all texts for vocabulary
        all_texts = list(job_texts) + list(candidate_texts)

        tfidf_config = self.config['models']['tfidf']

        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df'],
            stop_words='english',
            sublinear_tf=True
        )

        self.vectorizer.fit(all_texts)
        self.is_fitted = True

        self.logger.info(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def rank(self, job_texts, candidate_texts, top_k=10):
        """Rank candidates using TF-IDF + cosine similarity"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before ranking")

        # Transform texts
        job_tfidf = self.vectorizer.transform(job_texts)
        candidate_tfidf = self.vectorizer.transform(candidate_texts)

        # Calculate similarities
        similarities = cosine_similarity(job_tfidf, candidate_tfidf)

        return self._extract_top_k(similarities, 'tfidf', top_k)
