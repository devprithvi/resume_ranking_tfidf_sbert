import pandas as pd
import requests
from pathlib import Path
import logging


def load_config_new():
    return {
        'data': {
            'jobs_url': "https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english/resolve/main/data/train-00000-of-00001.parquet",
            'candidates_url': "https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english/resolve/main/data/train-00000-of-00001.parquet",
            'sample_size': 20000  #for 20000 rows
        },
        'models': {
            'tfidf': {
                'max_features': 20000,
                'ngram_range': [1, 2],
                'min_df': 5,  #  to filter noise with more data
                'max_df': 0.8
            },
            'sbert': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32
            },
            'hybrid': {
                'alpha': 0.7
            }
        },
        'evaluation': {
            'top_k': 10
        }
    }

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def download_data(self, force_download=False):
        """Download datasets if they don't exist"""
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)

        jobs_path = data_dir / "jobs_raw.parquet"
        candidates_path = data_dir / "candidates_raw.parquet"

        if not jobs_path.exists() or force_download:
            self._download_file(self.config['data']['jobs_url'], jobs_path)

        if not candidates_path.exists() or force_download:
            self._download_file(self.config['data']['candidates_url'], candidates_path)

        return jobs_path, candidates_path

    def _download_file(self, url, path):
        """Download file from URL"""
        self.logger.info(f"Downloading {url} to {path}")
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)

    def load_data(self):
        """Load and optionally sample data"""
        jobs_path, candidates_path = self.download_data()

        jobs_df = pd.read_parquet(jobs_path)
        self.logger.info(f"job data {jobs_df.head()}")
        self.logger.info(f"job data {jobs_df.info()}")

        candidates_df = pd.read_parquet(candidates_path)
        self.logger.info(f"candidates data {candidates_df.head()}")
        self.logger.info(f"candidates data {candidates_df.info()}")

        #Remove duplicate columns
        jobs_df = jobs_df.loc[:, ~jobs_df.columns.duplicated()]
        candidates_df = candidates_df.loc[:, ~candidates_df.columns.duplicated()]

        #Remove duplicate rows
        jobs_df = jobs_df.drop_duplicates().reset_index(drop=True)
        candidates_df = candidates_df.drop_duplicates().reset_index(drop=True)

        self.logger.info("After cleaning data (removed duplicate columns & rows)")
        self.logger.info(f"Jobs data size: {jobs_df.shape}")
        self.logger.info(f"Candidates data size: {candidates_df.shape}")

        # Sample data
        sample_size = self.config['data'].get('sample_size')
        if sample_size:
            jobs_df = jobs_df.sample(n=min(sample_size, len(jobs_df)), random_state=42)
            candidates_df = candidates_df.sample(n=min(sample_size, len(candidates_df)), random_state=42)

        self.logger.info(f"Loaded {len(jobs_df)} jobs and {len(candidates_df)} candidates")

        # Print cleaned dataframes
        print("\n--- Cleaned Jobs Data ---")
        print(jobs_df.head(20))  # show first 20 rows
        print(jobs_df.shape)  # show first 20 rows
        print("\n--- Cleaned Candidates Data ---")
        print(candidates_df.head(20))
        print(candidates_df.shape)

        return jobs_df, candidates_df


if __name__ == '__main__':
    config = load_config_new()
    loader = DataLoader(config)
    jobs_df, candidates_df = loader.load_data()

    print("\n--- Cleaned Jobs Data (first 10 rows) ---")
    print(jobs_df.head(10))
    print(jobs_df.shape)

    print("\n--- Cleaned Candidates Data (first 10 rows) ---")
    print(candidates_df.head(10))
    print(candidates_df.shape)

    print("\nData loaded successfully")



