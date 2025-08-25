import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from resume_ranking_env.resume_ranking_research.analysis.AdvancedRankerEvaluator import ImprovedAdvancedRankerEvaluator
from resume_ranking_env.resume_ranking_research.analysis.RankerComparasionAnaAnalysis import RankerComparator
from resume_ranking_env.resume_ranking_research.data.data_loader import DataLoader
from resume_ranking_env.resume_ranking_research.eda.ResumeRankingEDA import ResumeRankingEDA
from resume_ranking_env.resume_ranking_research.model.tfidf_ranker import TFIDFRanker
from resume_ranking_env.resume_ranking_research.preprocess.ImprovedDataPreprocessor import ImprovedDataPreprocessor, \
    EnhancedGroundTruthGenerator, extract_dataset_statistics
from resume_ranking_env.resume_ranking_research.preprocess.data.DataProcessingPlots import RankingPlots


class OptimizedSBERTRanker:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.candidate_embeddings = None
        self.logger = logging.getLogger(__name__)

    def fit(self, job_texts, candidate_texts):
        from sentence_transformers import SentenceTransformer

        sbert_config = self.config['models']['sbert']
        self.model = SentenceTransformer(sbert_config['model_name'])

        self.logger.info(f"Creating embeddings for {len(candidate_texts)} candidates...")

        # Optimized batch processing for 10K
        batch_size = sbert_config['batch_size']
        candidate_list = list(candidate_texts)

        embeddings = []
        total_batches = (len(candidate_list) + batch_size - 1) // batch_size

        for i in range(0, len(candidate_list), batch_size):
            batch_num = (i // batch_size) + 1
            batch = candidate_list[i:i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)

            # Progress logging
            if batch_num % 50 == 0:
                self.logger.info(f"Processed batch {batch_num}/{total_batches}")

            # Memory management every 100 batches
            if batch_num % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        self.candidate_embeddings = np.vstack(embeddings)

        # Normalize embeddings
        norms = np.linalg.norm(self.candidate_embeddings, axis=1, keepdims=True)
        self.candidate_embeddings = self.candidate_embeddings / norms

        self.logger.info(f"Created {self.candidate_embeddings.shape[0]} embeddings successfully")

    def rank(self, job_texts, candidate_texts, top_k=10):
        """Optimized ranking for 10K samples"""
        self.logger.info(f"Ranking {len(job_texts)} jobs against {len(candidate_texts)} candidates...")

        job_list = list(job_texts)
        batch_size = 16  # Moderate batch size for jobs

        all_results = []
        total_job_batches = (len(job_list) + batch_size - 1) // batch_size

        for i in range(0, len(job_list), batch_size):
            batch_num = (i // batch_size) + 1
            batch_jobs = job_list[i:i + batch_size]

            # Create job embeddings
            job_embeddings = self.model.encode(
                batch_jobs,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Normalize
            job_norms = np.linalg.norm(job_embeddings, axis=1, keepdims=True)
            job_embeddings = job_embeddings / job_norms

            # Calculate similarities using optimized numpy
            similarities = np.dot(job_embeddings, self.candidate_embeddings.T)

            # Extract top-k for this batch
            for batch_idx, job_similarities in enumerate(similarities):
                job_idx = i + batch_idx
                top_indices = np.argpartition(job_similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(job_similarities[top_indices])[::-1]]

                for rank, cand_idx in enumerate(top_indices):
                    all_results.append({
                        'job_id': job_idx,
                        'cand_id': int(cand_idx),
                        'sbert_score': float(job_similarities[cand_idx]),
                        'rank': rank + 1
                    })

            # Progress logging
            if batch_num % 20 == 0:
                self.logger.info(f"Ranked job batch {batch_num}/{total_job_batches}")

            # Memory cleanup
            del job_embeddings, similarities
            if batch_num % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return pd.DataFrame(all_results)


def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    return mem_mb


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('resume_ranking_10k.log'),
            logging.StreamHandler()
        ]
    )


def load_config():
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


def main():
    """
    Main function to execute comprehensive resume ranking pipeline.
    Compares TF-IDF and SBERT approaches with advanced evaluation metrics.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()

    start_time = time.time()
    start_memory = monitor_memory()

    # Initialize variables to ensure they're available throughout the pipeline
    tfidf_results = None
    sbert_results = None
    jobs_processed = None
    candidates_processed = None
    enhanced_ground_truth = None
    advanced_results = None

    try:
        # Create necessary directories for output
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/results").mkdir(parents=True, exist_ok=True)

        logger.info("Starting comprehensive resume ranking pipeline")
        logger.info(f"Initial memory usage: {start_memory:.1f} MB")

        # ==================== DATA LOADING ====================
        print("Loading dataset...")
        logger.info("Beginning data loading process")

        loader = DataLoader(config)
        jobs_df, candidates_df = loader.load_data()

        print(f"Successfully loaded {len(jobs_df):,} job postings and {len(candidates_df):,} candidate profiles")
        logger.info(f"Data loading completed: {len(jobs_df)} jobs, {len(candidates_df)} candidates")

        # ==================== DATA PREPROCESSING ====================
        print("\nStarting data preprocessing...")
        logger.info("Beginning preprocessing phase")

        preprocessor = ImprovedDataPreprocessor(config)
        jobs_processed = preprocessor.preprocess_dataframe(jobs_df, is_jobs=True)
        candidates_processed = preprocessor.preprocess_dataframe(candidates_df, is_jobs=False)

        # Extract and display dataset statistics
        print("Extracting dataset characteristics...")
        logger.info("Computing dataset statistics")
        dataset_stats = extract_dataset_statistics(jobs_processed, candidates_processed, logger)

        # ==================== VISUALIZATION GENERATION ====================
        print("Generating exploratory data analysis plots...")
        logger.info("Creating initial visualization plots")

        plots = RankingPlots(plots_dir="plots")
        plot_artifacts = plots.generate_all(jobs_processed, candidates_processed)

        print("Plot generation completed. Saved artifacts:")
        for plot_type, file_path in plot_artifacts.items():
            print(f"  - {plot_type}: {file_path}")

        # ==================== TEXT PREPARATION ====================
        print("\nPreparing text data for ranking algorithms...")
        logger.info("Preparing combined text representations")

        # Ensure combined_text column exists for both datasets
        if 'combined_text' not in jobs_processed.columns:
            jobs_processed['combined_text'] = (
                    jobs_processed.get('Position', '').astype(str) + ' ' +
                    jobs_processed.get('Long Description', jobs_processed.get('Description', '')).astype(str)
            )
        if 'combined_text' not in candidates_processed.columns:
            candidates_processed['combined_text'] = (
                    candidates_processed.get('Position', '').astype(str) + ' ' +
                    candidates_processed.get('CV', candidates_processed.get('Moreinfo', '')).astype(str)
            )

        # Extract text collections for ranking
        job_texts = jobs_processed['combined_text'].fillna('')
        candidate_texts = candidates_processed['combined_text'].fillna('')

        print(f"Text preparation completed. Memory usage: {monitor_memory():.1f} MB")
        logger.info(f"Text preparation completed, current memory: {monitor_memory():.1f} MB")

        # ==================== TF-IDF RANKING ====================
        print("\nExecuting TF-IDF ranking algorithm...")
        logger.info("Starting TF-IDF ranking phase")

        tfidf_ranker = TFIDFRanker(config)
        tfidf_ranker.fit(job_texts, candidate_texts)

        # Process ranking in batches for memory efficiency
        batch_size = 500
        job_batches = [job_texts[i:i + batch_size] for i in range(0, len(job_texts), batch_size)]
        tfidf_results_list = []

        print(f"Processing {len(job_batches)} batches for TF-IDF ranking...")
        for i, job_batch in enumerate(job_batches):
            if (i + 1) % 10 == 0:  # Progress indicator every 10 batches
                print(f"  Completed {i + 1}/{len(job_batches)} batches")

            batch_results = tfidf_ranker.rank(job_batch, candidate_texts)
            batch_results['job_id'] = batch_results['job_id'] + i * batch_size
            tfidf_results_list.append(batch_results)

        tfidf_results = pd.concat(tfidf_results_list, ignore_index=True)
        print(f"TF-IDF ranking completed. Generated {len(tfidf_results):,} job-candidate rankings")
        logger.info(f"TF-IDF ranking completed: {len(tfidf_results)} rankings generated")

        # Memory cleanup
        del tfidf_ranker
        gc.collect()
        logger.info(f"TF-IDF cleanup completed, memory: {monitor_memory():.1f} MB")

        # ==================== SBERT RANKING ====================
        print("\nExecuting SBERT semantic ranking algorithm...")
        logger.info("Starting SBERT ranking phase")

        sbert_ranker = OptimizedSBERTRanker(config)
        sbert_ranker.fit(job_texts, candidate_texts)

        # Process SBERT ranking in batches
        sbert_results_list = []
        print(f"Processing {len(job_batches)} batches for SBERT ranking...")
        for i, job_batch in enumerate(job_batches):
            if (i + 1) % 10 == 0:  # Progress indicator every 10 batches
                print(f"  Completed {i + 1}/{len(job_batches)} batches")

            batch_results = sbert_ranker.rank(job_batch, candidate_texts)
            batch_results['job_id'] = batch_results['job_id'] + i * batch_size
            sbert_results_list.append(batch_results)

        sbert_results = pd.concat(sbert_results_list, ignore_index=True)
        print(f"SBERT ranking completed. Generated {len(sbert_results):,} job-candidate rankings")
        logger.info(f"SBERT ranking completed: {len(sbert_results)} rankings generated")

        # Memory cleanup
        del sbert_ranker
        gc.collect()
        logger.info(f"SBERT cleanup completed, memory: {monitor_memory():.1f} MB")

        # ==================== SAVE INITIAL RESULTS ====================
        print("\nSaving ranking results to files...")
        logger.info("Saving ranking results")

        tfidf_results.to_csv("data/results/tfidf_results_comprehensive.csv", index=False)
        sbert_results.to_csv("data/results/sbert_results_comprehensive.csv", index=False)
        print("Ranking results saved successfully")

        # ==================== ENHANCED GROUND TRUTH GENERATION ====================
        print("\nGenerating enhanced ground truth for evaluation...")
        logger.info("Creating enhanced ground truth labels")

        gt_generator = EnhancedGroundTruthGenerator(jobs_processed, candidates_processed)

        # Balance between evaluation coverage and computational efficiency
        max_jobs_for_eval = min(2000, len(jobs_processed))
        job_ids_for_eval = list(range(max_jobs_for_eval))

        print(f"Creating ground truth for {max_jobs_for_eval:,} jobs with up to 300 candidates each...")
        enhanced_ground_truth = gt_generator.create_enhanced_ground_truth(
            job_ids=job_ids_for_eval,
            max_candidates_per_job=300
        )

        # Verify ground truth generation was successful
        if enhanced_ground_truth is None or len(enhanced_ground_truth) == 0:
            logger.error("Ground truth generation failed")
            print("Error: Ground truth generation failed. Cannot proceed with evaluation.")
            return None

        print(f"Enhanced ground truth created successfully for {len(enhanced_ground_truth):,} jobs")
        logger.info(f"Ground truth generation completed: {len(enhanced_ground_truth)} jobs")

        # ==================== EXPLORATORY DATA ANALYSIS ====================
        print("\nConducting exploratory data analysis...")
        logger.info("Running comprehensive EDA")

        eda = ResumeRankingEDA(
            jobs_df=jobs_processed,
            candidates_df=candidates_processed,
            results_df=tfidf_results,
            name="Comprehensive Resume Ranking Analysis",
            plots_dir="comprehensive_analysis_plots"
        )
        eda.generate_all_plots()
        eda.visualize_distributions('Primary Keyword_normalized', 'jobs', top_n=15)

        if 'Experience Years' in candidates_processed.columns:
            eda.visualize_distributions('Experience Years', 'candidates', bins=20)

        print("Exploratory data analysis completed")
        logger.info("EDA phase completed")

        # ==================== ENHANCED SAMPLE CREATION ====================
        print("\nCreating detailed result samples for analysis...")
        logger.info("Generating enhanced result samples")

        def create_enhanced_sample(results_df, method_name, sample_size=10):
            """
            Create detailed samples with comprehensive job and candidate information.

            Args:
                results_df: DataFrame containing ranking results
                method_name: Name of the ranking method (for column naming)
                sample_size: Number of top results to include

            Returns:
                DataFrame with detailed sample information
            """
            rows = []
            for _, row in results_df.head(sample_size).iterrows():
                try:
                    job_id = int(row['job_id'])
                    cand_id = int(row['cand_id'])

                    # Verify indices are within bounds
                    if job_id < len(jobs_processed) and cand_id < len(candidates_processed):
                        job_info = jobs_processed.iloc[job_id]
                        candidate_info = candidates_processed.iloc[cand_id]

                        rows.append({
                            'job_id': job_id,
                            'cand_id': cand_id,
                            'score': row[f'{method_name}_score'],
                            'rank': row['rank'],
                            'job_position': job_info.get('Position', 'N/A'),
                            'job_company': job_info.get('Company Name', 'N/A'),
                            'job_keywords': job_info.get('Primary Keyword_normalized', 'N/A'),
                            'job_domain': job_info.get('domain_category', 'N/A'),
                            'job_seniority': job_info.get('seniority_level', 'N/A'),
                            'job_description': str(job_info.get('Long Description',
                                                                job_info.get('Description', 'N/A')))[:500],
                            'candidate_position': candidate_info.get('Position', 'N/A'),
                            'candidate_keywords': candidate_info.get('Primary Keyword_normalized', 'N/A'),
                            'candidate_domain': candidate_info.get('domain_category', 'N/A'),
                            'candidate_seniority': candidate_info.get('seniority_level', 'N/A'),
                            'candidate_experience_years': candidate_info.get('exp_years_final',
                                                                             candidate_info.get('Experience Years',
                                                                                                'N/A')),
                            'candidate_experience_level': candidate_info.get('exp_level_final', 'unknown'),
                            'candidate_highlights': str(candidate_info.get('Highlights', 'N/A'))[:300],
                            'candidate_resume': str(candidate_info.get('CV',
                                                                       candidate_info.get('Moreinfo', 'N/A')))[:500]
                        })
                except Exception as e:
                    logger.warning(f"Error processing sample row: {e}")
                    continue

            return pd.DataFrame(rows)

        # Create enhanced samples for both methods
        tfidf_enhanced = create_enhanced_sample(tfidf_results, 'tfidf', sample_size=10)
        sbert_enhanced = create_enhanced_sample(sbert_results, 'sbert', sample_size=10)

        # hi thi is comment

        # Save enhanced samples for detailed analysis
        tfidf_enhanced.to_csv("data/results/tfidf_enhanced_samples.csv", index=False)
        sbert_enhanced.to_csv("data/results/sbert_enhanced_samples.csv", index=False)
        print("Enhanced result samples created and saved")

        # ==================== STANDARD RANKING COMPARISON ====================
        print("\nExecuting standard ranking comparison analysis...")
        logger.info("Running standard hjdjjjdd comparison between methods")




        comparator = RankerComparator(
            tfidf_results=tfidf_results,
            sbert_results=sbert_results,
            jobs_df=jobs_processed,
            candidates_df=candidates_processed,
            top_k=10,
            plots_dir="standard_comparison_plots"
        )
        standard_comparison_results = comparator.generate_comprehensive_report()
        print("Standard comparison analysis completed")

        # ==================== ADVANCED EVALUATION ====================
        print("\nInitiating advanced comprehensive evaluation...")
        logger.info("Starting advanced evaluation with comprehensive metrics")

        def run_advanced_evaluation():
            """
            Execute advanced evaluation with comprehensive metrics including
            MRR, NDCG, MAP, statistical significance testing, and correlation analysis.

            Returns:
                Dictionary containing all evaluation results
            """
            try:
                evaluator = ImprovedAdvancedRankerEvaluator(
                    tfidf_results=tfidf_results,
                    sbert_results=sbert_results,
                    jobs_df=jobs_processed,
                    candidates_df=candidates_processed,
                    enhanced_ground_truth=enhanced_ground_truth,
                    top_k=10,
                    plots_dir="advanced_evaluation_results"
                )

                return evaluator.run_comprehensive_evaluation()

            except Exception as e:
                logger.error(f"Error in advanced evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

        advanced_results = run_advanced_evaluation()

        if advanced_results is not None:
            print("Advanced evaluation completed successfully")
            logger.info("Advanced evaluation phase completed")
        else:
            print("Warning: Advanced evaluation encountered issues. Check logs for details.")
            logger.warning("Advanced evaluation failed")

        # ==================== PERFORMANCE SUMMARY ====================
        end_time = time.time()
        end_memory = monitor_memory()

        print("\n" + "=" * 80)
        print("COMPREHENSIVE RESUME RANKING PIPELINE COMPLETED")
        print("=" * 80)
        print(f"Total Processing Time: {end_time - start_time:.1f} seconds")
        print(f"Peak Memory Usage: {end_memory:.1f} MB")
        print(f"Memory Increase: {end_memory - start_memory:.1f} MB")
        print(f"TF-IDF Rankings Generated: {len(tfidf_results):,}")
        print(f"SBERT Rankings Generated: {len(sbert_results):,}")
        print(f"Ground Truth Jobs Evaluated: {len(enhanced_ground_truth):,}")
        print("=" * 80)

        logger.info(f"Pipeline completed successfully in {end_time - start_time:.1f} seconds")

        # ==================== RESULTS DISPLAY ====================
        print_enhanced_results_summary(tfidf_enhanced, sbert_enhanced)

        # ==================== FINAL CLEANUP AND RETURN ====================
        print("\nPipeline execution completed successfully!")
        print("Results and visualizations have been saved to respective directories:")
        print("  - data/results/ (ranking results and samples)")
        print("  - plots/ (initial exploratory visualizations)")
        print("  - advanced_evaluation_results/ (comprehensive metrics and plots)")
        print("  - standard_comparison_plots/ (method comparison analysis)")

        logger.info("Pipeline completed successfully, returning results")

        return {
            'tfidf_results': tfidf_results,
            'sbert_results': sbert_results,
            'enhanced_ground_truth': enhanced_ground_truth,
            'advanced_results': advanced_results,
            'standard_comparison': standard_comparison_results,
            'dataset_stats': dataset_stats
        }

    except Exception as e:
        logger.error(f"Critical pipeline error: {str(e)}")
        print(f"Error: Critical pipeline failure - {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Final cleanup operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        final_memory = monitor_memory()
        logger.info(f"Pipeline cleanup completed. Final memory usage: {final_memory:.1f} MB")


def print_enhanced_results_summary(tfidf_enhanced, sbert_enhanced):
    """
    Print a concise, human-readable summary of the top ranking results.

    Args:
        tfidf_enhanced: DataFrame with detailed TF-IDF results
        sbert_enhanced: DataFrame with detailed SBERT results
    """
    print("\n" + "=" * 100)
    print("TOP RANKING RESULTS SUMMARY")
    print("=" * 100)

    print("\nTF-IDF Top 3 Job-Candidate Matches:")
    print("-" * 50)
    for i, (_, result) in enumerate(tfidf_enhanced.head(3).iterrows(), 1):
        match_quality = "Excellent" if result['score'] > 0.5 else "Good" if result['score'] > 0.3 else "Fair"
        print(f"\n{i}. Rank {result['rank']} | Similarity Score: {result['score']:.4f} | Quality: {match_quality}")
        print(f"   Job: {result['job_position']} ({result['job_domain']}, {result['job_seniority']} level)")
        print(
            f"   Candidate: {result['candidate_position']} ({result['candidate_domain']}, {result['candidate_seniority']} level)")

        # Display experience match if available
        if result['candidate_experience_years'] != 'N/A':
            print(f"   Experience: {result['candidate_experience_years']} years")

    print("\nSBERT Top 3 Job-Candidate Matches:")
    print("-" * 50)
    for i, (_, result) in enumerate(sbert_enhanced.head(3).iterrows(), 1):
        match_quality = "Excellent" if result['score'] > 0.5 else "Good" if result['score'] > 0.3 else "Fair"
        print(f"\n{i}. Rank {result['rank']} | Similarity Score: {result['score']:.4f} | Quality: {match_quality}")
        print(f"   Job: {result['job_position']} ({result['job_domain']}, {result['job_seniority']} level)")
        print(
            f"   Candidate: {result['candidate_position']} ({result['candidate_domain']}, {result['candidate_seniority']} level)")

        # Display experience match if available
        if result['candidate_experience_years'] != 'N/A':
            print(f"   Experience: {result['candidate_experience_years']} years")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

from sklearn.ensemble import GradientBoostingRegressor


class LearningToRankRanker:
    def __init__(self):
        self.ranker = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )

    def fit(self, features, relevance_scores):
        self.ranker.fit(features, relevance_scores)

    def rank(self, job_features, candidate_features):
        scores = self.ranker.predict(candidate_features)
        return np.argsort(scores)[::-1]
