import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import ndcg_score, precision_score, recall_score
import random
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')
    # Advanced evaluation metrics for ranking systems including MRR, Precision@K, Recall@K, Hit Rate, and User Satisfaction

from scipy import stats
import random



class ImprovedAdvancedRankerEvaluator:
    def __init__(self, tfidf_results, sbert_results, jobs_df, candidates_df,
                 enhanced_ground_truth=None, top_k=10, plots_dir="advanced_metrics_plots"):
        self.tfidf_results = tfidf_results
        self.sbert_results = sbert_results
        self.jobs_df = jobs_df
        self.candidates_df = candidates_df
        self.enhanced_ground_truth = enhanced_ground_truth
        self.top_k = top_k
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Store job IDs for analysis
        self.job_ids = list(set(tfidf_results['job_id']).intersection(set(sbert_results['job_id'])))

        # Initialize results storage
        self.evaluation_results = {}

    def create_enhanced_ground_truth(self, method='multi_dimensional'):
        """
        Create sophisticated ground truth using multiple relevance criteria
        """
        if self.enhanced_ground_truth is not None:
            return self.enhanced_ground_truth

        ground_truth = {}

        for job_id in self.job_ids[:1000]:  # Limit for efficiency
            if job_id >= len(self.jobs_df):
                continue

            job_info = self.jobs_df.iloc[job_id]

            # Extract job characteristics
            job_domain = job_info.get('domain_category', 'unknown')
            job_seniority = job_info.get('seniority_level', 'unknown')
            job_keywords = str(job_info.get('Primary Keyword_normalized', '')).lower()
            job_position = str(job_info.get('Position', '')).lower()

            # Get candidates for this job
            job_candidates = set(
                self.tfidf_results[self.tfidf_results['job_id'] == job_id]['cand_id']
            ).union(
                set(self.sbert_results[self.sbert_results['job_id'] == job_id]['cand_id'])
            )

            relevance_scores = {}

            for cand_id in job_candidates:
                if cand_id >= len(self.candidates_df):
                    continue

                candidate_info = self.candidates_df.iloc[cand_id]

                # Multi-dimensional relevance scoring
                relevance = 0.0

                # 1. Domain Match (40% weight)
                cand_domain = candidate_info.get('domain_category', 'unknown')
                if job_domain != 'unknown' and cand_domain != 'unknown':
                    if job_domain == cand_domain:
                        relevance += 0.40
                    elif self._are_related_domains(job_domain, cand_domain):
                        relevance += 0.25

                # 2. Seniority Match (25% weight)
                cand_seniority = candidate_info.get('seniority_level', 'unknown')
                if job_seniority != 'unknown' and cand_seniority != 'unknown':
                    seniority_score = self._calculate_seniority_match(job_seniority, cand_seniority)
                    relevance += 0.25 * seniority_score

                # 3. Keyword Match (25% weight)
                cand_keywords = str(candidate_info.get('Primary Keyword_normalized', '')).lower()
                if job_keywords and cand_keywords:
                    keyword_score = self._calculate_keyword_similarity(job_keywords, cand_keywords)
                    relevance += 0.25 * keyword_score

                # 4. Experience Level Match (10% weight)
                job_exp = job_info.get('exp_years_final', None)
                cand_exp = candidate_info.get('exp_years_final', None)
                if job_exp is not None and cand_exp is not None:
                    exp_score = self._calculate_experience_match(job_exp, cand_exp)
                    relevance += 0.10 * exp_score

                # Only include if above threshold
                if relevance > 0.15:
                    relevance_scores[cand_id] = min(relevance, 1.0)

            ground_truth[job_id] = relevance_scores

        return ground_truth

    def _are_related_domains(self, domain1, domain2):
        """Check if two domains are related"""
        related_pairs = {
            ('software_dev', 'data_science'),
            ('design', 'marketing'),
            ('sales', 'marketing'),
            ('operations', 'consulting')
        }
        return (domain1, domain2) in related_pairs or (domain2, domain1) in related_pairs

    def _calculate_seniority_match(self, job_seniority, cand_seniority):
        """Calculate seniority level match score"""
        seniority_order = {'intern': 0, 'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4}

        if job_seniority not in seniority_order or cand_seniority not in seniority_order:
            return 0.5

        diff = abs(seniority_order[job_seniority] - seniority_order[cand_seniority])
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.7
        elif diff == 2:
            return 0.4
        else:
            return 0.1

    def _calculate_keyword_similarity(self, job_keywords, cand_keywords):
        """Calculate keyword similarity using Jaccard index"""
        job_words = set(job_keywords.split())
        cand_words = set(cand_keywords.split())

        if not job_words or not cand_words:
            return 0.0

        intersection = len(job_words.intersection(cand_words))
        union = len(job_words.union(cand_words))

        return intersection / union if union > 0 else 0.0

    def _calculate_experience_match(self, job_exp, cand_exp):
        """Calculate experience level match with smooth decay"""
        try:
            job_years = float(job_exp)
            cand_years = float(cand_exp)
            diff = abs(job_years - cand_years)
            return max(0, 1.0 - (diff / 10.0))  # Smooth decay over 10 years
        except:
            return 0.5

    def compute_advanced_mrr(self, results_df, ground_truth, threshold=0.2):
        """Enhanced MRR computation with confidence intervals"""
        reciprocal_ranks = []

        for job_id in self.job_ids:
            if job_id not in ground_truth or len(ground_truth[job_id]) == 0:
                continue

            job_results = results_df[results_df['job_id'] == job_id].sort_values('rank')

            for _, row in job_results.iterrows():
                cand_id = row['cand_id']
                if cand_id in ground_truth[job_id]:
                    if ground_truth[job_id][cand_id] >= threshold:
                        reciprocal_ranks.append(1.0 / row['rank'])
                        break
            else:
                reciprocal_ranks.append(0.0)

        mrr = np.mean(reciprocal_ranks)
        std_err = np.std(reciprocal_ranks) / np.sqrt(len(reciprocal_ranks))
        ci_lower = mrr - 1.96 * std_err
        ci_upper = mrr + 1.96 * std_err

        return {
            'mrr': mrr,
            'std_error': std_err,
            'confidence_interval': (ci_lower, ci_upper),
            'reciprocal_ranks': reciprocal_ranks,
            'num_queries': len(reciprocal_ranks)
        }

    def compute_ndcg_at_k(self, results_df, ground_truth, k_values=[1, 3, 5, 10]):
        """Compute Normalized Discounted Cumulative Gain"""
        ndcg_scores = {k: [] for k in k_values}

        for job_id in self.job_ids:
            if job_id not in ground_truth or len(ground_truth[job_id]) == 0:
                continue

            job_results = results_df[results_df['job_id'] == job_id].sort_values('rank')

            for k in k_values:
                top_k_results = job_results.head(k)

                # Calculate DCG
                dcg = 0
                for i, (_, row) in enumerate(top_k_results.iterrows()):
                    cand_id = row['cand_id']
                    relevance = ground_truth[job_id].get(cand_id, 0)
                    dcg += relevance / np.log2(i + 2)

                # Calculate IDCG (ideal DCG)
                ideal_relevances = sorted(ground_truth[job_id].values(), reverse=True)[:k]
                idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

                # Calculate NDCG
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores[k].append(ndcg)

        return {k: np.mean(scores) for k, scores in ndcg_scores.items()}

    def compute_map_at_k(self, results_df, ground_truth, k_values=[1, 3, 5, 10]):
        """Compute Mean Average Precision"""
        map_scores = {k: [] for k in k_values}

        for job_id in self.job_ids:
            if job_id not in ground_truth or len(ground_truth[job_id]) == 0:
                continue

            job_results = results_df[results_df['job_id'] == job_id].sort_values('rank')
            relevant_candidates = set(ground_truth[job_id].keys())

            for k in k_values:
                top_k_results = job_results.head(k)

                precision_sum = 0
                relevant_found = 0

                for i, (_, row) in enumerate(top_k_results.iterrows()):
                    cand_id = row['cand_id']
                    if cand_id in relevant_candidates:
                        relevant_found += 1
                        precision_sum += relevant_found / (i + 1)

                ap = precision_sum / len(relevant_candidates) if len(relevant_candidates) > 0 else 0
                map_scores[k].append(ap)

        return {k: np.mean(scores) for k, scores in map_scores.items()}

    def compute_ranking_correlation(self):
        """Compute ranking correlation between TF-IDF and SBERT"""
        correlations = []

        for job_id in self.job_ids:
            tfidf_job = self.tfidf_results[self.tfidf_results['job_id'] == job_id]
            sbert_job = self.sbert_results[self.sbert_results['job_id'] == job_id]

            # Get common candidates
            common_candidates = set(tfidf_job['cand_id']).intersection(set(sbert_job['cand_id']))

            if len(common_candidates) < 3:  # Need at least 3 for correlation
                continue

            # Create rankings for common candidates
            tfidf_ranks = []
            sbert_ranks = []

            for cand_id in common_candidates:
                tfidf_rank = tfidf_job[tfidf_job['cand_id'] == cand_id]['rank'].iloc[0]
                sbert_rank = sbert_job[sbert_job['cand_id'] == cand_id]['rank'].iloc[0]
                tfidf_ranks.append(tfidf_rank)
                sbert_ranks.append(sbert_rank)

            # Compute Spearman correlation
            if len(tfidf_ranks) > 1:
                corr, p_value = stats.spearmanr(tfidf_ranks, sbert_ranks)
                if not np.isnan(corr):
                    correlations.append(corr)

        return {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0,
            'correlations': correlations
        }

    def statistical_significance_test(self, tfidf_metric, sbert_metric):
        """Perform paired t-test for statistical significance"""
        if len(tfidf_metric) != len(sbert_metric):
            return {'significant': False, 'p_value': 1.0, 'reason': 'Unequal sample sizes'}

        if len(tfidf_metric) < 2:
            return {'significant': False, 'p_value': 1.0, 'reason': 'Insufficient data'}

        t_stat, p_value = stats.ttest_rel(sbert_metric, tfidf_metric)

        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat,
            'effect_size': np.mean(sbert_metric) - np.mean(tfidf_metric)
        }

    def generate_performance_plots(self, results):
        """Generate comprehensive performance visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Advanced Ranking Metrics Comparison', fontsize=16, fontweight='bold')

        # 1. MRR Comparison with Confidence Intervals
        ax1 = axes[0, 0]
        methods = ['TF-IDF', 'SBERT']
        mrr_values = [results['mrr']['tfidf']['mrr'], results['mrr']['sbert']['mrr']]
        ci_values = [results['mrr']['tfidf']['confidence_interval'], results['mrr']['sbert']['confidence_interval']]

        bars = ax1.bar(methods, mrr_values, color=['lightblue', 'lightcoral'], alpha=0.7)

        # Add error bars
        errors = [(ci[1] - ci[0]) / 2 for ci in ci_values]
        ax1.errorbar(methods, mrr_values, yerr=errors, fmt='none', color='black', capsize=5)

        ax1.set_title('Mean Reciprocal Rank (MRR)')
        ax1.set_ylabel('MRR Score')

        # Add significance annotation
        if 'statistical_tests' in results and 'mrr' in results['statistical_tests']:
            if results['statistical_tests']['mrr']['significant']:
                ax1.text(0.5, max(mrr_values) * 1.1, f"p = {results['statistical_tests']['mrr']['p_value']:.3f}*",
                         ha='center', fontweight='bold')

        # 2. NDCG@K Comparison
        ax2 = axes[0, 1]
        if 'ndcg' in results:
            k_values = list(results['ndcg']['tfidf'].keys())
            tfidf_ndcg = list(results['ndcg']['tfidf'].values())
            sbert_ndcg = list(results['ndcg']['sbert'].values())

            x = np.arange(len(k_values))
            width = 0.35

            ax2.bar(x - width / 2, tfidf_ndcg, width, label='TF-IDF', color='lightblue', alpha=0.7)
            ax2.bar(x + width / 2, sbert_ndcg, width, label='SBERT', color='lightcoral', alpha=0.7)
            ax2.set_title('NDCG@K')
            ax2.set_xlabel('K')
            ax2.set_ylabel('NDCG')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'@{k}' for k in k_values])
            ax2.legend()

        # 3. MAP@K Comparison
        ax3 = axes[0, 2]
        if 'map' in results:
            k_values = list(results['map']['tfidf'].keys())
            tfidf_map = list(results['map']['tfidf'].values())
            sbert_map = list(results['map']['sbert'].values())

            x = np.arange(len(k_values))

            ax3.bar(x - width / 2, tfidf_map, width, label='TF-IDF', color='lightblue', alpha=0.7)
            ax3.bar(x + width / 2, sbert_map, width, label='SBERT', color='lightcoral', alpha=0.7)
            ax3.set_title('MAP@K')
            ax3.set_xlabel('K')
            ax3.set_ylabel('MAP')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'@{k}' for k in k_values])
            ax3.legend()

        # 4. Ranking Correlation Distribution
        ax4 = axes[1, 0]
        if 'ranking_correlation' in results:
            correlations = results['ranking_correlation']['correlations']
            ax4.hist(correlations, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(np.mean(correlations), color='red', linestyle='--',
                        label=f'Mean: {np.mean(correlations):.3f}')
            ax4.set_title('Ranking Correlation Distribution')
            ax4.set_xlabel('Spearman Correlation')
            ax4.set_ylabel('Frequency')
            ax4.legend()

        # 5. Performance Improvement Heatmap
        ax5 = axes[1, 1]
        metrics = ['MRR', 'NDCG@5', 'MAP@5', 'Precision@5', 'Recall@5']
        improvements = []

        if all(metric in results for metric in ['mrr', 'ndcg', 'map', 'precision_recall']):
            mrr_imp = (results['mrr']['sbert']['mrr'] - results['mrr']['tfidf']['mrr']) / results['mrr']['tfidf'][
                'mrr'] * 100
            ndcg_imp = (results['ndcg']['sbert'][5] - results['ndcg']['tfidf'][5]) / results['ndcg']['tfidf'][
                5] * 100 if results['ndcg']['tfidf'][5] > 0 else 0
            map_imp = (results['map']['sbert'][5] - results['map']['tfidf'][5]) / results['map']['tfidf'][5] * 100 if \
            results['map']['tfidf'][5] > 0 else 0
            prec_imp = (results['precision_recall']['sbert']['precision_at_k'][5] -
                        results['precision_recall']['tfidf']['precision_at_k'][5]) / \
                       results['precision_recall']['tfidf']['precision_at_k'][5] * 100 if \
            results['precision_recall']['tfidf']['precision_at_k'][5] > 0 else 0
            rec_imp = (results['precision_recall']['sbert']['recall_at_k'][5] -
                       results['precision_recall']['tfidf']['recall_at_k'][5]) / \
                      results['precision_recall']['tfidf']['recall_at_k'][5] * 100 if \
            results['precision_recall']['tfidf']['recall_at_k'][5] > 0 else 0

            improvements = [mrr_imp, ndcg_imp, map_imp, prec_imp, rec_imp]

            colors = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax5.barh(metrics, improvements, color=colors, alpha=0.7)
            ax5.set_title('SBERT Performance Improvement (%)')
            ax5.set_xlabel('Percentage Improvement')
            ax5.axvline(0, color='black', linestyle='-', alpha=0.5)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, improvements)):
                ax5.text(val + (1 if val > 0 else -1), i, f'{val:.1f}%',
                         va='center', ha='left' if val > 0 else 'right')

        # 6. Statistical Significance Summary
        ax6 = axes[1, 2]
        if 'statistical_tests' in results:
            test_metrics = list(results['statistical_tests'].keys())
            p_values = [results['statistical_tests'][metric]['p_value'] for metric in test_metrics]
            significant = [p < 0.05 for p in p_values]

            colors = ['green' if sig else 'red' for sig in significant]
            bars = ax6.bar(test_metrics, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
            ax6.axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
            ax6.set_title('Statistical Significance (-log10(p-value))')
            ax6.set_ylabel('-log10(p-value)')
            ax6.set_xlabel('Metrics')
            ax6.legend()

            # Rotate x-axis labels
            plt.setp(ax6.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'comprehensive_advanced_metrics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        import numpy as np
        import pandas as pd

        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)  # Ensure Python bool, not numpy bool
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def run_comprehensive_evaluation(self):
        """Run complete advanced evaluation"""
        print("Starting comprehensive advanced evaluation...")

        # Create or use existing ground truth
        ground_truth = self.create_enhanced_ground_truth()
        print(f"Ground truth created/loaded: {len(ground_truth)} jobs with relevance data")

        # Compute all metrics
        results = {}

        # MRR with confidence intervals
        print("Computing MRR...")
        results['mrr'] = {
            'tfidf': self.compute_advanced_mrr(self.tfidf_results, ground_truth),
            'sbert': self.compute_advanced_mrr(self.sbert_results, ground_truth)
        }

        # NDCG
        print("Computing NDCG@K...")
        results['ndcg'] = {
            'tfidf': self.compute_ndcg_at_k(self.tfidf_results, ground_truth),
            'sbert': self.compute_ndcg_at_k(self.sbert_results, ground_truth)
        }

        # MAP
        print("Computing MAP@K...")
        results['map'] = {
            'tfidf': self.compute_map_at_k(self.tfidf_results, ground_truth),
            'sbert': self.compute_map_at_k(self.sbert_results, ground_truth)
        }

        # Precision/Recall (using existing implementation)
        print("Computing Precision/Recall@K...")
        results['precision_recall'] = {
            'tfidf': self.compute_precision_recall_at_k(self.tfidf_results, ground_truth),
            'sbert': self.compute_precision_recall_at_k(self.sbert_results, ground_truth)
        }

        # Ranking correlation
        print("Computing ranking correlation...")
        results['ranking_correlation'] = self.compute_ranking_correlation()

        # Statistical significance tests
        print("Running statistical significance tests...")
        results['statistical_tests'] = {}

        # MRR significance test
        results['statistical_tests']['mrr'] = self.statistical_significance_test(
            results['mrr']['tfidf']['reciprocal_ranks'],
            results['mrr']['sbert']['reciprocal_ranks']
        )

        # Generate comprehensive report
        self.print_comprehensive_report(results)

        # Generate plots
        print("Generating performance plots...")
        self.generate_performance_plots(results)

        # Save results - FIXED VERSION
        try:
            with open(self.plots_dir / 'evaluation_results.json', 'w') as f:
                results_serializable = self._make_json_serializable(results)
                json.dump(results_serializable, f, indent=2)
            print(f"Evaluation results saved to {self.plots_dir}/evaluation_results.json")
        except Exception as e:
            print(f"Warning: Could not save JSON results: {e}")
            # Still return results even if JSON save fails

        print(f"Evaluation complete! Results saved to {self.plots_dir}")
        return results



    def print_comprehensive_report(self, results):
        """Print detailed evaluation report"""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE ADVANCED EVALUATION REPORT")
        print("=" * 80)

        # MRR Results
        print(f"\n MEAN RECIPROCAL RANK (MRR):")
        tfidf_mrr = results['mrr']['tfidf']
        sbert_mrr = results['mrr']['sbert']

        print(f"  TF-IDF MRR: {tfidf_mrr['mrr']:.4f} ± {tfidf_mrr['std_error']:.4f}")
        print(f"  SBERT MRR:  {sbert_mrr['mrr']:.4f} ± {sbert_mrr['std_error']:.4f}")

        improvement = ((sbert_mrr['mrr'] - tfidf_mrr['mrr']) / tfidf_mrr['mrr']) * 100
        print(f"   SBERT Improvement: {improvement:+.2f}%")

        if 'statistical_tests' in results and 'mrr' in results['statistical_tests']:
            sig_test = results['statistical_tests']['mrr']
            significance = " Significant" if sig_test['significant'] else " Not significant"
            print(f"   Statistical Test: {significance} (p = {sig_test['p_value']:.4f})")

        # NDCG Results
        if 'ndcg' in results:
            print(f"\n NORMALIZED DISCOUNTED CUMULATIVE GAIN (NDCG):")
            for k in [1, 3, 5, 10]:
                if k in results['ndcg']['tfidf']:
                    tfidf_val = results['ndcg']['tfidf'][k]
                    sbert_val = results['ndcg']['sbert'][k]
                    improvement = ((sbert_val - tfidf_val) / tfidf_val) * 100 if tfidf_val > 0 else 0
                    print(f"  NDCG@{k} - TF-IDF: {tfidf_val:.4f}, SBERT: {sbert_val:.4f} ({improvement:+.2f}%)")

        # MAP Results
        if 'map' in results:
            print(f"\n MEAN AVERAGE PRECISION (MAP):")
            for k in [1, 3, 5, 10]:
                if k in results['map']['tfidf']:
                    tfidf_val = results['map']['tfidf'][k]
                    sbert_val = results['map']['sbert'][k]
                    improvement = ((sbert_val - tfidf_val) / tfidf_val) * 100 if tfidf_val > 0 else 0
                    print(f"  MAP@{k} - TF-IDF: {tfidf_val:.4f}, SBERT: {sbert_val:.4f} ({improvement:+.2f}%)")

        # Ranking Correlation
        if 'ranking_correlation' in results:
            corr_data = results['ranking_correlation']
            print(f"\n RANKING CORRELATION ANALYSIS:")
            print(
                f"  Mean Spearman Correlation: {corr_data['mean_correlation']:.4f} ± {corr_data['std_correlation']:.4f}")
            print(
                f"  Interpretation: {'Low correlation - methods capture different signals' if abs(corr_data['mean_correlation']) < 0.5 else 'Moderate to high correlation'}")

        print("\n" + "=" * 80)


    # Keep your existing methods but integrate them with the above improvements
    def compute_precision_recall_at_k(self, results_df, ground_truth, k_values=[1, 3, 5, 10], threshold=0.2):
        """Your existing method - keeping for compatibility"""
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}

        for job_id in self.job_ids:
            if job_id not in ground_truth or len(ground_truth[job_id]) == 0:
                continue

            relevant_candidates = set([
                cand_id for cand_id, score in ground_truth[job_id].items()
                if score >= threshold
            ])

            if len(relevant_candidates) == 0:
                continue

            job_results = results_df[results_df['job_id'] == job_id].sort_values('rank')

            for k in k_values:
                top_k_candidates = set(job_results.head(k)['cand_id'])
                true_positives = len(top_k_candidates.intersection(relevant_candidates))

                precision = true_positives / k if k > 0 else 0
                recall = true_positives / len(relevant_candidates) if len(relevant_candidates) > 0 else 0

                precision_at_k[k].append(precision)
                recall_at_k[k].append(recall)

        mean_precision = {k: np.mean(scores) for k, scores in precision_at_k.items()}
        mean_recall = {k: np.mean(scores) for k, scores in recall_at_k.items()}

        return {
            'precision_at_k': mean_precision,
            'recall_at_k': mean_recall,
            'precision_distribution': precision_at_k,
            'recall_distribution': recall_at_k
        }

