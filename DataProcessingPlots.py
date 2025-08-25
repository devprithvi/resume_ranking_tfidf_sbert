# =======================
# plotting_utils.py
# =======================
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RankingPlots:
    def __init__(self, plots_dir: str = "plots"):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    # ---------- helpers ----------
    def _save(self, name: str):
        path = os.path.join(self.plots_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def _vc(df: pd.Series, top_n=None):
        vc = df.value_counts(dropna=False)
        if top_n:
            vc = vc.head(top_n)
        return vc

    @staticmethod
    def _percent(s: pd.Series):
        total = s.sum() if isinstance(s, pd.Series) else len(s)
        return (s / total * 100.0).round(1)

    # ---------- plots ----------
    def seniority_distribution(self, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame):
        have_j = 'seniority_level' in jobs_df.columns
        have_c = 'seniority_level' in candidates_df.columns
        if not (have_j or have_c):
            return None, None

        order = ['intern','junior','mid','senior','lead','unknown']

        if have_j:
            s = jobs_df['seniority_level'].fillna('unknown')
            counts = s.value_counts().reindex(order, fill_value=0)
            perc = self._percent(counts)
            plt.figure()
            counts.plot(kind='bar')
            plt.title("Jobs — Seniority Distribution")
            plt.xlabel("Seniority")
            plt.ylabel("Count")
            for i, v in enumerate(counts):
                plt.text(i, v, f"{perc.iloc[i]}%", ha='center', va='bottom', fontsize=9, rotation=0)
            p1 = self._save("jobs_seniority_distribution.png")
        else:
            p1 = None

        if have_c:
            s = candidates_df['seniority_level'].fillna('unknown')
            counts = s.value_counts().reindex(order, fill_value=0)
            perc = self._percent(counts)
            plt.figure()
            counts.plot(kind='bar')
            plt.title("Candidates — Seniority Distribution")
            plt.xlabel("Seniority")
            plt.ylabel("Count")
            for i, v in enumerate(counts):
                plt.text(i, v, f"{perc.iloc[i]}%", ha='center', va='bottom', fontsize=9)
            p2 = self._save("candidates_seniority_distribution.png")
        else:
            p2 = None

        return p1, p2

    def domain_distribution(self, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame):
        have_j = 'domain_category' in jobs_df.columns
        have_c = 'domain_category' in candidates_df.columns
        if not (have_j or have_c):
            return None, None

        # stable order
        order = ['software_dev','data_science','marketing','design','finance','operations','sales','hr','unknown']

        if have_j:
            s = jobs_df['domain_category'].fillna('unknown')
            counts = s.value_counts().reindex(order, fill_value=0)
            perc = self._percent(counts)
            plt.figure()
            counts.plot(kind='bar')
            plt.title("Jobs — Domain Distribution")
            plt.xlabel("Domain")
            plt.ylabel("Count")
            for i, v in enumerate(counts):
                plt.text(i, v, f"{perc.iloc[i]}%", ha='center', va='bottom', fontsize=9, rotation=0)
            p1 = self._save("jobs_domain_distribution.png")
        else:
            p1 = None

        if have_c:
            s = candidates_df['domain_category'].fillna('unknown')
            counts = s.value_counts().reindex(order, fill_value=0)
            perc = self._percent(counts)
            plt.figure()
            counts.plot(kind='bar')
            plt.title("Candidates — Domain Distribution")
            plt.xlabel("Domain")
            plt.ylabel("Count")
            for i, v in enumerate(counts):
                plt.text(i, v, f"{perc.iloc[i]}%", ha='center', va='bottom', fontsize=9)
            p2 = self._save("candidates_domain_distribution.png")
        else:
            p2 = None

        return p1, p2

    def experience_histogram(self, candidates_df: pd.DataFrame, bins: int = 20, cap_years: float = 30.0):
        if 'exp_years_final' not in candidates_df.columns and 'Experience Years' not in candidates_df.columns:
            return None, None

        # prefer parsed years
        exp = candidates_df.get('exp_years_final', candidates_df.get('Experience Years')).astype(float)
        exp = exp.replace([np.inf, -np.inf], np.nan).dropna()
        exp = exp.clip(lower=0, upper=cap_years)

        plt.figure()
        plt.hist(exp, bins=bins)
        plt.title("Candidates — Experience (Years)")
        plt.xlabel("Years")
        plt.ylabel("Frequency")
        p = self._save("candidates_experience_hist.png")

        # summary table
        summary = pd.Series({
            "count": int(exp.shape[0]),
            "mean_years": float(exp.mean()) if exp.size else np.nan,
            "median_years": float(exp.median()) if exp.size else np.nan,
            "p25_years": float(exp.quantile(0.25)) if exp.size else np.nan,
            "p75_years": float(exp.quantile(0.75)) if exp.size else np.nan,
            "max_years": float(exp.max()) if exp.size else np.nan,
        })
        summary_path = os.path.join(self.plots_dir, "candidates_experience_summary.csv")
        summary.to_csv(summary_path)

        return p, summary_path

    def experience_by_seniority_boxplot(self, candidates_df: pd.DataFrame, cap_years: float = 30.0):
        need_cols = {'exp_years_final', 'seniority_level'}
        if not need_cols.issubset(set(candidates_df.columns)):
            return None

        df = candidates_df[['seniority_level','exp_years_final']].copy()
        df = df.dropna()
        df['exp_years_final'] = df['exp_years_final'].astype(float).clip(0, cap_years)

        order = ['intern','junior','mid','senior','lead','unknown']
        grouped = [df.loc[df['seniority_level']==k, 'exp_years_final'].values for k in order if k in df['seniority_level'].unique()]

        plt.figure()
        plt.boxplot(grouped, labels=[k for k in order if k in df['seniority_level'].unique()], showfliers=False)
        plt.title("Candidates — Experience by Seniority")
        plt.xlabel("Seniority")
        plt.ylabel("Experience (Years)")
        return self._save("candidates_experience_by_seniority_box.png")

    def top_keywords(self, df: pd.DataFrame, which: str = 'jobs', col: str = 'Primary Keyword_normalized', top_n: int = 15):
        if col not in df.columns:
            return None
        s = df[col].dropna().astype(str).str.strip().str.lower()
        counts = s.value_counts().head(top_n)

        plt.figure()
        counts.plot(kind='bar')
        plt.title(f"{which.capitalize()} — Top {top_n} Primary Keywords")
        plt.xlabel("Keyword")
        plt.ylabel("Count")
        for i, v in enumerate(counts):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
        return self._save(f"{which}_top{top_n}_primary_keywords.png")

    # ---------- orchestrator ----------
    def generate_all(self, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame):
        out = {}
        out['jobs_seniority'], out['cands_seniority'] = self.seniority_distribution(jobs_df, candidates_df)
        out['jobs_domain'], out['cands_domain'] = self.domain_distribution(jobs_df, candidates_df)
        out['cands_exp_hist'], out['cands_exp_summary'] = self.experience_histogram(candidates_df)
        out['cands_exp_box_by_seniority'] = self.experience_by_seniority_boxplot(candidates_df)
        out['jobs_keywords'] = self.top_keywords(jobs_df, which='jobs')
        out['cands_keywords'] = self.top_keywords(candidates_df, which='candidates')
        return out
