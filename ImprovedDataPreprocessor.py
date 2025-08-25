import logging

import numpy as np
import pandas as pd

from resume_ranking_env.resume_ranking_research.data.data_loader import load_config_new, DataLoader
from resume_ranking_env.resume_ranking_research.preprocessor import DataPreprocessor
import re
from collections import Counter
import math
from typing import Dict, List, Optional

import logging
import math
import re
from collections import defaultdict, Counter
from typing import Dict, List


class ImprovedDataPreprocessor(DataPreprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Seniority levels
        self.seniority_levels = [
            ("lead", [r"\blead\b", r"\bprincipal\b", r"\bdirector\b", r"\bhead\b", r"\bchief\b"]),
            ("senior", [r"\bsenior\b", r"\bmanager\b"]),
            ("mid", [r"\bmid\b", r"\bexperienced\b", r"\bspecialist\b", r"\banalyst\b", r"\bcoordinator\b"]),
            ("junior", [r"\bjunior\b", r"\bentry\b", r"\btrainee\b", r"\bintern\b", r"\bgraduate\b", r"\bassociate\b"]),
            ("intern", [r"\bintern\b"]),
        ]
        self.seniority_patterns = [
            (level, [re.compile(pat, flags=re.I) for pat in pats])
            for level, pats in self.seniority_levels
        ]

        # Domain categories
        base_domain_map = {
            'software_dev': [r"\bsoftware\b", r"\bdeveloper\b", r"\bengineer\b", r"\bprogrammer\b",
                             r"\bcoding\b", r"\bbackend\b", r"\bfront[- ]?end\b", r"\bfull[- ]?stack\b"],
            'data_science': [r"\bdata\b", r"\banalyst\b", r"\bscientist\b",
                             r"\bmachine learning\b", r"\bml\b", r"\bai\b",
                             r"\banalytics\b", r"\bbusiness intelligence\b", r"\bbi\b(?!\w)"],
            'design': [r"\bdesigner\b", r"\bui\b", r"\bux\b", r"\bgraphic\b",
                       r"\bvisual\b", r"\bcreative\b", r"\bproduct design\b"],
            'marketing': [r"\bmarketing\b", r"\bdigital marketing\b", r"\bseo\b",
                          r"\bsocial media\b", r"\bcontent\b", r"\bbrand(ing)?\b"],
            'sales': [r"\bsales\b", r"\bbusiness development\b", r"\baccount\b",
                      r"\bcustomer\b", r"\brelationship\b"],
            'hr': [r"\bhr\b", r"\bhuman resources\b", r"\brecruit(ment|er)\b", r"\btalent\b", r"\bpeople\b"],
            'finance': [r"\bfinance\b", r"\baccounting\b", r"\bfinancial\b", r"\bcontroller\b", r"\baudit(ing)?\b"],
            'operations': [r"\boperations?\b", r"\bproject manager\b", r"\bscrum\b", r"\bagile\b",
                           r"\bbusiness analyst\b"],
        }

        extend_domains = self.config.get("extend_domains", True)
        if extend_domains:
            base_domain_map.update({
                'education': [r"\bteacher\b", r"\bled?cturer\b", r"\bprofessor\b", r"\btutor\b", r"\bacademic\b",
                              r"\beducation(al)?\b", r"\bcurriculum\b"],
                'healthcare': [r"\bnurse\b", r"\bdoctor\b", r"\bphysician\b", r"\bclinician\b", r"\bmedical\b",
                               r"\bhospital\b", r"\bpharma(ceutical)?\b", r"\bclinical\b", r"\bhealth(care)?\b"],
                'legal': [r"\blawyer\b", r"\battorney\b", r"\bparalegal\b", r"\blegal\b", r"\bcounsel\b",
                          r"\bcompliance\b"],
                'consulting': [r"\bconsult(ant|ing)\b", r"\badvisor\b", r"\bstrategy\b", r"\bstrategist\b",
                               r"\bfreelance\b"],
            })

        self.domain_patterns = {dom: [re.compile(p, re.I) for p in pats] for dom, pats in base_domain_map.items()}

        # Keyword to domain mapping
        self.keyword_domain_map = {
            # software
            'python': 'software_dev', 'java': 'software_dev', 'javascript': 'software_dev',
            'frontend': 'software_dev', 'backend': 'software_dev', 'fullstack': 'software_dev',
            'c#': 'software_dev', 'php': 'software_dev', 'golang': 'software_dev', 'android': 'software_dev',
            'react': 'software_dev', 'node': 'software_dev', 'devops': 'software_dev',
            # data
            'data scientist': 'data_science', 'data analyst': 'data_science',
            'ml': 'data_science', 'ai': 'data_science', 'analytics': 'data_science',
            'business intelligence': 'data_science',
            # design
            'ux': 'design', 'ui': 'design', 'figma': 'design', 'graphic': 'design', 'product designer': 'design',
            # marketing
            'seo': 'marketing', 'smm': 'marketing', 'digital marketing': 'marketing',
            # sales
            'sales': 'sales', 'account manager': 'sales', 'business development': 'sales',
            # hr
            'recruiter': 'hr', 'talent acquisition': 'hr', 'hr': 'hr',
            # finance
            'finance': 'finance', 'accountant': 'finance', 'auditor': 'finance',
            # operations
            'project manager': 'operations', 'scrum master': 'operations', 'business analyst': 'operations',
            # extended
            'teacher': 'education', 'lecturer': 'education', 'professor': 'education',
            'nurse': 'healthcare', 'doctor': 'healthcare', 'lawyer': 'legal', 'attorney': 'legal',
            'consultant': 'consulting', 'strategy': 'consulting',
        }
        self.keyword_domain_map = {k.lower(): v for k, v in self.keyword_domain_map.items()}

        # Experience extraction patterns
        YEAR_TOKENS = r"(?:years?|yrs?|yoe)\b"
        MONTH_TOKENS = r"(?:months?|mos?)\b"
        RANGE_SEP = r"(?:-|to|–|—|~)"
        QUALIFIERS = r"(?:at\s+least|minimum|min\.?|more\s+than|over|about|approx\.?|~)?"

        self._re_range = re.compile(fr"{QUALIFIERS}\s*(\d+(?:\.\d+)?)\s*{RANGE_SEP}\s*(\d+(?:\.\d+)?)\s*{YEAR_TOKENS}",
                                    re.I)
        self._re_years = re.compile(
            fr"{QUALIFIERS}\s*(\d+(?:\.\d+)?)\s*\+?\s*{YEAR_TOKENS}(?:\s*(?:of)?\s*(?:experience|exp))?", re.I)
        self._re_months = re.compile(
            fr"{QUALIFIERS}\s*(\d+(?:\.\d+)?)\s*{MONTH_TOKENS}(?:\s*(?:of)?\s*(?:experience|exp))?", re.I)

        # Level bins for experience mapping
        self.LEVEL_BINS = [
            (0.0, 1.0, "intern"),
            (1.0, 3.0, "junior"),
            (3.0, 6.0, "mid"),
            (6.0, 10.0, "senior"),
            (10.0, 1e9, "lead"),
        ]

        # Fallback skill to domain mapping
        self.fallback_skill_to_domain = {
            'python': 'data_science', 'pandas': 'data_science', 'numpy': 'data_science',
            'machine learning': 'data_science', 'ml': 'data_science', 'ai': 'data_science',
            'java': 'software_dev', 'javascript': 'software_dev', 'typescript': 'software_dev',
            'react': 'software_dev', 'node': 'software_dev', 'spring': 'software_dev', 'devops': 'software_dev',
            'figma': 'design', 'ux': 'design', 'ui': 'design',
            'seo': 'marketing', 'google ads': 'marketing',
            'gaap': 'finance', 'accounting': 'finance', 'recruitment': 'hr',
            'curriculum': 'education', 'clinical': 'healthcare', 'legal': 'legal',
            'compliance': 'legal', 'consulting': 'consulting', 'strategy': 'consulting',
        }
        self.fallback_skill_to_domain = {k.lower(): v for k, v in self.fallback_skill_to_domain.items()}

        # Lead signals
        self._lead_signals = [re.compile(p, re.I) for p in [
            r"\bteam\s*lead\b", r"\blead\b", r"\bhead\b", r"\bdirector\b",
            r"\bmanager\b", r"\bengineering\s+manager\b", r"\bchief\b",
            r"\bprincipal\b", r"\bstaff\b", r"\barchitect\b"
        ]]

    @staticmethod
    def _first_num(x):
        """Extract first number from text"""
        if pd.isna(x):
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", str(x))
        return float(m.group(1)) if m else None

    def _extract_years_from_text(self, text: str) -> float:
        """Extract years of experience from text"""
        if not isinstance(text, str) or not text.strip():
            return np.nan

        s = text.lower().replace("–", "-").replace("—", "-")
        candidates = []

        # Extract ranges and convert to midpoint
        candidates += [(float(a) + float(b)) / 2.0 for a, b in self._re_range.findall(s)]
        # Extract single years
        candidates += [float(y) for y in self._re_years.findall(s)]
        # Extract months and convert to years
        candidates += [float(m) / 12.0 for m in self._re_months.findall(s)]

        return max(candidates) if candidates else np.nan

    def _years_to_level(self, y: float) -> str:
        """Convert years of experience to seniority level"""
        if pd.isna(y):
            return "unknown"
        for lo, hi, label in self.LEVEL_BINS:
            if lo <= y < hi:
                return label
        return "unknown"

    def _seniority_from_years(self, y: float) -> str:
        """Get seniority level from years of experience"""
        if pd.isna(y):
            return "unknown"
        for lo, hi, label in self.LEVEL_BINS:
            if lo <= y < hi:
                return label
        return "unknown"

    def _force_lead_if_signal(self, text: str, current: str) -> str:
        """Upgrade to 'lead' if strong lead/manager tokens are present"""
        if current in ("lead", "senior"):
            return current
        for pattern in self._lead_signals:
            if pattern.search(text or ""):
                return "lead"
        return current

    def _domain_by_regex(self, text: str) -> str:
        """Classify domain using regex patterns"""
        if not isinstance(text, str) or not text.strip():
            return 'unknown'

        scores = Counter()
        for domain, patterns in self.domain_patterns.items():
            hits = sum(len(pattern.findall(text)) for pattern in patterns)
            if hits:
                scores[domain] += hits

        if not scores:
            return 'unknown'

        max_val = max(scores.values())
        # Deterministic tie-breaking by domain name
        return sorted([k for k, v in scores.items() if v == max_val])[0]

    def _refine_unknown(self, row: pd.Series, extra_fields: List[str]) -> str:
        """Refine unknown domains using skill keywords"""
        blob = " ".join(str(row.get(c, "")) for c in extra_fields if c in row).lower()
        votes = Counter()

        for token, domain in self.fallback_skill_to_domain.items():
            if token in blob:
                votes[domain] += 1

        if not votes:
            return "unknown"

        domain, count = votes.most_common(1)[0]
        # Require at least 2 signals to reclassify (conservative)
        return domain if count >= 2 else "unknown"

    def extract_seniority_level(self, text: str) -> str:
        """Extract seniority level from text"""
        if not isinstance(text, str) or not text.strip():
            return 'unknown'

        for level, patterns in self.seniority_patterns:
            if any(pattern.search(text) for pattern in patterns):
                return level
        return 'unknown'

    def _domain_candidate(self, row: pd.Series) -> str:
        """Determine domain for candidate"""
        # 1) Primary Keyword first (structured)
        pk = str(row.get('Primary Keyword', '')).lower().strip()
        if pk in self.keyword_domain_map:
            return self.keyword_domain_map[pk]

        # 2) Regex over Position + Moreinfo + CV
        text = " ".join([
            str(row.get('Position', '')),
            str(row.get('Moreinfo', '')),
            str(row.get('CV', ''))
        ]).lower()

        domain = self._domain_by_regex(text)
        if domain != "unknown":
            return domain

        # 3) Conservative refinement using skills across fields
        return self._refine_unknown(row, ['Primary Keyword', 'Position', 'Moreinfo', 'CV'])

    def calculate_experience_gap(self, job_exp, candidate_exp):
        """Calculate experience gap score"""
        j = self._first_num(job_exp)
        c = self._first_num(candidate_exp)
        if j is None or c is None:
            return 0.5
        gap = abs(j - c)
        tau = 2.0
        return float(math.exp(-gap / tau))

    def preprocess_dataframe(self, df: pd.DataFrame, is_jobs: bool = True) -> pd.DataFrame:
        """Main preprocessing method"""
        try:
            df = super().preprocess_dataframe(df, is_jobs)

            # Find relevant columns
            title_col = next((c for c in ['Position', 'Title', 'Job Title', 'job_title', 'position']
                              if c in df.columns), None)
            desc_col = next((c for c in ['Long Description', 'Description', 'Job Description',
                                         'description', 'job_description', 'Details']
                             if c in df.columns), None)

            # ========= 1) EXPERIENCE EXTRACTION =========
            self.logger.info("Starting experience extraction...")

            # Extract experience from Moreinfo if available
            if 'Moreinfo' in df.columns:
                df['exp_years_from_moreinfo'] = df['Moreinfo'].apply(self._extract_years_from_text)
                self.logger.info(
                    f"Extracted experience from Moreinfo: {df['exp_years_from_moreinfo'].notna().sum()} non-null values")
            else:
                df['exp_years_from_moreinfo'] = np.nan

            # Find existing experience column
            exp_numeric_col = next((c for c in ['Experience Years', 'Exp Years', 'experience_years',
                                                'Experience', 'exp_years'] if c in df.columns), None)

            if exp_numeric_col:
                self.logger.info(f"Found experience column: {exp_numeric_col}")
                df['experience_years_numeric'] = df[exp_numeric_col].apply(self._first_num)

                # FIXED: Use proper fillna with actual values, not None
                # Fill NaN values in experience_years_numeric with values from exp_years_from_moreinfo
                if 'exp_years_from_moreinfo' in df.columns:
                    df['exp_years_final'] = df['experience_years_numeric'].fillna(df['exp_years_from_moreinfo'])
                else:
                    df['exp_years_final'] = df['experience_years_numeric']
            else:
                self.logger.info("No structured experience column found, using extracted values")
                df['exp_years_final'] = df.get('exp_years_from_moreinfo', np.nan)

            # Convert experience years to seniority levels
            if 'exp_years_final' in df.columns:
                df['exp_level_final'] = df['exp_years_final'].apply(self._years_to_level)

            # ========= 2) SENIORITY LEVEL EXTRACTION =========
            self.logger.info("Starting seniority level extraction...")

            # Extract seniority from title
            if title_col:
                df['seniority_level'] = df[title_col].astype(str).str.lower().apply(self.extract_seniority_level)
            else:
                df['seniority_level'] = 'unknown'

            # Upgrade to 'lead' if there are strong lead/manager signals
            if title_col:
                if desc_col:
                    combined_title_desc = (df[title_col].astype(str) + " " + df[desc_col].astype(str)).str.lower()
                else:
                    combined_title_desc = df[title_col].astype(str).str.lower()

                # Upgrade to 'lead' where signals exist
                df['seniority_level'] = [
                    self._force_lead_if_signal(txt, lvl)
                    for txt, lvl in zip(combined_title_desc, df['seniority_level'])
                ]

            # Fill remaining unknown/empty from years-of-experience
            if 'exp_years_final' in df.columns:
                by_years = df['exp_years_final'].apply(self._seniority_from_years)
                mask = (df['seniority_level'].isna() |
                        df['seniority_level'].eq('') |
                        df['seniority_level'].eq('unknown'))
                df.loc[mask, 'seniority_level'] = by_years[mask]

            # ========= 3) DOMAIN CLASSIFICATION =========
            self.logger.info("Starting domain classification...")
            df['domain_category'] = df.apply(self._domain_candidate, axis=1)

            self.logger.info("Preprocessing completed successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise



class EnhancedGroundTruthGenerator:
    """Enhanced Ground Truth Generator with improved error handling"""

    # Weights for different components
    W_KEYWORD = 0.35
    W_DOMAIN = 0.25
    W_SENIORITY = 0.20
    W_EXPER = 0.15
    W_TITLE = 0.5

    def __init__(self, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame, seed: int = 42):
        self.logger = logging.getLogger(__name__)
        self.jobs_df = jobs_df
        self.candidates_df = candidates_df
        self.seed = seed

        # Initialize string similarity function
        self._string_ratio = None
        try:
            from rapidfuzz import fuzz as rf_fuzz
            self._string_ratio = lambda a, b: rf_fuzz.token_set_ratio(a, b) / 100.0
            self.logger.info("Using RapidFuzz for string similarity")
        except ImportError:
            try:
                from fuzzywuzzy import fuzz
                self._string_ratio = lambda a, b: fuzz.token_set_ratio(a, b) / 100.0
                self.logger.info("Using FuzzyWuzzy for string similarity")
            except ImportError:
                self.logger.warning("No fuzzy library found; falling back to naive ratio.")
                self._string_ratio = self._naive_ratio

        # Seniority order consistent with preprocessor
        self._seniority_order = {'intern': 0, 'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4}

        # Domain relatedness mapping
        self._related_domains = {
            ('software_dev', 'data_science'),
            ('design', 'marketing'),
            ('sales', 'marketing'),
            ('operations', 'data_science'),
        }

    def create_enhanced_ground_truth(
            self,
            job_ids: List[int],
            max_candidates_per_job: int = 200,
            min_relevance: float = 0.10
    ) -> Dict[int, Dict[int, float]]:
        """
        Create enhanced ground truth with error handling
        Returns: { job_id: { cand_id: relevance in [0,1] } }
        """
        try:
            gt = {}
            total_jobs = len(job_ids)
            rng = np.random.default_rng(self.seed)

            self.logger.info(f"Starting ground truth generation for {total_jobs} jobs")

            for idx, job_id in enumerate(job_ids):
                if job_id >= len(self.jobs_df):
                    continue

                if total_jobs and idx % max(1, total_jobs // 10) == 0:
                    self.logger.info(f"Processing job {idx}/{total_jobs} ({idx / max(total_jobs, 1) * 100:.1f}%)")

                try:
                    job = self.jobs_df.iloc[job_id]

                    # Extract job characteristics
                    j_kw = self._norm(job.get('Primary Keyword_normalized', ''))
                    j_pos = self._norm(job.get('Position', ''))
                    j_desc = self._norm(job.get('Long Description', job.get('Description', '')))
                    j_dom = job.get('domain_category', 'unknown')
                    j_sen = job.get('seniority_level', 'unknown')
                    j_exp = self._coerce_years(job.get('exp_years_final', job.get('Experience Years', np.nan)))

                    # Candidate sampling (deterministic)
                    total_cands = len(self.candidates_df)
                    if max_candidates_per_job and total_cands > max_candidates_per_job:
                        cand_indices = rng.choice(total_cands, size=max_candidates_per_job, replace=False)
                    else:
                        cand_indices = np.arange(total_cands)

                    rels = {}

                    for cand_id in cand_indices:
                        try:
                            c = self.candidates_df.iloc[cand_id]

                            # Extract candidate characteristics
                            c_kw = self._norm(c.get('Primary Keyword_normalized', ''))
                            c_pos = self._norm(c.get('Position', ''))
                            c_cv = self._norm(c.get('CV', c.get('Moreinfo', '')))
                            c_dom = c.get('domain_category', 'unknown')
                            c_sen = c.get('seniority_level', 'unknown')
                            c_exp = self._coerce_years(c.get('exp_years_final', c.get('Experience Years', np.nan)))

                            # Calculate component scores
                            keyword_score = self._keyword_overlap(j_kw or j_pos, c_kw or c_pos)
                            if not keyword_score and j_kw:
                                keyword_score = 0.5 if j_kw in c_cv else 0.0

                            domain_score = self._domain_score(j_dom, c_dom)
                            seniority_score = self._seniority_score(j_sen, c_sen)
                            experience_score = self._experience_score(j_exp, c_exp)

                            title_score = 0.0
                            if j_pos and c_pos:
                                try:
                                    title_score = self._string_ratio(j_pos, c_pos)
                                except Exception as e:
                                    self.logger.warning(f"Title similarity error: {e}")
                                    title_score = 0.0

                            # Calculate final relevance
                            relevance = (
                                    keyword_score * self.W_KEYWORD +
                                    domain_score * self.W_DOMAIN +
                                    seniority_score * self.W_SENIORITY +
                                    experience_score * self.W_EXPER +
                                    title_score * self.W_TITLE
                            )

                            if relevance >= min_relevance:
                                rels[int(cand_id)] = float(min(1.0, max(0.0, relevance)))

                        except Exception as e:
                            self.logger.warning(f"Error processing candidate {cand_id}: {e}")
                            continue

                    gt[int(job_id)] = rels

                except Exception as e:
                    self.logger.warning(f"Error processing job {job_id}: {e}")
                    continue

            self.logger.info(f"Enhanced ground truth completed for {len(gt)} jobs")
            return gt

        except Exception as e:
            self.logger.error(f"Fatal error in ground truth generation: {e}")
            return {}

    @staticmethod
    def _norm(x) -> str:
        """Normalize text input"""
        return str(x).strip().lower() if isinstance(x, str) else ""

    @staticmethod
    def _token_set(s: str) -> set:
        """Extract tokens from string"""
        return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s))

    def _keyword_overlap(self, a: str, b: str) -> float:
        """Calculate Jaccard similarity between token sets"""
        A, B = self._token_set(a), self._token_set(b)
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union else 0.0

    def _domain_score(self, d1: str, d2: str) -> float:
        """Calculate domain matching score"""
        if d1 == 'unknown' or d2 == 'unknown':
            return 0.5
        if d1 == d2:
            return 1.0
        pair = tuple(sorted((d1, d2)))
        return 0.6 if pair in self._related_domains else 0.2

    def _seniority_score(self, j: str, c: str) -> float:
        """Calculate seniority matching score"""
        if j not in self._seniority_order or c not in self._seniority_order:
            return 0.5  # Unknown-safe neutral

        diff = abs(self._seniority_order[j] - self._seniority_order[c])
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.7
        elif diff == 2:
            return 0.4
        else:
            return 0.2

    @staticmethod
    def _coerce_years(x):
        """Extract numeric years from various formats"""
        if pd.isna(x):
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", str(x))
        return float(m.group(1)) if m else None

    @staticmethod
    def _experience_score(j, c, tau: float = 2.0) -> float:
        """Calculate experience gap score with exponential decay"""
        if j is None or c is None:
            return 0.5
        gap = abs(j - c)
        return float(math.exp(-gap / tau))

    @staticmethod
    def _naive_ratio(a: str, b: str) -> float:
        """Fallback string similarity function"""
        if not a or not b:
            return 0.0
        A, B = set(a.split()), set(b.split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)


def extract_dataset_statistics(jobs_df, candidates_df, logger=None):
    """
    Extract comprehensive dataset characteristics and preprocessing outcomes
    """
    global std_job_length_new, avg_job_length_new
    import json

    if logger is None:
        logger = logging.getLogger(__name__)

    print("\n" + "=" * 80)
    print("DATASET CHARACTERISTICS AND PREPROCESSING OUTCOMES")
    print("=" * 80)

    stats = {}

    # ============ DATA DISTRIBUTION AND REPRESENTATIVENESS ============

    print("\nProfessional Domain Distribution:")
    if 'domain_category' in jobs_df.columns:
        jobs_domain_counts = jobs_df['domain_category'].value_counts()
        jobs_domain_pct = jobs_df['domain_category'].value_counts(normalize=True) * 100

        print("Jobs by Domain:")
        for domain, count in jobs_domain_counts.items():
            pct = jobs_domain_pct[domain]
            formatted_domain = domain.title().replace('_', ' ')
            print(f"  {formatted_domain}: {count:,} jobs ({pct:.1f}%)")

        stats['jobs_domain_distribution'] = jobs_domain_counts.to_dict()

    if 'domain_category' in candidates_df.columns:
        cand_domain_counts = candidates_df['domain_category'].value_counts()
        cand_domain_pct = candidates_df['domain_category'].value_counts(normalize=True) * 100

        print("\nCandidates by Domain:")
        for domain, count in cand_domain_counts.items():
            pct = cand_domain_pct[domain]
            formatted_domain = domain.title().replace('_', ' ')
            print(f"  {formatted_domain}: {count:,} candidates ({pct:.1f}%)")

        stats['candidates_domain_distribution'] = cand_domain_counts.to_dict()

    # Experience Level Distribution
    print("\nExperience Level Distribution:")
    if 'seniority_level' in jobs_df.columns:
        jobs_seniority = jobs_df['seniority_level'].value_counts()
        jobs_sen_pct = jobs_df['seniority_level'].value_counts(normalize=True) * 100

        print("Jobs by Seniority:")
        for level, count in jobs_seniority.items():
            pct = jobs_sen_pct[level]
            print(f"  {level.title()} positions: {count:,} jobs ({pct:.1f}%)")

        stats['jobs_seniority_distribution'] = jobs_seniority.to_dict()

    if 'seniority_level' in candidates_df.columns:
        cand_seniority = candidates_df['seniority_level'].value_counts()
        cand_sen_pct = candidates_df['seniority_level'].value_counts(normalize=True) * 100

        print("\nCandidates by Seniority:")
        for level, count in cand_seniority.items():
            pct = cand_sen_pct[level]
            print(f"  {level.title()} level: {count:,} candidates ({pct:.1f}%)")

        stats['candidates_seniority_distribution'] = cand_seniority.to_dict()

    # ============ TEXT PROCESSING AND FEATURE EXTRACTION RESULTS ============

    print("\n" + "=" * 60)
    print("TEXT PROCESSING AND FEATURE EXTRACTION RESULTS")
    print("=" * 60)

    # Document Length and Complexity Statistics
    print("\nDocument Length and Complexity Statistics:")

    if 'combined_text' in jobs_df.columns:
        job_lengths = jobs_df['combined_text'].dropna().str.split().str.len()
        avg_job_length_new = job_lengths.mean()
        std_job_length_new = job_lengths.std()
        print(f"Average job description length: {avg_job_length_new:.0f} ± {std_job_length_new:.0f} words (post-cleaning)")
        stats['avg_job_length'] = avg_job_length_new
        stats['std_job_length'] = std_job_length_new

    if 'combined_text' in candidates_df.columns:
        resume_lengths = candidates_df['combined_text'].dropna().str.split().str.len()
        avg_resume_length = resume_lengths.mean()
        std_resume_length = resume_lengths.std()
        print(f"Average resume summary length: {avg_resume_length:.0f} ± {std_resume_length:.0f} words (post-cleaning)")
        stats['avg_resume_length'] = avg_resume_length
        stats['std_resume_length'] = std_resume_length

        # Combined document representations
        if 'combined_text' in jobs_df.columns:
            combined_avg = (avg_job_length_new + avg_resume_length) / 2
            combined_std = (std_job_length_new + std_resume_length) / 2
            print(f"Combined document representations: {combined_avg:.0f} ± {combined_std:.0f} words average")
            stats['combined_avg_length'] = combined_avg

    # Feature Extraction Success Rates
    print("\nFeature Extraction Success Rates:")

    if 'seniority_level' in jobs_df.columns:
        seniority_success = (jobs_df['seniority_level'] != 'unknown').sum()
        seniority_total = len(jobs_df)
        seniority_rate = (seniority_success / seniority_total) * 100
        print(f"Seniority level detection: {seniority_rate:.1f}% success rate")
        stats['seniority_detection_rate'] = seniority_rate

    if 'domain_category' in jobs_df.columns:
        domain_success = (jobs_df['domain_category'] != 'unknown').sum()
        domain_total = len(jobs_df)
        domain_rate = (domain_success / domain_total) * 100
        print(f"Domain classification: {domain_rate:.1f}% success rate")
        stats['domain_classification_rate'] = domain_rate

    if 'exp_years_final' in candidates_df.columns:
        exp_success = candidates_df['exp_years_final'].notna().sum()
        exp_total = len(candidates_df)
        exp_rate = (exp_success / exp_total) * 100
        print(f"Experience quantification: {exp_rate:.1f}% successful extraction rate")
        stats['experience_extraction_rate'] = exp_rate

    # Vocabulary overlap analysis
    if 'combined_text' in jobs_df.columns and 'combined_text' in candidates_df.columns:
        job_vocab = set()
        candidate_vocab = set()

        for text in jobs_df['combined_text'].dropna():
            job_vocab.update(str(text).lower().split())

        for text in candidates_df['combined_text'].dropna():
            candidate_vocab.update(str(text).lower().split())

        overlap = len(job_vocab.intersection(candidate_vocab))
        total_unique = len(job_vocab.union(candidate_vocab))
        overlap_pct = (overlap / total_unique) * 100

        print(f"Vocabulary overlap between jobs and resumes: {overlap_pct:.1f}%")
        stats['vocabulary_overlap_pct'] = overlap_pct

    # Save statistics to file
    stats['total_jobs'] = len(jobs_df)
    stats['total_candidates'] = len(candidates_df)

    # Save to JSON file
    with open('dataset_statistics.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, (dict, pd.Series)) else float(v) if pd.notna(v) else None
                   for k, v in stats.items()}, f, indent=2)

    logger.info("Dataset statistics saved to dataset_statistics.json")

    return stats


from pathlib import Path

if __name__ == '__main__':
    config = load_config_new()
    loader = DataLoader(config)
    jobs_df, candidates_df = loader.load_data()

    pre = ImprovedDataPreprocessor(config)

    # jobs (optional, unchanged)
    jobs_df = pre.preprocess_dataframe(jobs_df, is_jobs=True)

    # candidates: extract exp from Moreinfo and merge
    candidates_df = pre.preprocess_dataframe(candidates_df, is_jobs=False)

    # ---- persist processed data ----
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_out = out_dir / "candidates_enriched.parquet"
    jobs_out = out_dir / "jobs_enriched.parquet"

    candidates_df.to_parquet(candidates_out, index=False)
    jobs_df.to_parquet(jobs_out, index=False)

    cols = ['Position', 'Moreinfo', 'Experience Years',
            'exp_years_from_moreinfo', 'exp_years_final', 'exp_level_final']

    # Keep only columns that exist in df (in case some are missing)
    cols = [c for c in cols if c in candidates_df.columns]

    print("\n--- Extracted Experience Columns ---")

    print(candidates_df["exp_years_from_moreinfo"].head(20))  # first 20 rows
    print(candidates_df["Experience Years"].head(20))  # first 20 rows
    print(candidates_df["exp_years_final"].head(20))  # first 20 rows
    print(candidates_df["exp_level_final"].head(20))  # first 20 rows
    print(candidates_df[cols].head(20))  # first 20 rows
    # # ---- quick peek ----
    # cols = ['Position','Moreinfo','Experience Years',
    #         'exp_years_from_moreinfo','exp_years_final','exp_level_final']
    # print("\n--- Candidates (experience extraction preview) ---")
    # print(candidates_df[[c for c in cols if c in candidates_df.columns]].head(10))

    print("\nData processed & saved")











