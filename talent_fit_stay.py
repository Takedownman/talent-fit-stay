#!/usr/bin/env python3
"""
Talent Fit & Stay — Takedownman
Resume-to-job matching plus attrition-risk scoring.

Portfolio starter by Takedownman.
Synthetic attrition labels — not real HR data.
Do not train on age, gender, race, or disability.

Run:
  pip install scikit-learn numpy
  python talent_fit.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__author__ = "Takedownman"

SKILL_LEXICON = {
    "python", "java", "javascript", "typescript", "sql", "r", "c++", "c#",
    "go", "rust", "scala", "pandas", "numpy", "scikit-learn", "sklearn",
    "pytorch", "tensorflow", "keras", "xgboost", "lightgbm", "spark",
    "hadoop", "airflow", "dbt", "snowflake", "redshift", "bigquery",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux",
    "git", "ci/cd", "nlp", "llm", "transformers", "huggingface",
    "langchain", "fastapi", "flask", "django", "react", "node", "excel",
    "tableau", "power bi", "statistics", "experimentation", "a/b testing",
    "mlops", "feature store",
}

ROLE_HINTS = (
    "engineer", "scientist", "analyst", "manager", "director",
    "intern", "consultant", "developer", "architect", "researcher",
)

YEAR_SPAN = re.compile(
    r"(?:19|20)\d{2}\s*[-–—to]+\s*(?:(?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)
BARE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

SAMPLE_RESUME = """Jordan Hale
Senior Machine Learning Engineer

Experience
Staff ML Engineer, Northwind Labs, 2022-present
  PyTorch, FastAPI, AWS, Docker, Kubernetes. Shipped ranking models.

ML Engineer, Harbor Analytics, 2019-2022
  scikit-learn, pandas, SQL, Airflow, experiment design.

Data Scientist, Riverbank, 2016-2019
  Python, statistics, Tableau, A/B testing.

Skills: Python, SQL, PyTorch, scikit-learn, AWS, Docker, Kubernetes, NLP
"""

SAMPLE_JOB = """Machine Learning Engineer

We need someone who can take models to production.

Required: Python, SQL, PyTorch or TensorFlow, Docker, AWS, CI/CD.
Nice: Kubernetes, Airflow, NLP, experiment design.
"""


@dataclass
class ParsedDoc:
    text: str
    skills: set[str] = field(default_factory=set)
    years_mentioned: list[int] = field(default_factory=list)
    n_roles: int = 0
    career_span_years: float = 0.0
    last_tenure_years: float = 2.0
    has_management: bool = False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def extract_skills(text: str) -> set[str]:
    blob = _normalize(text)
    found: set[str] = set()
    for skill in SKILL_LEXICON:
        if re.search(rf"(?<![a-z]){re.escape(skill)}(?![a-z])", blob):
            found.add(skill)
    return found


def extract_years(text: str) -> list[int]:
    years = [int(m.group(0)) for m in BARE_YEAR.finditer(text)]
    return sorted({y for y in years if 1990 <= y <= 2026})


def estimate_roles(text: str) -> int:
    blob = _normalize(text)
    hits = sum(blob.count(h) for h in ROLE_HINTS)
    spans = YEAR_SPAN.findall(text)
    return max(len(spans), min(hits, 12), 1)


def estimate_span_and_tenure(years: list[int], n_roles: int) -> tuple[float, float]:
    if len(years) >= 2:
        span = float(max(years) - min(years))
        span = max(span, 1.0)
    elif years:
        span = 3.0
    else:
        span = 4.0
    last_tenure = max(span / max(n_roles, 1), 0.4)
    last_tenure = min(last_tenure, 8.0)
    return span, last_tenure


def parse_document(text: str) -> ParsedDoc:
    years = extract_years(text)
    n_roles = estimate_roles(text)
    span, last_tenure = estimate_span_and_tenure(years, n_roles)
    blob = _normalize(text)
    return ParsedDoc(
        text=text,
        skills=extract_skills(text),
        years_mentioned=years,
        n_roles=n_roles,
        career_span_years=span,
        last_tenure_years=last_tenure,
        has_management=any(
            w in blob for w in ("manager", "director", "lead", "head of")
        ),
    )


def hop_rate(resume: ParsedDoc) -> float:
    return resume.n_roles / max(resume.career_span_years, 1.0)


def build_features(resume: ParsedDoc, job: ParsedDoc):
    if not job.skills:
        jaccard, coverage, overlap, missing = 0.0, 0.0, set(), set()
    else:
        overlap = resume.skills & job.skills
        missing = job.skills - resume.skills
        jaccard = len(overlap) / len(resume.skills | job.skills)
        coverage = len(overlap) / len(job.skills)
    extra = len(resume.skills - job.skills)
    overqual = extra / max(len(job.skills), 1)
    values = [
        jaccard,
        coverage,
        float(len(resume.skills)),
        hop_rate(resume),
        resume.last_tenure_years,
        resume.career_span_years,
        overqual,
        1.0 if resume.has_management else 0.0,
    ]
    return values, overlap, missing


def make_synthetic_careers(n: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    skill_overlap = rng.beta(2.2, 2.0, n)
    skill_coverage = np.clip(skill_overlap + rng.normal(0, 0.08, n), 0, 1)
    n_resume_skills = rng.integers(3, 22, n).astype(float)
    hop = rng.gamma(2.0, 0.35, n)
    last_tenure = np.clip(rng.gamma(2.4, 1.1, n), 0.3, 12)
    career_span = np.clip(last_tenure * rng.uniform(1.2, 6.0, n), 1, 30)
    overqual = rng.gamma(1.4, 0.5, n)
    management = rng.binomial(1, 0.22, n).astype(float)

    logit = (
        -1.4
        + 1.6 * np.clip(hop - 0.7, 0, None)
        + 1.1 * np.clip(1.4 - last_tenure, 0, None)
        + 1.3 * (1.0 - skill_coverage)
        + 0.35 * np.clip(overqual - 1.0, 0, None)
        - 0.25 * management
        + rng.normal(0, 0.35, n)
    )
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack(
        [
            skill_overlap,
            skill_coverage,
            n_resume_skills,
            hop,
            last_tenure,
            career_span,
            overqual,
            management,
        ]
    )
    return X, y


def train_model(n: int = 4000):
    X, y = make_synthetic_careers(n)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_train, y_train)
    acc = float(model.score(X_test, y_test))
    return model, acc


def score(resume_text: str, job_text: str, model) -> dict:
    resume = parse_document(resume_text)
    job = parse_document(job_text)
    values, overlap, missing = build_features(resume, job)
    leave_risk = float(model.predict_proba(np.array(values).reshape(1, -1))[0, 1])
    return {
        "fit_score": round(values[1], 3),
        "stay_score": round(1.0 - leave_risk, 3),
        "leave_risk": round(leave_risk, 3),
        "hop_rate": round(values[3], 3),
        "last_tenure_years": round(values[4], 2),
        "overlap": sorted(overlap),
        "missing": sorted(missing),
    }


def format_report(report: dict) -> str:
    skills = ", ".join(report["overlap"]) or "(none)"
    missing = ", ".join(report["missing"]) or "(none)"
    return (
        "Talent Fit & Stay — Takedownman\n"
        f"  Job-fit (JD coverage): {report['fit_score']:.3f}\n"
        f"  Stay score:            {report['stay_score']:.3f}\n"
        f"  Leave risk:            {report['leave_risk']:.3f}\n"
        f"  Hop rate:              {report['hop_rate']:.3f} roles/year\n"
        f"  Last tenure (est.):    {report['last_tenure_years']:.2f} years\n"
        f"  Overlap:               {skills}\n"
        f"  Missing vs JD:         {missing}\n"
    )


if __name__ == "__main__":
    model, acc = train_model()
    print(f"[Takedownman] Trained synthetic attrition model. Holdout accuracy: {acc:.3f}\n")
    print(format_report(score(SAMPLE_RESUME, SAMPLE_JOB, model)))
