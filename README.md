# Talent Fit & Stay

**Resume-to-job matching + attrition-risk scoring**  
Portfolio project by **Takedownman**

A lightweight Python tool that:
1. Extracts skills and career signals from a resume and job description
2. Scores how well the resume covers the job requirements
3. Estimates the probability the candidate would leave (attrition risk) using a calibrated logistic regression model trained on synthetic data

---

## Features

- Skill extraction from free-text resumes and job descriptions
- Jaccard + coverage-based job-fit scoring
- Hop-rate and tenure estimation from year spans
- Calibrated leave-risk prediction
- Clean, readable report output
- Fully synthetic training data (no real employee records)

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and synthetic data generation |
| `scikit-learn` | Logistic regression, calibration, train/test split, scaling, pipelines |
| `re` (stdlib) | Skill and year extraction with regex |
| `dataclasses` (stdlib) | Clean structured parsing of documents |

No external APIs. No databases. Runs locally.

---

## Installation

```bash
pip install scikit-learn numpy

Usage
Bashpython talent_fit.py
The script will:

Train a calibrated logistic regression model on synthetic career data
Score the built-in sample resume against the sample job description
Print a short report with fit score, stay score, leave risk, and skill overlap


Example Output
text[Takedownman] Trained synthetic attrition model. Holdout accuracy: 0.XXX

Talent Fit & Stay — Takedownman
  Job-fit (JD coverage): 0.XXX
  Stay score:            0.XXX
  Leave risk:            0.XXX
  Hop rate:              X.XXX roles/year
  Last tenure (est.):    X.XX years
  Overlap:               python, sql, pytorch, ...
  Missing vs JD:         ...

How It Works

Parsing – Skills are matched against a curated lexicon. Years and role counts are estimated with regex and keyword heuristics.
Feature building – Jaccard similarity, skill coverage, hop rate, tenure, overqualification, and management signals are computed.
Model – A logistic regression model (with StandardScaler + CalibratedClassifierCV) is trained on 4,000 synthetic career profiles.
Scoring – The same features extracted from a real resume + job description are fed into the model to produce fit and leave-risk scores.


Important Notes

All attrition labels are synthetic. This is a portfolio demonstration, not a production HR system.
The model deliberately avoids age, gender, race, disability, or any protected attributes.
Skill extraction is lexicon-based and will miss skills that are phrased unusually.
Treat the leave-risk number as an experimental signal, not a hiring decision.


Author
Takedownman
GitHub portfolio project · AI / ML exploration
