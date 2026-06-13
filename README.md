# Candidate Recommender Diagnostics Demo

A portfolio-safe demo project showing how to diagnose and improve a simple skill-based candidate recommender.

The project uses **synthetic data only**. It does not contain any proprietary recruitment materials, company data, private candidate information.

## What this project shows

This repository demonstrates a practical ML/AI engineering workflow:

- understanding an existing baseline recommender,
- diagnosing failure modes using data,
- separating algorithm issues from data-quality and product issues,
- implementing two targeted improvements,
- validating the output before writing conclusions.

The goal is not to build a complex model from scratch. The goal is to show a disciplined diagnostic process around a simple recommender system.

## Problem overview

The baseline recommender matches candidates to projects using skill overlap.

For each candidate-project pair, it computes:

```text
score = number of shared skills
```

Then it returns the top 5 candidates for every project.

This baseline is easy to explain, but it has several weaknesses:

- candidates with many listed skills can dominate rankings,
- self-reported skills are treated the same as validated skills,
- required and nice-to-have project skills are treated equally,
- candidates already assigned to a project may still be recommended,
- raw scores are not calibrated business values.

## Implemented improvements

### 1. Assignment filter

Candidates already assigned to or completed on the same project are removed before ranking.

This prevents operationally useless recommendations such as recommending a person who is already on the project.

### 2. Weighted normalized skill match

The raw overlap score is replaced with a weighted cosine-like similarity.

The improved score uses:

- candidate skill source,
- candidate skill confidence,
- project skill criticality,
- skill rarity,
- normalization by profile size.

This reduces the advantage of candidates who have very large, noisy skill profiles.

## Repository structure

```text
.
├── data/
│   └── .gitkeep
├── scripts/
│   └── generate_synthetic_data.py
├── src/
│   ├── baseline_recommender.py
│   ├── diagnostics.py
│   ├── improved_recommender.py
│   └── validate_improvements.py
├── tests/
│   └── test_recommender_outputs.py
├── report.md
├── pyproject.toml
└── README.md
```

## Setup

Recommended setup with `uv`:

```bash
uv sync
```

Alternative setup with pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## How to run

Generate synthetic data:

```bash
uv run python scripts/generate_synthetic_data.py
```

Run the baseline recommender:

```bash
uv run python src/baseline_recommender.py
```

Run diagnostics:

```bash
uv run python src/diagnostics.py
```

Run the improved recommender:

```bash
uv run python src/improved_recommender.py
```

Validate the improvement:

```bash
uv run python src/validate_improvements.py
```

Run tests:

```bash
uv run pytest
```

## Expected outputs

After running the full workflow, the `data/` directory will contain generated CSV files, including:

```text
baseline_recommendations.csv
improved_recommendations.csv
```

These files are generated from synthetic data and are not committed by default.

## Design choices

The improved recommender is intentionally simple. It uses heuristics rather than a trained ML model because the purpose of this project is diagnostic clarity.

The most important design decisions are:

- keep the baseline easy to compare against,
- make every improvement tied to a diagnosed failure mode,
- validate before/after results,
- avoid overclaiming model quality.

## Limitations

This demo does not model all real staffing constraints. A production recommender would also need:

- candidate availability,
- project workload,
- required seniority,
- skill recency,
- domain experience,
- manager feedback,
- candidate preferences,
- business rules around staffing.

The system also cannot recover missing information. If a candidate has a skill but it is not present in the data, the recommender cannot use it.

## Why synthetic data?

This repository is designed for public portfolio use. The data is generated locally to avoid exposing any private or proprietary recruitment material.

## Author note

This project is a cleaned and generalized demo of a recommender-diagnostics workflow. It focuses on reasoning, validation, and communication rather than model complexity.
