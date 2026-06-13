from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
TOP_K = 5

SOURCE_WEIGHT = {
    "cv": 0.60,
    "assessment": 1.00,
    "peer_review": 1.10,
}

CRITICALITY_WEIGHT = {
    "required": 1.00,
    "nice_to_have": 0.35,
}


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "candidates": pd.read_csv(DATA_DIR / "candidates.csv"),
        "projects": pd.read_csv(DATA_DIR / "projects.csv"),
        "skills": pd.read_csv(DATA_DIR / "skills.csv"),
        "candidate_skills": pd.read_csv(DATA_DIR / "candidate_skills.csv"),
        "project_skills": pd.read_csv(DATA_DIR / "project_skills.csv"),
        "candidate_projects": pd.read_csv(DATA_DIR / "candidate_projects.csv"),
        "baseline_recommendations": pd.read_csv(DATA_DIR / "baseline_recommendations.csv"),
    }


def compute_skill_idf(candidate_skills: pd.DataFrame) -> dict[int, float]:
    n_candidates = candidate_skills["candidate_id"].nunique()

    skill_candidate_counts = (
        candidate_skills.drop_duplicates(["candidate_id", "skill_id"]).groupby("skill_id")["candidate_id"].nunique()
    )

    return {
        int(skill_id): math.log((1 + n_candidates) / (1 + count)) + 1
        for skill_id, count in skill_candidate_counts.items()
    }


def build_candidate_skill_weights(
    candidate_skills: pd.DataFrame,
    skill_idf: dict[int, float],
) -> dict[int, dict[int, float]]:
    df = candidate_skills.copy()

    df["source_weight"] = df["source"].map(SOURCE_WEIGHT).fillna(0.70)
    df["confidence"] = df["confidence"].fillna(0.50)
    df["idf"] = df["skill_id"].map(skill_idf).fillna(1.00)

    df["weight"] = df["source_weight"] * df["confidence"] * df["idf"]

    # If the same candidate-skill pair appears more than once, keep the strongest signal.
    df = df.groupby(["candidate_id", "skill_id"], as_index=False)["weight"].max()

    result: dict[int, dict[int, float]] = {}

    for candidate_id, group in df.groupby("candidate_id"):
        result[int(candidate_id)] = {int(row.skill_id): float(row.weight) for row in group.itertuples()}

    return result


def build_project_skill_weights(
    project_skills: pd.DataFrame,
    skill_idf: dict[int, float],
) -> dict[int, dict[int, float]]:
    df = project_skills.copy()

    df["criticality_weight"] = df["criticality"].map(CRITICALITY_WEIGHT).fillna(0.70)
    df["idf"] = df["skill_id"].map(skill_idf).fillna(1.00)

    df["weight"] = df["criticality_weight"] * df["idf"]

    result: dict[int, dict[int, float]] = {}

    for project_id, group in df.groupby("project_id"):
        result[int(project_id)] = {int(row.skill_id): float(row.weight) for row in group.itertuples()}

    return result


def weighted_cosine_score(
    candidate_weights: dict[int, float],
    project_weights: dict[int, float],
) -> float:
    common_skills = set(candidate_weights) & set(project_weights)

    if not common_skills:
        return 0.0

    numerator = sum(candidate_weights[skill_id] * project_weights[skill_id] for skill_id in common_skills)

    candidate_norm = math.sqrt(sum(weight**2 for weight in candidate_weights.values()))
    project_norm = math.sqrt(sum(weight**2 for weight in project_weights.values()))

    if candidate_norm == 0 or project_norm == 0:
        return 0.0

    return numerator / (candidate_norm * project_norm)


def get_already_on_project(candidate_projects: pd.DataFrame) -> set[tuple[int, int]]:
    already_on_project = candidate_projects[candidate_projects["status"].isin(["assigned", "completed"])][
        ["project_id", "candidate_id"]
    ].drop_duplicates()

    return {(int(row.project_id), int(row.candidate_id)) for row in already_on_project.itertuples()}


def generate_improved_recommendations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidate_skills = data["candidate_skills"]
    project_skills = data["project_skills"]
    candidate_projects = data["candidate_projects"]

    skill_idf = compute_skill_idf(candidate_skills)

    candidate_weights = build_candidate_skill_weights(candidate_skills, skill_idf)
    project_weights = build_project_skill_weights(project_skills, skill_idf)
    already_on_project = get_already_on_project(candidate_projects)

    rows = []

    for project_id, p_weights in project_weights.items():
        for candidate_id, c_weights in candidate_weights.items():
            if (project_id, candidate_id) in already_on_project:
                continue

            score = weighted_cosine_score(c_weights, p_weights)
            matched_skill_count = len(set(c_weights) & set(p_weights))

            rows.append(
                {
                    "project_id": project_id,
                    "candidate_id": candidate_id,
                    "score": score,
                    "matched_skill_count": matched_skill_count,
                }
            )

    scores = pd.DataFrame(rows)

    recommendations = (
        scores.sort_values(
            ["project_id", "score", "matched_skill_count", "candidate_id"],
            ascending=[True, False, False, True],
        )
        .groupby("project_id", as_index=False)
        .head(TOP_K)
        .copy()
    )

    recommendations["rank"] = recommendations.groupby("project_id").cumcount() + 1

    recommendations = recommendations[["project_id", "rank", "candidate_id", "score"]]
    recommendations["score"] = recommendations["score"].round(6)

    return recommendations


def print_top_for_project(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    project_id: int,
    title: str,
) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    result = (
        recommendations[recommendations["project_id"] == project_id]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values("rank")
    )

    print(result[["project_id", "rank", "candidate_id", "name", "level", "score"]].to_string(index=False))


def main() -> None:
    data = load_data()

    improved_recommendations = generate_improved_recommendations(data)

    output_path = DATA_DIR / "improved_recommendations.csv"
    improved_recommendations.to_csv(output_path, index=False)

    print(f"Saved {len(improved_recommendations)} recommendations to {output_path}")

    candidates = data["candidates"]
    baseline = data["baseline_recommendations"]

    print_top_for_project(
        baseline,
        candidates,
        project_id=2,
        title="Baseline: project_id=2",
    )
    print_top_for_project(
        improved_recommendations,
        candidates,
        project_id=2,
        title="Improved: project_id=2",
    )

    print_top_for_project(
        baseline,
        candidates,
        project_id=39,
        title="Baseline: project_id=39",
    )
    print_top_for_project(
        improved_recommendations,
        candidates,
        project_id=39,
        title="Improved: project_id=39",
    )

    print_top_for_project(
        baseline,
        candidates,
        project_id=15,
        title="Baseline: project_id=15",
    )
    print_top_for_project(
        improved_recommendations,
        candidates,
        project_id=15,
        title="Improved: project_id=15",
    )


if __name__ == "__main__":
    main()
