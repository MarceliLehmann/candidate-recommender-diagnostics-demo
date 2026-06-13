from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "candidates": pd.read_csv(DATA_DIR / "candidates.csv"),
        "candidate_skills": pd.read_csv(DATA_DIR / "candidate_skills.csv"),
        "candidate_projects": pd.read_csv(DATA_DIR / "candidate_projects.csv"),
        "baseline": pd.read_csv(DATA_DIR / "baseline_recommendations.csv"),
        "improved": pd.read_csv(DATA_DIR / "improved_recommendations.csv"),
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_df(df: pd.DataFrame, max_rows: int = 20) -> None:
    if df.empty:
        print("(empty)")
        return

    print(df.head(max_rows).to_string(index=False))


def validate_output_shape(baseline: pd.DataFrame, improved: pd.DataFrame) -> None:
    print_section("1. Output shape validation")

    print(f"Baseline rows: {len(baseline)}")
    print(f"Improved rows: {len(improved)}")

    improved_projects = improved["project_id"].nunique()
    improved_recommendations_per_project = (
        improved.groupby("project_id").size().reset_index(name="n_recommendations")
    )

    print(f"Improved unique projects: {improved_projects}")
    print("\nRecommendations per project:")
    print_df(improved_recommendations_per_project)

    bad_projects = improved_recommendations_per_project[
        improved_recommendations_per_project["n_recommendations"] != 5
    ]

    if bad_projects.empty:
        print("\nOK: every project has exactly 5 recommendations.")
    else:
        print("\nWARNING: some projects do not have exactly 5 recommendations:")
        print_df(bad_projects)


def validate_no_self_recommendations(
    improved: pd.DataFrame,
    candidate_projects: pd.DataFrame,
) -> None:
    print_section("2. Self-recommendation validation")

    already_on_project = candidate_projects[
        candidate_projects["status"].isin(["assigned", "completed"])
    ][["project_id", "candidate_id", "status"]].drop_duplicates()

    violations = improved.merge(
        already_on_project,
        on=["project_id", "candidate_id"],
        how="inner",
    ).sort_values(["project_id", "rank"])

    if violations.empty:
        print("OK: improved recommendations do not include assigned/completed candidates.")
    else:
        print("WARNING: improved recommendations still include assigned/completed candidates:")
        print_df(violations)


def compare_project(
    baseline: pd.DataFrame,
    improved: pd.DataFrame,
    candidates: pd.DataFrame,
    project_id: int,
    title: str,
) -> None:
    print_section(title)

    baseline_view = (
        baseline[baseline["project_id"] == project_id]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values("rank")
    )

    improved_view = (
        improved[improved["project_id"] == project_id]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values("rank")
    )

    print("\nBaseline:")
    print_df(
        baseline_view[
            ["project_id", "rank", "candidate_id", "name", "level", "score"]
        ]
    )

    print("\nImproved:")
    print_df(
        improved_view[
            ["project_id", "rank", "candidate_id", "name", "level", "score"]
        ]
    )


def validate_specific_cases(
    baseline: pd.DataFrame,
    improved: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    print_section("3. Specific complaint cases")

    cases = [
        {
            "project_id": 2,
            "candidate_id": 7,
            "case": "Joanna Kozłowska in ML project",
        },
        {
            "project_id": 39,
            "candidate_id": 78,
            "case": "Adam Nowakowski in React Native project",
        },
        {
            "project_id": 15,
            "candidate_id": 96,
            "case": "Adam Wójcik self-recommendation",
        },
    ]

    rows = []

    for case in cases:
        project_id = case["project_id"]
        candidate_id = case["candidate_id"]

        baseline_match = baseline[
            (baseline["project_id"] == project_id)
            & (baseline["candidate_id"] == candidate_id)
        ]

        improved_match = improved[
            (improved["project_id"] == project_id)
            & (improved["candidate_id"] == candidate_id)
        ]

        candidate_name = candidates.loc[
            candidates["candidate_id"] == candidate_id,
            "name",
        ].iloc[0]

        rows.append(
            {
                "case": case["case"],
                "project_id": project_id,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "baseline_rank": None
                if baseline_match.empty
                else int(baseline_match.iloc[0]["rank"]),
                "baseline_score": None
                if baseline_match.empty
                else baseline_match.iloc[0]["score"],
                "improved_rank": None
                if improved_match.empty
                else int(improved_match.iloc[0]["rank"]),
                "improved_score": None
                if improved_match.empty
                else improved_match.iloc[0]["score"],
            }
        )

    result = pd.DataFrame(rows)

    print_df(result)

    output_path = DATA_DIR / "improvement_validation_cases.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved validation cases to {output_path}")


def compare_hub_candidates(
    baseline: pd.DataFrame,
    improved: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_skills: pd.DataFrame,
) -> None:
    print_section("8. Hub candidate frequency comparison")

    baseline_counts = (
        baseline.groupby("candidate_id")
        .size()
        .reset_index(name="baseline_n_recommendations")
    )

    improved_counts = (
        improved.groupby("candidate_id")
        .size()
        .reset_index(name="improved_n_recommendations")
    )

    skill_counts = (
        candidate_skills.groupby("candidate_id")["skill_id"]
        .nunique()
        .reset_index(name="n_skills")
    )

    comparison = (
        baseline_counts.merge(improved_counts, on="candidate_id", how="outer")
        .fillna(0)
        .merge(skill_counts, on="candidate_id", how="left")
        .merge(candidates, on="candidate_id", how="left")
    )

    comparison["change"] = (
        comparison["improved_n_recommendations"]
        - comparison["baseline_n_recommendations"]
    )

    comparison = comparison.sort_values(
        ["baseline_n_recommendations", "n_skills"],
        ascending=[False, False],
    )

    print("\nMost recommended candidates in baseline and their improved counts:")
    print_df(
        comparison[
            [
                "candidate_id",
                "name",
                "level",
                "n_skills",
                "baseline_n_recommendations",
                "improved_n_recommendations",
                "change",
            ]
        ],
        max_rows=20,
    )

    baseline_corr = comparison["n_skills"].corr(
        comparison["baseline_n_recommendations"]
    )
    improved_corr = comparison["n_skills"].corr(
        comparison["improved_n_recommendations"]
    )

    print(f"\nBaseline correlation n_skills vs n_recommendations: {baseline_corr:.3f}")
    print(f"Improved correlation n_skills vs n_recommendations: {improved_corr:.3f}")


def main() -> None:
    data = load_data()

    candidates = data["candidates"]
    candidate_skills = data["candidate_skills"]
    candidate_projects = data["candidate_projects"]
    baseline = data["baseline"]
    improved = data["improved"]

    validate_output_shape(baseline, improved)

    validate_no_self_recommendations(
        improved=improved,
        candidate_projects=candidate_projects,
    )

    validate_specific_cases(
        baseline=baseline,
        improved=improved,
        candidates=candidates,
    )

    compare_project(
        baseline=baseline,
        improved=improved,
        candidates=candidates,
        project_id=2,
        title="5. Project comparison: ML Recommender - GovTech, project_id=2",
    )

    compare_project(
        baseline=baseline,
        improved=improved,
        candidates=candidates,
        project_id=39,
        title="6. Project comparison: React Native, project_id=39",
    )

    compare_project(
        baseline=baseline,
        improved=improved,
        candidates=candidates,
        project_id=15,
        title="7. Project comparison: self-recommendation, project_id=15",
    )

    compare_hub_candidates(
        baseline=baseline,
        improved=improved,
        candidates=candidates,
        candidate_skills=candidate_skills,
    )


if __name__ == "__main__":
    main()