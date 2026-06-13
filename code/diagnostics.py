from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "candidates": pd.read_csv(DATA_DIR / "candidates.csv"),
        "projects": pd.read_csv(DATA_DIR / "projects.csv"),
        "skills": pd.read_csv(DATA_DIR / "skills.csv"),
        "candidate_skills": pd.read_csv(DATA_DIR / "candidate_skills.csv"),
        "project_skills": pd.read_csv(DATA_DIR / "project_skills.csv"),
        "candidate_projects": pd.read_csv(DATA_DIR / "candidate_projects.csv"),
        "recommendations": pd.read_csv(DATA_DIR / "baseline_recommendations.csv"),
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


def build_raw_scores(
    candidate_skills: pd.DataFrame,
    project_skills: pd.DataFrame,
) -> pd.DataFrame:
    cand_skill_sets = candidate_skills.groupby("candidate_id")["skill_id"].apply(set).to_dict()

    proj_skill_sets = project_skills.groupby("project_id")["skill_id"].apply(set).to_dict()

    rows = []

    for project_id, p_skills in proj_skill_sets.items():
        for candidate_id, c_skills in cand_skill_sets.items():
            score = len(c_skills & p_skills)
            rows.append(
                {
                    "project_id": project_id,
                    "candidate_id": candidate_id,
                    "score": score,
                }
            )

    scores = pd.DataFrame(rows)

    scores = scores.sort_values(
        ["project_id", "score", "candidate_id"],
        ascending=[True, False, True],
    )

    scores["raw_rank"] = scores.groupby("project_id").cumcount() + 1

    return scores


def show_project_top(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    project_id: int,
    title: str,
) -> None:
    print_section(title)

    result = (
        recommendations[recommendations["project_id"] == project_id]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values("rank")
    )

    columns = ["project_id", "rank", "candidate_id", "name", "level", "score"]
    print_df(result[columns])


def show_candidate_skill_profile(
    candidate_skills: pd.DataFrame,
    skills: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_id: int,
) -> None:
    candidate = candidates[candidates["candidate_id"] == candidate_id].iloc[0]

    print(f"\nCandidate profile: {candidate['name']} (id={candidate_id})")
    print(f"Level: {candidate['level']}")
    print(
        f"Total unique skills: {candidate_skills[candidate_skills['candidate_id'] == candidate_id]['skill_id'].nunique()}"
    )

    skill_profile = (
        candidate_skills[candidate_skills["candidate_id"] == candidate_id]
        .merge(skills, on="skill_id", how="left")
        .sort_values(["source", "category", "name"])
    )

    print("\nSkill source distribution:")
    print_df(
        skill_profile.groupby("source").size().reset_index(name="n_skills").sort_values("n_skills", ascending=False)
    )

    print("\nCandidate skills:")
    columns = ["skill_id", "name", "category", "source", "confidence"]
    existing_columns = [col for col in columns if col in skill_profile.columns]
    print_df(skill_profile[existing_columns], max_rows=50)


def show_matched_skills(
    candidate_skills: pd.DataFrame,
    project_skills: pd.DataFrame,
    skills: pd.DataFrame,
    candidate_id: int,
    project_id: int,
) -> None:
    project_skill_subset = project_skills[project_skills["project_id"] == project_id][["skill_id", "criticality"]]

    matched = (
        candidate_skills[
            (candidate_skills["candidate_id"] == candidate_id)
            & (candidate_skills["skill_id"].isin(project_skill_subset["skill_id"]))
        ]
        .merge(skills, on="skill_id", how="left")
        .merge(project_skill_subset, on="skill_id", how="left")
        .sort_values(["criticality", "source", "name"])
    )

    print(f"\nMatched skills for candidate_id={candidate_id}, project_id={project_id}: {len(matched)}")

    columns = ["skill_id", "name", "category", "criticality", "source", "confidence"]
    existing_columns = [col for col in columns if col in matched.columns]
    print_df(matched[existing_columns], max_rows=50)
def show_precision_at_5_diagnostics(
    recommendations: pd.DataFrame,
    candidate_projects: pd.DataFrame,
) -> None:
    print_section("Complaint 6: precision@5 diagnostic")

    historical_all = (
        candidate_projects[["project_id", "candidate_id"]]
        .drop_duplicates()
        .assign(was_historical=True)
    )

    historical_completed = (
        candidate_projects[candidate_projects["status"] == "completed"][
            ["project_id", "candidate_id"]
        ]
        .drop_duplicates()
        .assign(was_completed=True)
    )

    eval_all = (
        recommendations
        .merge(historical_all, on=["project_id", "candidate_id"], how="left")
        .assign(was_historical=lambda d: d["was_historical"].fillna(False))
        .groupby("project_id", as_index=False)["was_historical"]
        .mean()
        .rename(columns={"was_historical": "precision_at_5_any_history"})
    )

    eval_completed = (
        recommendations
        .merge(historical_completed, on=["project_id", "candidate_id"], how="left")
        .assign(was_completed=lambda d: d["was_completed"].fillna(False))
        .groupby("project_id", as_index=False)["was_completed"]
        .mean()
        .rename(columns={"was_completed": "precision_at_5_completed_only"})
    )

    combined = eval_all.merge(eval_completed, on="project_id", how="left")

    print("Precision@5 variants:")
    print(f"Mean precision@5, any historical status: {combined['precision_at_5_any_history'].mean():.3f}")
    print(f"Median precision@5, any historical status: {combined['precision_at_5_any_history'].median():.3f}")
    print(f"Mean precision@5, completed only: {combined['precision_at_5_completed_only'].mean():.3f}")
    print(f"Median precision@5, completed only: {combined['precision_at_5_completed_only'].median():.3f}")

    print("\nProjects with lowest completed-only precision@5:")
    print_df(
        combined.sort_values("precision_at_5_completed_only").head(10),
        max_rows=10,
    )

    print(
        "\nInterpretation: this metric is biased because historical data contains "
        "only candidates who were proposed or assigned, not all candidates who could "
        "have been valid matches."
    )


def show_paulina_legacy_java_case(
    recommendations: pd.DataFrame,
    raw_scores: pd.DataFrame,
    candidate_skills: pd.DataFrame,
    project_skills: pd.DataFrame,
    candidates: pd.DataFrame,
    projects: pd.DataFrame,
    skills: pd.DataFrame,
) -> None:
    print_section("Complaint 7: Paulina Piotrowska and legacy Java visibility")

    paulina_candidates = candidates[
        candidates["name"].str.contains("Paulina Piotrowska", case=False, na=False)
    ]

    print("Candidate lookup:")
    print_df(paulina_candidates)

    if paulina_candidates.empty:
        print("Paulina Piotrowska not found in candidates.csv")
        return

    paulina_id = int(paulina_candidates.iloc[0]["candidate_id"])

    print(f"\nDetected Paulina candidate_id={paulina_id}")

    show_project_top(
        recommendations,
        candidates,
        project_id=16,
        title="Project 16: Legacy Maintenance - PaymentBank",
    )

    show_candidate_skill_profile(
        candidate_skills,
        skills,
        candidates,
        candidate_id=paulina_id,
    )

    show_matched_skills(
        candidate_skills,
        project_skills,
        skills,
        candidate_id=paulina_id,
        project_id=16,
    )

    paulina_recommendations = (
        recommendations[recommendations["candidate_id"] == paulina_id]
        .merge(projects, on="project_id", how="left")
        .sort_values(["project_id", "rank"])
    )

    print("\nAll baseline recommendations containing Paulina:")
    print_df(
        paulina_recommendations[
            ["project_id", "name", "status", "rank", "candidate_id", "score"]
        ],
        max_rows=20,
    )

    java_like_skills = skills[
        skills["name"].str.contains("Java|Spring|Struts", case=False, na=False)
    ]

    java_like_projects = project_skills[
        project_skills["skill_id"].isin(java_like_skills["skill_id"])
    ][["project_id"]].drop_duplicates()

    paulina_java_project_scores = (
        raw_scores[
            (raw_scores["candidate_id"] == paulina_id)
            & (raw_scores["project_id"].isin(java_like_projects["project_id"]))
        ]
        .merge(projects, on="project_id", how="left")
        .sort_values(["raw_rank", "score"], ascending=[True, False])
    )

    print("\nPaulina rank in Java-like projects:")
    columns = ["project_id", "name", "status", "score", "raw_rank"]
    existing_columns = [col for col in columns if col in paulina_java_project_scores.columns]
    print_df(paulina_java_project_scores[existing_columns], max_rows=20)


def show_score_interpretability(
    raw_scores: pd.DataFrame,
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    print_section("Complaint 8: score interpretability and top-K cutoff")

    score_summary = (
        raw_scores.groupby("project_id")["score"]
        .agg(["max", "mean", "median", "std"])
        .reset_index()
        .sort_values("max", ascending=False)
    )

    print("Score distribution per project:")
    print_df(score_summary, max_rows=15)

    top10 = (
        raw_scores[raw_scores["raw_rank"] <= 10]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values(["project_id", "raw_rank"])
    )

    print("\nExample top-10 for project_id=3 to show ties around top-5:")
    print_df(
        top10[top10["project_id"] == 3][
            ["project_id", "raw_rank", "candidate_id", "name", "level", "score"]
        ],
        max_rows=10,
    )

    print("\nExample top-10 for project_id=39 to show candidates just outside top-5:")
    print_df(
        top10[top10["project_id"] == 39][
            ["project_id", "raw_rank", "candidate_id", "name", "level", "score"]
        ],
        max_rows=10,
    )

    tie_counts = (
        recommendations.groupby(["project_id", "score"])
        .size()
        .reset_index(name="n_candidates_with_same_score_in_top5")
        .sort_values("n_candidates_with_same_score_in_top5", ascending=False)
    )

    print("\nTies inside top-5 recommendations:")
    print_df(tie_counts[tie_counts["n_candidates_with_same_score_in_top5"] > 1], max_rows=20)

    print(
        "\nInterpretation: raw score is a count of shared skills, not a calibrated "
        "probability or business value. A score of 10 is not necessarily twice as good "
        "as a score of 5."
    )


def show_source_confidence_diagnostics(
    recommendations: pd.DataFrame,
    candidate_skills: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    print_section("Complaint 9: source and confidence are ignored")

    source_distribution = (
        candidate_skills.groupby("source")
        .agg(
            n_edges=("skill_id", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )

    print("Global candidate skill source distribution:")
    print_df(source_distribution)

    recommended_candidate_ids = recommendations["candidate_id"].drop_duplicates()

    recommended_source_distribution = (
        candidate_skills[candidate_skills["candidate_id"].isin(recommended_candidate_ids)]
        .groupby("source")
        .agg(
            n_edges=("skill_id", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )

    print("\nSkill source distribution among recommended candidates:")
    print_df(recommended_source_distribution)

    candidate_source_profile = (
        candidate_skills
        .groupby(["candidate_id", "source"])
        .size()
        .reset_index(name="n_skills")
        .pivot(index="candidate_id", columns="source", values="n_skills")
        .fillna(0)
        .reset_index()
        .merge(candidates, on="candidate_id", how="left")
    )

    recommendation_counts = (
        recommendations.groupby("candidate_id")
        .size()
        .reset_index(name="n_recommendations")
    )

    candidate_source_profile = (
        candidate_source_profile
        .merge(recommendation_counts, on="candidate_id", how="left")
        .fillna({"n_recommendations": 0})
        .sort_values("n_recommendations", ascending=False)
    )

    print("\nTop recommended candidates with skill source counts:")
    columns = ["candidate_id", "name", "level", "cv", "assessment", "peer_review", "n_recommendations"]
    existing_columns = [col for col in columns if col in candidate_source_profile.columns]
    print_df(candidate_source_profile[existing_columns], max_rows=20)

    print(
        "\nInterpretation: baseline converts skills to plain sets of skill_id, so all "
        "sources and confidence values have zero impact on ranking."
    )

def main() -> None:
    data = load_data()

    candidates = data["candidates"]
    projects = data["projects"]
    skills = data["skills"]
    candidate_skills = data["candidate_skills"]
    project_skills = data["project_skills"]
    candidate_projects = data["candidate_projects"]
    recommendations = data["recommendations"]

    raw_scores = build_raw_scores(candidate_skills, project_skills)

    print_section("Dataset overview")
    print(f"Candidates: {len(candidates)}")
    print(f"Projects: {len(projects)}")
    print(f"Skills: {len(skills)}")
    print(f"Candidate-skill edges: {len(candidate_skills)}")
    print(f"Project-skill edges: {len(project_skills)}")
    print(f"Baseline recommendations: {len(recommendations)}")

    # Complaint 1: Joanna for ML project
    show_project_top(
        recommendations,
        candidates,
        project_id=2,
        title="Complaint 1: ML Recommender - GovTech, project_id=2",
    )
    show_candidate_skill_profile(candidate_skills, skills, candidates, candidate_id=7)
    show_matched_skills(
        candidate_skills,
        project_skills,
        skills,
        candidate_id=7,
        project_id=2,
    )

    # Complaint 2: React Native project and hub candidates
    show_project_top(
        recommendations,
        candidates,
        project_id=39,
        title="Complaint 2: React Native project, project_id=39",
    )

    rn_candidates_to_check = [78, 99, 100]
    rn_ranks = (
        raw_scores[(raw_scores["project_id"] == 39) & (raw_scores["candidate_id"].isin(rn_candidates_to_check))]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values("raw_rank")
    )

    print("\nRanks for mentioned React Native candidates:")
    print_df(rn_ranks[["project_id", "raw_rank", "candidate_id", "name", "level", "score"]])

    for candidate_id in rn_candidates_to_check:
        show_candidate_skill_profile(
            candidate_skills,
            skills,
            candidates,
            candidate_id=candidate_id,
        )
        show_matched_skills(
            candidate_skills,
            project_skills,
            skills,
            candidate_id=candidate_id,
            project_id=39,
        )

    # Complaint 3: Paweł Mazur invisible
    print_section("Complaint 3: Paweł Mazur, candidate_id=88")

    pawel_recommendations = recommendations[recommendations["candidate_id"] == 88].merge(
        candidates, on="candidate_id", how="left"
    )

    print("Paweł in baseline recommendations:")
    print_df(pawel_recommendations)

    show_candidate_skill_profile(candidate_skills, skills, candidates, candidate_id=88)

    pawel_scores = (
        raw_scores[(raw_scores["candidate_id"] == 88) & (raw_scores["score"] > 0)]
        .merge(projects, on="project_id", how="left")
        .sort_values(["score", "raw_rank"], ascending=[False, True])
    )

    print("\nProjects where Paweł has at least one matching skill:")
    columns = ["project_id", "name", "status", "score", "raw_rank"]
    existing_columns = [col for col in columns if col in pawel_scores.columns]
    print_df(pawel_scores[existing_columns], max_rows=20)

    # Complaint 4: SAP tie-breaking
    show_project_top(
        recommendations,
        candidates,
        project_id=3,
        title="Complaint 4: SAP project tie-breaking, project_id=3",
    )

    sap_scores = (
        raw_scores[raw_scores["project_id"] == 3]
        .merge(candidates, on="candidate_id", how="left")
        .sort_values(["score", "candidate_id"], ascending=[False, True])
    )

    print("\nTop 15 raw scores for SAP project:")
    print_df(
        sap_scores[["project_id", "raw_rank", "candidate_id", "name", "level", "score"]],
        max_rows=15,
    )

    # Complaint 5: self-recommendation on project 15
    show_project_top(
        recommendations,
        candidates,
        project_id=15,
        title="Complaint 5: self-recommendation, project_id=15",
    )

    already_on_project = candidate_projects[
        (candidate_projects["project_id"] == 15) & (candidate_projects["status"].isin(["assigned", "completed"]))
    ][["project_id", "candidate_id", "status", "feedback_score"]]

    project_15_check = (
        recommendations[recommendations["project_id"] == 15]
        .merge(candidates, on="candidate_id", how="left")
        .merge(already_on_project, on=["project_id", "candidate_id"], how="left")
        .sort_values("rank")
    )

    print("\nProject 15 baseline recommendations with assignment history:")
    columns = [
        "project_id",
        "rank",
        "candidate_id",
        "name",
        "level",
        "score",
        "status",
        "feedback_score",
    ]
    existing_columns = [col for col in columns if col in project_15_check.columns]
    print_df(project_15_check[existing_columns])

        # Complaint 6: precision@5 interpretation
    show_precision_at_5_diagnostics(
        recommendations=recommendations,
        candidate_projects=candidate_projects,
    )

    # Complaint 7: Paulina and legacy Java visibility
    show_paulina_legacy_java_case(
        recommendations=recommendations,
        raw_scores=raw_scores,
        candidate_skills=candidate_skills,
        project_skills=project_skills,
        candidates=candidates,
        projects=projects,
        skills=skills,
    )

    # Complaint 8: score interpretability and top-K cutoff
    show_score_interpretability(
        raw_scores=raw_scores,
        recommendations=recommendations,
        candidates=candidates,
    )

    # Complaint 9: source and confidence ignored
    show_source_confidence_diagnostics(
        recommendations=recommendations,
        candidate_skills=candidate_skills,
        candidates=candidates,
    )

    # General diagnostic: hub candidates
    print_section("General diagnostic: most frequently recommended candidates")

    recommendation_counts = recommendations.groupby("candidate_id").size().reset_index(name="n_recommendations")

    skill_counts = candidate_skills.groupby("candidate_id")["skill_id"].nunique().reset_index(name="n_skills")

    hub_analysis = (
        recommendation_counts.merge(skill_counts, on="candidate_id", how="left")
        .merge(candidates, on="candidate_id", how="left")
        .sort_values(["n_recommendations", "n_skills"], ascending=[False, False])
    )

    print_df(
        hub_analysis[["candidate_id", "name", "level", "n_skills", "n_recommendations"]],
        max_rows=20,
    )

    correlation = hub_analysis["n_skills"].corr(hub_analysis["n_recommendations"])
    print(f"\nCorrelation between number of skills and number of recommendations: {correlation:.3f}")

    print_section("Diagnostic summary")
    print("1. Baseline uses raw common-neighbor count, so candidates with many skills can dominate rankings.")
    print("2. High-degree candidates such as Joanna Kozłowska and Adam Nowakowski appear high in multiple unrelated projects.")
    print("3. Source, confidence, last_updated and project skill criticality are ignored because skills are converted to plain sets.")
    print("4. Ties are resolved by candidate_id, which has no business meaning and affects cases such as the SAP project.")
    print("5. Baseline does not filter candidates already assigned/completed on the same project, causing self-recommendations.")
    print("6. Paweł Mazur is not recommended anywhere because his niche skills have low overlap with most projects.")
    print("7. The React Native junior complaint is partly a data issue: candidates 99 and 100 have only one matching project skill.")
    print("8. Paulina Piotrowska's case is not clearly a bug: she ranks #1 for legacy projects but lower for modern Java projects.")
    print("9. Historical precision@5 is exposure-biased and should not be interpreted as true model quality.")
    print("10. Raw score is not calibrated: score=10 does not mean the candidate is twice as good as score=5.")
    print("11. Top-5 is an arbitrary cutoff; several projects have many tied candidates around the cutoff.")
    print("12. The strongest actionable fixes are assignment filtering and a weighted normalized score.")


if __name__ == "__main__":
    main()
