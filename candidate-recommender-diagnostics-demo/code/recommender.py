"""
Skill-Match Recommender v1.2
============================

Generuje rekomendacje top-K kandydatów dla każdego projektu na podstawie
wspólnych umiejętności (common-neighbor count w bipartite skill graph).

Autor: poprzedni członek zespołu (już nie pracuje)
Ostatnia modyfikacja: 2024-09

Stakeholderzy zgłosili problemy z jakością rekomendacji - zob. COMPLAINTS.md

Uruchomienie:
    python recommender.py
Wymaga: pandas
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
TOP_K = 5


# ---- 1. Wczytanie danych ----------------------------------------------------
print("Wczytuję dane...")
candidates         = pd.read_csv(DATA_DIR / "candidates.csv")
projects           = pd.read_csv(DATA_DIR / "projects.csv")
skills             = pd.read_csv(DATA_DIR / "skills.csv")
candidate_skills   = pd.read_csv(DATA_DIR / "candidate_skills.csv")
project_skills     = pd.read_csv(DATA_DIR / "project_skills.csv")
candidate_projects = pd.read_csv(DATA_DIR / "candidate_projects.csv")

print(f"  {len(candidates)} kandydatów, {len(projects)} projektów, {len(skills)} skilli")
print(f"  {len(candidate_skills)} edges kandydat-skill, "
      f"{len(project_skills)} edges projekt-skill")


# ---- 2. Budowa grafu bipartite candidate-skill ------------------------------
# Reprezentacja: dict {candidate_id: set(skill_ids)} oraz {project_id: set(skill_ids)}.
# Skille kandydata to jego sąsiedzi w bipartite graph;
# skille projektu to jego sąsiedzi.
print("Buduję graf bipartite candidate-skill...")

cand_skill_sets = (
    candidate_skills
    .groupby("candidate_id")["skill_id"]
    .apply(set)
    .to_dict()
)

# Skille projektów (required + nice_to_have traktowane razem)
proj_skill_sets = (
    project_skills
    .groupby("project_id")["skill_id"]
    .apply(set)
    .to_dict()
)


# ---- 3. Skoring: common neighbors w bipartite -------------------------------
# Dla każdej pary (kandydat, projekt) liczymy liczbę wspólnych skilli.
# To podstawowa miara podobieństwa w bipartite graphs:
#     score(c, p) = |N(c) ∩ N(p)|
# gdzie N(x) to zbiór sąsiadów wierzchołka x w bipartite candidate-skill graph.
print("Liczę common neighbors dla każdej pary kandydat × projekt...")

rows = []
for pid, p_skills in proj_skill_sets.items():
    for cid, c_skills in cand_skill_sets.items():
        score = len(c_skills & p_skills)
        rows.append((pid, cid, score))

all_scores = pd.DataFrame(rows, columns=["project_id", "candidate_id", "score"])


# ---- 4. Top-K per project ---------------------------------------------------
# Sortowanie: score malejąco, przy remisach candidate_id rosnąco.
print(f"Wybieram top-{TOP_K} kandydatów dla każdego projektu...")

recommendations = (
    all_scores
    .sort_values(
        ["project_id", "score", "candidate_id"],
        ascending=[True, False, True],
    )
    .groupby("project_id", as_index=False)
    .head(TOP_K)
)
recommendations["rank"] = (
    recommendations.groupby("project_id").cumcount() + 1
)
recommendations = recommendations[["project_id", "rank", "candidate_id", "score"]]


# ---- 5. Zapis wyników -------------------------------------------------------
out_path = DATA_DIR / "baseline_recommendations.csv"
recommendations.to_csv(out_path, index=False)
print(f"Zapisano {len(recommendations)} rekomendacji do {out_path}")


# ---- 6. Quick eval: precision@5 dla projektów completed ---------------------
# Sprawdzamy, ilu z top-5 rekomendowanych faktycznie zostało przydzielonych
# do tego projektu w danych historycznych.
print("\n=== Ewaluacja: precision@5 ===")
historical = (
    candidate_projects[["project_id", "candidate_id"]]
    .drop_duplicates()
    .assign(was_assigned=True)
)

eval_df = (
    recommendations
    .merge(historical, on=["project_id", "candidate_id"], how="left")
    .assign(was_assigned=lambda d: d["was_assigned"].fillna(False))
    .groupby("project_id", as_index=False)["was_assigned"]
    .mean()
    .rename(columns={"was_assigned": "precision_at_5"})
)

print(f"Średnie precision@5 across all projects: "
      f"{eval_df['precision_at_5'].mean():.3f}")
print(f"Mediana: {eval_df['precision_at_5'].median():.3f}")
print(f"Projekty z precision@5 >= 0.4: "
      f"{(eval_df['precision_at_5'] >= 0.4).sum()}/{len(eval_df)}")


# ---- 7. Diagnostic: którzy kandydaci są rekomendowani najczęściej? ----------
print("\n=== Top 10 najczęściej rekomendowanych kandydatów ===")
top_recommended = (
    recommendations
    .groupby("candidate_id")
    .size()
    .reset_index(name="n_recommendations")
    .merge(candidates, on="candidate_id")
    .sort_values("n_recommendations", ascending=False)
    .head(10)
    [["candidate_id", "name", "level", "n_recommendations"]]
)
print(top_recommended.to_string(index=False))
