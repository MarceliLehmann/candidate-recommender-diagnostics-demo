# Candidate Recommender Diagnostics Report

## 1. Baseline understanding

The baseline recommender represents candidates and projects as sets of skills.

A candidate is connected to the skills they have.  
A project is connected to the skills it requires.

For every candidate-project pair, the baseline computes:

```text
score(candidate, project) = number of shared skills
```

Then, for each project, candidates are sorted by descending score and the top 5 are returned.

This is a simple common-neighbor approach on a candidate-skill/project-skill graph. It is easy to explain, but it ignores several important signals:

- total number of skills listed by a candidate,
- source and confidence of candidate skills,
- required vs nice-to-have project skills,
- rarity of skills,
- candidate assignment history.

## 2. Diagnostics

I did not treat every observed issue as a separate bug. Several symptoms came from the same root cause: the baseline uses raw skill overlap and does not distinguish skill quality.

I focused on issues that can be reproduced in data and improved with small, clear changes.

### 2.1. Raw overlap favors broad profiles

The baseline rewards candidates for having many skills because every listed skill creates another chance to overlap with a project.

This creates a hub-candidate problem. A broad profile may rank highly across unrelated projects, even when the candidate is not the most relevant specialist.

In diagnostics, the most frequently recommended candidates were often those with larger skill profiles. The correlation between number of skills and number of recommendations was high enough to treat this as a systemic ranking issue.

**Classification:** algorithmic issue.  
**Root cause:** no normalization by candidate profile size.  
**Improvement:** replace raw overlap with a weighted normalized similarity score.

### 2.2. Skill quality is ignored

The baseline converts skill tables into plain sets of `skill_id`.

That removes useful metadata:

- skill source,
- confidence,
- last update,
- project skill criticality.

As a result, a self-reported skill has the same ranking impact as a validated skill. A nice-to-have skill has the same impact as a required skill.

This explains why noisy broad profiles can rank too high and why narrow specialists may be underrepresented.

**Classification:** algorithmic issue and data-quality issue.  
**Root cause:** metadata is lost before scoring.  
**Improvement:** use source, confidence, criticality, and skill rarity in scoring.

### 2.3. Already assigned candidates can be recommended

The baseline does not check assignment history before returning recommendations.

This can produce operationally useless results, for example recommending someone who is already assigned to or has already completed the same project.

**Classification:** process/algorithm bug.  
**Root cause:** no filter based on candidate-project history.  
**Improvement:** filter candidates with `assigned` or `completed` status for the same project.

### 2.4. Issues not solved directly

Some issues should not be treated as pure recommender bugs.

If a junior candidate has relevant experience but this skill is missing from the data, the recommender cannot infer it reliably. This is a data coverage problem.

If business users expect legacy technology experience to transfer to modern projects, the system needs a skill taxonomy or skill-similarity layer. Exact skill overlap cannot model transferability.

I left these as limitations rather than forcing them into the two implemented changes.

## 3. Improvements and validation

### 3.1. Improvement 1: assignment filter

**Hypothesis**

The recommender fails because it ignores assignment history.  
If candidates already assigned to or completed on the same project are removed before ranking, the output will be more useful operationally.

**Implementation**

The improved recommender builds a set of `(project_id, candidate_id)` pairs from assignment history where status is either:

- `assigned`,
- `completed`.

Those pairs are skipped before scoring and top-5 selection.

**Validation**

The validation script checks the full improved output and confirms that no assigned/completed candidate appears in recommendations.

The output shape is preserved:

- every project still receives 5 recommendations,
- the improved output keeps the same schema as the baseline.

**Honest assessment**

This fix solves a clear operational bug. It does not improve the quality of skill matching itself.

In some products, already assigned candidates may still be useful as context. In that case, the better UI choice would be to show them with a label instead of removing them completely.

### 3.2. Improvement 2: weighted normalized score

**Hypothesis**

The baseline fails because raw overlap rewards candidates with many skills and ignores skill quality.  
If I use a weighted normalized score, hub-candidate dominance should decrease.

**Implementation**

The improved recommender uses a cosine-like similarity over weighted skill profiles.

Candidate-side weights use:

- skill source,
- confidence,
- skill rarity.

Project-side weights use:

- required vs nice-to-have criticality,
- skill rarity.

The score is normalized, so a candidate with a long skill list does not automatically dominate.

**Validation**

The validation compares the relationship between number of skills and number of recommendations before and after improvement.

The key aggregate check is the correlation between:

```text
number of candidate skills
```

and:

```text
number of times the candidate appears in recommendations
```

This is the right aggregate metric for this specific issue because the diagnosed problem is hub-candidate dominance.

After improvement, this correlation drops clearly, showing that the new scoring method reduces the profile-size advantage.

**Honest assessment**

The new score is better aligned with the diagnosed issue, but it is still heuristic.

The source and criticality weights should be reviewed with business users. IDF-like weighting can also overpromote rare skills if those skills are outdated or not relevant enough.

This improvement also cannot fix missing skills. If a candidate has a skill in reality but it is not stored in the data, the recommender still cannot use it.

## 4. Reflection and limitations

### 4.1. Missing data

To improve the recommender further, I would need:

- candidate availability,
- workload and project allocation,
- required seniority,
- skill recency,
- candidate preferences,
- project domain and client context,
- stronger post-project feedback,
- records of candidates considered but not selected.

The biggest evaluation limitation is missing negative data. If a candidate was not historically proposed for a project, that does not mean they were a bad match.

### 4.2. Business questions

I would ask business stakeholders:

- Should the recommender optimize technical fit or staffing feasibility?
- Should already assigned candidates be hidden or shown with labels?
- How much should we trust CV skills compared with assessment and peer review?
- Should nice-to-have skills affect ranking or only explanations?
- Should the output be fixed top-5, or should users see top-10 with explanations?
- Should the system favor specialists or broad generalists?
- How should rare skills be handled?

### 4.3. Limits of static skill matching

A static skill graph only knows what is explicitly stored in the data.

It does not understand:

- skill transferability,
- skill recency,
- candidate availability,
- team fit,
- client preferences,
- real staffing constraints.

Changing the similarity metric can reduce some failure modes, but it cannot solve missing or outdated data.

### 4.4. Monitoring

I would monitor three groups of metrics.

**Operational checks**

- number of recommendations per project,
- self-recommendation rate,
- projects with no suitable candidates,
- score distribution,
- number of ties.

**Ranking quality**

- acceptance rate of recommended candidates,
- rejection reasons,
- conversion from recommendation to assignment,
- post-project feedback,
- stability after data updates.

**Bias and coverage**

- concentration of recommendations among a few candidates,
- visibility of niche specialists,
- visibility of junior/new candidates,
- relationship between number of skills and recommendation frequency.

I would also add a feedback loop where staffing users can mark recommendations as useful, not useful, or missing, with a short reason.

## 5. AI usage

I used AI tools as support for structuring the diagnostic workflow, naming issue classes, and improving the clarity of the report.

AI was not treated as a source of truth. Each important conclusion was checked against code output or validation results.

The main risk of using AI was overconfident interpretation without data verification. To reduce that risk, I tied conclusions to reproducible diagnostics and before/after validation.
