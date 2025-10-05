# YT-CLASSIFY — Mini-Sprint Spec

> **Scope:** Start from zero with only `channels.json` (fields: `id`, `title`, `description`, `videoCount`). Produce a portable, data-driven classifier that groups channels into human-sensible topics for a digest UI. No prior heuristics, no external APIs.

---

## 1) Mission

Design a **portable, data-driven channel classification system** that groups YouTube channels into human-sensible topics using only `channels.json`. The system must be fast, explainable, reproducible, and generalize to any user’s dataset without hand-tuned rules.

## 2) Primary Goal

From `channels.json`, produce a structured output that organizes channels into:

1. **Umbrella categories** (broad topics that cover ≥95% of channels).
2. **Sub-clusters** within each umbrella (tight, meaningful groupings).

The result powers a UI where users quickly **recognize**, **find**, and **select** channels for their digest.

---

## 3) Success Criteria (Definition of Done)

- **Coverage:** ≥ 90% of channels assigned to an umbrella; ≥ 70% to a sub-cluster.
- **Cohesion & Separation:** avg cosine(sim(channel, own-cluster centroid)) ≥ 0.35 and avg margin vs. 2nd-best centroid ≥ 0.05 (TF-IDF baseline, no embeddings).
- **Balance:** No umbrella > 40% of total; sub-clusters not all “Other”.
- **Explainability:** Each assignment exposes top terms/features (why it grouped there).
- **Determinism:** Same input → same output (fixed seeds/config); reproducible builds.
- **Performance:** ≤ 5s on 1k channels on laptop-class hardware, offline.
- **Portability:** No dataset-specific rules; only a minimal global stopword list.

---

## 4) Non-Goals (for this sprint)

- No manual overrides, no circle packing/tiling, no image or thumbnail work.
- No external APIs or scraping.
- No evolving ontologies: umbrella list is fixed for the sprint.
- No per-user tuning or hand curation (ensure generalization).

---

## 5) Inputs & Outputs

### Input
`channels.json` with at least:
- `id` (string), `title` (string), `description` (string; may be empty), `videoCount` (int).

### Output (JSON)
```json
{
  "umbrellas": [
    {
      "name": "<Umbrella>",
      "clusters": [
        {
          "label": "<Auto-labeled sub-cluster>",
          "explain": { "topTerms": ["term1","term2","term3"] },
          "items": [
            {
              "id": "...",
              "title": "...",
              "description": "...",
              "videoCount": 123,
              "score": { "sim": 0.51, "margin": 0.09 }
            }
          ]
        }
      ]
    }
  ],
  "unclassified": [ /* channels that failed thresholds */ ]
}
```

### Output (CSV)
For quick UI integration:
```
cluster_id,cluster_label,id,title,description,sim,margin,videoCount
```
Where `cluster_id` is deterministic (e.g., `UmbrellaSlug::SubSlug` or a stable index).

---

## 6) Umbrella Taxonomy (Fixed for Sprint)

**24 umbrellas intended to cover ≥95% of channels:**

- News & Commentary  
- Science & Technology  
- Business & Finance  
- Music & Musicians  
- Film & TV  
- Gaming  
- Sports  
- Health & Wellness  
- Food & Cooking  
- Travel & Places  
- Education & Learning  
- Arts & Design  
- DIY & Making  
- Home & Garden  
- Auto & Vehicles  
- Aviation & Flight  
- Weather & Climate  
- History & Culture  
- Reading & Literature  
- Spirituality & Philosophy  
- Fashion & Beauty  
- Pets & Animals  
- Comedy & Entertainment  
- Podcasts & Interviews

> **Portability principle:** Avoid dataset-specific keywords; rely on TF-IDF and a tiny global stopword set. Umbrellas act as broad anchors only.

---

## 7) Algorithm (No External Dependencies)

### 7.1 Tokenization & TF-IDF
- Construct text as `(title + title + description)` (title ×2 for vague titles).
- Tokenize (lowercase, strip URLs/punct, remove global stopwords).
- Build **TF-IDF** with smoothed IDF. Keep vectors unit-normalized.

### 7.2 Umbrella Assignment (portable)
- **Option A (Baseline):** Lexicon *hints* only (1–2 generic stems per umbrella) with **title×3, desc×1.8**. If inconclusive → `Unclassified`.
- **Option B (Strictly data-driven alternative):** Train a tiny one-vs-rest linear classifier from *confident* seeds discovered automatically (no custom rules). (Optional; keep off by default to remain rule-free.)

### 7.3 Centroid Reassignment (description-first)
- Compute umbrella centroids from already-assigned channels (min members: **5**).
- Reassign each `Unclassified` to nearest centroid if **cosine ≥ 0.20** and **gap ≥ 0.05**.

### 7.4 Sub-clustering (within each umbrella)
- Use **term-seeded** clustering (no libs): choose K by size (1–6), seed by top TF-IDF terms, assign by term weights, **merge tiny clusters (<6)** into **Other (Umbrella)**.

### 7.5 Auto-labels & Explanations
- Sub-cluster label = top 2–3 distinctive terms (proper-cased).
- For every item, store `topTerms` contributing the most weight to its assignment.

### 7.6 Handling Name-Only Channels (“Jack Smith”)
- Let **description** drive classification. If description is missing/very short:
  1) rely on title duplication (title ×2),  
  2) compare to umbrella **prototype vectors** (generic seed texts),  
  3) else keep `Unclassified` (explicitly surfaced).

---

## 8) Evaluation Harness

**Metrics**
- Coverage, Cohesion, Separation, Balance (as in §3).

**Reports**
- Per-umbrella and per-cluster tables: size, top terms, 5 sample titles.
- CSV export for the UI: `cluster_id, cluster_label, id, title, description, sim, margin, videoCount`.

**Comparisons**
- Baseline **100% TF-IDF like-with-like** vs. **Umbrella→Centroid→Sub-clusters**.

**QA**
- Random 50-channel sample with explanations printed for spot checks.

---

## 9) Risks & Mitigations

- **Short/empty descriptions:** allow `Unclassified`; optional cached AI finish pass **later** (post-sprint).  
- **Mega-umbrellas:** cap sub-clusters at 6; keep **Other (Umbrella)** to prevent fragmentation.  
- **Overfitting:** forbid dataset-specific rules; use fixed seeds; evaluate on random splits.

---

## 10) Deliverables

- `yt-classify.md` (this spec)  
- `classify.py` (single no-dep script)  
- `run.sh` (entrypoint)  
- Sample outputs: `out/clusters.json`, `out/clusters.csv`, `out/report.md`  
- **Human review page:** `out/report.html` (see §11)  
- `README.md` with quickstart & metrics

---

## 11) Human-Review HTML (`out/report.html`)

**Purpose:** A fast, zero-dependency page to review each run across all channels.

**Inputs:** Loads `out/clusters.json` (schema in §5). Works offline.

**Layout & Behavior:**
- **Summary header**: umbrellas count, channel count, sub-cluster coverage %, unclassified count.
- **One panel per umbrella** in a single column (scroll). Each panel shows:
  - Header: `Umbrella Name • #channels • #sub-clusters`
  - **Rows for each sub-cluster** with a label chip (name + count) and a **mini bubble strip** (small circles sized by `sqrt(videoCount)`).
  - Clicking a sub-cluster **expands** a list of channels (title + optional description snippet).
- **Unclassified panel** at the end (if any).
- **Controls (sticky):** search (filters titles/descriptions), sort (by count/label), toggle snippets.

**Constraints:**
- **No frameworks**; plain HTML/CSS/JS.
- Must render in < 200ms on 1k channels on a laptop-class machine.
- Deterministic: no external network requests.

**Starter template:** This repository MUST include a working `out/report.html` that reads `clusters.json` from the same folder and renders the view as specified. (A reference template is provided alongside this spec.)

---

## 12) Milestones (2 days)

- **D0–0.5:** Baseline TF-IDF like-with-like + metrics.  
- **D0.5–1.0:** Umbrella centroids + reassignment thresholds tuned.  
- **D1.0–1.5:** Sub-clustering + labels/explanations + CSV/JSON export.  
- **D1.5–2.0:** HTML review page + metrics sweep, finalize thresholds, write report.

---

## 13) LLM Prompt Contract (Optional, later)

> **Task:** From `channels.json`, assign each channel to a fixed umbrella and a coherent sub-cluster using only title/description. When uncertain, output `Other (Umbrella)` or `Unclassified`. Provide per-item explanations (top terms) and per-cluster labels derived from distinctive terms. Optimize for coverage, cohesion, separation, and balanced clusters. Output follows the schema in §5.

---

## 14) Config Knobs (tunable)

- Title duplication factor (default **2**).  
- Centroid thresholds: `SIM_MIN` (default **0.20**), `GAP_MIN` (default **0.05**).  
- Sub-cluster K rule (1–6) and tiny-cluster merge threshold (default **6**).  
- Stopword list (global; minimal).  
- Random seed (default **42**).

---

## 15) Repo Skeleton

```
/yt-classify
  ├── yt-classify.md
  ├── README.md
  ├── classify.py
  ├── run.sh
  ├── channels.json           # input
  └── out/
      ├── clusters.json
      ├── clusters.csv
      ├── report.md
      └── report.html
```

---

## 16) Acceptance

- Meets **Success Criteria** (§3) on an unseen channels.json sample of ≥1k entries.  
- Code is single-file, dependency-free (Python stdlib), deterministic, and documented.  
- Outputs integrate cleanly into the digest UI (CSV + JSON).  
- **HTML review page renders correctly** from `report.html` with only `clusters.json` present.

---

**End of spec.**
