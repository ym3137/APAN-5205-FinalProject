# Using NLP to Find Keywords to Predict Viral Topics on Hacker News

**APAN 5205 Applied Machine Learning II — Group 2**  
Yun-Chien (Athena) Huang · Yike (Ryan) Meng · Leanne Chung · Joe Wang · Feiyue Gai

---

## Project Overview

Hacker News (HN) engagement is highly unequal where a small fraction of posts capture most upvotes while the majority receive almost none. This project uses NLP and unsupervised clustering to identify which *topic clusters* produce disproportionately viral posts, and builds predictive regression models to quantify how well post titles predict engagement scores.

Rather than building a black-box popularity predictor, our goal is interpretability: cluster posts into named topics, extract the keywords that define each cluster, and explain which content areas drive virality on HN.

---

## Repository Structure

```
├── data/
│   ├── hn_stories_raw.csv          # Raw data pulled from Algolia HN API (17,274 rows)
│   ├── hn_stories_clean.csv        # Cleaned dataset used as input to all notebooks (14,065 rows)
│   └── hn_with_clusters.csv        # Output of EDA+Clustering notebook; input to predictive model
│
├── hn_data_eda_clustering.ipynb    # Main notebook: EDA + TF-IDF + K-Means clustering
├── hn_supplementary_eda.ipynb      # Supplementary notebook: trends, author analysis, word clouds
├── predictive_model.ipynb          # Predictive model: Decision Tree + Random Forest regression
│
└── README.md
```

---

## Notebooks

### 1. EDA Clustering `hn_data_eda_clustering.ipynb`
**Author:** Yike (Ryan) Meng

The primary analysis notebook. Covers exploratory data analysis, NLP preprocessing, TF-IDF vectorization, and two-stage K-Means clustering.

**Sections:**
- **Section 1 — EDA:** Score distribution (raw vs. log1p), story type volume and average scores, posting time patterns (hour and day of week), title length vs. score
- **Section 2 — Text Preprocessing:** Strips `Show HN:` / `Ask HN:` prefixes, removes HN-specific stopwords and generic high-frequency words, applies WordNet lemmatization
- **Section 3 — K-Means Clustering:** Elbow method to select k=6 for Stage 1; sub-clusters the dominant catchall group into 3 sub-groups (Stage 2); produces 8 final interpretable clusters
- **Section 4 — Cluster Analysis:** Manually names each cluster based on top TF-IDF terms; compares average score by cluster; generates cluster × story type heatmap
- **Section 5 — Export:** Saves `data/hn_with_clusters.csv` for use by the predictive model notebook

**Output file:** `data/hn_with_clusters.csv`

---

### 2.  Supplementary Analysis `hn_supplementary_eda.ipynb`
**Author:** Leanne Chung

Extends the main EDA with additional visualizations not included in the primary notebook.

**Sections:**
- **Section 1 — Monthly Posting Trends:** Post volume and average/median score over 14 months (Feb 2025–Mar 2026); monthly breakdown by story type
- **Section 2 — Story Type Deep Dive:** Title length distributions by story type (boxplot); log1p(score) distributions by story type (violin plot); proportion of posts with URL vs. body text per type
- **Section 3 — Author Activity Analysis:** Top 15 most active authors by post count; average score for top authors vs. platform average; scatter of post count vs. avg score for authors with ≥5 posts
- **Section 4 — Text Analysis:** Top 20 most frequent words in HN titles (bar chart); overall word cloud; per-type word clouds for the top 4 story types; normalized word frequency heatmap across all story types

**Input file:** `data/hn_stories_clean.csv`

---

### 3. Predictive Modeling `predictive_model.ipynb`
**Author:** Yun-Chien (Athena) Huang

Trains and evaluates regression models to predict log1p(score) from post title text.

**Sections:**
- **Read Data:** Loads `hn_with_clusters.csv`, creates `log_score = log1p(score)` target
- **Custom Stopword List:** Same custom HN stopwords used in the clustering notebook to ensure consistent preprocessing
- **Text Cleaning:** Strips HN prefixes, lowercases, removes URLs/numbers/punctuation, lemmatizes tokens
- **Split Data:** Time-based train/test split at `2026-01-30` to simulate real deployment conditions and prevent data leakage
- **Document Term Matrix:** CountVectorizer (unigrams, sklearn English stopwords) fitted on training titles only
- **Decision Tree Regressor:** `max_depth=10`, `min_samples_leaf=20`; tree structure visualized via `plot_tree()`
- **Random Forest + GridSearchCV:** 3-fold CV over 24 combinations of `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`; best model evaluated on test set

**Input file:** `data/hn_with_clusters.csv`

---

## Data

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| `hn_stories_raw.csv` | 17,274 | 9 | Raw API pull: title, URL, author, timestamp, score, num_comments |
| `hn_stories_clean.csv` | 14,065 | 20 | Cleaned dataset with engineered features (story_type, hour, is_weekend, etc.) |
| `hn_with_clusters.csv` | 14,065 | 21 | Cleaned dataset + `cluster` column (integer 0–7) from K-Means |

**Data source:** [Algolia HN Search API](https://hn.algolia.com/api) — public, no authentication required  
**Time range:** February 27, 2025 – March 2, 2026

### Key Variables

| Variable | Type | Description |
|---|---|---|
| `title` | string | Post headline — primary NLP input |
| `score` | int | Upvotes received (primary outcome) |
| `num_comments` | int | Comment count (secondary outcome) |
| `story_type` | categorical | 10 types: external, github, show_hn, ask_hn, news_major, research, blog_personal, text_post, other |
| `created_at` | datetime | UTC timestamp |
| `hour` | int | Hour of submission (UTC) |
| `day_of_week` | int | 0 = Monday, 6 = Sunday |
| `is_weekend` | int | 1 if Saturday or Sunday |
| `title_length` | int | Character count of title |
| `log_score` | float | log1p(score) — used as model target |
| `cluster` | int | K-Means cluster ID (0–7), added by clustering notebook |

---

## Methodology Summary

### Text Preprocessing Pipeline
All three notebooks share a consistent title cleaning approach:
1. Strip `Show HN:`, `Ask HN:`, `Tell HN:`, `Launch HN:` prefixes
2. Lowercase all text
3. Remove URLs, numbers, and punctuation
4. Remove standard English stopwords (NLTK) + custom HN-specific stopwords (~50 additional terms)
5. Apply WordNet lemmatization (e.g., `years → year`, `running → run`)

### Clustering (Two-Stage K-Means)
- **Stage 1:** TF-IDF vectorizer (500 features, unigrams + bigrams, `min_df=4`, `max_df=0.85`, `sublinear_tf=True`) → normalize → KMeans with k=6 (chosen by elbow method)
- **Stage 2:** Largest Stage 1 cluster sub-clustered with KMeans k=3
- **Result:** 8 final named topic clusters

### Predictive Models
- **Target:** `log1p(score)`
- **Features:** CountVectorizer bag-of-words on cleaned titles (fit on training set only)
- **Split:** Time-based at `2026-01-30` (no random shuffle — prevents leakage)
- **Models:** Decision Tree Regressor, Random Forest + GridSearchCV
- **Metric:** Root Mean Squared Error (RMSE) on log1p scale

---

## Results

### Topic Clusters and Average Score

| Cluster | Size | Top Keywords | Avg Score |
|---|---|---|---|
| Tech Industry & Business News | 219 | business, tech, company | 30.2 |
| Politics, Society & World Events | 12,836 | world, politics, news | 22.1 |
| Open Source Libraries & Dev Tools | 145 | rust, cli, library | 18.7 |
| Software Engineering & Programming | 255 | software, engineering, developer | 16.7 |
| Apps, Games & Products | 98 | app, game, product | 15.5 |
| Data, Science & Academic Research | 115 | data, research, science | 9.0 |
| Security, Privacy & Infrastructure | 452 | privacy, security, infrastructure | 8.8 |
| Help, Career & Discussion | 220 | help, career, discussion | 5.9 |

### Model Performance

| Model | Train RMSE | Test RMSE |
|---|---|---|
| Decision Tree (max_depth=10) | 1.194 | 1.111 |
| Random Forest + GridSearchCV | 1.183 | **1.108** |

Both models predict on the `log1p(score)` scale. An RMSE of ~1.11 reflects that title text alone captures non-trivial but limited variance in post engagement. Adding cluster labels, story type, and temporal features is the expected next step to improve performance.

---

## Setup and Requirements

### Python Version
Python 3.8+

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
```

### NLTK Data
The notebooks download required NLTK data automatically on first run:
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Running the Notebooks (in order)

```
1. hn_data_eda_clustering.ipynb      # produces data/hn_with_clusters.csv
2. hn_supplementary_eda.ipynb        # standalone, reads data/hn_stories_clean.csv
3. predictive_model.ipynb            # reads data/hn_with_clusters.csv
```

> **Important:** `hn_data_eda_clustering.ipynb` must be run before `predictive_model.ipynb` because it generates the cluster labels used as input.

---

## Key Findings

- **Score is power-law distributed.** Mean = 21, median = 3, max = 2,936. 72.8% of posts score 5 or below. The `log1p` transform is essential for modeling.
- **Topic cluster is a strong engagement signal.** Average scores span a 5× range across clusters (5.9 to 30.2) — far more predictive than title length (Pearson r ≈ –0.03).
- **Story format adds independent signal.** External links (avg 25.6) and GitHub repos (22.5) outperform Show HN (8.6) regardless of topic.
- **One cluster dominates.** 89.5% of posts fall in the broad "Politics, Society & World Events" cluster, reflecting the challenge of clustering short, diverse text.
- **Title text alone has limited but real predictive power.** Both models achieve RMSE ≈ 1.11, providing a strong baseline for richer feature-enriched models.

---

## References

- Algolia. *Hacker News Search API Documentation*. https://hn.algolia.com/api
- Geng, Z., Yuan, Y., & Wang, C. (2016). *Predicting Popularity of Posts on Hacker News*. CS229. https://cs229.stanford.edu/proj2016/report/GengYuanWang-PredictingPopularityOfPostsOnHackerNews-report.pdf
- Hacker News. *Official HN API (Firebase)*. https://github.com/HackerNews/API
- Keneshloo, Y. et al. (2016). *Predicting the Popularity of News Articles*. SIAM SDM. https://people.cs.vt.edu/naren/papers/sdm2016.pdf
- Maguire, J. & Michelson, S. (2012). *Predicting the Popularity of Social News Posts*. CS229. https://cs229.stanford.edu/proj2012/MaguireMichelson-PredictingThePopularityOfSocialNewsPosts.pdf
- Stoddard, G. (2021). Popularity Dynamics and Intrinsic Quality in Reddit and Hacker News. *ICWSM*, 9(1), 416–425. https://doi.org/10.1609/icwsm.v9i1.14636
