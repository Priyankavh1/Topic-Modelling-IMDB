# Topic Modelling for IMDb Reviews

A hybrid NLP pipeline that combines **Latent Dirichlet Allocation (LDA)** topic modelling with **VADER sentiment analysis** on the IMDb movie reviews dataset. The project uncovers latent themes across reviews and evaluates sentiment classification performance using standard metrics.

---

## Overview

Movie reviews carry rich, layered information — opinions about acting, plot, direction, music, and more. This project applies unsupervised topic modelling to discover what reviewers are actually talking about, alongside lexicon-based sentiment analysis to determine whether they liked it. The combination offers a fuller picture of audience perception than sentiment alone.

---

## Pipeline

```
Raw IMDb Reviews
      ↓
Text Preprocessing (lowercase, HTML tag removal, stopword removal, lemmatization)
      ↓
      ├──── LDA Topic Modelling (5 topics, top 10 keywords each)
      │
      └──── VADER Sentiment Analysis → Polarity Score → Sentiment Label
                                              ↓
                                    Evaluation vs. VADER-derived ground truth
                                    (Accuracy, Precision, Recall, F1)
```

---

## Features

- **Text Preprocessing** — lowercasing, HTML `<br>` tag removal, non-alphabetic character stripping, stopword filtering, and WordNet lemmatization
- **LDA Topic Modelling** — extracts 5 latent topics from the review corpus using `CountVectorizer` + `LatentDirichletAllocation`, displaying top 10 keywords per topic
- **VADER Sentiment Analysis** — assigns compound polarity scores and maps them to `positive`, `negative`, or `neutral` labels
- **Sentiment Distribution Visualization** — bar chart of sentiment class distribution across the dataset
- **Classification Metrics** — Accuracy, Precision (macro), Recall (macro), and F1-Score (macro)

---

## Tech Stack

| Component | Library |
|---|---|
| Data Handling | `pandas` |
| Text Preprocessing | `nltk` (stopwords, tokenizer, WordNetLemmatizer) |
| Topic Modelling | `sklearn` (CountVectorizer, LDA) |
| Sentiment Analysis | `vaderSentiment` |
| Evaluation Metrics | `sklearn.metrics` |
| Visualization | `matplotlib`, `seaborn` |

---

## Dataset

**IMDb Movie Reviews Dataset** (`imdb_dataset.csv`)  
A widely used benchmark dataset containing 50,000 movie reviews labeled as positive or negative.  
Available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

> The dataset file is not included in this repository. Download it from Kaggle and place it in the project root before running the notebook.

---

## Installation

```bash
pip install pandas nltk scikit-learn vaderSentiment matplotlib seaborn
```

Also download the required NLTK resources (handled in the notebook):

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

---

## Usage

1. Clone the repository and place `imdb_dataset.csv` in the project root
2. Open `SMDM_project.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells in order

---

## Sentiment Labeling

VADER compound scores are mapped as follows:

| Compound Score | Label |
|---|---|
| > 0 | Positive |
| < 0 | Negative |
| = 0 | Neutral |

---

## Evaluation

Model performance is measured by comparing VADER-predicted labels against VADER-derived ground truth labels across the full dataset, using macro-averaged metrics to account for class imbalance.

Metrics reported: **Accuracy**, **Precision**, **Recall**, **F1-Score**

---

## Project Structure

```
├── SMDM_project.ipynb     # Main analysis notebook
├── imdb_dataset.csv       # Dataset (not included — download separately)
└── README.md
```

---

## Course Context

Developed as part of the **Social Media Data Mining** coursework at **Syracuse University**.
