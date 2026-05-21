# Error Pattern Discovery in Introductory Programming: An AI-Driven Pipeline

> A proof-of-concept pipeline combining structural code analysis, clustering, and LLM interpretation to detect recurring error patterns in introductory Java programming submissions and generate plausible misconception hypotheses for educators.

---

## Overview

Introductory programming courses consistently exhibit high failure and dropout rates. A significant contributing factor is the persistence of student misconceptions about core concepts in programming, which often go undetected at scale and are rarely addressed by existing automated tools, which tend to focus on correctness rather than the conceptual origins of errors.

This project implements an end-to-end pipeline that moves beyond correctness assessment toward structural error pattern discovery. Applied to ~51,000 Java submissions from [IBM's Project CodeNet](https://github.com/IBM/Project_CodeNet), the pipeline:

1. Parses submissions into Abstract Syntax Trees and trains a neural network to detect structurally erroneous code
2. Extracts and filters attention-weighted subtree vectors that the model associates with incorrectness
3. Reduces these high-dimensional vectors and clusters them to identify recurring structural error patterns
4. Feeds representative cluster samples to an LLM, which generates natural-language misconception hypotheses for each cluster

The system is designed to give educators a population-level diagnostic instrument in order to assist with instructor judgment. It aims to narrow the diagnostic search space by surfacing structural patterns that warrant closer inspection.

>The LLM-generated outputs should be understood as exploratory misconception hypotheses, not verified diagnoses. They are intended to guide educator attention, not serve as autonomous instructional recommendations.

---

## Pipeline Architecture

```
Raw Java Submissions (CodeNet)
         │
         ▼
┌─────────────────────────┐
│  1. AST Parsing &       │  java_to_s_expression_parser.py
│     Subtree Extraction  │  subtree_extractor.py
│     & Tokenization      │  subtree_tokenizer.py
└────────────┬────────────┘
             │  PyTorch tensors (.pt files)
             ▼
┌─────────────────────────┐
│  2. Error Detection     │  SANN_model.py
│     (Modified SANN)     │
│     - Binary classification (correct/incorrect)
│     - Sigmoid attention weights per subtree
│     - Entropy regularization
└────────────┬────────────┘
             │  64-dim subtree vectors (attention weight > 0.5)
             ▼
┌─────────────────────────┐
│  3. Dimensionality      │  UMAP_HDBSCAN.py
│     Reduction & Clustering│
│     - UMAP (64D → 2D)  │
│     - HDBSCAN clustering│
└────────────┬────────────┘
             │  12 structural error clusters
             ▼
┌─────────────────────────┐
│  4. LLM Interpretation  │  groq_cluster_interpreter.py
│     (Groq API)          │
│     - 3-step chained prompting per cluster
│     - Misconception hypothesis generation
└────────────┬────────────┘
             │
             ▼
    JSON reports + HTML summaries
    (one per cluster, for educator review)
```

---

## Key Results

| Stage | Metric | Value |
|---|---|---|
| Error Detection | Accuracy | 0.72 |
| Error Detection | F1-Score | 0.65 |
| Dimensionality Reduction | Trustworthiness | 0.8936 |
| Dimensionality Reduction | Continuity | 0.8340 |
| Clustering | Silhouette Coefficient | 0.3801 |
| Clustering | Davies-Bouldin Index | 0.7709 |
| Clusters Produced | — | 12 |
| Clusters with Specific Findings | — | 3 / 7 interpreted |

The three clusters that yielded educationally grounded hypotheses identified: **reference equality misuse** (`==` vs `.equals()`), **insufficient algorithmic reasoning** (primality checking, O(n²) complexity), and **Scanner input-handling errors** (`nextInt()`/`nextLine()` ordering). All three are consistent with documented Java misconceptions in the computing education literature (Sorva, 2012).

---

## Repository Structure

```
├── Processing/
│   ├── SANN_model.py               # Full SANN architecture, training loop, and feature extraction
│   ├── UMAP_HDBSCAN.py             # Dimensionality reduction + clustering + evaluation metrics
│   ├── groq_cluster_interpreter.py # 3-step LLM prompting via Groq API
│   ├── java_to_s_expression_parser.py  # Fault-tolerant Java → AST parsing (Tree-sitter)
│   ├── subtree_extractor.py        # Recursive subtree extraction from ASTs
│   ├── subtree_tokenizer.py        # Vocabulary construction + tensor serialization
│   ├── correctness_label_normalizer.py  # Accuracy imputation + binary label assignment
│   ├── metadata_organizer.py       # Dataset metadata structuring and filtering
│   ├── cluster_analyzer.py         # Post-clustering analysis and sample selection
│   ├── manual_review_organizer.py  # Organizes cluster samples for manual inspection
│   ├── file_locator.py             # Utility for locating submission files by ID
│   ├── json_to_html_converter.py   # Converts LLM JSON outputs to readable HTML reports
│   └── Trials/                     # Experimental scripts from earlier pipeline iterations
│
├── Data/                           # Excluded from version control (.gitignore)
│   ├── submission_tensors/         # PyTorch .pt files (one per submission)
│   ├── node_vocabulary.json        # Syntax element → integer ID mapping
│   ├── global_subtree_vocabulary.json  # Subtree structure → global ID mapping
│   ├── submissions_metadata_labels_cleaned.csv
│   ├── incorrect_source_code_vectors.csv
│   ├── incorrect_attention_weights.csv
│   ├── error_clusters_mapped.csv
│   └── groq_cluster_analysis/     # LLM output JSON files (one per cluster)
│
└── README.md
```

### Processing Scripts Descriptions

| Script | Role in Pipeline |
|---|---|
| `java_to_s_expression_parser.py` | Parses raw Java source code into ASTs using Tree-sitter's fault-tolerant Java grammar binding. Outputs S-expression representations. |
| `subtree_extractor.py` | Recursively extracts every node and its immediate children from each AST, serializing each as a structured textual subtree. |
| `subtree_tokenizer.py` | Builds two vocabularies (node-level and subtree-level) and converts each submission into a padded PyTorch tensor of integer IDs. |
| `correctness_label_normalizer.py` | Imputes missing Accuracy values from Status labels, removes intermediate accuracy entries, and produces a clean binary correctness label. |
| `metadata_organizer.py` | Filters and restructures CodeNet metadata CSVs for the selected Java subset. |
| `SANN_model.py` | Full pipeline: model definition (two-way embedding + LSTM + sigmoid attention + entropy regularization), training loop with Adamax optimizer and early stopping, test evaluation, and feature extraction via forward hook on the `fc_combine` layer. |
| `UMAP_HDBSCAN.py` | Applies attention-weight threshold (> 0.5) to filter subtree vectors, runs UMAP reduction (64D → 2D), applies HDBSCAN clustering, evaluates both stages with quality metrics, and saves cluster assignments. |
| `cluster_analyzer.py` | Analyzes cluster distributions, computes cluster statistics, and prepares data for LLM interpretation. |
| `manual_review_organizer.py` | Randomly samples submissions per cluster and exports structured CSV for manual inspection or LLM prompting. |
| `groq_cluster_interpreter.py` | Runs the 3-step chained prompting strategy (individual analysis → cluster analysis → misconception inference) via the Groq API for each cluster. Includes retry logic with exponential backoff. |
| `json_to_html_converter.py` | Converts the JSON cluster analysis outputs from Groq into readable, styled HTML reports for educator review. |
| `file_locator.py` | Utility script for locating submission files by submission ID across the dataset directory. |

---

## Tech Stack

- **Language:** Python 3
- **Parsing:** [Tree-sitter](https://github.com/tree-sitter/tree-sitter) with Java grammar binding
- **Deep Learning:** PyTorch
- **Dimensionality Reduction:** [UMAP-learn](https://github.com/lmcinnes/umap)
- **Clustering:** [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) via scikit-learn
- **LLM Inference:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Dataset:** [IBM Project CodeNet](https://github.com/IBM/Project_CodeNet)

---

## Setup

```bash
# Clone the repository
git clone https://github.com/maryamelnahas/Bachelor-Project.git
cd Bachelor-Project

# Install dependencies
pip install torch tree-sitter umap-learn hdbscan scikit-learn pandas numpy groq tqdm

# Install Tree-sitter Java grammar
pip install tree-sitter-java
```

> The `Data/` directory is excluded from version control due to file size. To reproduce results, download the Java subset of the first 500 problems in [Project CodeNet](https://github.com/IBM/Project_CodeNet) and run the processing scripts in the order listed in the pipeline diagram above.

A Groq API key is required for the interpretation stage. Set it in `groq_cluster_interpreter.py` or as an environment variable.

---

## Academic Context

This project was developed as part of a Bachelor's thesis in Business Informatics at The German University in Cairo, 2026.

**Thesis title:** *AI-Driven Framework for the Detection and Interpretation of Error Patterns in Introductory Programming*


