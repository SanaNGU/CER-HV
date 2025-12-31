# CER-HV: A CER-Based Human-in-the-Loop Framework for Cleaning Arabic-Script Handwritten Text Recognition Datasets (Arabic, Urdu, Persian, Ajami and Pashto)

This repository contains the implementation of **CER-HV (CER-based Ranking with Human Verification)**, a two-stage framework for identifying and correcting mislabeled or out-of-scope samples in handwritten text recognition (HTR) datasets.  
The framework combines model-based scoring using Character Error Rate (CER) with targeted human inspection, enabling efficient dataset cleaning for both modern and historical Arabic-script corpora.

---

## 📌 Framework Overview

The figure below illustrates the full CER-HV pipeline, consisting of:

1. **Automated Noise Detection** using a CRNN and CER-based ranking  
2. **Human-in-the-Loop Verification** of top-ranked error candidates

<p align="center">
  <img src="images/crhv_framework.png" width="900">
</p>

---

## 📌 Repository Structure


```text
├── src/                        # Core source code for CER-HV
│   ├── trainer.py              # Training script (Stage 1: CRNN training)
│   ├── evaluate.py             # Evaluation script (generates predictions)
│   ├── model.py                # CRNN model definition
│   ├── error_classification.py #  
│   ├── prepare_files_to_EA.py  # Prepares ranked CER lists for Stage 2 (HITL)
│   ├── configs/                # YAML configuration files for each dataset
│   ├── utils/                  # Utility functions 
│   └──README.md                  

├── error-analysis/             # Error analysis and visualization
│   ├── ajami-visualise-samples-with-errors.ipynb   
│   ├── khatt-visualise-samples-with-errors.ipynb   
│   ├── muharaf-visualise-samples-with-errors.ipynb   
│   ├── nust-visualise-samples-with-errors.ipynb  
│   ├── PHTI-visualise-samples-with-errors.ipynb   
│   ├── cleaning-effect-remove.ipynb  
│   ├── error_counts.py   
│   └── cleaning-effect-remove.py

├── images/                     # Framework diagrams, result tables, figures

├── saved_models/               # Model checkpoints (not included in repo)

├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files and folders excluded from Git
```

---

# The code for implementing CER-HV is in [src](src/README.md) 



# 🔍 Notebooks for visualizing each error category wit examples for each dataset

| Dataset | Notebook |
|---------|----------|
| **Muharaf**  | [muharaf_review.ipynb](error-analysis/muharaf-visualise-samples-with-errors.ipynb) |
| **KHATT**    | [khatt_review.ipynb](error-analysis/khatt-visualise-samples-with-errors) |
| **Ajami**    | [ajami_review.ipynb](error-analysis/ajami-visualise-samples-with-errors.ipynb) |
| **PHTI**     | [phti_review.ipynb](error-analysis/PHTI-visualise-samples-with-errors.ipynb) |
| **Nust-UHWR**| [phti_review.ipynb](error-analysis/nust-visualise-samples-with-errors.ipynb) |

# 🔍 Notebooks to see the effect of data cleaning on CER and WER for all datasets

[Cleaning effects ](error-analysis/cleaning-effect-remove.ipynb) |

## 📊 Results Overview

Below is the summary of CRNN benchmarking results across six datasets (KHATT, Muharaf, PHTI,PHTD, NUST-UHWR, Ajami).

<p align="center">
  <img src="images/CRNN-results.png" width="800">
</p>

The reported CER values are shown as **mean ± standard deviation** across multiple training runs.

---
## 📊 CER Before and After Cleaning

The following figure summarizes the effect of cleaning mislabeled samples on the evaluation splits for all datasets.

<p align="center">
  <img src="images/before and after.png" width="750">
</p>
