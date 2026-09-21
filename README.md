# IncidentPrediction

> **Accepted Paper | [IUKM 2026](https://www.jaist.ac.jp/IUKM/IUKM2026/index.php)**  
> To appear in Springer Lecture Notes in Artificial Intelligence (LNAI), an LNCS subseries | Scopus-indexed book series | SJR 2024 Q2

This repository contains the source code accompanying the paper:

**Abstract Reasoning-Driven Prediction and Root Cause Identification of Device Incidents in Core Internet Infrastructure**

Springer Nature Link: [https://link.springer.com/chapter/10.1007/978-981-92-5644-0_17](https://link.springer.com/chapter/10.1007/978-981-92-5644-0_17)

Full paper link: [Abstract Reasoning-Driven Prediction and Root Cause Identification of Device Incidents in Core Internet Infrastructure](https://fsd-labs.github.io/Network-Incident-Prediction/Abstract_Reasoning_Driven_Prediction_and_Root_Cause_Discovery_of_Device_Incidents_in_Core_Internet_Infrastructure.pdf)

**Data source:** [ML (LGBM) dataset](https://huggingface.co/datasets/FSD-LAB/noc-incident-dataset-for-ml) & [ART dataset](https://huggingface.co/datasets/FSD-LAB/noc-incident-dataset-for-dl)

## Overview

The code is organized around the three main technical components used in the paper:

- `ART/`: the proposed Abstract Reasoning LogTransformer for proactive incident prediction and concept-level root-cause analysis.
- `traditionalML/`: the LightGBM model representing a conventional feature-engineered machine-learning approach.
- `drain3_multiprocess/`: the length-sharded multiprocessing Drain3 implementation used to make template mining practical on high-volume network logs.

The full workflow starts from raw device observability streams, converts raw logs into stable templates, builds incident-prediction datasets, and compares a traditional tabular model against the proposed concept-token transformer.

## Paper Workflow to Code

1. Template mining:
   `drain3_multiprocess/` parses and clusters raw syslog messages into log templates. These templates become concept tokens for ART.
2. Traditional model:
   `traditionalML/` builds TF-IDF, temporal log-count, priority-score, and SNMP rolling features, then trains/evaluates LightGBM.
3. ART model:
   `ART/` builds concept-token sequences, trains a causal transformer with supervised incident prediction and next-concept prediction, and exports prediction and RCA artifacts.

## Repository Layout

```text
IncidentPrediction/
├── ART/                       # Proposed ART model and evaluation pipeline
├── traditionalML/             # LightGBM feature-engineered approach
├── drain3_multiprocess/       # Multiprocess Drain3 template mining
└── README.md
```

## Results Context

In the manuscript, experiments are conducted on production Juniper router and switch logs with sparse incident labels. The LightGBM traditional ML approach provides a strong predictive model for structured and text-derived features, while ART is introduced to add concept-level reasoning and root-cause ranking. The Drain3 multiprocess component supports this pipeline by reducing the template-mining bottleneck on large log streams.

## Data and Artifacts

Large datasets, trained checkpoints, and production-scale intermediate artifacts are intentionally not stored directly in this source tree. Use the dataset link above once released, then update each subproject config with local paths.

The code expects data in Parquet-based intermediate formats, with fields such as device/IP, timestamp or window end, normalized log message or template ID, and incident label depending on the pipeline stage.

## Subproject READMEs

Start from the README in the component you want to reproduce:

- [ART](ART/README.md)
- [traditionalML](traditionalML/README.md)
- [drain3_multiprocess](drain3_multiprocess/README.md)
