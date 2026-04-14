# Traditional ML Approach

This folder contains the LightGBM-based traditional machine learning approach used in the paper:

**Abstract Reasoning-Driven Prediction and Root Cause Identification of Device Incidents in Core Internet Infrastructure**

This approach represents a strong feature-engineered solution for device incident prediction. It uses structured SNMP-derived features, normalized syslog text features, temporal log-volume features, and a LightGBM classifier.

## Role in the Paper

The paper compares ART with a strong non-sequential tabular approach based on LightGBM. This folder implements that traditional ML pipeline:

- Parse and normalize syslog fields.
- Aggregate events into 5-minute device windows.
- Build TF-IDF features from normalized log messages.
- Add log-volume, priority-score, and SNMP rolling statistics.
- Train and evaluate a LightGBM classifier under severe class imbalance.

The reported LightGBM approach reaches competitive AUC-ROC and F1 performance. Its bag-of-tokens representation is effective for prediction, while ART is designed to add concept-level root-cause reasoning on top of competitive forecasting performance.

## Pipeline

1. Collect raw SNMP and syslog inputs with scripts under `src/collect_data/`.
2. Run preprocessing under `src/preprocess/`:
   - parse syslog message structure
   - normalize free-text content
   - aggregate logs into 5-minute windows
   - create log-count ratio and priority-score ratio features
   - compute SNMP rolling statistics
   - join future incident labels
   - export merged train/test-ready Parquet data
3. Split data chronologically with `train_test_split.py`.
4. Train LightGBM with `src/model_pipeline/train.py`.
5. Evaluate with `src/model_pipeline/evaluate.py`.

## Repository Layout

```text
traditionalML/
├── src/
│   ├── collect_data/          # Raw data collection helpers
│   ├── preprocess/            # Spark preprocessing and chronological splitting
│   ├── model_pipeline/        # LightGBM train/evaluate/inference scripts
│   ├── config/config.yaml     # Data and model path configuration
│   └── utils/                 # Spark, plotting, config, and evaluation helpers
└── artifacts/
    └── hyperparams.yaml       # LightGBM search space and experiment records
```

## Model Features

- Text: TF-IDF unigram/bigram features from `combined_message`.
- Numeric: rolling SNMP metrics, log-count ratios, and priority-score ratios.
- Categorical: remaining non-numeric context fields encoded with one-hot encoding.
- Classifier: `lightgbm.LGBMClassifier`.
- Optional tuning: Optuna search over LightGBM hyperparameters.
- Experiment tracking: MLflow logging for model and evaluation artifacts.

## Running

Update `src/config/config.yaml` with local or released dataset paths first.

```bash
cd traditionalML
bash src/preprocess/run_pipeline.sh
bash src/model_pipeline/_train_pipeline.sh
```

The checked-in scripts are research snapshots and may contain environment-specific paths. Before running on a new machine, review `ROOT_DIR`, virtual environment paths, and the date split constants inside `src/preprocess/` and `src/model_pipeline/`.

## Outputs

Expected generated outputs include:

- preprocessed SNMP/syslog Parquet data
- merged labeled window-level Parquet data
- chronological train/test Parquet splits
- LightGBM `final_model.joblib`
- feature importance arrays
- ROC/PR, confusion matrix, gain chart, and prediction reports
