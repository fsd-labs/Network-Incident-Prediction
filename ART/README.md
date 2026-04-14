# ART: Abstract Reasoning LogTransformer

This folder contains the main deep-learning model used in the paper:

**Abstract Reasoning-Driven Prediction and Root Cause Identification of Device Incidents in Core Internet Infrastructure**

ART is a causal, concept-token transformer for proactive device incident prediction and root cause analysis. Raw network logs are first mapped to Drain3 log templates. Each template is treated as a concept token, so the model reasons over stable log concepts rather than high-entropy raw text tokens.

## Role in the Paper

ART is the proposed model in the deep-learning branch of the study. It extends the LogBERT-style template-token modeling idea with three changes:

- Causal next-concept prediction instead of bidirectional masked log-key prediction.
- Supervised incident classification for short-horizon incident forecasting.
- A learnable linear aggregation head whose token weights are used as an association proxy for root-cause ranking.

The paper reports that ART reaches competitive incident prediction performance compared with the LightGBM traditional ML approach while adding concept-level evidence for operational root-cause analysis.

## Pipeline

1. Parse raw logs into log templates with Drain3.
2. Build concept-token and device vocabularies.
3. Convert per-device log streams into time-window sequences.
4. Train ART with a multitask objective:
   `L_total = L_incident + 0.1 * L_causal + 1e-4 * L_l1`
5. Run prediction and export ROC-AUC, gain-chart, loss, and root-cause analysis artifacts.

## Repository Layout

```text
ART/
├── main_exec.py                     # Main execution entrypoint
├── experimental_space/              # Default run configuration
├── executors/                       # Vocabulary, training, prediction, and visualization executors
├── model_architecture/              # Transformer, HF wrapper, preprocessing, vocab, and utilities
├── drain3/                          # Drain3 configs and saved parser state used by this model snapshot
├── artifacts/                       # Sample vocabularies and generated evaluation figures
├── temp_setup/                      # Environment setup files
├── mapping_cls_pd.py                # Log/template mapping and sequence preparation utilities
├── visualize_loss.py                # Trainer loss visualization helper
└── run.sh                           # Convenience run script
```

## Key Files

- `model_architecture/hf/modeling.py`: Hugging Face-compatible ART sequence classifier.
- `model_architecture/modelling/logbert_model.py`: core LogBERT/ART model components.
- `model_architecture/modelling/interpolate.py`: post-encoder sequence normalization used before aggregation.
- `model_architecture/preprocessing/sequence_mp.py`: multiprocessing per-device sequence generation.
- `executors/hf_train.py`: Hugging Face Trainer-based training loop.
- `executors/hf_predict.py`: inference and evaluation artifact generation.
- `experimental_space/args_builder.py`: default paths and hyperparameters.

## Artifacts

The checked-in `artifacts/` folder contains small model-side assets and figures useful for reproducing the code flow:

- `vocab.pkl`: concept-token vocabulary.
- `dev_vocab.pkl`: device vocabulary.
- `roc_auc/`: ROC-AUC visualization.
- `gain_chart/`: precision and coverage chart.
- `visualize_loss/`: loss component plots.

Training checkpoints and large datasets should be provided externally through the dataset release.

## Running

Install dependencies from `temp_setup/requirements.txt`, then configure data and artifact paths in `experimental_space/args_builder.py`.

```bash
pip install -r temp_setup/requirements.txt
python main_exec.py --mode vocab
python main_exec.py --mode train
python main_exec.py --mode predict
```

The research snapshot may set the runtime mode directly inside `main_exec.py` for a specific experiment. Check `parse_and_run()` before launching a new run.

## Expected Inputs

ART expects per-device log-template sequences with labels. The main fields used across the preprocessing path are:

- device identifier or IP address
- timestamp or 5-minute window end
- log template ID / concept token
- binary incident label for the future prediction horizon

Dataset and checkpoint links are intentionally externalized from this source release.
