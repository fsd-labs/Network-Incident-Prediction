# evaluate_refactor.py
import os, sys, glob, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score
)
import joblib
from datetime import datetime
import pytz
vntz = pytz.timezone("Asia/Ho_Chi_Minh")

ROOT_DIR = r"/home/ubuntu/AnomalyDetection/dev_refactor"
sys.path.insert(0, ROOT_DIR)
from src.utils.helper_func import load_config, save_config, load_data, plot_roc_pr, plot_imp, plot_cm, return_percentile_gain_chart, to_builtin

LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")
HYPERPARAMS_DIR = os.path.join(ROOT_DIR, "artifacts", "hyperparams.yaml")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(LOG_DIR, f"eval_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}.log"))])
logger = logging.getLogger(__name__)

def main(args):
    # load config
    test_path = "/home/ubuntu/AnomalyDetection/dev_refactor/data/processed_extra_label_t11_from_20_t12_to_29/final_merged_data_latest"
    model_path = "/home/ubuntu/AnomalyDetection/dev_refactor/artifacts/model/model_v29/baseline/final_model.joblib"
    out_dir = "/home/ubuntu/AnomalyDetection/dev_refactor/data/inference/v29_t11_from_20_t12_to_29"

    key_for_params = "baseline"
    params_config = load_config(HYPERPARAMS_DIR)
    parent_run_id = params_config.get(f"model_v{args.model_ver}", {}).get(key_for_params, {}).get("parent_run_id", None)
    if not parent_run_id:
        logger.error("Parent run ID not found in hyperparams.yaml; MLflow linkage may be incomplete")
        raise
    logger.info(f"Parent MLflow run ID: {parent_run_id}")
    
    with mlflow.start_run(run_id=parent_run_id) as run:
        current_time = datetime.now(vntz).strftime("%Y%m%d_%H%M")
        mlflow.log_param(f"eval_timestamp_{current_time}", datetime.now(vntz).strftime("%Y-%m-%d %H:%M"))
        mlflow.log_param(f"test_data_path_{current_time}", test_path)
        mlflow.log_param(f"model_path_{current_time}", model_path)
        logger.info(f"Loading model from: {model_path}")
        pipe = joblib.load(model_path)

        logger.info(f"Loading test data from: {test_path}")
        test_df = load_data(test_path)

        X_te = test_df.drop(columns=[args.label_col])
        y_te = test_df[args.label_col].astype(int).values

        logger.info("Scoring test set...")
        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Predictions dump
        pred_df = pd.DataFrame({
            "ip_address": test_df["ip_address"].values,
            "window_5min_end": test_df["window_5min_end"].values,
            "y_true": y_te,
            # "y_pred": y_pred,
            "y_proba": y_prob
        })
        pred_df.to_parquet(os.path.join(out_dir, "predictions_v29.parquet"))
        logger.info(f"Saved predictions to {out_dir}/predictions_v29.parquet")

    logger.info(f"Evaluation artifacts saved under: {out_dir}")
    logger.info("DONE")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the label column in test data")
    parser.add_argument("-mv", "--model_ver", type=int, default=29, help="Model version number")
    parser.add_argument("-hpt", "--hypertune", action="store_true", help="Whether the model was hypertuned")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)