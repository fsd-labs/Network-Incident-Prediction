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
MODEL_DIR = os.path.join(ROOT_DIR, "artifacts", "model")
REPORT_DIR = os.path.join(ROOT_DIR, "artifacts", "reports")
HYPERPARAMS_DIR = os.path.join(ROOT_DIR, "artifacts", "hyperparams.yaml")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(LOG_DIR, f"eval_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}.log"))])
logger = logging.getLogger(__name__)

def main(args):
    # load config
    cfg = load_config()
    test_path = cfg.get("test_data_path")
    
    latest_model_dir = cfg.get("latest_model_path")
    if not latest_model_dir or not os.path.isdir(latest_model_dir):
        raise RuntimeError("Config 'latest_model_path' is missing or not a directory")
    # logger.info(f"latest_model_path: {latest_model_dir}")
    # model_path = os.path.join(latest_model_dir, "final_model.joblib")
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(latest_model_dir)), f"model_v{args.model_ver}", f"baseline" if not args.hypertune else "with_hypertune")
    model_path = os.path.join(model_dir, "final_model.joblib")
    logger.info(f"Constructed model path: {model_path}")

    # Map evaluation outputs to the same subfolder pattern as training
    rel_part = os.path.relpath(model_dir, MODEL_DIR)
    out_dir = os.path.join(REPORT_DIR, rel_part)
    os.makedirs(out_dir, exist_ok=True)

    key_for_params = "hpt" if args.hypertune else "baseline"
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

        # filter  ip by date include label 1 
        # test_df['date_only'] = pd.to_datetime(test_df['window_5min_end']).dt.date
        # mask = test_df.groupby(['ip_address', 'date_only'])['label'].transform('max') == 1
        # test_df_filtered = test_df[mask].copy()
        # test_df_filtered = test_df_filtered.drop(columns=['date_only'])
        # print(f"Trước khi lọc: {len(test_df)} dòng")
        # print(f"Sau khi lọc: {len(test_df_filtered)} dòng")

        # total_records = len(test_df_filtered)
        # positive_records = test_df_filtered['label'].sum() # Vì label là 0 hoặc 1, sum chính là số lượng số 1
        # pos_ratio = (positive_records / total_records) * 100
        # print(f"--- Thống kê tập Train sau khi lọc ---")
        # print(f"Tổng số bản ghi: {total_records}")
        # print(f"Số bản ghi nhãn 1: {positive_records}")
        # print(f"Tỷ lệ nhãn 1: {pos_ratio:.2f}%")
        # test_df = test_df_filtered

        # import json
        # with open("/home/ubuntu/AnomalyDetection/dev_refactor/data/label/agg_ips_daily.json", "r") as f:
        #     agg_ips_daily = json.load(f)
        # test_df["date_utc7"] = test_df["window_5min_end"].dt.tz_localize("UTC").dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.astype(str)
        # # map dateutc7 to agg_ips_daily to get list of ips for each date, then check if ip_address in that list to create label
        # test_df["agg_ips"] = test_df["date_utc7"].map(agg_ips_daily)
        # test_df["agg_ips"] = test_df["agg_ips"].apply(lambda x: x if isinstance(x, (list, set, tuple)) else [])
        # test_df["flag"] = test_df.apply(lambda r: int(r["ip_address"] in r["agg_ips"]), axis=1)
        # test_df = test_df[test_df["flag"] == 1].drop(columns=["date_utc7", "agg_ips", "flag"])
        # logger.info(f"After filtering with agg_ips_daily, test_df shape: {test_df.shape}")

        pre = pipe.named_steps["pre"]
        print("remainder:", getattr(pre, "remainder", None))
        print("transformers:", [(name, trans, cols) for name, trans, cols in pre.transformers])

        # create null cols if not exist in train_df
        miss_cat_cols = pipe.named_steps["pre"].transformers_[2][2]
        required_cols = pipe.named_steps["pre"].transformers_[1][2] + miss_cat_cols
        missing_cols = []
        for col in required_cols:
            if col not in test_df.columns:
                test_df[col] = np.nan
                logger.info(f"Added missing column '{col}' with NaN values to test_df")
                missing_cols.append(col)
                
        # Force categorical columns to object dtype to match OneHotEncoder expectations.
        # This prevents sklearn from taking numeric unknown-check paths on mixed-type categories.
        for col in miss_cat_cols:
            if col in test_df.columns:
                test_df[col] = test_df[col].astype("object")
        logger.info(f"Total missing columns added: {len(missing_cols)}")

        if args.label_col not in test_df.columns:
            raise ValueError(f"Label column '{args.label_col}' not found in test data")

        X_te = test_df.drop(columns=[args.label_col])
        y_te = test_df[args.label_col].astype(int).values

        logger.info("Scoring test set...")
        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics + reports
        te_auc = roc_auc_score(y_te, y_prob)
        rep = classification_report(y_te, y_pred, digits=4)
        logger.info(f"Test ROC AUC: {te_auc:.6f}")
        logger.info("Classification report (test):\n" + rep)

        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(f"ROC_AUC: {te_auc:.6f}\n\n")
            f.write(rep)

        # Visuals
        plot_roc_pr(y_te, y_prob, os.path.join(out_dir, "roc_pr_test.png"))
        plot_cm(y_te, y_pred, os.path.join(out_dir, "confusion_matrix.png"))

        # Feature importance (requires selected_features saved at training time)
        sel_path = os.path.join(latest_model_dir, "selected_features.npy")
        if os.path.exists(sel_path):
            selected_features = np.load(sel_path, allow_pickle=True)
            try:
                plot_imp(pipe.named_steps["clf"], selected_features, os.path.join(out_dir, "feature_importance.png"))
            except Exception as e:
                logger.warning(f"plot_imp failed: {e}")
        else:
            logger.warning("selected_features.npy not found; skipping feature importance plot")

        # Predictions dump with IP and timestamp for daily analysis
        pred_df = pd.DataFrame({
            "ip_address": test_df["ip_address"].values if "ip_address" in test_df.columns else None,
            "window_5min_end": test_df["window_5min_end"].values if "window_5min_end" in test_df.columns else None,
            "y_true": y_te,
            "y_pred": y_pred,
            "y_proba": y_prob
        })
        # Add date column for daily aggregation
        if "window_5min_end" in test_df.columns:
            pred_df["date"] = pd.to_datetime(pred_df["window_5min_end"]).dt.date
        pred_df.to_parquet(os.path.join(out_dir, "predictions.parquet"))
        logger.info(f"Saved predictions to {out_dir}/predictions.parquet with columns: {pred_df.columns.tolist()}")

        # Percentile gain chart
        return_percentile_gain_chart(
            pred_df, true_col="y_true", y_pred="y_pred", y_proba="y_proba",
            number_of_thresholds=140,
            save_fig =True,
            output_dir=out_dir,
            plot_name="Precision & Coverage by Decile (Test)"
        )

        # save artifacts to mlflow
        mlflow.log_artifacts(out_dir, artifact_path="evaluation_artifacts")
        logger.info("Logged evaluation artifacts to MLflow")
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