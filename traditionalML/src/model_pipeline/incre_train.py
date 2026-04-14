import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

import os, sys, warnings, logging, yaml, re, glob, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score
)
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy import sparse
from sklearn.impute import SimpleImputer
import mlflow
import json
import hashlib
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
import joblib
import argparse
from datetime import datetime
from sklearn.base import clone
import pytz
vntz = pytz.timezone("Asia/Ho_Chi_Minh")
from scipy.sparse import issparse, csr_matrix


ROOT_DIR = r"/home/ubuntu/AnomalyDetection/dev_refactor"
sys.path.insert(0, ROOT_DIR)
from src.utils.helper_func import load_config, save_config, plot_imp, load_data, load_hyperparameters, to_builtin

LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

IP_DIR = os.path.join(ROOT_DIR, "data", "split_data", "train_ip")

MODEL_DIR = os.path.join(ROOT_DIR, "artifacts", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
HYPERPARAMS_DIR = os.path.join(ROOT_DIR, "artifacts", "hyperparams.yaml")

REPORT_DIR = os.path.join(ROOT_DIR, "artifacts", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(LOG_DIR, f"_incre_train_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}.log"))])
logger = logging.getLogger(__name__)

run_id = None
parent_run_id = None

def _to_matrix(X):
    if issparse(X) and not isinstance(X, csr_matrix):
        return X.tocsr()
    return X


def train_with_incremental_learning(df_train, pipe, label_col, df_valid, add_rounds = 100, nested=False):
    try:
        old_clf = pipe.named_steps["clf"]
        if not hasattr(old_clf, "booster_"):
            raise ValueError("baseline model not fitted")
        # train
        yn = df_train[label_col].astype(int).values
        Xn = df_train.drop(columns=[label_col])
        Xn = pipe.named_steps["pre"].transform(Xn)
        Xn = pipe.named_steps["select"].transform(Xn)
        Xn = _to_matrix(Xn)

        # valid
        Xv = df_valid.drop(columns=[label_col])
        yv = df_valid[label_col].astype(int).values
        Xv = pipe.named_steps["pre"].transform(Xv)
        Xv = pipe.named_steps["select"].transform(Xv)
        Xv = _to_matrix(Xv)
        eval_set = [(Xv, yv)]

        mlflow.end_run()
        with mlflow.start_run(run_name="incremental_training", nested=nested) as run:
            mlflow.set_tag("model_type", "incremental")
            global run_id
            run_id = run.info.run_id
            logger.info(f"MLflow run_id: {run_id}")

            prev_n = int(old_clf.get_params().get("n_estimators"))
            logger.info(f"Previous n_estimators: {prev_n}")
            logger.info(f"Adding {add_rounds} more estimators.")

            new_clf = clone(old_clf).set_params(n_estimators=add_rounds)
            logger.info(f"Old model: {old_clf.booster_.num_trees()} trees before incremental training.")

            new_clf.fit(Xn, yn, init_model=old_clf, eval_set=eval_set, eval_metric=["auc", "binary_logloss"],
                        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
                          )
            logger.info(f"New model: {new_clf.booster_.num_trees()} trees after incremental training.")
            pipe.set_params(clf=new_clf)
            new_n = int(new_clf.get_params().get("n_estimators"))
            logger.info(f"New n_estimators in clf: {new_n}")
            logger.info(f"Check pipe with new clf: {pipe.named_steps['clf'].booster_.num_trees()} trees.")

            pipe.named_steps["clf"].set_params(n_estimators=prev_n + add_rounds)

            try:
                if len(np.unique(yv)) < 2:
                    logger.warning("Validation set has only one class present. ROC AUC cannot be computed.")
                    roc_auc_fulltrain = None
                else:
                    roc_auc_fulltrain = roc_auc_score(yv, pipe.named_steps['clf'].predict_proba(Xv)[:,1])
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC on validation set: {e}")
                roc_auc_fulltrain = None

            try:
                mlflow.log_params({"roc_auc_fulltrain": roc_auc_fulltrain})
                mlflow.log_params({k.replace("clf__",""): v for k, v in pipe.get_params().items() if k.startswith("clf__")})
                logger.info(f"roc_auc_fulltrain: {roc_auc_fulltrain}")
                mlflow.log_params(
                    {"tree_before_incremental": old_clf.booster_.num_trees(),
                     "tree_after_incremental": new_clf.booster_.num_trees()}
                )
            except Exception as e:
                logger.error(f"Error logging parameters to MLflow: {e}")
                pass
            return pipe, roc_auc_fulltrain
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        mlflow.end_run() 
        raise
        
def main(args):
    # Load config
    general_config = load_config()
    params_config = load_hyperparameters(HYPERPARAMS_DIR)

    # load fitted model
    BASELINE_MODEL_SAVEDIR = os.path.join(MODEL_DIR, f"model_v{args.prev_model_ver}", "baseline")
    pipe = joblib.load(os.path.join(BASELINE_MODEL_SAVEDIR, "final_model.joblib"))
    logger.info(f"Loaded baseline model from {BASELINE_MODEL_SAVEDIR}")

    # list ip train 
    # list_ip_train = pd.read_json(f"{IP_DIR}/train_{args.model_ver}.json").squeeze().tolist()
    # print(f"List {len(list_ip_train)} ip ", list_ip_train)

    # load train data
    incre_train_data_path = "/home/ubuntu/AnomalyDetection/dev_refactor/data/processed_extra_label_t12/final_merged_data_latest"
    if incre_train_data_path is None:
        logger.info(r"Use default train_data_path from general_config")
        incre_train_data_path = general_config.get("train_data_path")

    train_df = load_data(incre_train_data_path)
    # train_df = train_df[train_df["ip_address"].isin(list_ip_train)]

    before = train_df.memory_usage(deep=True).sum() / 1024**2
    print(f"Trước khi downcast: {before:.2f} MB")

    # down cast
    for col in train_df.select_dtypes(include=['number']).columns:
        if pd.api.types.is_integer_dtype(train_df[col]):
            train_df[col] = pd.to_numeric(train_df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(train_df[col]):
            train_df[col] = pd.to_numeric(train_df[col], downcast='float')

    after = train_df.memory_usage(deep=True).sum() / 1024**2
    print(f"Sau khi downcast: {after:.2f} MB")

    # create null cols if not exist in train_df
    miss_cat_cols = pipe.named_steps["pre"].transformers_[2][2]
    required_cols = pipe.named_steps["pre"].transformers_[1][2] + miss_cat_cols
    missing_cols = []
    for col in required_cols:
        if col not in train_df.columns:
            train_df[col] = np.nan
            logger.info(f"Added missing column '{col}' with NaN values to train_df")
            missing_cols.append(col)
            if col in miss_cat_cols:
                train_df[col] = train_df[col].astype("object")
    logger.info(f"Total missing columns added: {len(missing_cols)}")

    label_rate = train_df[args.label_col].mean()
    logger.info(f"Training data label rate: {label_rate:.6f} ({train_df[args.label_col].sum()} positive samples out of {len(train_df)} total samples)")

    # define training scenario
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"incremental_script_run_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}") as parent_run:
        global parent_run_id
        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent MLflow run_id: {parent_run_id}")
        mlflow.set_tag("run_type", "script_execution")

        params_to_log = vars(args)
        params_to_log.update({
            "positive_ratio": train_df[args.label_col].mean(),
            "timestamp": datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')
        })
        
        # Log all parameters
        mlflow.log_params(params_to_log)
        params_config.setdefault(f"model_v{args.model_ver}", {})

        split_time = pd.to_datetime("2025-12-16 00:00:00", format="%Y-%m-%d %H:%M:%S") #"2025-11-12 00:00:00"
        # split_time = pd.to_datetime(args.cut_off_time, format="%Y-%m-%d_%H:%M:%S", utc=True)
        logger.info(f"split_time {split_time}")
        if split_time == "":
            train_df_sorted = train_df.sort_values(by="window_5min_end")
            split_index = int(len(train_df_sorted) * 0.7)

            train_df_split = train_df_sorted.iloc[:split_index]
            valid_df = train_df_sorted.iloc[split_index:]
        else:
            train_df_split = train_df[train_df['window_5min_end'] < split_time]
            valid_df = train_df[train_df['window_5min_end'] >= split_time]

        model, roc_auc_fulltrain = train_with_incremental_learning(
            df_train=train_df_split,
            pipe=pipe,
            label_col=args.label_col,
            df_valid= valid_df,
            nested=True
        )
        clf_params = model.named_steps["clf"].get_params()
        params_config[f"model_v{args.model_ver}"]["baseline"] = {
            # "params": {k: v for k, v in model.get_params().items() if k.startswith("clf__")},
            "params": clf_params,
            "roc_auc_fulltrain": roc_auc_fulltrain,
            "parent_run_id": parent_run_id,
            "train_run_id": run_id,
            "timestamp": params_to_log["timestamp"]
        }
        
        save_config(to_builtin(params_config), HYPERPARAMS_DIR)
    
    flag_hpt = "baseline"
    FINAL_MODEL_SAVEDIR = os.path.join(MODEL_DIR, f"model_v{args.model_ver}", flag_hpt)
    os.makedirs(FINAL_MODEL_SAVEDIR, exist_ok=True)
    joblib.dump(model, os.path.join(FINAL_MODEL_SAVEDIR, "final_model.joblib"))
    logger.info(f"Final model saved to {FINAL_MODEL_SAVEDIR}")

    # feature importance
    full_features = model.named_steps['pre'].get_feature_names_out()
    mask_feats = model.named_steps['select'].get_support()
    selected_features = full_features[mask_feats]
    np.save(os.path.join(FINAL_MODEL_SAVEDIR, "selected_features.npy"), selected_features)
    logger.info(f"Selected features saved to {FINAL_MODEL_SAVEDIR}/selected_features.npy")

    general_config["latest_model_path"] = FINAL_MODEL_SAVEDIR
    save_config(to_builtin(general_config))
    logger.info("Training pipeline completed successfully.")

def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Model Training Pipeline")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name")
    parser.add_argument("-mv", "--model_ver", type=int, default=None, help="Model version number")
    parser.add_argument("-pmv", "--prev_model_ver", type=int, default=None, help="Previous fitted model version number to load baseline from")
    parser.add_argument("-en", "--experiment_name", type=str, default="anomaly_detection_experiment", help="MLflow experiment name")
    parser.add_argument("-cot", "--cut_off_time", type=str, default="", help="Train test split time")

    return parser.parse_args()


def get_lastest_model_version():
    import re
    pattern = re.compile(r"model_v(\d+)$")
    max_version_model = 0

    for name in os.listdir(MODEL_DIR):
        full_path = os.path.join(MODEL_DIR, name)
        if os.path.isdir(full_path):
            match = pattern.match(name)
            if match:
                ver = int(match.group(1))
                if ver > max_version_model:
                    max_version_model = ver
    return max_version_model
        

if __name__ == "__main__":
    args = parse_args()
    if args.model_ver is None:
        prev_model_version = get_lastest_model_version()

        if prev_model_version == 0:
            logger.info(f"NOT FOUND MODEL PREV")
            sys.exit(1)

        args.prev_model_ver = prev_model_version
        args.model_ver = 30

        logger.info(f"Prev model {prev_model_version}")
        logger.info(f"Current model {args.model_ver}")


    main(args)