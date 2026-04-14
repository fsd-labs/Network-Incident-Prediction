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
from sklearn.base import clone, BaseEstimator, TransformerMixin
import pytz

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None

vntz = pytz.timezone("Asia/Ho_Chi_Minh")

ROOT_DIR = r"/home/ubuntu/AnomalyDetection/dev_refactor"
sys.path.insert(0, ROOT_DIR)
from src.utils.helper_func import load_config, save_config, plot_imp, load_data, load_hyperparameters, to_builtin

LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

IP_DIR = os.path.join(ROOT_DIR, "data", "split_data", "train_ip")
os.makedirs(IP_DIR, exist_ok=True)


MODEL_DIR = os.path.join(ROOT_DIR, "artifacts", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
HYPERPARAMS_DIR = os.path.join(ROOT_DIR, "artifacts", "hyperparams.yaml")

REPORT_DIR = os.path.join(ROOT_DIR, "artifacts", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(LOG_DIR, f"train_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}.log"))])
logger = logging.getLogger(__name__)

run_id = None
parent_run_id = None

class LoggingStep(BaseEstimator, TransformerMixin):
    def __init__(self, name="Step"):
        self.name = name
    
    def fit(self, X, y=None):
        logger.info(f"[{self.name}] Starting FIT")
        return self
    
    def transform(self, X, y=None):
        logger.info(f"[{self.name}] Starting TRANSFORM - Shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


def _build_optuna_distributions(space_dict):
    dists = {}
    for k, cfg in space_dict.items():
        t = cfg["type"]
        low, high = cfg["low"], cfg["high"]
        log = cfg.get("log", False)
        if t == "int":
            dists[f"clf__{k}"] = IntDistribution(low=low, high=high, step=1, log=log)
        else:
            dists[f"clf__{k}"] = FloatDistribution(low=low, high=high, log=log)
    return dists

def build_smote_step(y, use_upsampling=False):
    if not use_upsampling:
        return None

    if SMOTE is None or ImbPipeline is None:
        raise ImportError("Flag -usamp/--upsampling requires `imbalanced-learn`. Please install it before running.")

    class_counts = pd.Series(y).value_counts()
    minority_count = int(class_counts.min()) if not class_counts.empty else 0
    if minority_count < 2:
        logger.warning("SMOTE is skipped because the minority class has fewer than 2 samples.")
        return None

    k_neighbors = min(5, minority_count - 1)
    logger.info(f"SMOTE upsampling enabled with k_neighbors={k_neighbors}")
    return ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors))

def build_pipeline(train_df, args, exclude_cols, message_col, numeric_cols, categorical_cols):
    if len(categorical_cols+ numeric_cols + message_col + exclude_cols) != train_df.shape[1]:
        raise ValueError("Some columns are not included in the pipeline, check your column names")
    else:
        logger.info(f"Pipeline will use {len(categorical_cols)} categorical, {len(numeric_cols)} numeric, and {len(message_col)} text columns")
        
    ## define scale_pos_weight for LightGBM
    raw_scale_pos_weight = (train_df[args.label_col] == 0).sum() / (train_df[args.label_col] == 1).sum()
    scale_pos_weight = 1.0 if args.upsampling else raw_scale_pos_weight
    logger.info(f"Computed scale_pos_weight: {scale_pos_weight}")
    if args.upsampling:
        logger.info(f"Original scale_pos_weight before SMOTE: {raw_scale_pos_weight}")

    preprocessor = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=args.token, token_pattern=r"[^ ]+", lowercase=True, dtype=np.float32), *message_col),
        ("num", SimpleImputer(strategy="constant", fill_value = -1, add_indicator=True), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop="first", dtype=np.int8), categorical_cols)],
        sparse_threshold=1.0)

    pipe_steps = [
        ("pre", preprocessor),
    ]

    smote_step = build_smote_step(train_df[args.label_col], args.upsampling)
    if smote_step is not None:
        pipe_steps.append(smote_step)

    pipe_steps.append(
        ("clf", LGBMClassifier(
            random_state=42,
            # scale_pos_weight=scale_pos_weight,
            # n_estimators=400,
            # learning_rate=0.05,
            # num_leaves=31,  
            # max_depth=-1,     
            # min_child_samples=100, 
            # subsample=0.8,
            # subsample_freq=1,
            # colsample_bytree=0.7,
            n_jobs=-1,
        ))
    )

    pipe_cls = ImbPipeline if args.upsampling else Pipeline
    pipe = pipe_cls(pipe_steps)
    return pipe, scale_pos_weight

def train_baseline(df_train, pipe, label_col, nested=False):
    try:
        mlflow.end_run()
        with mlflow.start_run(run_name="baseline_model", nested=nested) as run:
            mlflow.set_tag("model_type", "baseline")
            global run_id
            run_id = run.info.run_id
            logger.info(f"MLflow run_id: {run_id}")
            pipe.fit(df_train, df_train[label_col])
            mlflow.sklearn.log_model(pipe, f"model_v{args.model_ver}_baseline")
            logger.info("Model training completed and logged to MLflow.")

            logger.info("Computing roc_auc_fulltrain...")
            roc_auc_fulltrain = roc_auc_score(df_train[label_col].astype(int).values, pipe.predict_proba(df_train.drop(columns=[label_col]))[:,1])
            try:
                mlflow.log_params({"roc_auc_fulltrain": roc_auc_fulltrain})
                mlflow.log_params({k.replace("clf__",""): v for k, v in pipe.get_params().items() if k.startswith("clf__")})
                logger.info(f"roc_auc_fulltrain: {roc_auc_fulltrain}")
                logger.info("Training completed successfully.")
            except Exception as e:
                logger.error(f"Error logging parameters to MLflow: {e}")
                pass
            return pipe, roc_auc_fulltrain
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        mlflow.end_run() 
        raise

def train_hypertune(
    df_train_full, pipe, label_col, hypertune_space,
    n_days_valid=7, n_trials=30, run_name="optuna_fixed_valid", nested=False
):
    anchor_date = df_train_full["window_5min_end"].max() - pd.Timedelta(days=n_days_valid)
    train_mask = (df_train_full["window_5min_end"] <= anchor_date)

    X_full = df_train_full.drop(columns=[label_col])
    y_full = df_train_full[label_col].astype(int).values

    test_fold = np.where(train_mask.values, -1, 0)
    cv = PredefinedSplit(test_fold)

    mlflow.log_params({
        "n_days_valid": n_days_valid,
        "anchor_date": anchor_date.strftime('%Y-%m-%d %H:%M:%S'),
        "train_size": train_mask.sum(),
        "valid_size": (~train_mask).sum(),
        "n_trials": n_trials
    })
    dists = _build_optuna_distributions(hypertune_space)
    try:
        mlflow.end_run()
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            mlflow.set_tag("model_type", "hypertune")
            global run_id
            run_id = run.info.run_id
            logger.info(f"MLflow run_id: {run_id}")
            search = OptunaSearchCV(
                estimator=pipe,
                param_distributions=dists,
                cv=cv,
                scoring="roc_auc",
                n_trials=n_trials,
                n_jobs=-1,
                random_state=0,
                refit=False,
                verbose = 1
            )
            search.fit(X_full, y_full)
            logger.info("HYPERPARAMETER TUNING COMPLETED.")

            best_params = search.best_params_
            best_score = float(search.best_score_)
            logger.info(f"Best params: {best_params}")
            logger.info(f"Best CV score: {best_score}")

            try:
                mlflow.log_params({k.replace("clf__",""): v for k, v in best_params.items()})
                mlflow.log_metric("valid_roc_auc", best_score)
            except Exception as e:
                logger.error(f"Error logging parameters to MLflow: {e}")
                pass

            # refit full train with best params
            best_pipe = clone(pipe).set_params(**best_params)
            best_pipe.fit(X_full, y_full)
            logger.info("Refit best model on full training data.")
            mlflow.sklearn.log_model(best_pipe, f"model_v{args.model_ver}_hpt_best")
            logger.info("Best model logged to MLflow.")

            # reeval on full train
            roc_auc_fulltrain = roc_auc_score(y_full, best_pipe.predict_proba(X_full)[:,1])

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        mlflow.end_run() 
        raise
    return best_pipe, best_params, best_score, roc_auc_fulltrain

def main(args):
    # Load config
    general_config = load_config()
    params_config = load_hyperparameters(HYPERPARAMS_DIR)
    hypertune_space = params_config.get("hyperparameter_space", {})

    # load train data
    train_df = load_data(general_config.get("train_data_path"))

    # # pd.set_option('display.max_columns', None)
    # sorted_columns = sorted(train_df.columns.tolist())
    # print(f"Total column before train {len(sorted_columns)}")
    # i = 0 
    # for col in sorted_columns:
    #     print(f"{i} , {col}")
    #     i += 1

    # filter  ip by date include label 1 
    # ips_with_label_1 = train_df.groupby('ip_address')['label'].max()
    # ips_to_keep = ips_with_label_1[ips_with_label_1 == 1].index
    # n_ips_before = train_df['ip_address'].nunique()
    # train_df = train_df[train_df['ip_address'].isin(ips_to_keep)].copy()
    # # Thống kê sau khi lọc
    # n_ips_after = train_df['ip_address'].nunique()
    # n_label_1 = train_df['label'].sum()
    # n_total_records = len(train_df)
    # pos_ratio = (n_label_1 / n_total_records) * 100 if n_total_records > 0 else 0

    # print("-" * 30)
    # print(f"LỌC DỮ LIỆU THEO IP (IP có ít nhất một nhãn 1):")
    # print(f"- Số lượng IP ban đầu: {n_ips_before:,}")
    # print(f"- Số lượng IP giữ lại: {n_ips_after:,}")
    # print(f"- Số lượng IP bị loại bỏ: {n_ips_before - n_ips_after:,}")
    # print(f"- Tổng số bản ghi sau lọc: {n_total_records:,}")
    # print(f"- Tỷ lệ nhãn 1 hiện tại: {pos_ratio:.2f}%")
    # print("-" * 30)
    # return

    # n_ips_before = train_df['ip_address'].nunique()
    # n_records_before = len(train_df)

    # train_df['date_only'] = pd.to_datetime(train_df['window_5min_end']).dt.date
    # mask = train_df.groupby(['ip_address', 'date_only'])['label'].transform('max') == 1
    # train_df_filtered = train_df[mask].copy()
    # train_df_filtered = train_df_filtered.drop(columns=['date_only'])
    # print(f"Trước khi lọc: {len(train_df)} dòng")
    # print(f"Sau khi lọc: {len(train_df_filtered)} dòng")

    # n_ips_after = train_df_filtered['ip_address'].nunique()
    # n_records_after = len(train_df_filtered)
    # n_label_1 = train_df_filtered['label'].sum()

    # total_records = len(train_df_filtered)
    # positive_records = train_df_filtered['label'].sum() # Vì label là 0 hoặc 1, sum chính là số lượng số 1
    # pos_ratio = (positive_records / total_records) * 100
    # print(f"--- Thống kê tập Train sau khi lọc ---")
    # print(f"Tổng số bản ghi: {total_records}")
    # print(f"Số bản ghi nhãn 1: {positive_records}")
    # print(f"Tỷ lệ nhãn 1: {pos_ratio:.2f}%")
    # print("\nTHỐNG KÊ SAU KHI LỌC (Chỉ giữ IP/Ngày có nhãn 1):")
    # print(f"- Số lượng IP còn lại: {n_ips_after:,} (Giảm {((n_ips_before - n_ips_after)/n_ips_before)*100:.2f}%)")
    # print(f"- Số bản ghi còn lại: {n_records_after:,}")
    # print(f"- Số bản ghi nhãn 1: {n_label_1:,}")
    # train_df = train_df_filtered
    # return

    before = train_df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Trước khi downcast: {before:.2f} MB")

    # down cast
    for col in train_df.select_dtypes(include=['number']).columns:
    # Nếu là số nguyên (int)
        if pd.api.types.is_integer_dtype(train_df[col]):
            train_df[col] = pd.to_numeric(train_df[col], downcast='integer')
        # Nếu là số thực (float)
        elif pd.api.types.is_float_dtype(train_df[col]):
            train_df[col] = pd.to_numeric(train_df[col], downcast='float')

    
    # Downcast như trên
    after = train_df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Sau khi downcast: {after:.2f} MB")

    exclude_cols = ["ip_address", "window_5min_end", "label"]
    message_col = ["combined_message"]
    numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    categorical_cols = [col for col in train_df.columns if col not in numeric_cols + exclude_cols + message_col]

    # build pipeline
    logger.info(f"Start build pipeline")
    pipe, scale_pos_weight = build_pipeline(train_df, args, exclude_cols, message_col, numeric_cols, categorical_cols)
    logger.info(f"Done")
    # define training scenario
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"script_run_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}") as parent_run:
        global parent_run_id
        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent MLflow run_id: {parent_run_id}")
        mlflow.set_tag("run_type", "script_execution")

        params_to_log = vars(args)
        params_to_log.update({
            "scale_pos_weight": scale_pos_weight,
            "num_categorical_cols": len(categorical_cols),
            "num_numeric_cols": len(numeric_cols),
            "num_text_cols": len(message_col),
            "positive_ratio": train_df[args.label_col].mean(),
            "timestamp": datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')
        })
        
        # Log all parameters
        mlflow.log_params(params_to_log)
        params_config.setdefault(f"model_v{args.model_ver}", {})
        if args.hypertune:
            run_name = f"tune_{args.model_ver}_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}"
            model, best_params, best_score, roc_auc_fulltrain = train_hypertune(
                df_train_full=train_df,
                pipe=pipe,
                label_col=args.label_col,
                hypertune_space=hypertune_space,
                n_days_valid=args.n_days_valid,
                n_trials=args.n_trials,
                run_name=run_name,
                nested=True
            )
            params_config[f"model_v{args.model_ver}"]["hpt"] = {
                "best_params": {k.replace("clf__",""): v for k, v in best_params.items()},
                "roc_auc_fulltrain": roc_auc_fulltrain,
                "best_valid_roc_auc": best_score,
                "n_trials": args.n_trials,
                "parent_run_id": parent_run_id,
                "train_run_id": run_id,
                "timestamp": params_to_log["timestamp"]
            }

            save_config(to_builtin(params_config), HYPERPARAMS_DIR)
            
        else:
            logger.info(f"Start train")
            model, roc_auc_fulltrain = train_baseline(
                df_train=train_df,
                pipe=pipe,
                label_col=args.label_col,
                nested=True
            )
            logger.info(f"End train")
            params_config[f"model_v{args.model_ver}"]["baseline"] = {
                "params": {k: v for k, v in model.get_params().items() if k.startswith("clf__")},
                "roc_auc_fulltrain": roc_auc_fulltrain,
                "parent_run_id": parent_run_id,
                "train_run_id": run_id,
                "timestamp": params_to_log["timestamp"]
            }
            save_config(to_builtin(params_config), HYPERPARAMS_DIR)
    
    flag_hpt = "with_hypertune" if args.hypertune else "baseline"
    FINAL_MODEL_SAVEDIR = os.path.join(MODEL_DIR, f"model_v{args.model_ver}", flag_hpt)
    os.makedirs(FINAL_MODEL_SAVEDIR, exist_ok=True)
    joblib.dump(model, os.path.join(FINAL_MODEL_SAVEDIR, "final_model.joblib"))
    logger.info(f"Final model saved to {FINAL_MODEL_SAVEDIR}")

    # feature importance
    try: 
        full_features = model.named_steps['pre'].get_feature_names_out()
        # mask_feats = model.named_steps['select'].get_support()
        # selected_features = full_features[mask_feats]
        np.save(os.path.join(FINAL_MODEL_SAVEDIR, "selected_features.npy"), full_features)
        logger.info(f"Selected features saved to {FINAL_MODEL_SAVEDIR}/selected_features.npy")

        # Get feature importances from LightGBM classifier
        feature_importances = model.named_steps['clf'].feature_importances_
        np.save(os.path.join(FINAL_MODEL_SAVEDIR, "feature_importances.npy"), feature_importances)
        logger.info(f"Feature importances saved to {FINAL_MODEL_SAVEDIR}/feature_importances.npy")

        general_config["latest_model_path"] = FINAL_MODEL_SAVEDIR
        save_config(to_builtin(general_config))
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error during feature importance extraction: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("-tok", "--token", type=int, default=1_048_576, help="Token size for TF-IDF Feature")
    parser.add_argument("-sel_k", "--select_k", type=int, default=16_384, help="Number of top features to select")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name")
    parser.add_argument("-mv", "--model_ver", type=int, default=1, help="Model version number")
    parser.add_argument("-hpt", "--hypertune", action="store_true", help="Whether to perform hyperparameter tuning")
    parser.add_argument("-usamp", "--upsampling", action="store_true", help="Enable SMOTE upsampling before model training")
    parser.add_argument("-ntr", "--n_trials", type=int, default=5, help="Number of trials for hyperparameter tuning")
    parser.add_argument("-ndv", "--n_days_valid", type=int, default=7, help="Number of days for validation")
    parser.add_argument("-en", "--experiment_name", type=str, default="anomaly_detection_experiment", help="MLflow experiment name")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
