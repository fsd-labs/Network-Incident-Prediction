set -e

source /home/ubuntu/AnomalyDetection/venv310/bin/activate

CUR_DIR="/home/ubuntu/AnomalyDetection/dev_refactor/src/model_pipeline"
echo "Current directory: $CUR_DIR"

LOG_DIR="$CUR_DIR/logs"
mkdir -p "$LOG_DIR"

#TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TIMESTAMP=$(TZ="Asia/Ho_Chi_Minh" date +"%Y%m%d_%H%M")
echo "$TIMESTAMP"

MODEL_VER=
echo "Using MODEL_VER: $MODEL_VER"

# Script 1
echo "RUNNING MODEL TRAINING"
# python3 -u "$CUR_DIR/train.py" -mv $MODEL_VER -usamp -hpt -ntr 10 -ndv 2 > "$LOG_DIR/train_MV${MODEL_VER}_${TIMESTAMP}.log" 2>&1
python3 -u "$CUR_DIR/train.py" -mv $MODEL_VER > "$LOG_DIR/train_MV${MODEL_VER}_${TIMESTAMP}.log" 2>&1
echo "==> Model training completed. Logs saved in $LOG_DIR/train_MV${MODEL_VER}_${TIMESTAMP}.log"

# Script 2
echo "RUNNING MODEL EVALUATION"
python3 -u "$CUR_DIR/evaluate.py" -mv $MODEL_VER -hpt > "$LOG_DIR/evaluate_MV${MODEL_VER}_${TIMESTAMP}.log" 2>&1
echo "==> Model evaluation completed. Logs saved in $LOG_DIR/evaluate_MV${MODEL_VER}_${TIMESTAMP}.log"

