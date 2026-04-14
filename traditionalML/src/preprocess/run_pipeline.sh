set -e
cd /home/ubuntu/AnomalyDetection/dev_refactor/src/preprocess

CUR_DIR=$(pwd)
echo "Current directory: $CUR_DIR"

mkdir -p "$CUR_DIR/logs/new_lab"

echo "Running the preprocessing pipeline..."

source /home/ubuntu/AnomalyDetection/venv310/bin/activate

unset SPARK_HOME

# RUN GENERATE LABEL
echo "Step 1: Generating labels..."
python3 "$CUR_DIR/generate_labels.py" > "$CUR_DIR/logs/generate_labels.log" 2>&1

# RUN E2E PRE
echo "Step 2: Running end-to-end preprocessing..."
python -u "$CUR_DIR/e2e_preprocess.py" -sys -merge > "$CUR_DIR/logs/new_lab/e2e_preprocess.log" 2>&1

# # RUN TRAIN TEST SPLIT
echo "Step 3: Splitting data into train and test sets..."
python -u "$CUR_DIR/train_test_split.py" > "$CUR_DIR/logs/test_only_paper_without_snmp/train_test_split_21_nov_to_30_dec.log" 2>&1

echo "Preprocessing pipeline completed."