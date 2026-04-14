set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
CURRENT_DIR="$SCRIPT_DIR"
MODEL_NAME="deviceincidents_v2.7.8"
LOGDIR="$CURRENT_DIR/logs/$MODEL_NAME"


mkdir -p "$LOGDIR"

nohup python -u main_exec.py > "$LOGDIR/single_run_$TIMESTAMP.log" 2>&1 &
sleep 2
tail -f "$LOGDIR/single_run_$TIMESTAMP.log"
