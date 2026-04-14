# run snmp collect
cd /home/ast/AnomalyDetection/dev_refactor/src/collect_data/

CUR_DIR=$(pwd)
echo "Current dir: $CUR_DIR"

unset https_proxy && unset http_proxy

python3 "$CUR_DIR/collect_snmp_s3_new.py" > "$CUR_DIR/collect_snmp_s3_remaining.log" 2>&1

python3 "$CUR_DIR/collect_sys.py" > "$CUR_DIR/collect_sys_remaining.log" 2>&1
echo "Finished collecting data"