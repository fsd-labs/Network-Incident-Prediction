import os

import jsonpickle  # type: ignore[import]
import pandas as pd

from drain3 import LengthShardedTemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)


def get_cluster_sizes(template_miner):
    if not getattr(template_miner, "_use_processes", False):
        drains = getattr(template_miner, "_drains_by_token_count", {}).values()
        return [cluster.size for drain in drains for cluster in drain.clusters]

    drains_state, _, _ = template_miner._collect_snapshot_data()
    cluster_sizes = []
    for drain_state in drains_state.values():
        drain = jsonpickle.loads(drain_state, keys=True)
        cluster_sizes.extend(cluster.size for cluster in drain.clusters)
    return cluster_sizes


def main() -> None:
    config = TemplateMinerConfig()
    config.load(f"{current_folder}/drain3.ini")
    persistence = FilePersistence("drain3_state_m_11_12.bin")
    config.profiling_enabled = True

    template_miner = LengthShardedTemplateMiner(persistence_handler=persistence, config=config)

    for month in [11, 12]:
        for day in range(1, 32, 1):
            if month == 11 and day < 21:
                print(f"Skip {month} - {day}")
                continue
            for hour in range(0, 24, 1):
                parquet_path = (
                    f"/home/ast/raw/noc-syslog-2/2025-{month:02d}-{day:02d}/data_{hour:02d}-00-00.parquet"
                )
                if not os.path.exists(parquet_path):
                    print(f"Not exists {parquet_path}")
                    continue

                df = pd.read_parquet(parquet_path, engine="pyarrow")
                df = df[["syslog_timestamp", "message", "@timestamp", "host", "logsource"]]
                total_rows = len(df)
                print(f"Day {day} Hour {hour}: {total_rows}")
                print(df.head(5))

                logs = df["message"].tolist()
                batch_results = template_miner.add_log_messages(logs)

                records = df.to_dict(orient="records")
                for record, rs in zip(records, batch_results):
                    record["cluster_id"] = rs["cluster_id"]
                    record["template_mined"] = rs["template_mined"]

                # Convert list to DataFrame and write parquet
                batch_result_df = pd.DataFrame(records)
                output = f"{current_folder}/temp/month={month:02d}/batch_{day:02d}_{hour:02d}.parquet"
                os.makedirs(os.path.dirname(output), exist_ok=True)
                batch_result_df.to_parquet(output, engine="pyarrow", index=False)
                print(f"Write done {output}")

                cluster_sizes = get_cluster_sizes(template_miner)
                if cluster_sizes:
                    min_size = min(cluster_sizes)
                    max_size = max(cluster_sizes)
                    avg_size = sum(cluster_sizes) / len(cluster_sizes)
                    total_clusters = len(cluster_sizes)
                    print(f"Tong so cluster       : {total_clusters}")
                    print(f"Kich thuoc nho nhat   : {min_size}")
                    print(f"Kich thuoc lon nhat   : {max_size}")
                    print(f"Kich thuoc trung binh : {avg_size:.2f}")
                else:
                    print("Khong co cluster nao trong danh sach.")

    template_miner.close()


if __name__ == "__main__":
    main()
