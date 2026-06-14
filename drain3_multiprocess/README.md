# Drain3 Multiprocess Template Mining

This folder contains the modified Drain3 implementation used to speed up template mining for the paper:

**Abstract Reasoning-Driven Prediction and Root Cause Identification of Device Incidents in Core Internet Infrastructure**

The ART pipeline depends on high-quality log templates. On carrier-scale syslog streams, default single-process Drain3 parsing can become the dominant preprocessing cost. This module adds length-sharded, multiprocessing template mining so template induction can scale to the production log volume used in the study.

## Role in the Paper

The paper uses Drain3 to convert raw Juniper device logs into stable log templates, then treats those templates as concept tokens for ART. This folder implements the throughput-oriented parser variant described in the ART implementation section:

- Normalize and mask high-entropy fields before template induction.
- Route messages by token count after normalization.
- Maintain independent Drain shards for different token lengths.
- Process shards in worker processes.
- Persist a combined sharded snapshot for reproducible parsing.

In the reported workflow, this design reduced end-to-end template mining on 30 days of logs from an estimated 20-day run to under 20 hours.

## Main Changes from Upstream Drain3

- `LengthShardedTemplateMiner` shards Drain state by token count.
- `shard_workers` enables multiprocessing worker processes.
- `max_clusters_per_token_count` applies an LRU limit per token-length shard.
- `add_log_message_tokens()` allows pre-tokenized input to avoid repeated tokenization.
- Snapshot persistence stores all shard states and the global cluster-id counter.
- `TemplateMiner` guards against loading sharded snapshots as standard Drain snapshots.

## Repository Layout

```text
drain3_multiprocess/
├── drain3/
│   ├── length_sharded_template_miner.py  # Multiprocess length-sharded miner
│   ├── drain.py                          # Tokenized add path and cluster-id hooks
│   ├── template_miner_config.py          # Sharding config keys
│   └── template_miner.py                 # Compatibility guard for standard miner
├── drain3_state.py                       # Example production parsing driver
├── examples/                             # Original and sample Drain3 demos
├── tests/                                # Drain3 tests
├── pyproject.toml                        # Package metadata
└── deploy_new_ver.sh                     # Deployment helper
```

## Configuration

Add these keys to the `[DRAIN]` section of a Drain3 config:

```ini
[DRAIN]
max_clusters_per_token_count = 2000
shard_workers = 4
```

Use `shard_workers > 1` to enable multiprocessing. Keep `TemplateMiner` for upstream-compatible single-process behavior.

## Quick Usage

```python
from drain3 import LengthShardedTemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
config.load("drain3.ini")
config.drain_shard_workers = 4

persistence = FilePersistence("drain3_state.bin")
miner = LengthShardedTemplateMiner(config=config, persistence_handler=persistence)

results = miner.add_log_messages([
    "rpd[1234]: %DAEMON-3-JTASK_SEND_RECV_ERROR Send msg call failed",
    "mib2d[5678]: %DAEMON-3-MIB2D_COUNTER_DECREASING counter decreasing",
])

miner.save_state("manual checkpoint")
miner.close()
```

## Quick Cluster Exploration

Use this as the sharded-snapshot equivalent of
`ART/drain3/explore_cluster_id.py`. Multiprocess snapshots must be loaded with
`LengthShardedTemplateMiner`, not the standard `TemplateMiner`, because clusters
are stored under token-count shards instead of `template_miner.drain.clusters`.
For read-only exploration, set `drain_shard_workers = 1` so the saved sharded
snapshot is loaded in-process and can be traversed without worker output.

```python
from drain3 import LengthShardedTemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
config.load("drain3.ini")
config.drain_shard_workers = 1

miner = LengthShardedTemplateMiner(
    config=config,
    persistence_handler=FilePersistence("drain3_state.bin"),
)

clusters = []
templates = []
for token_count, drain in miner._drains_by_token_count.items():
    for cluster in drain.clusters:
        template = cluster.get_template()
        clusters.append({
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "token_count": token_count,
            "latest_ts": miner.get_cluster_latest_ts(cluster.cluster_id),
            "template": template,
        })
        templates.append(template)

for row in sorted(clusters, key=lambda r: r["size"], reverse=True)[:20]:
    print(
        f'Cluster {row["cluster_id"]} '
        f'with size {row["size"]} '
        f'tokens={row["token_count"]} '
        f'latest_ts={row["latest_ts"]}:'
    )
    print(f'Template: {row["template"]}')

print(f"TOTAL CLUSTER/TEMP: {len(clusters)}")
print(f"Total unique template: {len(set(templates))}")
```

If the miner is still running with `shard_workers > 1`, call
`miner._collect_snapshot_data()` first and deserialize each returned Drain
state with `jsonpickle.loads(...)`; in multiprocessing mode the parent process
does not own the live Drain shards directly.

## Snapshot Format

The sharded miner persists a single snapshot containing all Drain shards:

```json
{
  "version": 1,
  "drains": {
    "12": "<jsonpickle Drain object>",
    "19": "<jsonpickle Drain object>"
  },
  "total_cluster_count": 12345,
  "cluster_id_counter": 34567,
  "cluster_latest_ts": {
    "101": 1735689600
  }
}
```

`drains` maps token length to serialized Drain instances. Standard `TemplateMiner` snapshots are not interchangeable with this sharded format.

## Notes

- Sharding is implemented for the standard Drain engine, not `JaccardDrain`.
- In multiprocessing mode, Drain state lives inside workers; use snapshot collection helpers rather than reading a single parent-side `drain` object.
- Batch output order is aligned to input order.
- On Windows, wrap multiprocessing usage with `if __name__ == "__main__":`.
