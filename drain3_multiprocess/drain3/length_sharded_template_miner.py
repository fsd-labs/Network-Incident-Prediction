# SPDX-License-Identifier: MIT

import base64
import logging
import multiprocessing
import threading
import time
import zlib
from itertools import count
from typing import List, Mapping, MutableMapping, NamedTuple, Optional, Sequence, Tuple, Union

import jsonpickle  # type: ignore[import]
import sys
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CUR_DIR))

from drain3.drain import Drain, LogCluster
from drain3.masking import LogMasker
from drain3.persistence_handler import PersistenceHandler
from drain3.simple_profiler import SimpleProfiler, NullProfiler
from drain3.template_miner_config import TemplateMinerConfig


logger = logging.getLogger(__name__)

ResultMap = Mapping[str, Union[str, int]]


class _LocalClusterIdGenerator:
    def __init__(self, start: int = 0) -> None:
        self._counter = count(start + 1)
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return next(self._counter)


class _SharedClusterIdGenerator:
    def __init__(self, counter: multiprocessing.Value, lock: multiprocessing.Lock) -> None:
        self._counter = counter
        self._lock = lock

    def __call__(self) -> int:
        with self._lock:
            self._counter.value += 1
            return self._counter.value


class _WorkerDrainConfig(NamedTuple):
    sim_th: float
    depth: int
    max_children: int
    extra_delimiters: Sequence[str]
    parametrize_numeric_tokens: bool


def _worker_loop(task_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 drain_config: _WorkerDrainConfig,
                 max_clusters_per_token_count: Optional[int],
                 param_str: str,
                 cluster_id_counter: multiprocessing.Value,
                 cluster_id_lock: multiprocessing.Lock,
                 profiling_enabled: bool) -> None:
    drains: MutableMapping[int, Drain] = {}
    cluster_id_supplier = _SharedClusterIdGenerator(cluster_id_counter, cluster_id_lock)

    while True:
        task = task_queue.get()
        op = task.get("op")
        if op == "shutdown":
            return
        if op == "add":
            for index, token_count, tokens, ts in task["items"]:
                drain = drains.get(token_count)
                if drain is None:
                    profiler = SimpleProfiler() if profiling_enabled else NullProfiler()
                    drain = Drain(
                        sim_th=drain_config.sim_th,
                        depth=drain_config.depth,
                        max_children=drain_config.max_children,
                        max_clusters=max_clusters_per_token_count,
                        extra_delimiters=drain_config.extra_delimiters,
                        profiler=profiler,
                        param_str=param_str,
                        parametrize_numeric_tokens=drain_config.parametrize_numeric_tokens,
                        cluster_id_supplier=cluster_id_supplier
                    )
                    drains[token_count] = drain
                cluster, change_type = drain.add_log_message_tokens(tokens)
                result_queue.put({
                    "op": "add_result",
                    "index": index,
                    "change_type": change_type,
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.size,
                    "template_mined": cluster.get_template(),
                    "timestamp": ts
                })
        elif op == "match":
            token_count = task["token_count"]
            drain = drains.get(token_count)
            if drain is None:
                cluster = None
            else:
                cluster = drain.match(task["content_tokens"], task["full_search_strategy"])
            result_queue.put({
                "op": "match_result",
                "index": task["index"],
                "cluster": cluster
            })
        elif op == "batch_match":
            full_search_strategy = task.get("full_search_strategy", "never")
            for index, token_count, content in task["items"]:
                drain = drains.get(token_count)
                if drain is None:
                    cluster = None
                else:
                    cluster = drain.match(content, full_search_strategy)
                result_queue.put({
                    "op": "batch_match_result",
                    "index": index,
                    "cluster": cluster
                })
        elif op == "snapshot":
            drains_state: MutableMapping[int, str] = {}
            cluster_count = 0
            total_cluster_size = 0

            worker_index = task.get("worker_index", -1)
            drain_count = len(drains)

            for token_count, drain in drains.items():
                drains_state[token_count] = jsonpickle.dumps(drain, keys=True)
                cluster_count += len(drain.clusters)
                total_cluster_size += drain.get_total_cluster_size()
            result_queue.put({
                "op": "snapshot_result",
                "worker_index": worker_index,
                "drain_count": drain_count,
                "drains": drains_state,
                "cluster_count": cluster_count,
                "total_cluster_size": total_cluster_size
            })
        elif op == "load":
            drains_state = task.get("drains", {})
            drains = {}
            profiler = SimpleProfiler() if profiling_enabled else NullProfiler()
            cluster_count = 0
            total_cluster_size = 0
            for token_count, drain_state in drains_state.items():
                drain = jsonpickle.loads(drain_state, keys=True)
                drain.cluster_id_supplier = cluster_id_supplier
                drain.profiler = profiler
                drains[int(token_count)] = drain
                cluster_count += len(drain.clusters)
                total_cluster_size += drain.get_total_cluster_size()
            result_queue.put({
                "op": "load_result",
                "cluster_count": cluster_count,
                "total_cluster_size": total_cluster_size
            })
        else:
            raise ValueError(f"Unsupported operation: {op}")


class LengthShardedTemplateMiner:
    """
    Shard Drain template mining by token count (length) and optionally process
    multiple length branches in parallel when handling batches.
    """

    def __init__(self,
                 persistence_handler: Optional[PersistenceHandler] = None,
                 config: Optional[TemplateMinerConfig] = None,
                 max_workers: Optional[int] = None) -> None:
        if config is None:
            raise ValueError("TemplateMinerConfig must be provided")

        if config.engine != "Drain":
            raise ValueError("Length sharding is supported only for the Drain engine")

        self.config = config
        self._profiling_enabled = self.config.profiling_enabled
        self.persistence_handler = persistence_handler
        self.last_save_time = time.time()

        self.masker = LogMasker(self.config.masking_instructions, self.config.mask_prefix, self.config.mask_suffix)
        self.param_str = f"{self.config.mask_prefix}*{self.config.mask_suffix}"
        self._total_cluster_count = 0
        self._cluster_count_lock = threading.Lock()

        per_token_max = self.config.drain_max_clusters_per_token_count
        if per_token_max is None:
            per_token_max = self.config.drain_max_clusters
        self._max_clusters_per_token_count = per_token_max

        if max_workers is None:
            max_workers = self.config.drain_shard_workers or None
        self._max_workers = max_workers
        self._use_processes = self._max_workers is not None and self._max_workers > 1

        self._drains_by_token_count: MutableMapping[int, Drain] = {}
        self._cluster_id_generator = _LocalClusterIdGenerator()
        self._cluster_latest_ts: MutableMapping[int, int] = {}  # cluster_id -> latest timestamp
        self._process_ctx: Optional[multiprocessing.context.BaseContext] = None
        self._task_queues: List[multiprocessing.Queue] = []
        self._result_queue: Optional[multiprocessing.Queue] = None
        self._workers: List[multiprocessing.Process] = []
        self._cluster_id_counter: Optional[multiprocessing.Value] = None
        self._cluster_id_lock: Optional[multiprocessing.Lock] = None
        self._closed = False

        self._drain_config = _WorkerDrainConfig(
            sim_th=self.config.drain_sim_th,
            depth=self.config.drain_depth,
            max_children=self.config.drain_max_children,
            extra_delimiters=self.config.drain_extra_delimiters,
            parametrize_numeric_tokens=self.config.parametrize_numeric_tokens
        )

        if self._use_processes:
            self._start_workers()

        if self.persistence_handler is not None:
            self.load_state()

    def _get_content_as_tokens(self, content: str) -> Sequence[str]:
        content = content.strip()
        for delimiter in self.config.drain_extra_delimiters:
            content = content.replace(delimiter, " ")
        return content.split()

    def load_state(self) -> None:
        print("Checking for saved state")

        assert self.persistence_handler is not None

        state = self.persistence_handler.load_state()
        if state is None:
            print("Saved state not found")
            return

        if self.config.snapshot_compress_state:
            state = zlib.decompress(base64.b64decode(state))

        snapshot = jsonpickle.loads(state, keys=True)
        drains_state = snapshot.get("drains", {})
        self._total_cluster_count = snapshot.get("total_cluster_count", 0)
        cluster_id_counter = snapshot.get("cluster_id_counter", 0)
        self._cluster_latest_ts = {int(k): v for k, v in snapshot.get("cluster_latest_ts", {}).items()}

        if self._use_processes:
            if self._cluster_id_counter is not None and self._cluster_id_lock is not None:
                with self._cluster_id_lock:
                    self._cluster_id_counter.value = max(self._cluster_id_counter.value, cluster_id_counter)
            self._load_drains_to_workers(drains_state)
        else:
            self._cluster_id_generator = _LocalClusterIdGenerator(start=cluster_id_counter)
            self._drains_by_token_count = {}
            profiler = SimpleProfiler() if self._profiling_enabled else NullProfiler()
            for token_count, drain_state in drains_state.items():
                drain = jsonpickle.loads(drain_state, keys=True)
                drain.cluster_id_supplier = self._cluster_id_generator
                drain.profiler = profiler
                self._drains_by_token_count[int(token_count)] = drain
            if self._total_cluster_count == 0 and drains_state:
                self._total_cluster_count = sum(len(drain.clusters) for drain in self._drains_by_token_count.values())

        print(f"Restored {self._total_cluster_count} clusters")

    def save_state(self, snapshot_reason: str) -> None:
        assert self.persistence_handler is not None

        drains_state, cluster_count, total_cluster_size = self._collect_snapshot_data()
        snapshot = {
            "version": 1,
            "drains": drains_state,
            "total_cluster_count": cluster_count,
            "cluster_id_counter": self._get_cluster_id_counter_value(),
            "cluster_latest_ts": dict(self._cluster_latest_ts)
        }

        state = jsonpickle.dumps(snapshot, keys=True).encode("utf-8")
        if self.config.snapshot_compress_state:
            state = base64.b64encode(zlib.compress(state))

        print(f"Saving state of {cluster_count} clusters "
                    f"with {total_cluster_size} messages, {len(state)} bytes, "
                    f"reason: {snapshot_reason}")
        self.persistence_handler.save_state(state)

    def get_snapshot_reason(self, change_type: str, cluster_id: int) -> Optional[str]:
        if change_type != "none":
            return f"{change_type} ({cluster_id})"

        diff_time_sec = time.time() - self.last_save_time
        if diff_time_sec >= self.config.snapshot_interval_minutes * 60:
            return "periodic"

        return None

    def _maybe_save_state(self, results: Sequence[ResultMap]) -> None:
        if self.persistence_handler is None:
            return

        snapshot_reason = None
        for result in reversed(results):
            change_type = result["change_type"]
            cluster_id = result["cluster_id"]
            snapshot_reason = self.get_snapshot_reason(change_type, cluster_id)
            if snapshot_reason is not None and change_type != "none":
                break

        if snapshot_reason:
            self.save_state(snapshot_reason)
            self.last_save_time = time.time()

    def _get_drain(self, token_count: int) -> Drain:
        if self._use_processes:
            raise RuntimeError("In-process drains are disabled when multiprocessing is enabled")
        drain = self._drains_by_token_count.get(token_count)
        if drain is None:
            profiler = SimpleProfiler() if self._profiling_enabled else NullProfiler()
            drain = Drain(
                sim_th=self.config.drain_sim_th,
                depth=self.config.drain_depth,
                max_children=self.config.drain_max_children,
                max_clusters=self._max_clusters_per_token_count,
                extra_delimiters=self.config.drain_extra_delimiters,
                profiler=profiler,
                param_str=self.param_str,
                parametrize_numeric_tokens=self.config.parametrize_numeric_tokens,
                cluster_id_supplier=self._cluster_id_generator
            )
            self._drains_by_token_count[token_count] = drain
        return drain

    def _format_result_values(self,
                              change_type: str,
                              cluster_id: int,
                              cluster_size: int,
                              template_mined: str,
                              timestamp: Optional[int] = None) -> ResultMap:
        with self._cluster_count_lock:
            if change_type == "cluster_created":
                self._total_cluster_count += 1
            cluster_count = self._total_cluster_count
        
        ts = timestamp if timestamp is not None else int(time.time())
        current_ts = self._cluster_latest_ts.get(cluster_id, 0)
        if ts > current_ts:
            self._cluster_latest_ts[cluster_id] = ts
        
        return {
            "change_type": change_type,
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "template_mined": template_mined,
            "cluster_count": cluster_count,
            "latest_ts": self._cluster_latest_ts[cluster_id]
        }

    def add_log_message(self, 
                        log_message: str, 
                        timestamp: Optional[int] = None) -> ResultMap:
        if timestamp is None:
            timestamp = int(time.time())
        
        masked_content = self.masker.mask(log_message)
        content_tokens = self._get_content_as_tokens(masked_content)
        token_count = len(content_tokens)
        
        if not self._use_processes:
            drain = self._get_drain(token_count)
            cluster, change_type = drain.add_log_message_tokens(content_tokens)
            result = self._format_result_values(change_type, cluster.cluster_id, 
                                                cluster.size, cluster.get_template(), timestamp)
            # self._maybe_save_state([result])
            return result

        results = self._process_add_items([(0, token_count, content_tokens, timestamp)], 1)
        # self._maybe_save_state(results)
        return results[0]

    def match(self, log_message: str, full_search_strategy: str = "never") -> Optional[LogCluster]:
        masked_content = self.masker.mask(log_message)
        content_tokens = self._get_content_as_tokens(masked_content)
        token_count = len(content_tokens)
        if not self._use_processes:
            drain = self._drains_by_token_count.get(token_count)
            if drain is None:
                return None
            return drain.match(content_tokens, full_search_strategy)

        if self._result_queue is None:
            raise RuntimeError("Worker processes are not initialized")

        worker_index = self._worker_index(token_count)
        self._task_queues[worker_index].put({
            "op": "match",
            "index": 0,
            "token_count": token_count,
            "content_tokens": content_tokens,
            "full_search_strategy": full_search_strategy
        })
        result = self._result_queue.get()
        if result.get("op") != "match_result":
            raise RuntimeError("Unexpected result type while waiting for match")
        return result.get("cluster")

    def batch_match(self,
                    log_messages: Sequence[str],
                    full_search_strategy: str = "never") -> Sequence[Optional[LogCluster]]:
        """
        Match multiple log messages against existing clusters in parallel.
        No new clusters will be created as a result of this call.

        :param log_messages: List of log messages to match
        :param full_search_strategy: when to perform full cluster search.
            (1) "never" is the fastest, will always perform a tree search [O(log(n)] but might produce
            false negatives (wrong mismatches) on some edge cases;
            (2) "fallback" will perform a linear search [O(n)] among all clusters with the same token count,
            but only in case tree search found no match;
            (3) "always" is the slowest. It will select the best match among all known clusters.
        :return: List of matched clusters (or None for each message that found no match)
        """
        if not log_messages:
            return []

        if not self._use_processes:
            return [self.match(message, full_search_strategy) for message in log_messages]

        # Prepare items: (index, token_count, masked_content)
        items: List[Tuple[int, int, Sequence[str]]] = []
        for index, message in enumerate(log_messages):
            masked_content = self.masker.mask(message)
            content_tokens = self._get_content_as_tokens(masked_content)
            items.append((index, len(content_tokens), content_tokens))

        return self._process_match_items(items, len(log_messages), full_search_strategy)

    def _process_match_items(self,
                             items: Sequence[Tuple[int, int, Sequence[str]]],
                             result_size: int,
                             full_search_strategy: str) -> Sequence[Optional[LogCluster]]:
        """
        Process batch match items across worker processes.

        :param items: List of (index, token_count, content_tokens) tuples
        :param result_size: Expected number of results
        :param full_search_strategy: Strategy for full search
        :return: List of matched clusters
        """
        if self._result_queue is None:
            raise RuntimeError("Worker processes are not initialized")

        # Group items by worker based on token count
        items_by_worker: List[List[Tuple[int, int, Sequence[str]]]] = [[] for _ in range(self._max_workers or 0)]
        for index, token_count, content in items:
            worker_index = self._worker_index(token_count)
            items_by_worker[worker_index].append((index, token_count, content))

        # Send tasks to workers
        for worker_index, worker_items in enumerate(items_by_worker):
            if worker_items:
                self._task_queues[worker_index].put({
                    "op": "batch_match",
                    "items": worker_items,
                    "full_search_strategy": full_search_strategy
                })

        # Collect results
        expected_results = len(items)
        results: List[Optional[LogCluster]] = [None] * result_size
        received = 0
        while received < expected_results:
            result = self._result_queue.get()
            if result.get("op") != "batch_match_result":
                raise RuntimeError("Unexpected result type while waiting for batch_match")
            index = result["index"]
            results[index] = result.get("cluster")
            received += 1

        return results

    def add_log_messages(self,
                         log_messages: Sequence[str],
                         timestamps: Optional[Sequence[Optional[int]]] = None,
                         max_workers: Optional[int] = None) -> Sequence[ResultMap]:
        if not log_messages:
            return []

        if not self._use_processes:
            if timestamps is None:
                return [self.add_log_message(message) for message in log_messages]
            return [self.add_log_message(msg, ts) for msg, ts in zip(log_messages, timestamps)]

        workers = self._max_workers if max_workers is None else max_workers
        if workers is not None and workers != self._max_workers:
            raise ValueError("Dynamic worker counts are not supported in multiprocessing mode")

        now = int(time.time())
        items: List[Tuple[int, int, Sequence[str], int]] = []
        for index, message in enumerate(log_messages):
            masked_content = self.masker.mask(message)
            content_tokens = self._get_content_as_tokens(masked_content)
            ts = timestamps[index] if timestamps is not None and timestamps[index] is not None else now
            items.append((index, len(content_tokens), content_tokens, ts))

        results = self._process_add_items(items, len(log_messages))
        self._maybe_save_state(results)
        return results

    def close(self) -> None:
        if not self._use_processes or self._closed:
            return
        for task_queue in self._task_queues:
            task_queue.put({"op": "shutdown"})
        for worker in self._workers:
            worker.join()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _start_workers(self) -> None:
        self._process_ctx = multiprocessing.get_context("spawn")
        assert self._max_workers is not None
        self._task_queues = [self._process_ctx.Queue() for _ in range(self._max_workers)]
        self._result_queue = self._process_ctx.Queue()
        self._cluster_id_counter = self._process_ctx.Value("i", 0)
        self._cluster_id_lock = self._process_ctx.Lock()
        for idx in range(self._max_workers):
            worker = self._process_ctx.Process(
                target=_worker_loop,
                args=(
                    self._task_queues[idx],
                    self._result_queue,
                    self._drain_config,
                    self._max_clusters_per_token_count,
                    self.param_str,
                    self._cluster_id_counter,
                    self._cluster_id_lock,
                    self._profiling_enabled
                )
            )
            worker.start()
            self._workers.append(worker)

    def _worker_index(self, token_count: int) -> int:
        assert self._max_workers is not None
        return token_count % self._max_workers

    def _process_add_items(self,
                           items: Sequence[Tuple[int, int, Sequence[str], int]],
                           result_size: int) -> Sequence[ResultMap]:
        if self._result_queue is None:
            raise RuntimeError("Worker processes are not initialized")

        items_by_worker: List[List[Tuple[int, int, Sequence[str], int]]] = [[] for _ in range(self._max_workers or 0)]
        for index, token_count, tokens, ts in items:
            worker_index = self._worker_index(token_count)
            items_by_worker[worker_index].append((index, token_count, tokens, ts))

        for worker_index, worker_items in enumerate(items_by_worker):
            if worker_items:
                self._task_queues[worker_index].put({"op": "add", "items": worker_items})

        expected_results = len(items)
        results: List[Optional[ResultMap]] = [None] * result_size
        received = 0
        while received < expected_results:
            result = self._result_queue.get()
            if result.get("op") != "add_result":
                raise RuntimeError("Unexpected result type while waiting for add")
            index = result["index"]
            results[index] = self._format_result_values(
                result["change_type"],
                result["cluster_id"],
                result["cluster_size"],
                result["template_mined"],
                result.get("timestamp")
            )
            received += 1

        return [result for result in results if result is not None]

    def _collect_snapshot_data(self) -> Tuple[MutableMapping[int, str], int, int]:
        drains_state: MutableMapping[int, str] = {}
        cluster_count = 0
        total_cluster_size = 0

        if not self._use_processes:
            for token_count, drain in self._drains_by_token_count.items():
                drains_state[token_count] = jsonpickle.dumps(drain, keys=True)
                cluster_count += len(drain.clusters)
                total_cluster_size += drain.get_total_cluster_size()
            return drains_state, cluster_count, total_cluster_size

        if self._result_queue is None:
            raise RuntimeError("Worker processes are not initialized")

        for wi, task_queue in enumerate(self._task_queues):
            task_queue.put({"op": "snapshot", "worker_index": wi})

        print("============ COLLECTING SNAPSHOT DATA FROM WORKERS ============")

        for _ in range(len(self._task_queues)):
            start_time = time.time()
            result = self._result_queue.get()
            if result.get("op") != "snapshot_result":
                raise RuntimeError("Unexpected result type while waiting for snapshot")
            drains_state.update(result.get("drains", {}))
            cluster_count += result.get("cluster_count", 0)
            total_cluster_size += result.get("total_cluster_size", 0)

            print(f"--- WORKER {result.get('worker_index')} SNAPSHOT ---")
            print(f"Number of drains: {result.get('drain_count')}")
            print(f"Number of clusters: {result.get('cluster_count'):,.0f}")
            print(f"Total cluster size: {result.get('total_cluster_size'):,.0f}")
            print(f"PRINTING TREE FOR EACH DRAIN:")
            for token_count, drain_state in result.get("drains", {}).items():
                drain = jsonpickle.loads(drain_state, keys=True)
                print(f"--- DRAIN for token count: {token_count} ---")
                drain.print_tree()
            end_time = time.time()
            print(f"Worker snapshot collection time: {end_time - start_time:,.2f} seconds")
            print("\n\n")
        print("===============================================================")
        
        return drains_state, cluster_count, total_cluster_size

    def _get_cluster_id_counter_value(self) -> int:
        if self._use_processes:
            if self._cluster_id_counter is None or self._cluster_id_lock is None:
                return 0
            with self._cluster_id_lock:
                return self._cluster_id_counter.value

        max_cluster_id = 0
        for drain in self._drains_by_token_count.values():
            if drain.clusters_counter > max_cluster_id:
                max_cluster_id = drain.clusters_counter
        return max_cluster_id

    def _load_drains_to_workers(self, drains_state: Mapping[int, str]) -> None:
        if not drains_state:
            return
        if self._result_queue is None:
            raise RuntimeError("Worker processes are not initialized")

        drains_by_worker: List[MutableMapping[int, str]] = [
            {} for _ in range(self._max_workers or 0)
        ]
        for token_count, drain_state in drains_state.items():
            worker_index = self._worker_index(int(token_count))
            drains_by_worker[worker_index][int(token_count)] = drain_state

        expected = 0
        loaded_cluster_count = 0
        for worker_index, worker_drains in enumerate(drains_by_worker):
            if worker_drains:
                self._task_queues[worker_index].put({"op": "load", "drains": worker_drains})
                expected += 1

        for _ in range(expected):
            result = self._result_queue.get()
            if result.get("op") != "load_result":
                raise RuntimeError("Unexpected result type while waiting for load")
            loaded_cluster_count += result.get("cluster_count", 0)

        if self._total_cluster_count == 0 and drains_state:
            self._total_cluster_count = loaded_cluster_count

    def get_cluster_latest_ts(self, cluster_id: int) -> int:
        """Get the latest timestamp for a specific cluster."""
        return self._cluster_latest_ts.get(cluster_id, 0)

    def get_all_cluster_timestamps(self) -> Mapping[int, int]:
        """Get all cluster timestamps as a mapping of cluster_id -> timestamp."""
        return self._cluster_latest_ts
