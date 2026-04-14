import json
import logging
import os
import sys
from pathlib import Path

import jsonpickle

from model_architecture.vocab_builder import VocabBuilder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIG_D3 = PROJECT_ROOT.parent / "DeviceIncidents" / "drain3"

D3_REPO = PROJECT_ROOT.parent / "AnomalyDetection" / "drain3_training" / "drain3_multiprocess"
sys.path.insert(0, str(D3_REPO))

from drain3 import LengthShardedTemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


class TokenVocabConstructor:
    def __init__(self, args):
        self.args = args
        self.drain3_dir = ORIG_D3

    @staticmethod
    def iter_clusters(tm):
        if not getattr(tm, "_use_processes", False):
            for drain in tm._drains_by_token_count.values():
                for cluster in drain.clusters:
                    yield cluster
            return

        drains_state, _, _ = tm._collect_snapshot_data()
        for drain_state in drains_state.values():
            drain = jsonpickle.loads(drain_state, keys=True)
            for cluster in drain.clusters:
                yield cluster

    def _resolve_stats_path(self):
        stats_path = getattr(self.args, "cluster_token_stats_path", None)
        if stats_path:
            return stats_path

        vocab_path = getattr(self.args, "vocab_path", "")
        if vocab_path:
            return str(Path(vocab_path).with_name("cluster_token_stats.json"))

        return "cluster_token_stats.json"

    def load_tokenized_templates(self):
        logger.info("Loading clusters directly from Drain3 state.")
        logger.info("Drain3 folder: %s", self.drain3_dir)

        config = TemplateMinerConfig()
        config.load(str(self.drain3_dir / "drain3_v2.ini"))
        persistence = FilePersistence(
            str(self.drain3_dir / "drain3_state_m_11_12_v2__12_batch_21_01.bin")
        )
        config.profiling_enabled = False
        template_miner = LengthShardedTemplateMiner(config=config, persistence_handler=persistence)

        # tokenized_templates = []
        cluster_token_stats = []
        unique_tokens = set()
        cluster_count = 0

        for cluster in self.iter_clusters(template_miner):
            cluster_count += 1
            template = str(cluster.get_template())
            tokens = template.split()
            # tokenized_templates.append([[token] for token in tokens])
            unique_tokens.update(tokens)
            cluster_token_stats.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "token_counts": len(tokens),
                }
            )

        logger.info("Total clusters: %s", cluster_count)
        logger.info("Total unique tokens from templates: %s", len(unique_tokens))

        lst_unique_tokens = [[token] for token in unique_tokens]

        return lst_unique_tokens, cluster_token_stats

    def run(self):
        logger.info("Building token-level vocabulary...")
        texts, cluster_token_stats = self.load_tokenized_templates()
        logger.info("Total templates for vocab: %s", len(texts))

        vocab = VocabBuilder(texts, min_freq=self.args.min_freq)
        logger.info("Vocabulary size: %s", len(vocab))

        os.makedirs(os.path.dirname(self.args.vocab_path), exist_ok=True)
        logger.info("Saving vocabulary to %s", self.args.vocab_path)
        vocab.save_vocab(self.args.vocab_path)

        stats_path = self._resolve_stats_path()
        stats_dir = os.path.dirname(stats_path)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(cluster_token_stats, f, ensure_ascii=False, indent=2)

        logger.info("Saved cluster token statistics to %s", stats_path)
