import logging
import os, sys
from pathlib import Path
from model_architecture.vocab_builder import VocabBuilder
import jsonpickle

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Reuse the original DeviceIncidents/drain3 assets to avoid duplicating binaries
ORIG_D3 = PROJECT_ROOT.parent / "DeviceIncidents" / "drain3"

D3_REPO = PROJECT_ROOT.parent / "AnomalyDetection" / "drain3_training" / "drain3_multiprocess"
sys.path.insert(0, str(D3_REPO))

from drain3 import LengthShardedTemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


class VocabConstructor:
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

    def load_cluster_map(self):
        logger.info("Loading clusters directly from Drain3 state.")
        logger.info("Drain3 folder: %s", self.drain3_dir)
        config = TemplateMinerConfig()
        config.load(str(self.drain3_dir / "drain3_v2.ini"))
        persistence = FilePersistence(
            str(self.drain3_dir / "drain3_state_m_11_12_v2__12_batch_21_01.bin")
        )
        config.profiling_enabled = False
        template_miner = LengthShardedTemplateMiner(config=config, persistence_handler=persistence)

        temps = []
        total = 0
        cluster_ids = []
        for cluster in self.iter_clusters(template_miner):
            total += 1
            # print(f"Cluster {cluster.cluster_id} with size {cluster.size}:")
            # print(f"Template: {cluster.get_template()}")
            temps.append([str(cluster.get_template())])
            cluster_ids.append(cluster.cluster_id)

        logger.info("Total clusters: %s", total)
        logger.info("Total unique template: %s", len(set(t[0] for t in temps)))
        return temps

    def run(self):
        logger.info("Building vocabulary...")
        texts = self.load_cluster_map()
        logger.info("Total log keys for vocab: %s", len(texts))
        vocab = VocabBuilder(texts, min_freq=self.args.min_freq)
        logger.info("Vocabulary size: %s", len(vocab))
        os.makedirs(os.path.dirname(self.args.vocab_path), exist_ok=True)
        logger.info("Saving vocabulary to %s", self.args.vocab_path)
        vocab.save_vocab(self.args.vocab_path)
