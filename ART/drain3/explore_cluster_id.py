import os
import pandas as pd
from drain3.template_miner_config import TemplateMinerConfig
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence


current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)

config = TemplateMinerConfig()
config.load(f"{current_folder}/drain3.ini")
persistence = FilePersistence(f"{current_folder}/drain3_state_m_45__4_batch_26_06.bin")
config.profiling_enabled = False
template_miner = TemplateMiner(config=config, persistence_handler=persistence)


# In ra cluster_id + template
total_cluster = len(template_miner.drain.clusters)
temps = []
for cluster in template_miner.drain.clusters:
    print(f"Cluster {cluster.cluster_id} with size {cluster.size}:")
    print(f"Template: {cluster.get_template()}")
    temps.append(cluster.get_template())

print(f"TOTAL CLUSTER/TEMP: {total_cluster}")
print(f"Total unique template: {len(set(temps))}")
