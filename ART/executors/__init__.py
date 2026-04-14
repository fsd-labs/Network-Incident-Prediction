"""Helper executors for vocab building, sequence prep, HF training and prediction."""

from .vocab_executor import VocabConstructor
from .vocab_exec_tok import TokenVocabConstructor
from .prepare_data import SequencePreper
from .hf_train import train_with_args
from .hf_predict import predict_with_args
from .hf_visualize_weight import debug_with_args
from .dev_vocab_executor import create_dev_vocab

class ModelTrainer:
    def __init__(self, args):
        self.args = args

    def fit(self):
        train_with_args(self.args)


class Predictor:
    def __init__(self, args):
        self.args = args

    def run(self):
        predict_with_args(self.args)
    
class Debug:
    def __init__(self, args):
        self.args = args

    def run(self):
        debug_with_args(self.args)


__all__ = [
    "VocabConstructor",
    "TokenVocabConstructor"
    "SequencePreper",
    "train_with_args",
    "predict_with_args",
    "ModelTrainer",
    "Predictor",
    "create_dev_vocab",
    "debug_with_args",
]
