import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import torch.distributed as dist

import pytz

from executors import ModelTrainer, Predictor, VocabConstructor, SequencePreper, create_dev_vocab, TokenVocabConstructor, Debug
from experimental_space.args_builder import build_parser
from model_architecture.helper_utils import time_logger
from model_architecture.vocab_builder import WordVocab, DeviceVocab

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(os.path.dirname(str(PROJECT_ROOT)))
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

log_file = LOG_DIR / f"execution_{datetime.now(VN_TZ).strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")],
)
logger = logging.getLogger(__name__)

def get_dist_info():
    """Get distributed info from environment variables (set by accelerate/torchrun)"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    logger.info(f"[MAIN EXEC] world_size: {world_size}, rank: {rank}")
    return world_size, rank

def sync_processes():
    """Synchronize all processes if running distributed"""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

@time_logger(time_span="min")
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "vocab":
        # VocabConstructor(args).run()
        TokenVocabConstructor(args).run()
        return

    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found at {args.vocab_path}. Please run in 'vocab' mode first."
        )
    WordVocab.load_vocab(args.vocab_path)

    if args.mode == "dev_vocab":
        create_dev_vocab()
        return

    if not os.path.exists(args.device_vocab_path):
        raise FileNotFoundError(
            f"Device Vocabulary file not found at {args.device_vocab_path}. Please run dev_vocab_executor first."
        )
    DeviceVocab.load_vocab(args.device_vocab_path)

    if args.mode == "train":
        ModelTrainer(args).fit()
    elif args.mode == "predict":
        Predictor(args).run()
    elif args.mode == "seqprep":
        SequencePreper(args).run()
    elif args.mode == "debug":
        Debug(args).run()
    else:
        raise ValueError("mode must be one of: vocab, seqprep, train, predict")


def parse_and_run():
    parser = build_parser()
    args = parser.parse_args()

    world_size, rank = get_dist_info()
    is_distributed = world_size > 1
    is_main = rank == 0

    args.mode = "predict"

    # Preprocessing tasks: only run on main process
    if args.mode in ["vocab", "dev_vocab", "seqprep"]:
        if is_main:
            main(args)
        if is_distributed:
            import torch
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            sync_processes()
        return  

    # Training/prediction
    if args.mode in ["train", "predict", "debug"]:
        main(args)

    # args.mode = "predict"
    # main(args)


if __name__ == "__main__":
    parse_and_run()
