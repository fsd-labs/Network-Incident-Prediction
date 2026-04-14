from torch.utils.data import Sampler
import numpy as np
import torch.distributed as dist
import os

class BucketBatchSampler(Sampler):
    """
    Bucket-based batch sampler for variable-length sequences.
    
    CRITICAL for DDP: This sampler ensures all ranks have the same number of batches
    by padding with repeated indices if needed. This prevents cudaErrorIllegalAddress
    errors that occur when ranks have mismatched batch counts.
    """
    def __init__(self, data_source, batch_size, sort_key=None, drop_last=False, seed=None, mode="train", resume_from_epoch=0):
        # PyTorch Sampler API differs across versions:
        # some accept data_source in __init__, others inherit object.__init__ only.
        try:
            super().__init__(data_source)
        except TypeError:
            super().__init__()
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = seed if seed is not None else 42
        self.sort_key = sort_key or (lambda idx: len(data_source.log_corpus[idx][0]))
        self.lengths = np.array([self.sort_key(i) for i in range(len(data_source))], dtype=np.int64)
        self.epoch = 0 + resume_from_epoch
        self.mode = mode

    def _get_dist(self):
        if "WORLD_SIZE" in os.environ:
            try:
                ws = int(os.environ["WORLD_SIZE"])
                rk = int(os.environ.get("RANK", 0))
                return ws, rk
            except ValueError:
                pass

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()

        return 1, 0

    def __iter__(self):
        self.epoch += 1
        print(f"BucketBatchSampler epoch {self.epoch}")
        n = len(self.data_source)
        
        # Sort by length for bucketing
        idx_sorted = np.argsort(self.lengths, kind="mergesort")
        batches = [idx_sorted[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        
        # Use consistent RNG across all ranks for shuffling
        if self.mode == "train":
            rng = np.random.default_rng(self.seed + self.epoch)
        else:  # eval mode
            rng = np.random.default_rng(self.seed)
        rng.shuffle(batches)
        
        world_size, rank = self._get_dist()
        print(f"[SAMPLER] WORLD SIZE: {world_size}, RANK: {rank}, TOTAL BATCHES: {len(batches)}")
        
        if world_size > 1:
            # Pad batches to be divisible by world_size
            total_batches = len(batches)
            padded_size = ((total_batches + world_size - 1) // world_size) * world_size
            
            if padded_size > total_batches:
                print(f"[SAMPLER] Padding from {total_batches} to {padded_size} batches for rank {rank}")
                # Pad with repeated batches (they will produce duplicate gradients but prevent crashes)
                extra_needed = padded_size - total_batches
                for i in range(extra_needed):
                    batches.append(batches[i % total_batches])
            
            # Now slice evenly
            batches = batches[rank::world_size]
            print(f"[SAMPLER] RANK {rank}: {len(batches)} batches after distribution")
        
        for b in batches:
            yield b.tolist()

    def __len__(self):
        total = len(self.lengths) // self.batch_size if self.drop_last else (len(self.lengths) + self.batch_size - 1) // self.batch_size
        world_size, rank = self._get_dist()
        if world_size <= 1:
            return total
        # Return padded length
        padded_total = ((total + world_size - 1) // world_size)
        return padded_total
