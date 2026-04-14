import os
import sys
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain

import pandas as pd
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ARC_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ARC_DIR)

from vocab_builder import DeviceVocab
import warnings
warnings.filterwarnings("ignore")

GLOBAL_DF = None
GLOBAL_MODE = None
GLOBAL_WINDOW_SIZE = None
GLOBAL_HAS_LABEL = None


def _init_worker(df, mode, window_size, has_label):
    global GLOBAL_DF, GLOBAL_MODE, GLOBAL_WINDOW_SIZE, GLOBAL_HAS_LABEL
    GLOBAL_DF = df
    GLOBAL_MODE = mode
    GLOBAL_WINDOW_SIZE = window_size
    GLOBAL_HAS_LABEL = has_label


def _gen_sequences_fixed_size(group_df, window_size, has_label):
    flat_ids = list(chain.from_iterable(group_df['log_key_ids']))
    flat_devs = list(chain.from_iterable(group_df['device_ids']))
    ts_list = group_df['timestamp'].tolist()
    lab_list = group_df['label'].tolist() if has_label else None

    seqs = []
    n = len(flat_ids)
    w = int(window_size)
    for i in range(0, n, w):
        k = flat_ids[i:i+w]
        if not k:
            continue
        t = ts_list[i:i+w]
        d = flat_devs[i:i+w]
        if has_label:
            y = 1 if any(lab_list[i:i+w]) else 0
            seqs.append((k, t, d, y))
        else:
            seqs.append((k, t, d))
    return seqs


def _gen_sequences_time_window(group_df, window_size, has_label):
    if isinstance(window_size, float):
        rule = f'{int(window_size * 60)}S'
    else:
        rule = f'{int(window_size)}T'

    out = []
    for _, win in group_df.groupby(pd.Grouper(key='timestamp', freq=rule, label='left', closed='left')):
        if win.empty:
            continue
        win = win.sort_values('timestamp')
        ids_win = list(chain.from_iterable(win['log_key_ids']))
        if not ids_win:
            continue
        # ts_win = win['timestamp'].tolist()
        # ts_win = []
        ts_end = win['timestamp'].max()
        ceil_ts = ts_end.ceil(rule)
        ceil_sec = int(ceil_ts.timestamp())
        ts_win = [ceil_sec]

        devs_win = list(chain.from_iterable(win['device_ids']))
        devs_win = [devs_win[0]] if devs_win else []

        if has_label:
            y = 1 if any(win['label'].astype(int)) else 0
            out.append((ids_win, ts_win, devs_win, y))
        else:
            out.append((ids_win, ts_win, devs_win))
    return out

def _gen_sequences_time_window_overlap(group_df, window_size, has_label):
    if isinstance(window_size, float):
        win_td = pd.Timedelta(seconds=int(window_size * 60))
        step_rule = '300S'
    else:
        win_td = pd.Timedelta(minutes=int(window_size))
        step_rule = '5T'

    group_df = group_df.sort_values('timestamp')
    ts = group_df['timestamp']

    start_end = ts.min().ceil(step_rule) + pd.Timedelta(minutes=5)
    last_end  = ts.max().ceil(step_rule)
    end_points = pd.date_range(start=start_end, end=last_end, freq=step_rule)

    out = []
    for end in end_points:
        start = end - win_td

        win = group_df[(ts >= start) & (ts <= end)]
        if win.empty:
            continue

        ids_win = list(chain.from_iterable(win['log_key_ids']))
        if not ids_win:
            continue

        ts_win = [int(end.timestamp())]

        devs_win = list(chain.from_iterable(win['device_ids']))
        devs_win = [devs_win[0]] if devs_win else []

        # for label, only find whether any event happens within the latest 5min
        label_win = win[(ts > (end - pd.Timedelta(minutes=5))) & (ts <= end)]

        if has_label:
            y = 1 if any(label_win['label'].astype(int)) else 0
            out.append((ids_win, ts_win, devs_win, y))
        else:
            out.append((ids_win, ts_win, devs_win))

    return out


def _process_one_ip(args):
    ip, idx = args
    df_ip = GLOBAL_DF.loc[idx]
    cols = ['timestamp', 'log_key_ids', 'device_ids'] + (['label'] if GLOBAL_HAS_LABEL else [])
    df_ip = df_ip[cols].sort_values('timestamp')
    if GLOBAL_MODE == 'fixed_size':
        return _gen_sequences_fixed_size(df_ip, GLOBAL_WINDOW_SIZE, GLOBAL_HAS_LABEL)
    elif GLOBAL_MODE == 'time_window':
        # return _gen_sequences_time_window(df_ip, GLOBAL_WINDOW_SIZE, GLOBAL_HAS_LABEL)
        return _gen_sequences_time_window_overlap(df_ip, GLOBAL_WINDOW_SIZE, GLOBAL_HAS_LABEL)
    else:
        raise ValueError("mode must be 'fixed_size' or 'time_window'")


def generate_sequences_mp(df, window_size, mode, device_vocab=None, max_workers=None, vocab=None):
    if df.empty:
        print("DataFrame is empty, no sequences generated.")
        if device_vocab is None:
            device_vocab = DeviceVocab(pd.Series([], dtype=object))
        return [], device_vocab
    
    if vocab is None:
        raise AssertionError("missing vocab")
    
    if device_vocab is None:
        raise AssertionError("missing device_vocab")

    # df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    if 'log_key' not in df.columns:
        raise AssertionError("missing log_key col")
    if 'ip' not in df.columns:
        raise AssertionError("missing ip col")

    stoi = device_vocab.stoi
    unk = device_vocab.unk_index

    # df['log_key_ids'] = df['log_key'].astype(str).map(lambda s: [vocab.stoi.get(s,vocab.unk_index)])
    df['log_key_ids'] = df['log_key'].map(lambda s: [s])
    df['device_id'] = df['ip'].map(lambda s: stoi.get(s, unk)).astype(int)
    df['device_ids'] = df['device_id'].map(lambda i: [i])

    has_label = 'label' in df.columns
    if has_label:
        df['label'] = df['label'].astype(int)

    grouped = df.groupby('ip', sort=False).groups
    tasks = list(grouped.items())

    if max_workers is None:
        cpu_count = os.cpu_count() - 1 or 1
        max_workers = max(4, cpu_count)

    print(f"Processing {len(tasks)} IPs with {max_workers} workers...")

    sequences_all = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(df, mode, window_size, has_label),
    ) as ex:
        results = ex.map(_process_one_ip, tasks, chunksize=1)
        for seqs in tqdm(results, total=len(tasks), desc="Generating sequences"):
            sequences_all.extend(seqs)

    print(f"Generated {len(sequences_all)} sequences using '{mode}' mode (multiprocess, per-IP).")
    return sequences_all, device_vocab
