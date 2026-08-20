import os
import re
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append("../")
from logparser import Drain


class Config:
    WINDOWS_DIR = "../../../processed_v2/logbert_windows/"
    OUTPUT_DIR = "./output/"

    DRAIN_DEPTH = 4
    DRAIN_ST = 0.4

    FIXED_LEN = 64
    PAD_TOKEN = 0

    TRAIN_RATIO = 0.9
    SEED = 777

    TRAIN_FILE = "train"
    VAL_FILE = "val_normal"
    TEST_NORMAL_FILE = "test_normal"
    TEST_ABNORMAL_FILE = "test_abnormal"


def preprocess_log(line):
    line = re.sub(r"^\w+\s+\d+\s+\d+:\d+:\d+", "", line)
    line = re.sub(r"^\d{2}/\d{2}/\d{4}-\d+:\d+:\d+\.\d+", "", line)
    line = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d+:\d+:\d+", "", line)
    return line.strip()


def load_jsonl(path):
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                windows.append(json.loads(line))
    return windows


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    windows_dir = Path(Config.WINDOWS_DIR)

    train_normal_windows = load_jsonl(windows_dir / "train_normal.jsonl")
    train_anomaly_windows = load_jsonl(windows_dir / "train_anomaly.jsonl")
    test_normal_windows = load_jsonl(windows_dir / "test_normal.jsonl")
    test_anomaly_windows = load_jsonl(windows_dir / "test_abnormal.jsonl")

    print(f"loaded windows: train_normal={len(train_normal_windows)} "
          f"train_anomaly={len(train_anomaly_windows)} "
          f"test_normal={len(test_normal_windows)} "
          f"test_abnormal={len(test_anomaly_windows)}")

    all_contents = []
    all_sources = []

    for split_name, windows in [("train_normal", train_normal_windows),
                                 ("train_anomaly", train_anomaly_windows),
                                 ("test_normal", test_normal_windows),
                                 ("test_abnormal", test_anomaly_windows)]:
        for w_idx, window in enumerate(windows):
            for c_idx, content in enumerate(window["contents"]):
                cleaned = preprocess_log(content)
                if cleaned:
                    all_contents.append(cleaned)
                    all_sources.append((split_name, w_idx, c_idx))

    print(f"{len(all_contents):,} content strings, running Drain...")

    content_file = os.path.join(Config.OUTPUT_DIR, "content_only.log")
    with open(content_file, "w") as f:
        for content in all_contents:
            f.write(content + "\n")

    parser = Drain.LogParser(
        log_format="<Content>",
        indir=Config.OUTPUT_DIR,
        outdir=Config.OUTPUT_DIR,
        depth=Config.DRAIN_DEPTH,
        st=Config.DRAIN_ST,
        rex=[r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}", r":\d+", r"\d+", r"[a-f0-9]{8,}"]
    )
    parser.parse("content_only.log")

    structured = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "content_only.log_structured.csv"))
    print(f"{structured['EventId'].nunique()} unique templates")

    templates_file = os.path.join(Config.OUTPUT_DIR, "content_only.log_templates.csv")
    templates_df = pd.read_csv(templates_file)
    templates_df.sort_values(by=["Occurrences"], ascending=False, inplace=True)

    event_map = {event: idx + 1 for idx, event in enumerate(templates_df["EventId"])}

    event_ids = structured["EventId"].tolist()
    mapped_ids = [event_map.get(eid, 0) for eid in event_ids]

    window_sequences = {}
    for i, (split_name, w_idx, c_idx) in enumerate(all_sources):
        key = (split_name, w_idx)
        if key not in window_sequences:
            window_sequences[key] = []
        window_sequences[key].append(mapped_ids[i])

    def build_sequences(split_name, windows, label):
        sequences = []
        labels = []
        for w_idx, window in enumerate(windows):
            key = (split_name, w_idx)
            seq = window_sequences.get(key, [])

            if len(seq) > Config.FIXED_LEN:
                seq = seq[:Config.FIXED_LEN]
            else:
                seq = seq + [Config.PAD_TOKEN] * (Config.FIXED_LEN - len(seq))

            sequences.append(seq)
            labels.append(label)
        return sequences, labels

    train_normal_seqs, train_normal_labels = build_sequences("train_normal", train_normal_windows, 0)
    train_anomaly_seqs, train_anomaly_labels = build_sequences("train_anomaly", train_anomaly_windows, 1)
    test_normal_seqs, test_normal_labels = build_sequences("test_normal", test_normal_windows, 0)
    test_anomaly_seqs, test_anomaly_labels = build_sequences("test_abnormal", test_anomaly_windows, 1)

    n = len(train_normal_seqs)
    indices = list(range(n))
    rng = np.random.RandomState(Config.SEED)
    rng.shuffle(indices)

    train_size = int(n * Config.TRAIN_RATIO)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    final_train_seqs = [train_normal_seqs[i] for i in train_idx]
    final_train_labels = [0] * len(final_train_seqs)

    val_seqs = [train_normal_seqs[i] for i in val_idx]
    val_labels = [0] * len(val_seqs)

    print(f"train={len(final_train_seqs)} val={len(val_seqs)} "
          f"test_normal={len(test_normal_seqs)} test_abnormal={len(test_anomaly_seqs)}")

    def write_sequences(sequences, labels, filepath, include_label=False):
        with open(filepath, "w") as f:
            for seq, label in zip(sequences, labels):
                f.write(" ".join(str(x) for x in seq))
                if include_label:
                    f.write("\t" + str(label))
                f.write("\n")

    write_sequences(final_train_seqs, final_train_labels,
                    os.path.join(Config.OUTPUT_DIR, Config.TRAIN_FILE), include_label=False)
    write_sequences(val_seqs, val_labels,
                    os.path.join(Config.OUTPUT_DIR, Config.VAL_FILE), include_label=False)
    write_sequences(test_normal_seqs, test_normal_labels,
                    os.path.join(Config.OUTPUT_DIR, Config.TEST_NORMAL_FILE), include_label=True)
    write_sequences(test_anomaly_seqs, test_anomaly_labels,
                    os.path.join(Config.OUTPUT_DIR, Config.TEST_ABNORMAL_FILE), include_label=True)

    print(f"wrote sequences to {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
