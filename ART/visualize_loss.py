import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _first_present(item, keys):
    for key in keys:
        if key in item:
            return item[key]
    return None


def _to_dataframe(log_history, train_keys, eval_keys, train_scale=1.0):
    rows = []

    for item in log_history:
        epoch = item.get("epoch")
        if epoch is None:
            continue

        train_loss = _first_present(item, train_keys)
        eval_loss = _first_present(item, eval_keys)

        if train_loss is not None:
            rows.append({
                "epoch": epoch,
                "split": "train",
                "loss": train_loss / train_scale if epoch != 0 else train_loss
            })

        if eval_loss is not None:
            rows.append({
                "epoch": epoch,
                "split": "valid",
                "loss": eval_loss
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(["split", "epoch"]).reset_index(drop=True)


def _set_xlim(axis, series):
    if not series.empty:
        axis.set_xlim(0, series.max())


def _plot_train_valid_subplots(df, title, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    split_to_axis = {"train": axes[0], "valid": axes[1]}

    for split, axis in split_to_axis.items():
        part = df[df["split"] == split]
        color = "tab:blue" if split == "train" else "tab:orange"

        if part.empty:
            axis.text(0.5, 0.5, f"No {split} data", ha="center", va="center")
            axis.set_ylabel("Loss")
            axis.grid(True, linestyle="--", alpha=0.4)
            continue

        # line
        axis.plot(
            part["epoch"],
            part["loss"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=split,
        )

        # best point
        best_idx = part["loss"].idxmin()
        best_point = part.loc[best_idx]

        axis.scatter(
            best_point["epoch"],
            best_point["loss"],
            color="red",
            zorder=5,
            label=f"best={best_point['loss']:.4f}",
        )

        # xlim
        _set_xlim(axis, part["epoch"])

        axis.set_title(f"{title} - {split.capitalize()}")
        axis.set_ylabel("Loss")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

    axes[-1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_combined(df, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))

    for split, color in (("train", "tab:blue"), ("valid", "tab:orange")):
        part = df[df["split"] == split]
        if part.empty:
            continue

        ax.plot(
            part["epoch"],
            part["loss"],
            marker="o",
            linewidth=1.8,
            label=split,
            color=color,
        )

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        _set_xlim(ax, df["epoch"])

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize train/valid loss curves from Hugging Face Trainer state.json"
    )
    parser.add_argument("--input", required=True, help="Path to state.json")
    parser.add_argument("--output-dir", required=True, help="Directory to save output plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    total_df = _to_dataframe(
        log_history,
        train_keys=["loss", "initial_train_loss"],
        eval_keys=["eval_loss", "initial_eval_loss"],
        train_scale=32.0,
    )

    cls_df = _to_dataframe(
        log_history,
        train_keys=["train_loss_cls", "initial_train_loss_cls"],
        eval_keys=["eval_loss_cls", "initial_eval_loss_cls"],
    )

    l1_df = _to_dataframe(
        log_history,
        train_keys=["train_loss_l1", "initial_train_loss_l1"],
        eval_keys=["eval_loss_l1", "initial_eval_loss_l1"],
    )

    mlm_df = _to_dataframe(
        log_history,
        train_keys=["train_loss_mlm", "initial_train_loss_mlm"],
        eval_keys=["eval_loss_mlm", "initial_eval_loss_mlm"],
    )

    # bật/tắt tùy nhu cầu
    _plot_train_valid_subplots(total_df, "Total Loss", output_dir / "total_loss.png")
    _plot_train_valid_subplots(cls_df, "CLS Loss", output_dir / "cls_loss.png")
    _plot_combined(l1_df, "L1 Loss", output_dir / "l1_loss.png")
    _plot_combined(mlm_df, "Causal Loss", output_dir / "mlm_loss.png")

    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()