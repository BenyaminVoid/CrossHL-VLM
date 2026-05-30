import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from data import HSILidarDataset
from model.CrossHL_model import CrossHL_Transformer
from prompts import get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CrossHL-VLM ablation runs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", default="Trento")
    parser.add_argument("--root", default=".")
    parser.add_argument("--make-tsne", action="store_true")
    parser.add_argument("--tsne-pct", type=float, default=0.10)
    parser.add_argument(
        "--tsne-label",
        default=None,
        help="Optional split label for t-SNE, e.g. '20-shot'. Overrides --tsne-pct.",
    )
    parser.add_argument("--tsne-iter", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def read_summary(run_dir):
    path = Path(run_dir) / "summary.csv"
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["pct_label"], row["experiment"])
        groups[key].append(row)

    out = []
    for (dataset, pct_label, experiment), values in sorted(groups.items()):
        oa = np.array([float(v["oa"]) for v in values], dtype=float)
        aa = np.array([float(v["aa"]) for v in values], dtype=float)
        kappa = np.array([float(v["kappa"]) for v in values], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "pct_label": pct_label,
                "experiment": experiment,
                "n": len(values),
                "oa_mean": oa.mean(),
                "oa_std": oa.std(),
                "aa_mean": aa.mean(),
                "aa_std": aa.std(),
                "kappa_mean": kappa.mean(),
                "kappa_std": kappa.std(),
            }
        )
    return out


def paired_deltas(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["pct_label"], row["iteration"])
        grouped[key][row["experiment"]] = row

    deltas = []
    for (dataset, pct_label, iteration), values in sorted(grouped.items()):
        baseline = values.get("baseline")
        if baseline is None:
            continue
        base_oa = float(baseline["oa"])
        base_aa = float(baseline["aa"])
        base_kappa = float(baseline["kappa"])
        for experiment, row in sorted(values.items()):
            deltas.append(
                {
                    "dataset": dataset,
                    "pct_label": pct_label,
                    "iteration": iteration,
                    "experiment": experiment,
                    "oa": float(row["oa"]),
                    "aa": float(row["aa"]),
                    "kappa": float(row["kappa"]),
                    "delta_oa": float(row["oa"]) - base_oa,
                    "delta_aa": float(row["aa"]) - base_aa,
                    "delta_kappa": float(row["kappa"]) - base_kappa,
                }
            )
    return deltas


def summarize_deltas(delta_rows):
    groups = defaultdict(list)
    for row in delta_rows:
        groups[(row["dataset"], row["pct_label"], row["experiment"])].append(row)

    out = []
    for (dataset, pct_label, experiment), values in sorted(groups.items()):
        delta_oa = np.array([v["delta_oa"] for v in values], dtype=float)
        delta_aa = np.array([v["delta_aa"] for v in values], dtype=float)
        delta_kappa = np.array([v["delta_kappa"] for v in values], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "pct_label": pct_label,
                "experiment": experiment,
                "n": len(values),
                "delta_oa_mean": delta_oa.mean(),
                "delta_oa_std": delta_oa.std(),
                "delta_aa_mean": delta_aa.mean(),
                "delta_aa_std": delta_aa.std(),
                "delta_kappa_mean": delta_kappa.mean(),
                "delta_kappa_std": delta_kappa.std(),
            }
        )
    return out


def plot_oa_trends(aggregate_rows, out_dir):
    default_order = ["1%", "5%", "10%", "15%", "20%", "25%", "100%"]
    experiments = [
        "baseline",
        "name",
        "name_low_lambda",
        "spectral",
        "spectral_low_lambda",
        "spectral_lidar",
        "spectral_lidar_low_lambda",
        "spectral_lidar_high_lambda",
    ]
    styles = {
        "baseline": ("#d62728", "o", "-"),
        "name": ("#7f7f7f", "D", ":"),
        "name_low_lambda": ("#8c8c8c", "d", ":"),
        "spectral": ("#1f77b4", "s", "--"),
        "spectral_low_lambda": ("#17becf", "s", "--"),
        "spectral_lidar": ("#2ca02c", "^", "-."),
        "spectral_lidar_low_lambda": ("#98df8a", "^", "-."),
        "spectral_lidar_high_lambda": ("#9467bd", "P", "-"),
    }

    rows_by_key = {
        (row["pct_label"], row["experiment"]): row for row in aggregate_rows
    }
    labels = sorted({row["pct_label"] for row in aggregate_rows}, key=split_sort_key)
    order = [label for label in default_order if label in labels]
    order.extend(label for label in labels if label not in order)
    x_labels = [pct for pct in order if any((pct, exp) in rows_by_key for exp in experiments)]
    x = np.arange(len(x_labels))

    plt.figure(figsize=(9, 6))
    for exp in experiments:
        means = []
        stds = []
        for pct in x_labels:
            row = rows_by_key.get((pct, exp))
            means.append(np.nan if row is None else float(row["oa_mean"]))
            stds.append(0.0 if row is None else float(row["oa_std"]))
        if np.all(np.isnan(means)):
            continue
        color, marker, linestyle = styles[exp]
        plt.errorbar(
            x,
            means,
            yerr=stds,
            label=exp,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=7,
            capsize=5,
        )

    plt.xticks(x, x_labels)
    plt.xlabel("Few-shot setting")
    plt.ylabel("Overall accuracy (%)")
    plt.title("CrossHL-VLM ablation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "oa_trends.png", dpi=300)
    plt.close()


def load_model(checkpoint, dataset, root, device):
    train_dataset = HSILidarDataset(root, dataset=dataset, split="train")
    num_classes = len(torch.unique(train_dataset.lbls))
    model = CrossHL_Transformer(
        FM=16,
        NC=train_dataset.hs_image.shape[1],
        NCLidar=train_dataset.lidar_image.shape[1],
        Classes=num_classes,
        patchsize=train_dataset.hs_image.shape[-1],
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def extract_cls_features(model, loader, device):
    features = []
    labels = []
    with torch.no_grad():
        for hsi, lidar, label in loader:
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            _, feat = model(hsi, lidar, return_features=True)
            features.append(feat.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(features), np.concatenate(labels)


def split_sort_key(label):
    if label.endswith("-shot"):
        try:
            return (0, int(label.replace("-shot", "")))
        except ValueError:
            return (0, label)
    if label.endswith("%"):
        try:
            return (1, float(label.replace("%", "")))
        except ValueError:
            return (1, label)
    return (2, label)


def checkpoint_for(rows, experiment, split_label, iteration):
    for row in rows:
        if (
            row["experiment"] == experiment
            and row["pct_label"] == split_label
            and int(row["iteration"]) == iteration
        ):
            return Path(row["checkpoint"])
    return None


def plot_tsne(rows, args, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsne_label = args.tsne_label or f"{int(round(args.tsne_pct * 100))}%"
    class_names = get_class_names(args.dataset)
    test_dataset = HSILidarDataset(args.root, dataset=args.dataset, split="test")
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    experiments = ["baseline", "name", "spectral", "spectral_lidar"]
    sample_size = min(args.num_samples, len(test_dataset))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_dataset), sample_size, replace=False)

    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 5))
    if len(experiments) == 1:
        axes = [axes]

    for ax, experiment in zip(axes, experiments):
        checkpoint = checkpoint_for(rows, experiment, tsne_label, args.tsne_iter)
        if checkpoint is None or not checkpoint.exists():
            ax.set_title(f"Missing {experiment}")
            ax.axis("off")
            continue

        model = load_model(checkpoint, args.dataset, args.root, device)
        features, labels = extract_cls_features(model, loader, device)
        embedded = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            perplexity=min(30, max(5, sample_size // 20)),
        ).fit_transform(features[sample_idx])
        sampled_labels = labels[sample_idx]

        for class_id, class_name in enumerate(class_names):
            mask = sampled_labels == class_id
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                s=16,
                alpha=0.65,
                label=class_name,
            )
        ax.set_title(experiment)
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(class_names))
    fig.suptitle(
        f"CLS feature t-SNE, {tsne_label} labels, iteration {args.tsne_iter}"
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = tsne_label.replace("%", "pct").replace(" ", "_")
    plt.savefig(out_dir / f"tsne_cls_{safe_label}.png", dpi=300)
    plt.close()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    rows = read_summary(run_dir)
    aggregate_rows = aggregate(rows)
    delta_rows = paired_deltas(rows)
    delta_summary = summarize_deltas(delta_rows)
    write_csv(out_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(out_dir / "paired_deltas.csv", delta_rows)
    write_csv(out_dir / "paired_delta_summary.csv", delta_summary)
    plot_oa_trends(aggregate_rows, out_dir)
    if args.make_tsne:
        plot_tsne(rows, args, out_dir)
    print(f"Analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
