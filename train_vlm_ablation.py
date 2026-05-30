import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader

from data import (
    HSILidarDataset,
    apply_spectral_perturbation,
    make_fewshot_subset,
    make_kshot_subset,
)
from model.CrossHL_model import CrossHL_Transformer
from prompts import (
    build_text_prototypes,
    get_class_names,
    prototype_similarity_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CrossHL-VLM few-shot ablations with CLIP text regularization."
    )
    parser.add_argument("--dataset", default="Trento")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pcts", nargs="+", type=float, default=None)
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[],
        help="Optional K-shot settings. Example: --shots 10 20 30.",
    )
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline", "name", "spectral", "spectral_lidar"],
        choices=[
            "baseline",
            "name",
            "name_low_lambda",
            "spectral",
            "spectral_low_lambda",
            "spectral_lidar",
            "spectral_lidar_low_lambda",
            "spectral_lidar_high_lambda",
        ],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 matmul/convolutions on NVIDIA Ampere+ GPUs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN settings. Slower, but stricter reproducibility.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--lambda-sem", type=float, default=0.01)
    parser.add_argument("--name-lambda", type=float, default=0.01)
    parser.add_argument("--low-lambda", type=float, default=0.003)
    parser.add_argument("--high-lambda", type=float, default=0.03)
    parser.add_argument(
        "--semantic-start-epoch",
        type=int,
        default=0,
        help="Start semantic regularization after this many epochs.",
    )
    parser.add_argument("--lambda-warmup-epochs", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--center-prototypes",
        action="store_true",
        help="Center CLIP text prototypes across classes before normalization.",
    )
    parser.add_argument("--freeze-pct-threshold", type=float, default=0.0)
    parser.add_argument("--spectral-noise-std", type=float, default=0.0)
    parser.add_argument("--spectral-gain-std", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--split-seed", type=int, default=14)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-best-by-test", action="store_true")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(args):
    torch.backends.cudnn.deterministic = bool(args.deterministic)
    torch.backends.cudnn.benchmark = not bool(args.deterministic)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        if args.allow_tf32:
            torch.set_float32_matmul_precision("high")


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size, shuffle, args):
    use_cuda = getattr(args, "device_type", "cpu") == "cuda"
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def fmt_float(value):
    return f"{value:g}"


def pct_tag(pct):
    return f"pct{int(round(pct * 100))}"


def split_tag(args):
    if getattr(args, "current_shots", None) is not None:
        return f"k{args.current_shots}"
    return pct_tag(args.current_pct)


def split_label(args):
    if getattr(args, "current_shots", None) is not None:
        return f"{args.current_shots}-shot"
    return f"{int(round(args.current_pct * 100))}%"


def experiment_config(name, args):
    configs = {
        "baseline": (None, 0.0),
        "name": ("name", args.name_lambda),
        "name_low_lambda": ("name", args.low_lambda),
        "spectral": ("spectral", args.lambda_sem),
        "spectral_low_lambda": ("spectral", args.low_lambda),
        "spectral_lidar": ("spectral_lidar", args.lambda_sem),
        "spectral_lidar_low_lambda": ("spectral_lidar", args.low_lambda),
        "spectral_lidar_high_lambda": ("spectral_lidar", args.high_lambda),
    }
    prompt_mode, lambda_sem = configs[name]
    tag = f"{name}_{split_tag(args)}_lam{fmt_float(lambda_sem)}"
    return tag, prompt_mode, lambda_sem


def experiment_prompt_mode(name):
    prompt_modes = {
        "baseline": None,
        "name": "name",
        "name_low_lambda": "name",
        "spectral": "spectral",
        "spectral_low_lambda": "spectral",
        "spectral_lidar": "spectral_lidar",
        "spectral_lidar_low_lambda": "spectral_lidar",
        "spectral_lidar_high_lambda": "spectral_lidar",
    }
    return prompt_modes[name]


def maybe_center_prototypes(prototypes):
    centered = prototypes - prototypes.mean(dim=0, keepdim=True)
    return F.normalize(centered, dim=-1)


def evaluate(model, loader, device, num_classes, use_amp=False):
    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for hsi, lidar, labels in loader:
            hsi = hsi.to(device, non_blocking=True)
            lidar = lidar.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(hsi, lidar)
            pred = torch.argmax(logits, dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    conf = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    class_totals = conf.sum(axis=1)
    per_class = np.full(num_classes, np.nan, dtype=np.float64)
    valid_classes = class_totals > 0
    per_class[valid_classes] = (
        np.diag(conf)[valid_classes] / class_totals[valid_classes]
    ) * 100.0
    aa = float(np.nanmean(per_class)) if np.any(valid_classes) else 0.0
    per_class = np.nan_to_num(per_class, nan=0.0)
    return {
        "oa": accuracy_score(y_true, y_pred) * 100.0,
        "aa": aa,
        "kappa": cohen_kappa_score(y_true, y_pred) * 100.0,
        "per_class": per_class,
        "confusion": conf,
    }


def copy_state_dict_to_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_epoch_log(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_run(
    args,
    exp_name,
    run_tag,
    prompt_mode,
    lambda_sem,
    train_loader,
    test_loader,
    text_prototypes,
    class_names,
    device,
    dims,
    checkpoint_dir,
    log_dir,
):
    set_seed(args.seed + args.iter_num)
    model = CrossHL_Transformer(
        FM=dims["FM"],
        NC=dims["NC"],
        NCLidar=dims["NCLidar"],
        Classes=dims["Classes"],
        patchsize=dims["patchsize"],
    ).to(device)

    freeze_early = (
        args.freeze_pct_threshold > 0
        and getattr(args, "current_pct", None) is not None
        and args.current_pct <= args.freeze_pct_threshold
    )
    if freeze_early:
        model.freeze_early_layers()

    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    loss_func = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    epoch_rows = []
    best_metric = -1.0
    best_state = None
    best_eval = None
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        if epoch < args.semantic_start_epoch:
            lambda_scale = 0.0
        elif args.lambda_warmup_epochs > 0:
            warmup_epoch = epoch - args.semantic_start_epoch + 1
            lambda_scale = min(1.0, float(warmup_epoch) / args.lambda_warmup_epochs)
        else:
            lambda_scale = 1.0
        current_lambda = lambda_sem * lambda_scale

        running_ce = 0.0
        running_sem = 0.0
        running_total = 0.0
        batches = 0

        for hsi, lidar, labels in train_loader:
            hsi = hsi.to(device, non_blocking=True)
            lidar = lidar.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            hsi = apply_spectral_perturbation(
                hsi,
                noise_std=args.spectral_noise_std,
                gain_std=args.spectral_gain_std,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                if text_prototypes is None or current_lambda <= 0:
                    logits = model(hsi, lidar)
                    loss_sem = torch.tensor(0.0, device=device)
                else:
                    logits, img_embed = model(hsi, lidar, return_embed=True)
                    semantic_logits = torch.matmul(img_embed, text_prototypes.T)
                    semantic_logits = semantic_logits / args.temperature
                    loss_sem = F.cross_entropy(semantic_logits, labels)

                loss_ce = loss_func(logits, labels)
                loss = loss_ce + current_lambda * loss_sem

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_ce += loss_ce.item()
            running_sem += loss_sem.item()
            running_total += loss.item()
            batches += 1

        scheduler.step()

        should_eval = (
            (epoch + 1) % args.eval_interval == 0
            or epoch == args.epochs - 1
            or args.epochs <= args.eval_interval
        )
        eval_result = None
        if should_eval:
            eval_result = evaluate(
                model,
                test_loader,
                device,
                len(class_names),
                use_amp=args.amp,
            )
            if args.save_best_by_test and eval_result["oa"] > best_metric:
                best_metric = eval_result["oa"]
                best_state = copy_state_dict_to_cpu(model)
                best_eval = eval_result

        row = {
            "epoch": epoch + 1,
            "experiment": exp_name,
            "run_tag": run_tag,
            "pct": args.current_pct,
            "shots": "" if getattr(args, "current_shots", None) is None else args.current_shots,
            "split_label": split_label(args),
            "iteration": args.iter_num,
            "lambda_sem": current_lambda,
            "loss_ce": running_ce / max(1, batches),
            "loss_sem": running_sem / max(1, batches),
            "loss_total": running_total / max(1, batches),
            "test_oa": "" if eval_result is None else eval_result["oa"],
            "test_aa": "" if eval_result is None else eval_result["aa"],
            "test_kappa": "" if eval_result is None else eval_result["kappa"],
        }
        epoch_rows.append(row)

        if should_eval:
            print(
                f"{run_tag} | iter {args.iter_num} | epoch {epoch + 1}: "
                f"CE={row['loss_ce']:.4f}, Sem={row['loss_sem']:.4f}, "
                f"OA={eval_result['oa']:.2f}"
            )

    if args.save_best_by_test and best_state is not None:
        model.load_state_dict(best_state)
        final_eval = best_eval
        selection = "best_test"
    else:
        final_eval = evaluate(
            model,
            test_loader,
            device,
            len(class_names),
            use_amp=args.amp,
        )
        selection = "final_epoch"

    elapsed = time.time() - start_time
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"net_params_CrossHL_{run_tag}_Iter{args.iter_num}.pkl"
    torch.save(model.state_dict(), checkpoint_path)

    log_dir.mkdir(parents=True, exist_ok=True)
    write_epoch_log(log_dir / f"epochs_{run_tag}_Iter{args.iter_num}.csv", epoch_rows)
    np.savetxt(
        log_dir / f"confusion_{run_tag}_Iter{args.iter_num}.csv",
        final_eval["confusion"],
        delimiter=",",
        fmt="%d",
    )

    summary = {
        "dataset": args.dataset,
        "experiment": exp_name,
        "prompt_mode": "" if prompt_mode is None else prompt_mode,
        "run_tag": run_tag,
        "pct": args.current_pct,
        "shots": "" if getattr(args, "current_shots", None) is None else args.current_shots,
        "pct_label": split_label(args),
        "split_label": split_label(args),
        "iteration": args.iter_num,
        "split_seed": args.current_split_seed,
        "model_seed": args.seed + args.iter_num,
        "lambda_sem": lambda_sem,
        "amp": args.amp and device.type == "cuda",
        "allow_tf32": args.allow_tf32 and device.type == "cuda",
        "device": str(device),
        "freeze_early": freeze_early,
        "selection": selection,
        "oa": final_eval["oa"],
        "aa": final_eval["aa"],
        "kappa": final_eval["kappa"],
        "train_samples": args.current_train_samples,
        "seconds": elapsed,
        "checkpoint": str(checkpoint_path),
    }
    for idx, class_name in enumerate(class_names):
        summary[f"acc_{class_name}"] = final_eval["per_class"][idx]

    return summary


def main():
    args = parse_args()
    configure_torch(args)
    if args.pcts is None:
        args.pcts = [] if args.shots else [0.01, 0.05, 0.10]
    if args.include_full and 1.0 not in args.pcts:
        args.pcts.append(1.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device_type = device.type
    print(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    print(
        f"AMP: {args.amp and device.type == 'cuda'} | "
        f"TF32: {args.allow_tf32 and device.type == 'cuda'} | "
        f"deterministic: {args.deterministic}"
    )
    class_names = get_class_names(args.dataset)
    train_full = HSILidarDataset(args.root, dataset=args.dataset, split="train")
    test_dataset = HSILidarDataset(args.root, dataset=args.dataset, split="test")
    test_loader = make_loader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        args=args,
    )

    dims = {
        "FM": 16,
        "NC": train_full.hs_image.shape[1],
        "NCLidar": train_full.lidar_image.shape[1],
        "Classes": len(torch.unique(train_full.lbls)),
        "patchsize": train_full.hs_image.shape[-1],
    }
    if dims["Classes"] != len(class_names):
        raise ValueError(
            f"Dataset labels contain {dims['Classes']} classes, but "
            f"{len(class_names)} class names are configured."
        )

    needs_clip = any(name != "baseline" for name in args.experiments)
    text_cache = {}
    if needs_clip:
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
        )
        tokenizer = open_clip.get_tokenizer(args.clip_model)
        clip_model = clip_model.to(device).eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        sim_path = run_dir / "prompt_similarity.csv"
        active_prompt_modes = {
            mode for mode in (experiment_prompt_mode(e) for e in args.experiments)
            if mode is not None
        }
        for prompt_mode in ["name", "spectral", "spectral_lidar"]:
            if prompt_mode in active_prompt_modes:
                prototypes = build_text_prototypes(
                    clip_model,
                    tokenizer,
                    args.dataset,
                    prompt_mode,
                    device,
                )
                stats = prototype_similarity_stats(prototypes)
                append_csv_row(
                    sim_path,
                    {
                        "dataset": args.dataset,
                        "prompt_mode": prompt_mode,
                        "prototype_transform": "raw",
                        "mean_offdiag": stats["mean_offdiag"],
                        "max_offdiag": stats["max_offdiag"],
                    },
                )
                if args.center_prototypes:
                    prototypes = maybe_center_prototypes(prototypes)
                    centered_stats = prototype_similarity_stats(prototypes)
                    append_csv_row(
                        sim_path,
                        {
                            "dataset": args.dataset,
                            "prompt_mode": prompt_mode,
                            "prototype_transform": "centered",
                            "mean_offdiag": centered_stats["mean_offdiag"],
                            "max_offdiag": centered_stats["max_offdiag"],
                        },
                    )
                    print(
                        f"{prompt_mode}: raw mean/max="
                        f"{stats['mean_offdiag']:.3f}/{stats['max_offdiag']:.3f}; "
                        f"centered mean/max="
                        f"{centered_stats['mean_offdiag']:.3f}/{centered_stats['max_offdiag']:.3f}"
                    )
                else:
                    print(
                        f"{prompt_mode}: mean inter-class cosine="
                        f"{stats['mean_offdiag']:.3f}, max={stats['max_offdiag']:.3f}"
                    )
                text_cache[prompt_mode] = prototypes

    summary_path = run_dir / "summary.csv"

    split_settings = [("pct", pct) for pct in args.pcts]
    split_settings.extend(("shot", shot) for shot in args.shots)

    for split_kind, split_value in split_settings:
        args.current_pct = split_value if split_kind == "pct" else None
        args.current_shots = split_value if split_kind == "shot" else None
        for iter_num in range(args.iterations):
            args.iter_num = iter_num
            args.current_split_seed = args.split_seed + iter_num
            if split_kind == "shot":
                train_subset, counts = make_kshot_subset(
                    train_full,
                    shots=split_value,
                    seed=args.current_split_seed,
                )
            else:
                train_subset, counts = make_fewshot_subset(
                    train_full,
                    pct=split_value,
                    seed=args.current_split_seed,
                )
            args.current_train_samples = len(train_subset)
            train_loader = make_loader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                args=args,
            )

            print(
                f"\n{args.dataset} | {split_tag(args)} | iter {iter_num} | "
                f"samples={len(train_subset)} | per_class={counts}"
            )

            for exp_name in args.experiments:
                run_tag, prompt_mode, lambda_sem = experiment_config(exp_name, args)
                text_prototypes = None
                if prompt_mode is not None:
                    text_prototypes = text_cache[prompt_mode].to(device)

                summary = train_one_run(
                    args,
                    exp_name=exp_name,
                    run_tag=run_tag,
                    prompt_mode=prompt_mode,
                    lambda_sem=lambda_sem,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    text_prototypes=text_prototypes,
                    class_names=class_names,
                    device=device,
                    dims=dims,
                    checkpoint_dir=run_dir / "checkpoints" / args.dataset,
                    log_dir=run_dir / "logs" / args.dataset,
                )
                append_csv_row(summary_path, summary)
                print(
                    f"FINAL {run_tag} Iter{iter_num}: "
                    f"OA={summary['oa']:.2f}, AA={summary['aa']:.2f}, "
                    f"Kappa={summary['kappa']:.2f}"
                )

    print(f"\nFinished. Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
