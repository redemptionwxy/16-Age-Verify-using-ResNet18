"""
train.py — train the age regressor, then dump (true_age, pred_age, path) for
evaluate.py to run the challenge-age analysis and per-image error inspection.

Dataset-aware training:
  - FG-NET   (~1k images, ~82 subjects) → strong augmentation, longer training
  - UTKFace  (~24k images, no subject overlap) → standard, shorter
  - combined  (FG-NET + UTKFace) → balanced sampling, age-weighted loss

Key refinements (v2):
  - Age-weighted L1 loss: errors near the under-16 boundary cost more.
  - Balanced batch sampling: prevents larger datasets from dominating.
  - --exclude-csv: feed back manually-reviewed bad labels to skip.
  - Per-age-bin MAE logging: see which age ranges the model struggles with.

Usage:
    python train.py --dataset fgnet    --root Datasets/FGNET/images
    python train.py --dataset utkface  --root Datasets/UTKFace/UTKFace
    python train.py --dataset combined --fgnet-root Datasets/FGNET/images \\
                                        --utkface-root Datasets/UTKFace/UTKFace

    # With refinements:
    python train.py --dataset combined \\
        --fgnet-root Datasets/FGNET/images \\
        --utkface-root Datasets/UTKFace/UTKFace \\
        --boundary-weight 3.0 --balance \\
        --exclude-csv bad_labels.csv --name combined_v2

Then:
    python evaluate.py {name}_val_preds.csv
"""
import argparse
import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import (parse_fgnet, parse_utkface, subject_disjoint_split,
                  AgeDataset)
from model import build_model


# ---------------------------------------------------------------------------
# Dataset-specific defaults
# ---------------------------------------------------------------------------
DATASET_DEFAULTS = {
    "fgnet": {
        "epochs": 80, "lr": 1e-4,  "weight_decay": 1e-3,
        "bs": 16, "patience": 20,
        "description": "small (~1k images, ~82 subjects)",
    },
    "utkface": {
        "epochs": 20, "lr": 3e-4,  "weight_decay": 1e-4,
        "bs": 64, "patience": 7,
        "description": "large (~24k+ images, no subject overlap)",
    },
    "combined": {
        "epochs": 25, "lr": 3e-4,  "weight_decay": 1e-4,
        "bs": 64, "patience": 10,
        "description": "FG-NET + UTKFace (~25k images)",
    },
}

# Age bins for per-range logging
AGE_BINS = [(0, 11), (11, 16), (16, 25), (25, 45), (45, 120)]


# ---------------------------------------------------------------------------
# Age-weighted L1 loss — errors near the boundary cost more
# ---------------------------------------------------------------------------
def age_weighted_l1_loss(pred, target, boundary=16, weight=2.0, focus_width=15):
    """
    L1 loss where samples with true age within [boundary - focus_width,
    boundary + focus_width] get multiplied by `weight`.

    Default: ages 1–31 get 2× weight, tapering to 1× outside that range.
    Use --boundary-weight to adjust (1.0 = uniform L1).
    """
    base_loss = torch.abs(pred - target)
    if weight == 1.0:
        return base_loss.mean()

    t = target.detach()
    # Gaussian-style weighting centred on the boundary
    sigma = focus_width / 2.0                       # width of the bell
    w = 1.0 + (weight - 1.0) * torch.exp(-((t - boundary) ** 2) / (2 * sigma ** 2))
    return (base_loss * w).mean()


# ---------------------------------------------------------------------------
# Dataset loading with optional exclusion list
# ---------------------------------------------------------------------------
def load_dataset(args):
    """Return (train_items, val_items) — each item is (path, age, subj_id, tag)."""
    # --- load excluded paths ------------------------------------------------
    excluded = set()
    if args.exclude_csv:
        with open(args.exclude_csv, newline="") as f:
            for row in csv.DictReader(f):
                excluded.add(row["path"])
        print(f"  Excluded {len(excluded)} images from {args.exclude_csv}")

    if args.dataset == "combined":
        fg = [(p, a, s, "fgnet") for p, a, s in parse_fgnet(args.fgnet_root)
              if p not in excluded]
        ut = [(p, a, s, "utkface") for p, a, s in parse_utkface(args.utkface_root)
              if p not in excluded]
        fg_train, fg_val = subject_disjoint_split(fg, 0.2, seed=0)
        ut_train, ut_val = subject_disjoint_split(ut, 0.2, seed=0)
        train = fg_train + ut_train
        val   = fg_val   + ut_val
        print(f"  FG-NET   : {len(fg)} images ({len(fg_train)} train / "
              f"{len(fg_val)} val), {len({s for _,_,s,_ in fg})} subjects")
        print(f"  UTKFace  : {len(ut)} images ({len(ut_train)} train / "
              f"{len(ut_val)} val)")
    else:
        parse = parse_fgnet if args.dataset == "fgnet" else parse_utkface
        raw = [(p, a, s) for p, a, s in parse(args.root) if p not in excluded]
        items = [(p, a, s, args.dataset) for p, a, s in raw]
        train, val = subject_disjoint_split(items, 0.2, seed=0)

    return train, val


def make_balanced_sampler(items, boundary=16):
    """
    Build a WeightedRandomSampler that oversamples young faces and
    under-represented datasets.

    Strategy:
      - True age < boundary       → 3× base weight  (safety-critical range)
      - True age < boundary + 9   → 2× base weight  (boundary-adjacent)
      - FG-NET images             → 2× extra weight (small dataset, prevent drowning)
    """
    weights = []
    for item in items:
        age = item[1]
        tag = item[3] if len(item) > 3 else ""
        w = 1.0
        if age < boundary:
            w *= 3.0
        elif age < boundary + 9:
            w *= 2.0
        if tag == "fgnet":
            w *= 2.0
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def run(args):
    # --- resolve defaults --------------------------------------------------
    defaults = DATASET_DEFAULTS[args.dataset]
    epochs   = args.epochs        if args.epochs        is not None else defaults["epochs"]
    lr       = args.lr            if args.lr            is not None else defaults["lr"]
    wd       = args.weight_decay  if args.weight_decay  is not None else defaults["weight_decay"]
    bs       = args.bs            if args.bs            is not None else defaults["bs"]
    patience = args.patience      if args.patience      is not None else defaults["patience"]
    name     = args.name          if args.name          is not None else args.dataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load --------------------------------------------------------------
    print(f"Dataset: {args.dataset} ({defaults['description']})")
    train_items, val_items = load_dataset(args)

    n_subjects = len({s for _, _, s, _ in train_items + val_items})
    print(f"  images    : {len(train_items) + len(val_items)} total "
          f"({len(train_items)} train / {len(val_items)} val)")
    print(f"  subjects  : {n_subjects} (split is subject-disjoint)")
    print(f"  device    : {dev}")
    print(f"  backbone  : {args.backbone}")
    print(f"  epochs    : {epochs}  |  lr: {lr}  |  wd: {wd}  |  bs: {bs}")
    print(f"  patience  : {patience}  |  amp: {args.amp}")
    print(f"  loss      : age-weighted L1 (boundary_weight={args.boundary_weight})")
    print(f"  balance   : {args.balance}")
    print(f"  output    : {name}_model.pt  |  {name}_val_preds.csv")

    if args.dataset == "utkface":
        aligned = os.path.join(args.root, "utkface_aligned_cropped")
        if os.path.isdir(aligned):
            print("  ⚠ WARNING: utkface_aligned_cropped/ found under root.")

    # --- data loaders ------------------------------------------------------
    train_ds = AgeDataset(train_items, train=True)
    sampler = make_balanced_sampler(train_items) if args.balance else None
    tr = DataLoader(train_ds, batch_size=bs,
                    sampler=sampler, shuffle=(sampler is None),
                    num_workers=args.workers,
                    pin_memory=(dev == "cuda"))
    va = DataLoader(AgeDataset(val_items, train=False), batch_size=bs,
                    shuffle=False, num_workers=args.workers,
                    pin_memory=(dev == "cuda"))

    # --- model, optimiser, scheduler, loss --------------------------------
    net = build_model(args.backbone).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(epochs // 4, 5), T_mult=2, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda") if args.amp and dev == "cuda" else None

    # --- training loop -----------------------------------------------------
    best_mae = float("inf")
    best_epoch = 0
    stale = 0

    for ep in range(epochs):
        # ---- train ----
        net.train()
        train_loss = 0.0
        for x, y, _ in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = age_weighted_l1_loss(
                        net(x), y,
                        weight=args.boundary_weight)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = age_weighted_l1_loss(
                    net(x), y,
                    weight=args.boundary_weight)
                loss.backward()
                opt.step()
            train_loss += loss.item() * x.size(0)
        sched.step()
        train_loss /= len(tr.dataset)

        # ---- validate ----
        net.eval()
        ts, ps, paths = [], [], []
        with torch.no_grad():
            for x, y, p in va:
                pred = net(x.to(dev)).cpu().numpy().ravel()
                ts.extend(y.numpy().ravel())
                ps.extend(pred)
                paths.extend(p)
        ts_arr = np.array(ts)
        ps_arr = np.array(ps)
        mae = float(np.abs(ts_arr - ps_arr).mean())

        # Per-age-bin MAE
        bin_strs = []
        for lo, hi in AGE_BINS:
            mask = (ts_arr >= lo) & (ts_arr < hi)
            if mask.sum() > 0:
                bin_mae = float(np.abs(ts_arr[mask] - ps_arr[mask]).mean())
                bin_strs.append(f"[{lo:>2}-{hi:<3}]:{bin_mae:.2f}")

        lr_now = opt.param_groups[0]["lr"]
        marker = ""
        if mae < best_mae:
            best_mae = mae
            best_epoch = ep + 1
            stale = 0
            torch.save(net.state_dict(), f"{name}_model.pt")
            marker = " ★"
        else:
            stale += 1

        print(f"epoch {ep+1:>3}/{epochs}  "
              f"loss {train_loss:.3f}  "
              f"val_mae {mae:.2f}  "
              f"lr {lr_now:.2e}{marker}  "
              f"{' '.join(bin_strs)}")

        if stale >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    # --- finalise -----------------------------------------------------------
    print(f"\nBest val MAE: {best_mae:.2f} yrs @ epoch {best_epoch}")

    net.load_state_dict(torch.load(f"{name}_model.pt", map_location=dev))

    # Full validation pass for final predictions
    net.eval()
    ts, ps, paths = [], [], []
    with torch.no_grad():
        for x, y, p in va:
            pred = net(x.to(dev)).cpu().numpy().ravel()
            ts.extend(y.numpy().ravel())
            ps.extend(pred)
            paths.extend(p)

    path_tag = {}
    for it in val_items:
        path_tag[it[0]] = it[3] if len(it) > 3 else ""

    csv_path = f"{name}_val_preds.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_age", "pred_age", "error", "dataset"])
        for p, t, pred in zip(paths, ts, ps):
            err = pred - t
            w.writerow([p, t, pred, round(err, 2), path_tag.get(p, "")])

    print(f"Saved {csv_path} ({len(ts)} rows)")
    print(f"→ Run:  python evaluate.py {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train an age regressor with dataset-aware defaults.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python train.py --dataset fgnet --root Datasets/FGNET/images\n"
               "  python train.py --dataset utkface --root Datasets/UTKFace/UTKFace\n"
               "  python train.py --dataset combined \\\n"
               "      --fgnet-root Datasets/FGNET/images \\\n"
               "      --utkface-root Datasets/UTKFace/UTKFace \\\n"
               "      --boundary-weight 3.0 --balance")

    ap.add_argument("--dataset", choices=["fgnet", "utkface", "combined"],
                    required=True)
    ap.add_argument("--root", default=None,
                    help="Path to image directory (fgnet / utkface only).")
    ap.add_argument("--fgnet-root", default=None,
                    help="Path to FG-NET images (combined only).")
    ap.add_argument("--utkface-root", default=None,
                    help="Path to UTKFace images (combined only).")
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "mobilenet"])
    ap.add_argument("--name", default=None,
                    help="Output file prefix (default: dataset name).")
    # Hyperparams
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    # Refinement flags
    ap.add_argument("--boundary-weight", type=float, default=2.0,
                    help="Weight multiplier for ages near boundary "
                         "(1.0=uniform, 3.0=strong). Default: 2.0.")
    ap.add_argument("--balance", action="store_true",
                    help="Use WeightedRandomSampler to oversample young faces "
                         "and minority datasets.")
    ap.add_argument("--exclude-csv", default=None,
                    help="CSV of manually-reviewed bad labels to exclude "
                         "(must have a 'path' column).")
    args = ap.parse_args()

    if args.dataset == "combined":
        if not args.fgnet_root or not args.utkface_root:
            ap.error("--dataset combined requires --fgnet-root AND --utkface-root")
    else:
        if not args.root:
            ap.error(f"--dataset {args.dataset} requires --root")

    run(args)
