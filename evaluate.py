"""
evaluate.py — the heart of the project.

Given per-user (true_age, predicted_age) pairs, evaluate an age-assurance
system around the UK under-16 boundary.

Two things are computed:
  1. Binary performance AT the boundary (treat >=16 as adult).
  2. The CHALLENGE-AGE buffer sweep. This is the regulatory mechanism:
     a user is admitted without a second check only if predicted age >=
     challenge_age. Raising the challenge age above 16 trades friction
     (16+ users pushed to a step-up check) for safety (fewer under-16s
     slipping through). This sweep is the deliverable a regulator cares about.

The model is treated as a black box that outputs an age. Everything here
is model-agnostic, so it works identically on FG-NET, UTKFace, or a
commercial vendor's outputs.

Usage:
    python evaluate.py val_preds.csv
    python evaluate.py val_preds.csv --boundary 18 --target-leak 0.05
    python evaluate.py val_preds.csv --plot sweep.png
    python evaluate.py val_preds.csv --worst 20          # top-20 errors
    python evaluate.py val_preds.csv --leakers            # under-16s misclassified as adult
"""
import argparse
import csv
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Core metrics (unchanged — they are the spec)
# ---------------------------------------------------------------------------

def binary_at_boundary(true_age, pred_age, boundary=16):
    """Naive cut: predict 'adult' iff predicted age >= boundary."""
    true_age = np.asarray(true_age, float)
    pred_age = np.asarray(pred_age, float)
    true_adult = true_age >= boundary
    pred_adult = pred_age >= boundary

    acc = float((true_adult == pred_adult).mean())
    minors = ~true_adult
    adults = true_adult
    far = float((pred_adult & minors).sum() / max(minors.sum(), 1))
    frr = float((~pred_adult & adults).sum() / max(adults.sum(), 1))
    mae = float(np.abs(true_age - pred_age).mean())
    return {"accuracy": acc, "underage_pass_rate": far,
            "adult_block_rate": frr, "mae": mae}


def challenge_age_sweep(true_age, pred_age, boundary=16,
                        challenge_ages=range(16, 27)):
    """
    For each candidate challenge age c:
      - leak_rate:   fraction of TRUE under-16s admitted with no step-up
                     (predicted age >= c). This is the safety risk.
      - adult_stepup: fraction of TRUE 16+ users forced into a second
                     check (predicted < c). This is the friction cost.
      - total_stepup: overall fraction of users sent to step-up.
    """
    true_age = np.asarray(true_age, float)
    pred_age = np.asarray(pred_age, float)
    minor = true_age < boundary
    adult = ~minor
    rows = []
    for c in challenge_ages:
        admitted = pred_age >= c
        leak = float((admitted & minor).sum() / max(minor.sum(), 1))
        adult_stepup = float((~admitted & adult).sum() / max(adult.sum(), 1))
        total_stepup = float((~admitted).mean())
        rows.append({"challenge_age": int(c), "buffer_years": int(c - boundary),
                     "leak_rate": leak, "adult_stepup": adult_stepup,
                     "total_stepup": total_stepup})
    return rows


def recommend_buffer(sweep_rows, target_leak=0.01):
    """Smallest challenge age whose leak_rate is at or below the target."""
    ok = [r for r in sweep_rows if r["leak_rate"] <= target_leak]
    return ok[0] if ok else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(true_age, pred_age, boundary=16, target_leak=0.01):
    b = binary_at_boundary(true_age, pred_age, boundary)
    print(f"MAE: {b['mae']:.2f} yrs | binary acc @ {boundary}: "
          f"{b['accuracy']*100:.1f}%")
    print(f"  under-{boundary} wrongly admitted: {b['underage_pass_rate']*100:.1f}%")
    print(f"  {boundary}+ wrongly blocked:       {b['adult_block_rate']*100:.1f}%\n")
    sweep = challenge_age_sweep(true_age, pred_age, boundary)
    print(f"{'chal':>4} {'buf':>4} {'leak%':>7} {'adult step-up%':>15} "
          f"{'total step-up%':>16}")
    for r in sweep:
        print(f"{r['challenge_age']:>4} {r['buffer_years']:>4} "
              f"{r['leak_rate']*100:>6.1f} {r['adult_stepup']*100:>14.1f} "
              f"{r['total_stepup']*100:>15.1f}")
    rec = recommend_buffer(sweep, target_leak)
    if rec:
        print(f"\nTo keep under-{boundary} leak <= {target_leak*100:.0f}%: "
              f"challenge age {rec['challenge_age']} "
              f"({rec['buffer_years']}-yr buffer), "
              f"costing {rec['adult_stepup']*100:.0f}% of adults a step-up.")
    else:
        print(f"\nNo challenge age in range hits "
              f"leak <= {target_leak*100:.0f}%.")
    return sweep


# ---------------------------------------------------------------------------
# Per-image error analysis
# ---------------------------------------------------------------------------

def show_worst_errors(paths, true_age, pred_age, datasets, n=20):
    """Print the N images with the largest absolute error."""
    err = np.abs(pred_age - true_age)
    idx = np.argsort(err)[::-1][:n]
    print(f"\n── Top {n} worst absolute errors ──")
    print(f"{'path':<70} {'true':>5} {'pred':>6} {'err':>6}  {'dataset'}")
    for i in idx:
        ds = datasets[i] if datasets is not None else ""
        print(f"{paths[i]:<70} {true_age[i]:>5.0f} {pred_age[i]:>6.1f} "
              f"{err[i]:>6.1f}  {ds}")


def show_worst_leakers(paths, true_age, pred_age, datasets, boundary=16, n=20):
    """Print the N under-boundary images predicted oldest (the leakers)."""
    mask = true_age < boundary
    true_u = true_age[mask]
    pred_u = pred_age[mask]
    paths_u = np.array(paths)[mask]
    ds_u = np.array(datasets)[mask] if datasets is not None else None

    # Sort by predicted age descending — worst leakers first
    idx = np.argsort(pred_u)[::-1][:n]
    print(f"\n── Top {n} under-{boundary} images predicted as oldest (leakers) ──")
    print(f"{'path':<70} {'true':>5} {'pred':>6} {'err':>6}  {'dataset'}")
    for i in idx:
        ds = ds_u[i] if ds_u is not None else ""
        print(f"{paths_u[i]:<70} {true_u[i]:>5.0f} {pred_u[i]:>6.1f} "
              f"{pred_u[i] - true_u[i]:>6.1f}  {ds}")


def show_by_dataset(paths, true_age, pred_age, datasets, boundary=16):
    """Break down MAE and leak rate per dataset tag."""
    tags = sorted(set(datasets))
    if len(tags) <= 1:
        return
    print(f"\n── Per-dataset breakdown ──")
    print(f"{'dataset':<12} {'count':>6} {'MAE':>6}  "
          f"'under-{boundary} leak'")
    for tag in tags:
        mask = np.array([d == tag for d in datasets])
        t = true_age[mask]; p = pred_age[mask]
        mae = float(np.abs(t - p).mean())
        minors = t < boundary
        leak = float(((p >= boundary) & minors).sum() / max(minors.sum(), 1))
        print(f"  {tag:<10} {len(t):>6} {mae:>5.1f}  {leak*100:>5.1f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(sweep_rows, boundary, save_path):
    """Save a matplotlib figure of the challenge-age sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.", file=sys.stderr)
        return

    ages = [r["challenge_age"] for r in sweep_rows]
    leak = [r["leak_rate"] * 100 for r in sweep_rows]
    adult = [r["adult_stepup"] * 100 for r in sweep_rows]
    total = [r["total_stepup"] * 100 for r in sweep_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ages, leak, "o-", color="tab:red", linewidth=2,
            label=f"Under-{boundary} leak (safety risk)")
    ax.plot(ages, adult, "s-", color="tab:orange", linewidth=2,
            label=f"{boundary}+ step-up (friction)")
    ax.plot(ages, total, "D--", color="tab:blue", linewidth=1.5,
            label="Total step-up rate")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6,
               label="1% target leak")
    ax.axvline(boundary, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Challenge Age", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(f"Challenge-Age Buffer Sweep (boundary = {boundary})",
                 fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_ylim(bottom=-1)
    ax.set_xlim(min(ages) - 0.5, max(ages) + 0.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved sweep plot → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a CSV with true_age,pred_age columns (and optional path,error,dataset)."""
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"{path} is empty or missing columns.")

    true = np.array([float(row["true_age"]) for row in rows])
    pred = np.array([float(row["pred_age"]) for row in rows])

    paths = [row.get("path", "") for row in rows]
    datasets = [row.get("dataset", "") for row in rows]

    return true, pred, paths, datasets


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate age-assurance predictions against a boundary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python evaluate.py val_preds.csv\n"
               "  python evaluate.py val_preds.csv --boundary 18\n"
               "  python evaluate.py val_preds.csv --plot sweep.png\n"
               "  python evaluate.py val_preds.csv --worst 30\n"
               "  python evaluate.py val_preds.csv --leakers")
    ap.add_argument("preds_csv", help="CSV with true_age,pred_age columns.")
    ap.add_argument("--boundary", type=int, default=16,
                    help="Age boundary for binary classification (default: 16).")
    ap.add_argument("--target-leak", type=float, default=0.01,
                    help="Target max underage leak rate (default: 0.01 = 1%%).")
    ap.add_argument("--min-chal", type=int, default=16,
                    help="Start of challenge-age sweep (default: 16).")
    ap.add_argument("--max-chal", type=int, default=26,
                    help="End of challenge-age sweep, inclusive (default: 26).")
    ap.add_argument("--plot", default=None, metavar="PATH",
                    help="Save challenge-age sweep plot (requires matplotlib).")
    ap.add_argument("--save-csv", default=None, metavar="PATH",
                    help="Save sweep table rows to CSV.")
    ap.add_argument("--worst", type=int, default=0, metavar="N",
                    help="Show top-N worst absolute errors with paths.")
    ap.add_argument("--leakers", action="store_true",
                    help="Show worst under-age leakers (under-boundary predicted oldest).")
    args = ap.parse_args()

    true, pred, paths, datasets = load_csv(args.preds_csv)
    print(f"Loaded {len(true)} predictions from {args.preds_csv}")
    print(f"  true age range: {true.min():.0f} – {true.max():.0f}")
    print(f"  pred age range: {pred.min():.1f} – {pred.max():.1f}\n")

    sweep_rows = print_report(true, pred, args.boundary, args.target_leak)

    # Per-dataset breakdown (only meaningful if dataset column populated)
    if any(datasets):
        show_by_dataset(paths, true, pred, datasets, args.boundary)

    # Per-image error inspection
    if args.worst:
        show_worst_errors(paths, true, pred,
                          datasets if any(datasets) else None, args.worst)

    if args.leakers:
        show_worst_leakers(paths, true, pred,
                           datasets if any(datasets) else None, args.boundary)

    if args.plot:
        plot_sweep(sweep_rows, args.boundary, args.plot)

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"Saved sweep CSV → {args.save_csv}")


if __name__ == "__main__":
    main()
