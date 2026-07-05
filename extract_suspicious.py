"""
extract_suspicious.py — copy suspicious predictions into a review/ folder
so you can manually inspect the images and verify labels.

Three error categories are extracted:

  1. extreme_error   — |pred - true| >= THRESHOLD (default 20 yrs).
                       These are almost certainly mislabeled.

  2. under16_leakers — true age < 16  but predicted >= LEAK_AGE (default 21).
                       The dangerous cases an age-assurance system misses.
                       (challenge-age 21 = 5-yr buffer, a reasonable cutoff)

  3. over60_dropped  — true age >= 60 but predicted <= DROP_AGE (default 25).
                       Likely mislabeled older adults.

Images are copied (not moved) into review/<category>/ with the error info
baked into the filename for easy sorting.

Usage:
    python extract_suspicious.py combined_val_preds.csv
    python extract_suspicious.py fgnet_val_preds.csv --error-threshold 15
    python extract_suspicious.py utkface_val_preds.csv --leak-age 20
"""
import argparse
import csv
import os
import shutil
import sys


def extract(args):
    out_dir = args.output or "review"
    categories = {
        "extreme_error": [],
        "under16_leakers": [],
        "over60_dropped": [],
    }

    # --- load predictions --------------------------------------------------
    with open(args.preds_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} predictions from {args.preds_csv}")

    # --- classify ----------------------------------------------------------
    for row in rows:
        true_age = float(row["true_age"])
        pred_age = float(row["pred_age"])
        path = row.get("path", "")
        err = pred_age - true_age
        abs_err = abs(err)

        if not path or not os.path.exists(path):
            continue

        tag = row.get("dataset", "")

        # Extreme absolute error → likely mislabeled
        if abs_err >= args.error_threshold:
            categories["extreme_error"].append((path, true_age, pred_age, err, tag))

        # Under-16 predicted significantly above boundary → dangerous leaker
        if true_age < 16 and pred_age >= args.leak_age:
            categories["under16_leakers"].append((path, true_age, pred_age, err, tag))

        # Older adults predicted as very young → likely mislabeled
        if true_age >= 60 and pred_age <= args.drop_age:
            categories["over60_dropped"].append((path, true_age, pred_age, err, tag))

    # --- copy files --------------------------------------------------------
    total_copied = 0
    for cat_name, items in categories.items():
        if not items:
            print(f"\n  {cat_name}: 0 images (none matched)")
            continue

        cat_dir = os.path.join(out_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        # Sort by absolute error descending
        items.sort(key=lambda x: abs(x[3]), reverse=True)

        print(f"\n  {cat_name}: {len(items)} images → {cat_dir}/")
        for path, true, pred, err, tag in items:
            # Build informative filename:
            #   original: 042A28.JPG
            #   becomes:   fgnet_T6_P24_E18_042A28.JPG
            base = os.path.basename(path)
            stem, ext = os.path.splitext(base)
            new_name = f"{tag}_T{int(true)}_P{int(pred)}_E{int(abs(err))}_{stem}{ext}"
            dst = os.path.join(cat_dir, new_name)
            if not os.path.exists(dst):
                shutil.copy2(path, dst)
                total_copied += 1

    # --- summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Copied {total_copied} images to {os.path.abspath(out_dir)}/")
    print(f"Categories:")
    for cat_name, items in categories.items():
        if items:
            print(f"  {cat_name}/  — {len(items)} images")

    # Quick stats on each category
    if categories["extreme_error"]:
        ages_true = [it[1] for it in categories["extreme_error"]]
        print(f"\n  extreme_error true-age range: {min(ages_true):.0f}–{max(ages_true):.0f}")
        tags = set(it[4] for it in categories["extreme_error"])
        print(f"  datasets affected: {tags}")

    if categories["under16_leakers"]:
        tags = set(it[4] for it in categories["under16_leakers"])
        print(f"\n  under16_leakers dataset breakdown:")
        for t in sorted(tags):
            count = sum(1 for it in categories["under16_leakers"] if it[4] == t)
            print(f"    {t}: {count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract suspicious predictions into a review folder.")
    ap.add_argument("preds_csv", help="CSV from train.py with path,true_age,pred_age columns.")
    ap.add_argument("--output", "-o", default="review",
                    help="Output directory (default: review/).")
    ap.add_argument("--error-threshold", type=float, default=20,
                    help="Absolute error >= this → extreme_error (default: 20 yrs).")
    ap.add_argument("--leak-age", type=float, default=21,
                    help="True age < 16 & pred >= this → under16_leakers (default: 21).")
    ap.add_argument("--drop-age", type=float, default=25,
                    help="True age >= 60 & pred <= this → over60_dropped (default: 25).")
    extract(ap.parse_args())
