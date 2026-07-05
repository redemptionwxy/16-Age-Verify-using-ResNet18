"""
data.py — dataset parsing + leakage-safe splitting.

FG-NET filenames look like '001A02.JPG' -> subject 001, age 02.
  (some have a suffix letter, e.g. '001A02a.JPG'; handled below.)
UTKFace filenames look like '[age]_[gender]_[race]_[datetime].jpg'.

CRITICAL: FG-NET has ~82 subjects with many photos each. A random
image split leaks identity across train/test and inflates accuracy.
We split by SUBJECT so no person appears on both sides.
"""
import os
import re
import glob
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    _TORCH = True
except Exception:                       # lets evaluate/demo run without torch
    _TORCH = False


def parse_fgnet(root):
    """Return list of (path, age, subject_id)."""
    items = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            name = os.path.basename(p)
            m = re.match(r"(\d{3})A(\d{2})", name, re.IGNORECASE)
            if m:
                items.append((p, int(m.group(2)), m.group(1)))
    if not items:
        raise FileNotFoundError(f"No FG-NET images under {root}")
    return items


def parse_utkface(root):
    """Return list of (path, age, subject_id=path) — UTKFace has no subjects."""
    items = []
    for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
        m = re.match(r"(\d+)_\d+_\d+", os.path.basename(p))
        if m:
            items.append((p, int(m.group(1)), os.path.basename(p)))
    if not items:
        raise FileNotFoundError(f"No UTKFace images under {root}")
    return items


def subject_disjoint_split(items, test_frac=0.2, seed=0):
    """Split so each subject_id is entirely in train OR test.

    items may be 3-tuples (path, age, subject_id) or
    4-tuples (path, age, subject_id, tag) — subject_id is always index 2.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted({it[2] for it in items})
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_frac))
    test_subj = set(subjects[:n_test])
    train = [it for it in items if it[2] not in test_subj]
    test = [it for it in items if it[2] in test_subj]
    return train, test


if _TORCH:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def build_transforms(train=True, size=224):
        if train:
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    class AgeDataset(Dataset):
        def __init__(self, items, train=True, size=224):
            self.items = items          # each item: (path, age, subject_id, [dataset_tag])
            self.tf = build_transforms(train, size)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            item = self.items[i]
            path, age = item[0], item[1]
            img = Image.open(path).convert("RGB")
            return self.tf(img), torch.tensor([float(age)]), path
