# 16AgeVerify

## Goal: 

The challenge-age sweep with leak-rate/adult-friction trade-off is directly aligned with real regulation (UK Ofcom/ICO age-assurance guidance). Academic papers report MAE — they don't answer "how many under-16s get through vs. how many adults are annoyed?" This is the answer a regulator actually wants.

Age-assurance system for the UK under-16 boundary — predict age from a face photo, then determine whether a challenge-age buffer is needed to keep underage leaks ≤ 1%.

## What we've built

- train.py	

Dataset-aware training, age-weighted L1 loss, balanced sampling, AMP, best-checkpoint, --exclude-csv, per-age-bin logging
- evaluate.py	

Binary boundary metrics, challenge-age sweep, per-dataset breakdown, per-image error tracing (--worst, --leakers)
- extract_suspicious.py	

Copies misclassified images to review/ for manual label audit
- data.py	

Subject-disjoint split (no identity leakage), image-extension filter, path tracking

## Best results so far
Dataset |	MAE ↓	|Binary acc @ 16	| Under-16 leak @ 16
| --- | --- | --- | --- |
FG-NET only |	3.82 |	90.6%	| 10.8%
UTKFace only	| 4.62 |	98.1%	| 5.8%
Combined v2 |	4.62 |	97.9%	| 5.6%

Bottom line: at the raw boundary, ~5.6% of under-16s slip through. To hit ≤1% leak, we need a 10-year buffer (challenge age 26), which forces 32% of adults into a secondary check.
