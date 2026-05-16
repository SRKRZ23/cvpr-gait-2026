"""
CVPR 2026 Children Gait Challenge
Training Data Analysis

Computes per-type EVGS statistics, item rates, and training-test overlap.
Run this first to understand the dataset before building predictions.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

ITEMS = list(range(1, 18))
T1_TEST = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
T2_TEST = [4, 6, 7, 13, 26, 35, 39, 42, 50]

with open(DATA / "track1_train.json") as f:
    t1_train = json.load(f)
with open(DATA / "track2_train.json") as f:
    t2_train = json.load(f)

t1_by_id = {p["patient_id"]: p for p in t1_train}
t2_by_id = {p["patient_id"]: p for p in t2_train}

print(f"Track 1 training: {len(t1_train)} patients")
print(f"Track 2 training: {len(t2_train)} patients")

# ── Training-test overlap ─────────────────────────────────────────────────────
t1_test_known_type = {
    pid: {
        "left": t2_by_id[pid]["left"]["gait_subtype"],
        "right": t2_by_id[pid]["right"]["gait_subtype"],
    }
    for pid in T1_TEST if pid in t2_by_id
}

print(f"\nTrack 1 test patients with known gait type ({len(t1_test_known_type)}):")
for pid, sides in sorted(t1_test_known_type.items()):
    print(f"  P{pid}: L={sides['left']}, R={sides['right']}")

t2_test_known_evgs = [pid for pid in T2_TEST if pid in t1_by_id]
print(f"\nTrack 2 test patients with Track 1 training data ({len(t2_test_known_evgs)}):")
print(f"  {t2_test_known_evgs}")

# ── Per-type statistics ───────────────────────────────────────────────────────
overlap = set(t1_by_id.keys()) & set(t2_by_id.keys())
type_vecs = defaultdict(list)
for pid in overlap:
    for side in ["left", "right"]:
        gtype = t2_by_id[pid][side]["gait_subtype"]
        vec = np.array([t1_by_id[pid][side][str(k)] for k in ITEMS])
        type_vecs[gtype].append(vec)

print(f"\n=== Per-Type EVGS Statistics (N={len(overlap)} overlapping patients) ===")
for gtype in sorted(type_vecs.keys()):
    vecs = np.array(type_vecs[gtype])
    means = vecs.mean(axis=0)
    n = len(vecs)
    total_mean = vecs.sum(axis=1).mean()
    total_std = vecs.sum(axis=1).std()
    print(f"\n{gtype} (N={n} sides, avg_total={total_mean:.1f} ± {total_std:.1f}):")
    elevated = [(i+1, f"{m:.2f}") for i, m in enumerate(means) if m >= 0.3]
    print(f"  Items with P(deviation)>=0.3: {elevated}")

# ── Global item statistics ────────────────────────────────────────────────────
print("\n=== Global Per-Item Deviation Rates ===")
item_pos = defaultdict(int)
item_total = defaultdict(int)
for p in t1_train:
    for side in ["left", "right"]:
        for k in ITEMS:
            item_total[k] += 1
            item_pos[k] += p[side][str(k)]

print("Item  Rate  Majority")
for k in ITEMS:
    rate = item_pos[k] / item_total[k]
    majority = 1 if rate >= 0.5 else 0
    print(f"  {k:2d}   {rate:.3f}   {majority}")

# ── Total score distribution ──────────────────────────────────────────────────
print("\n=== Patient Total Score Distribution ===")
patient_totals = []
for p in t1_train:
    total = sum(p["left"][str(k)] + p["right"][str(k)] for k in ITEMS)
    patient_totals.append(total)
arr = np.array(patient_totals)
print(f"  N={len(arr)}, mean={arr.mean():.1f}, std={arr.std():.1f}, "
      f"min={arr.min()}, max={arr.max()}, median={np.median(arr):.1f}")
