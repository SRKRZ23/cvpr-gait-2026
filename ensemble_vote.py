"""
CVPR 2026 Children Gait Challenge
Multi-Model Ensemble Consensus Voting

Aggregates predictions from multiple independent video-based models.
High-confidence consensus (>= threshold) used for final item assignment.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
SUBMISSIONS.mkdir(exist_ok=True)

ITEMS = list(range(1, 18))
T1_TEST = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
T2_TEST = [4, 6, 7, 13, 26, 35, 39, 42, 50]


def load_csv_predictions(csv_path):
    """Load a submission CSV into a dict: patient_id -> item -> value."""
    preds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["ID"]
            if not pid.startswith("track1-"):
                continue
            patient_num = int(pid.split("-")[1])
            preds[patient_num] = {}
            for side, prefix in [("left", "L"), ("right", "R")]:
                preds[patient_num][side] = {}
                for k in ITEMS:
                    val = row.get(f"{prefix}{k}", "0")
                    try:
                        preds[patient_num][side][k] = int(val)
                    except (ValueError, TypeError):
                        preds[patient_num][side][k] = 0
    return preds


def consensus_vote(model_preds_list, threshold=0.5):
    """
    Majority vote across models.
    Returns: {patient_id: {side: {item: voted_value}}}
    """
    result = {}
    for pid in T1_TEST:
        result[pid] = {"left": {}, "right": {}}
        for side in ["left", "right"]:
            for k in ITEMS:
                votes = []
                for model_preds in model_preds_list:
                    if pid in model_preds and side in model_preds[pid]:
                        votes.append(model_preds[pid][side].get(k, 0))
                if votes:
                    vote_rate = sum(votes) / len(votes)
                    result[pid][side][k] = 1 if vote_rate >= threshold else 0
                else:
                    result[pid][side][k] = 0
    return result


def load_track2_predictions(csv_path):
    """Load Track 2 gait subtype predictions."""
    preds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["ID"]
            if not pid.startswith("track2-"):
                continue
            patient_num = int(pid.split("-")[1])
            preds[patient_num] = {
                "left": row.get("Left_gait_subtype", "type3"),
                "right": row.get("Right_gait_subtype", "type3"),
            }
    return preds


def build_final_csv(t1_preds, t2_preds, outpath):
    """Build final submission CSV."""
    header = (["ID"] +
              [f"L{k}" for k in ITEMS] +
              [f"R{k}" for k in ITEMS] +
              ["Total", "Left_gait_subtype", "Right_gait_subtype"])

    rows = []

    # Track 1
    for pid in sorted(T1_TEST):
        L = [t1_preds[pid]["left"].get(k, 0) for k in ITEMS]
        R = [t1_preds[pid]["right"].get(k, 0) for k in ITEMS]
        total = sum(L) + sum(R)
        rows.append([f"track1-{pid}"] + L + R + [total, -1, -1])

    # Track 2
    for pid in sorted(T2_TEST):
        if pid in t2_preds:
            left_type = t2_preds[pid]["left"]
            right_type = t2_preds[pid]["right"]
        else:
            left_type = right_type = "type3"
        rows.append([f"track2-{pid}"] + [-1]*17 + [-1]*17 + [-1, left_type, right_type])

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Submission written: {outpath}")


def compute_model_agreement(model_preds_list):
    """Report per-patient, per-item agreement rate across models."""
    print("\n=== Model Agreement Analysis ===")
    for pid in T1_TEST:
        for side in ["left", "right"]:
            low_agreement_items = []
            for k in ITEMS:
                votes = [
                    m[pid][side].get(k, 0)
                    for m in model_preds_list
                    if pid in m and side in m[pid]
                ]
                if votes:
                    agree = max(sum(votes), len(votes) - sum(votes)) / len(votes)
                    if agree < 0.75:
                        low_agreement_items.append((k, f"{agree:.0%}"))
            if low_agreement_items:
                print(f"  P{pid} {side}: low agreement items: {low_agreement_items}")


if __name__ == "__main__":
    # Load all available model predictions
    model_files = sorted(ROOT.glob("models/*.csv"))
    if not model_files:
        print("No model CSV files found in models/. Using final submission directly.")
        # Use the pre-built final submission
        import shutil
        src = ROOT / "submissions" / "final_submission.csv"
        if src.exists():
            print(f"Final submission already exists: {src}")
        else:
            print("Please add model CSV files to models/ directory.")
        exit(0)

    print(f"Loading {len(model_files)} model predictions...")
    model_preds_list = []
    for f in model_files:
        preds = load_csv_predictions(f)
        model_preds_list.append(preds)
        print(f"  Loaded: {f.name} ({len(preds)} patients)")

    # Compute agreement analysis
    compute_model_agreement(model_preds_list)

    # Majority vote
    print("\n=== Consensus Voting (threshold=0.5) ===")
    voted = consensus_vote(model_preds_list, threshold=0.5)

    # Load Track 2 from best available source
    t2_src = ROOT / "submissions" / "final_submission.csv"
    t2_preds = load_track2_predictions(t2_src) if t2_src.exists() else {}

    # Build output
    outpath = SUBMISSIONS / "ensemble_submission.csv"
    build_final_csv(voted, t2_preds, outpath)

    print("\nDone. Review ensemble_submission.csv before submitting.")
