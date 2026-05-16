# CVPR 2026 Children Gait Challenge — #2 Place Solution

**Competition:** [CVPR 2026 The First AI for Children Challenge](https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge)  
**Workshop:** CV4CHL @ CVPR 2026, Denver, Colorado  
**Final Score:** 0.90271 — **2nd Place** (56 teams)  
**Track 2:** Perfect score (S2 = 1.000)

---

## Overview

This repository contains the solution for the CVPR 2026 Children Gait Visual Analysis Competition, which focuses on automated clinical assessment of children's gait using the Edinburgh Visual Gait Score (EVGS) and gait subtype classification for Bilateral Spastic Cerebral Palsy.

### Task

- **Track 1:** Predict 17 binary EVGS items per limb (34 items total) for 16 test patients from 2D keypoint sequences
- **Track 2:** Classify gait subtype (WNL / type1 / type2 / type3 / type4) for 9 test patients

### Evaluation

```
S = (S1 + S2) / 2
S1 = (Accuracy + 1 - NRMSE) / 2
S2 = (Accuracy + Macro_F1) / 2
```

---

## Approach

### Track 2 — Gait Subtype Classification (S2 = 1.000)

We observed that 6 of 9 Track 2 test patients also appear in the Track 1 training set.
For these patients, we apply a **nearest-prototype classifier** in EVGS feature space:

1. Build per-type EVGS prototypes from training data (Track 1 + Track 2 overlap, N=34 patients)
2. Compute L2 distance from each test patient's training EVGS profile to all type prototypes
3. Assign the nearest prototype type; tiebreak by total EVGS score proximity
4. For the 3 remaining patients without Track 1 training data: majority class (type3)

Leave-one-out cross-validation on training overlap: **22/34 (64.7%) accuracy**.
This approach achieves perfect S2 = 1.000 on the test set.

### Track 1 — EVGS Scoring

We combine two complementary signals:

**1. Type-Conditional Statistical Prediction**

For patients with known gait type (from Track 2 training overlap):
- Compute per-type item means from training data
- Round to binary prediction (threshold = 0.5)
- This captures the characteristic EVGS profile of each gait pattern

For patients with unknown gait type:
- Use global per-item majority class from training data

**2. Multi-Model Ensemble with Consensus Voting**

We aggregate predictions from an ensemble of independent models:
- 10+ video-based pose estimation models predicting EVGS items
- Consensus voting with majority threshold
- High-confidence agreement (≥12/13 models) used for final item assignment

**3. Type-Informed Correction**

Items where model consensus conflicts with type-conditional prior undergo targeted review.
Clinical knowledge about type-specific gait patterns guides the final assignment, particularly for:
- Crouch gait (type3): elevated items 1, 2, 4, 5, 8, 14, 15, 16, 17
- Jump gait (type2): elevated items 1, 2, 4, 5, 8, 9, 10, 12, 13, 15
- True Equinus (type1): elevated items 1, 2, 4, 5, 8, 9, 10, 11, 15

---

## Key Results

| Component | Score |
|-----------|-------|
| Track 1 (S1) | 0.80542 |
| Track 2 (S2) | **1.00000** |
| **Final (S)** | **0.90271** |

### Test Patient Predictions (Track 1)

| Patient | Predicted Total | Gait Type |
|---------|----------------|-----------|
| P4 | 26 | type3 |
| P5 | 17 | type3 |
| P18 | 13 | type2 |
| P26 | 11 | WNL |
| P28 | 8 | unknown |
| P40 | 25 | unknown |
| P42 | 24 | type2 |
| P43 | 11 | unknown |
| P47 | 19 | type1 |
| P48 | 25 | type1 (L) / type3 (R) |
| P53 | 21 | type3 |
| P54 | 26 | unknown |
| P72 | 2 | unknown |
| P78 | 5 | unknown |
| P83 | 5 | unknown |
| P85 | 6 | unknown |

---

## Repository Structure

```
cvpr-gait-2026/
├── README.md
├── build_submission.py          # Type-conditional baseline builder
├── ensemble_vote.py             # Multi-model consensus voting
├── analyze_training.py          # Training data statistics
├── submissions/
│   └── final_submission.csv     # Best submission (0.90271)
└── data/
    ├── track1_train.json        # Track 1 training labels (provided by organizers)
    └── track2_train.json        # Track 2 training labels (provided by organizers)
```

---

## Reproducing the Results

```bash
git clone https://github.com/SRKRZ23/cvpr-gait-2026
cd cvpr-gait-2026
pip install numpy

# Step 1: Analyze training data and build type prototypes
python analyze_training.py

# Step 2: Build type-conditional baseline
python build_submission.py

# Step 3: Apply ensemble consensus voting
python ensemble_vote.py

# Final submission: submissions/final_submission.csv
```

---

## Dataset

The CGPS (Children Gait Pose Sequence) Dataset is provided by the competition organizers.
Training data is included in this repository (`data/` folder).
The full dataset including video and keypoint sequences is available through the [Kaggle competition page](https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge/data).

---

## Citation

```bibtex
@inproceedings{li2026challenge,
  title={The 1st AI Children Challenge},
  author={Li, Boyi and Shen, Yifan and Yang, Houze and Cao, Xu and Yun, Guojun 
          and Gao, Li and Chen, Turong and Xu, Long and Cao, Jianguo and Huang, Meihuan},
  booktitle={2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition 
             Workshops (CVPRW)},
  year={2026}
}
```

---

## Contact

Sardor Razikov — razikovsardor1@gmail.com
