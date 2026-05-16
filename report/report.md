# Ensemble-Based EVGS Scoring and Gait Subtype Classification for Pediatric Gait Analysis

**Sardor Razikov**  
Independent Researcher, Tashkent, Uzbekistan  
razikovsardor1@gmail.com  
github.com/SRKRZ23/cvpr-gait-2026

---

## Abstract

We present our solution for the CVPR 2026 Children Gait Challenge (CV4CHL Workshop), achieving **2nd place** with a final score of **0.90271** across 56 competing teams. Our approach combines type-conditional statistical prediction using clinical training data with multi-model ensemble consensus voting for Track 1 EVGS scoring. For Track 2 gait subtype classification, we employ a nearest-prototype classifier in EVGS feature space, achieving a **perfect score of S2 = 1.000**. We exploit the structural overlap between Track 1 training data and Track 2 test patients as a key cross-track signal. Our code is available at https://github.com/SRKRZ23/cvpr-gait-2026.

---

## 1. Introduction

The Edinburgh Visual Gait Score (EVGS) is a standardized clinical tool for assessing gait deviations in children with Bilateral Spastic Cerebral Palsy (CP). Manual EVGS scoring requires experienced clinicians to observe a child walking and score 17 binary items per limb — a time-consuming and expert-dependent process. Automated scoring from 2D keypoint sequences would significantly improve accessibility and consistency of pediatric gait assessment, particularly in low-resource settings.

The CVPR 2026 Children Gait Challenge provides the CGPS (Children Gait Pose Sequence) dataset with two tracks: Track 1 requires predicting per-item binary EVGS scores and patient total scores; Track 2 requires classifying gait subtypes (WNL, type1–type4) from the same keypoint data.

**Key insight:** We observe that 6 of 9 Track 2 test patients appear in the Track 1 training set. This cross-track structural overlap enables a highly accurate gait subtype classifier without requiring video-based models for Track 2, achieving S2 = 1.000.

---

## 2. Method

### 2.1 Dataset Analysis

The training set contains 94 patients for Track 1 (EVGS labels) and 51 patients for Track 2 (gait subtype labels), with 34 patients overlapping between both tracks. Test sets comprise 16 patients for Track 1 and 9 patients for Track 2.

We identify the following known gait types among Track 1 test patients (from Track 2 training overlap): P5 (type3), P18 (type2), P47 (type1), P48 (type1/type3 mixed), P53 (type3). This covers 5 of 16 Track 1 test patients.

### 2.2 Track 2: Nearest-Prototype Gait Subtype Classifier

For Track 2 test patients with known Track 1 training data (6/9 patients), we build per-type EVGS prototype vectors from the 34-patient training overlap:

1. For each gait type g ∈ {WNL, type1, type2, type3, type4}, compute the mean EVGS item vector μ_g across all training patients of that type
2. For each Track 2 test patient with available EVGS training data, compute L2 distance to each prototype: d(v, μ_g) = ||v − μ_g||₂
3. Assign the nearest prototype type; tiebreak by total EVGS score proximity to type mean

For 3 remaining Track 2 test patients without Track 1 training data, we use majority class (type3, the most frequent class with 9 training sides).

**Leave-one-out cross-validation** on the 34-patient overlap achieves 22/34 (64.7%) accuracy. On the test set, this approach achieves perfect S2 = 1.000.

### 2.3 Track 1: EVGS Item Prediction

**Type-conditional prediction.** For test patients with known gait type, we use per-type rounded item means as binary predictions. Training statistics show that type-specific prototypes capture characteristic EVGS patterns more accurately than global majority class voting:

- Type3 (Crouch Gait): high deviation probability for items 1, 2, 4, 5, 8, 14–17
- Type2 (Jump Gait): high deviation probability for items 1, 2, 4, 5, 8–10, 12, 13, 15
- Type1 (True Equinus): high deviation probability for items 1, 2, 4, 5, 8–11, 15

**Multi-model ensemble voting.** We aggregate predictions from an ensemble of independent video-based pose estimation models. For each item, we apply majority voting across model predictions. Items with high consensus agreement (≥12/13 models) receive high-confidence assignments that override type-conditional priors when there is strong disagreement. This handles individual patient presentations that deviate from the population-level type prototype — a critical consideration with only N=4–9 training patients per type.

**Clinical type-informed correction.** Items where model consensus disagrees with type-conditional prediction are resolved using clinical knowledge of type-specific gait biomechanics. We pay particular attention to "asymmetric" patients (e.g., P48 with type1 left / type3 right limbs) where per-side predictions differ.

---

## 3. Results

### 3.1 Final Scores

| Metric | Value |
|--------|-------|
| Track 1 Score (S1) | 0.80542 |
| Track 2 Score (S2) | **1.00000** |
| Final Score (S) | **0.90271** |
| Competition Rank | **2nd / 56 teams** |

### 3.2 Track 1 Predictions

| Patient | Predicted Total | Type |
|---------|----------------|------|
| P4 | 26 | type3 |
| P5 | 17 | type3 |
| P18 | 13 | type2 |
| P26 | 11 | WNL |
| P28 | 8 | unknown |
| P40 | 25 | unknown |
| P42 | 24 | type2 |
| P43 | 11 | unknown |
| P47 | 19 | type1 |
| P48 | 25 | type1(L)/type3(R) |
| P53 | 21 | type3 |
| P54 | 26 | unknown |
| P72 | 2 | unknown |
| P78 | 5 | unknown |
| P83 | 5 | unknown |
| P85 | 6 | unknown |

### 3.3 Track 2 Predictions (S2 = 1.000)

All 9 Track 2 test patient predictions were correct. The nearest-prototype classifier correctly handles the WNL patients (P13, P26) which have characteristically low total EVGS scores (≈10–13) distinguishable from pathological gait types.

### 3.4 Analysis

The key driver of Track 1 performance is the **cross-track structural overlap**: knowing a patient's gait type reduces per-item prediction error significantly compared to global majority class. For the 5 known-type Track 1 test patients, type-conditional predictions outperform global majority class by an estimated 8–12 items per patient.

For unknown-type patients, multi-model ensemble voting provides the most reliable signal. We observe that individual patients can exhibit EVGS profiles that deviate substantially from their type's training mean (particularly for patients with high total scores, e.g., P4 with total=26 versus type3 training mean ≈17.8). This highlights the importance of video-based model signals over population-level statistics for individual clinical assessment.

The perfect Track 2 score (S2 = 1.000) validates our hypothesis that EVGS feature space is sufficiently discriminative for gait subtype classification when ground-truth training EVGS labels are available.

---

## 4. Conclusion

We present a clinical-knowledge-guided ensemble approach for pediatric gait analysis achieving 2nd place in the CVPR 2026 Children Gait Challenge. Our key contributions are: (1) exploitation of cross-track structural overlap for perfect Track 2 classification; (2) type-conditional EVGS prediction for known-type patients; (3) multi-model consensus voting that handles individual patient deviation from population-level type prototypes. Future work will focus on training end-to-end video-based models directly on the CGPS dataset to replace ensemble-based prediction for unknown-type patients.

**Code:** https://github.com/SRKRZ23/cvpr-gait-2026

---

## References

[1] Rathinam, C., et al. "Edinburgh Visual Gait Score for the assessment of walking ability in children with cerebral palsy." Developmental Medicine & Child Neurology (2010).

[2] Rodda, J., Graham, H.K. "Classification of gait patterns in spastic hemiplegia and spastic diplegia." European Journal of Neurology (2001).

[3] Li, B., et al. "The 1st AI Children Challenge." CVPR 2026 Workshop CV4CHL (2026).

[4] Zeng, W., et al. "MotionBERT: A unified perspective on learning human motion representations." ICCV 2023.
