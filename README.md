Markdown# MedPlug
**Standardized and Reproducible Benchmarking of Medical Image Segmentation**

[Research Paper](https://drive.google.com/file/d/19ztNvJ3WWUTH76GH6a37FAy6qCVXWQn0/view?usp=drive_link) · [Demo Video](https://drive.google.com/file/d/1XNMTAUU8XPLFVRQNTnd2orrWQ2o62kZk/view?usp=drive_link)

---

## Overview

Reported performance numbers in medical image segmentation are frequently incomparable. Two models evaluated on the same dataset can differ not only in architecture but also in data splits, preprocessing pipelines, checkpoint selection rules, decision thresholds, metric implementations, and handling of empty predictions. When these factors change together, it becomes impossible to attribute performance differences to the model itself.

MedPlug addresses this problem directly. It is a modular, plugin-based framework that encodes a fair benchmarking protocol as software. The framework does not introduce a new network architecture. Its contribution is a reproducible experimental contract that keeps every factor except the model under control and makes every reported number independently verifiable from stored artifacts.

The design is guided by a simple principle:

> A performance difference can be attributed to architecture only when the surrounding experimental protocol is identical.

---

## The Problem in Formal Terms

Segmentation performance can be expressed as a function of multiple experimental factors:

\[
R = f(D,\; P,\; M,\; L,\; O,\; K,\; T,\; E)
\]

where  
- \(D\) = dataset and split  
- \(P\) = preprocessing  
- \(M\) = model  
- \(L\) = loss  
- \(O\) = optimizer and schedule  
- \(K\) = checkpoint selection rule  
- \(T\) = thresholding policy  
- \(E\) = evaluation protocol  

In most published work several of these arguments change simultaneously. MedPlug isolates \(M\) by fixing the remaining factors through explicit interfaces, a verified dataset contract, a single checkpoint rule, unique run identifiers, and a post-run audit layer that recomputes every metric from per-image outputs.

---

## Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| Explicit interfaces | Dataset, preprocessing, and model are registered as plugins. Training and evaluation code never hard-codes a particular network. |
| Tensor contract | Every model must map \(N \times C \times H \times W\) → \(N \times 1 \times H \times W\). |
| Dataset contract | Preflight validation of schema, split sizes, class counts, DICOM mapping, and RLE decoding before any training begins. |
| Fixed checkpoint rule | Best checkpoint is selected exclusively by positive-case validation Dice. |
| Run isolation | Every experiment receives a unique run ID and a private directory containing configuration, logs, checkpoints, metrics, predictions, and reports. |
| Independent verification | After each run the framework recomputes all reported metrics from stored per-image tables and records agreement or mismatch. |

---

## What the Audit Layer Detects

The audit layer is the practical core of MedPlug. From stored files alone it has already exposed:

- validation–test identifier overlap (42 of 174 test images also present in validation)
- degenerate image-level metrics caused by an inappropriate decision rule
- disagreement between the run report status and independent verification
- channel-count inconsistency between configuration and archived checkpoints
- incomplete training logs and missing artifacts on earlier runs

These findings demonstrate that protocol faults can be discovered systematically rather than by manual inspection.

---

## Experimental Demonstration (SIIM-ACR Pneumothorax)

Four compact models were trained from scratch under two deliberately separated protocols. Because the protocols differ in data composition, input channels, and early-stopping policy, the results are not ranked against one another. They serve to exercise the framework and to illustrate the diagnostic value of positive-case metrics and the audit layer.

**Protocol A** (full contract split containing negatives)  
- Compact U-Net  
- Positive-case validation Dice ≈ 0.617  
- All-case test Dice = 0.132  
→ The model predicts foreground on nearly every healthy image; the aggregate score alone would have been misleading.

**Protocol B** (positive-only diagnostic split)  
| Model                | Test Dice | 95 % Bootstrap Interval |
|----------------------|-----------|--------------------------|
| Attention U-Net      | 0.670     | [0.637, 0.702]          |
| TransUNet-style      | 0.666     | [0.633, 0.700]          |
| SegFormer-style      | 0.455     | [0.432, 0.477]          |

A paired bootstrap comparison between Attention U-Net and the TransUNet-style network yields a mean difference of 0.004 with interval [−0.005, 0.012]. The interval contains zero; no difference is demonstrated.

All three Protocol-B models under-segment (recall 0.33–0.41). Mean predicted mask area is smaller than mean ground-truth mask area.

---

## Metrics

**Primary (pixel-level)**  
Dice, IoU, Precision, Recall, Specificity, MCC, FPR, FNR, Pixel Accuracy, HD95.

Positive-case Dice and IoU are always computed. On a positive-only test set they coincide with the all-case values.

**Uncertainty**  
Mean per-image Dice is accompanied by a 95 % percentile bootstrap interval obtained from 5 000 resamples of the test cases. Paired model comparisons use the same bootstrap procedure on identical images.

**Image-level metrics**  
Produced but currently flagged as uninterpretable under the positive-only diagnostic protocol. A redesigned decision rule is required before these numbers can be used for model comparison.

Empty-prediction conventions are stated explicitly in the code and documentation: both-empty images receive Dice/IoU = 1; a non-empty target with an empty prediction receives an HD95 penalty equal to the image diagonal.

---

## Installation

```bash
git clone https://github.com/pranay-3108/MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation.git
cd MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation
pip install -r requirements.txt
pytestDataset ConnectionThe dataset remains external to the repository. Point the configuration at a single root folder:Textdataset_root/
├── data/
│   ├── train-rle.csv
│   ├── val-rle.csv
│   └── test-rle.csv
├── images/
│   ├── dicom-images-train/
│   └── dicom-images-test/
└── aligned_dataset/          # optional
    ├── images/
    └── masks/YAMLdataset:
  plugin: siim_acr
  root: "path/to/dataset_root"In source_mode: auto the framework prefers pre-aligned image/mask pairs when available and otherwise falls back to DICOM + RLE.Recommended Workflow
Launch the control panelBashpython app.py
Configure model, hyperparameters, and metrics in the browser interface.
Click Verify & Prepare PowerShell Run. Preflight checks are executed and the exact training command is written to disk.
Execute the saved command in a separate terminal. run_pipeline.py owns the complete workflow: preflight → training → evaluation → audit → report generation.
Re-open the control panel to inspect metrics, validation history, bootstrap intervals, prediction overlays, and the full artifact browser.
Command-line alternative:Bashpython preflight.py --config configs/your_config.yaml
python run_pipeline.py --config configs/your_config.yamlResume an interrupted run:Bashpython train.py --config configs/your_config.yaml --resumeProject StructureTextapp.py, run_pipeline.py, train.py, predict.py   entry points
preflight.py, evaluate_checkpoint.py            verification & re-evaluation
config.py, eval.py, data_loader.py, utils.py    core logic
experiment_manager.py, research_trace.py        run bookkeeping & audit
core/, plugins/, models/                        plugin system
configs/                                        experiment configurations
experiments/                                    run results and indexes
checkpoints/                                    model weights
tools/                                          maintenance utilities
tests/                                          test suiteCurrent Limitations
Experiments to date use a single random seed; seed-level variance is not yet quantified.
Protocol A and Protocol B are not interchangeable; no architecture ranking across the four models is claimed.
The positive-only diagnostic split cannot measure false-positive behaviour on healthy radiographs.
Image-level metrics under the present decision rule are not interpretable on a positive-only test set.
Validation–test identifier overlap discovered by the audit layer must be eliminated in future runs.
Full experimental artifacts from the paper (checkpoints, per-image tables) are not yet publicly released.
A multi-seed comparison under a single shared protocol is planned once leakage checks, image-level rule redesign, and shared-split infrastructure are complete.CitationIf you use MedPlug or refer to the experimental findings, please cite:Pranay Ugale, Manindra Dharani, Gaurav Paradhe, Mahamuni Kotangale, Ashwini Pise.

“MedPlug: Standardized and Reproducible Benchmarking of Medical Image Segmentation.” 2026.ContactPranay Ugale

Department of CSE – Data Science

Tulsiramji Gaikwad-Patil College of Engineering & Technology, Nagpur, India

pranayugale31@gmail.com
