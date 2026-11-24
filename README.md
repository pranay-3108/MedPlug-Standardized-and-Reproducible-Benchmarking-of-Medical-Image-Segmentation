# MedPlug

**Standardized and Reproducible Benchmarking of Medical Image Segmentation**

[Research Paper](https://drive.google.com/file/d/19ztNvJ3WWUTH76GH6a37FAy6qCVXWQn0/view?usp=drive_link) · [Demo Video](https://drive.google.com/file/d/1XNMTAUU8XPLFVRQNTnd2orrWQ2o62kZk/view?usp=drive_link) · [GitHub Repository](https://github.com/pranay-3108/MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation)

---

## MedPlug in 6 seconds

MedPlug is a modular benchmarking framework for medical image segmentation. It turns an experimental protocol into an executable contract: validate the dataset before training, use a shared model interface, apply a fixed checkpoint-selection rule, isolate every run, and independently recompute metrics from saved per-image outputs.

**The central contribution is reproducibility and auditability, not a new segmentation architecture.** The framework was demonstrated on SIIM-ACR pneumothorax segmentation using four compact networks across two deliberately separate experimental protocols.

| At a glance | Details |
|---|---|
| Task | Pneumothorax segmentation from chest X-rays |
| Dataset | SIIM-ACR Pneumothorax Segmentation |
| Framework | Modular, plugin-based benchmark pipeline |
| Model interface | `N × C × H × W` input → `N × 1 × H × W` logits |
| Models | U-Net, Attention U-Net, TransUNet-style, SegFormer-style |
| Evaluation | Pixel-level metrics, positive-case metrics, case-level bootstrap intervals, run audit |
| Reproducibility record | Configurations, logs, checkpoints, predictions, hashes, metrics, and audit outputs per run |

## My contributions

I independently carried out the research and technical work for MedPlug, with the exception of the user-facing web design and `app.py`, which were developed collaboratively. The distinction below reflects the work I personally owned:

- **Research framing and protocol design:** formulated the reproducibility problem, defined the experimental controls, established the shared model tensor contract, and designed the two protocols used to exercise the framework.
- **Framework architecture:** designed and implemented the plugin-oriented structure that separates dataset handling, preprocessing, model construction, training, evaluation, and reporting.
- **Dataset engineering:** implemented the SIIM-ACR data integration, RLE mask decoding, DICOM-to-image mapping, aligned image-mask loading support, and dataset preflight checks for schema, split counts, class counts, file mapping, and mask decoding.
- **Model integration and implementation:** implemented and registered the compact U-Net, Attention U-Net, TransUNet-style, and SegFormer-style model variants under the common input/output contract.
- **Training pipeline:** implemented the model-independent training workflow, loss composition, optimizer and learning-rate schedule setup, validation flow, checkpoint persistence, and fixed positive-case validation Dice selection rule.
- **Evaluation methodology:** implemented pixel-level and image-level metric calculations, positive-case reporting, explicit empty-mask conventions, and per-image result export.
- **Run management and traceability:** implemented unique run identifiers, isolated experiment directories, configuration snapshots, run manifests, training logs, checkpoint records, and artifact bookkeeping.
- **Independent audit and verification:** implemented post-run recomputation of reported metrics from stored per-image outputs, hash-based artifact records, and checks that identify inconsistencies between run reports and verification results.
- **Experimental analysis:** investigated split overlap, metric degeneracy, checkpoint/configuration inconsistencies, incomplete run records, and the limitations of the available evidence; analyzed the results and bootstrap intervals.
- **Research documentation:** prepared the technical analysis and research reporting, including the limitations and the protocol required for a definitive multi-model benchmark.

**Collaborative work:** the web interface design and `app.py` control-panel implementation were collaborative. I do not present those two parts as solely my work. The research, benchmark methodology, core framework and supporting engineering work listed above were completed independently by me.

---

## The problem MedPlug addresses

Medical image segmentation scores are difficult to compare when the experiments use different data splits, preprocessing pipelines, model configurations, checkpoint-selection rules, thresholds, or metric implementations. If several of these factors change at once, a performance difference cannot reliably be attributed to the architecture.

MedPlug represents the benchmark result as a function of the experimental conditions:

\[
R = f(D, P, M, L, O, K, T, E)
\]

where:

- \(D\): dataset and split
- \(P\): preprocessing
- \(M\): model architecture
- \(L\): loss function
- \(O\): optimizer and learning-rate schedule
- \(K\): checkpoint-selection rule
- \(T\): thresholding policy
- \(E\): evaluation protocol

A fair architecture comparison should vary \(M\) while holding the other factors constant. MedPlug encodes these controls in software and records the artifacts needed to inspect whether a run followed the intended protocol.

## Core design

| Principle | MedPlug implementation |
|---|---|
| Explicit plugin interfaces | Dataset, preprocessing, and model components are registered independently. The training and evaluation workflow does not hard-code one particular network. |
| Common tensor contract | Every model accepts `N × C × H × W` and returns `N × 1 × H × W` logits. |
| Dataset contract | Preflight validation checks annotation schema, split sizes, class counts, DICOM mappings, and representative RLE decoding before training. |
| Fixed checkpoint rule | The selected checkpoint is the one with the best positive-case validation Dice. |
| Run isolation | Every experiment receives a unique run ID and its own directory for configuration, logs, checkpoints, metrics, predictions, visualizations, and reports. |
| Independent verification | Metrics are recomputed from saved per-image outputs and compared with the values recorded by the run. Mismatches are retained as audit findings. |

### Execution flow

```text
YAML configuration
        ↓
Plugin registry
(dataset / preprocessing / model)
        ↓
Dataset contract and preflight validation
        ↓
DICOM + RLE loading or aligned image-mask loading
        ↓
Normalization, resizing, channel adaptation
        ↓
Model tensor contract
(N × C × H × W → N × 1 × H × W)
        ↓
Training and validation
(best checkpoint by positive-case validation Dice)
        ↓
Single test evaluation
        ↓
Run artifacts and independent audit
        ↓
Metrics, verification status, and report
```

## Dataset and preprocessing

The case study uses the **SIIM-ACR Pneumothorax Segmentation** dataset. The source data pairs DICOM chest radiographs with image identifiers and run-length-encoded (RLE) pneumothorax masks.

The published stage-1 training release contains 10,675 images: 2,379 positive and 8,296 negative. The framework's contract split records 10,675 training images, 854 validation images (172 positive, 682 negative), and 854 test images (174 positive, 680 negative). The paper notes that the provenance and disjointness of these additional validation and test lists were not fully documented; the audit findings below therefore matter when interpreting the results.

### RLE mask reconstruction

RLE annotations are decoded into binary masks. Multiple runs associated with one image are combined with a logical OR, and images without pneumothorax annotations receive an all-zero mask. A round-trip encode/decode check was also used in an earlier audit.

### Preprocessing

Each image is normalized using its own minimum and maximum intensity:

\[
\hat{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \epsilon}
\]

Images are resized to `256 × 256` using bilinear interpolation. Masks are resized with nearest-neighbor interpolation to preserve binary labels. Depending on the configuration, grayscale images are supplied as one channel or replicated into three identical channels.

The reduction to `256 × 256` can remove fine structures. This is particularly relevant for pneumothorax because the target masks are small relative to the full image.

### Dataset layout

Keep the dataset outside the Git repository and configure MedPlug to point to its root directory.

```text
dataset_root/
├── data/
│   ├── train-rle.csv
│   ├── val-rle.csv
│   └── test-rle.csv
├── images/
│   ├── dicom-images-train/
│   └── dicom-images-test/
└── aligned_dataset/              # optional
    ├── images/
    └── masks/
```

Example dataset configuration:

```yaml
dataset:
  plugin: siim_acr
  root: "path/to/dataset_root"
```

When `source_mode: auto` is enabled in the configuration, the framework prefers pre-aligned image/mask pairs when available and otherwise falls back to DICOM images with RLE annotations.

## Models and experimental protocols

Four compact models were registered under the same tensor interface. They were trained from scratch; the Transformer-based implementations are described as **-style** networks because they are compact project implementations and are not the original published architectures or pretrained checkpoints.

| Model | Parameters | Protocol |
|---|---:|---|
| U-Net | 117,073 (one-channel checkpoint) | A |
| Attention U-Net | 1,948,500 | B |
| TransUNet-style | 3,506,465 | B |
| SegFormer-style | 2,792,769 | B |

Two protocols were deliberately kept separate. Their results must not be pooled or treated as a four-model ranking.

| Setting | Protocol A | Protocol B |
|---|---|---|
| Models | U-Net | Attention U-Net, TransUNet-style, SegFormer-style |
| Train / validation / test | 10,675 / 854 / 854 | 1,903 / 476 / 174 |
| Negative images | Included | None; positive-only diagnostic split |
| Input channels | Configuration records 3; archived checkpoint has 1 (unresolved inconsistency) | 1 |
| Early stopping | Enabled, patience 10 | Disabled |
| Epochs / seed | 25 / 42 | 25 / 42 |
| Main purpose | Exercise the full contract split | Diagnostic comparison on positive cases |

Protocol B uses the 2,379 positive training images, shuffled with seed 42 and divided into 1,903 training and 476 validation images. Its 174-image test list is positive-only; the script that generated this list was not preserved, and the test list overlaps the validation list. This is a diagnostic protocol, not a replacement for a benchmark with a verified, disjoint test set containing both positive and negative cases.

### Shared training settings

The paper records the following common settings where applicable:

- Combined objective: `0.5 × BCE + 0.5 × Dice loss`
- Optimizer: AdamW
- Learning-rate schedule: cosine annealing
- Learning rate and weight decay: `1e-4`
- Batch size: 2, with two gradient-accumulation steps (effective batch size 4)
- Training duration: 25 epochs
- Random seed: 42
- Checkpoint selection: maximum positive-case validation Dice
- Inference threshold: 0.50
- Input resolution: `256 × 256`

The protocols still differ in split composition, input-channel settings, early-stopping policy, and hardware. Their results are therefore reported separately.

## Experimental results

### Protocol B: positive-only diagnostic test set

The following results are from 174 positive test images. Since every image in this test set is positive, all-case and positive-case Dice/IoU coincide.

| Metric | Attention U-Net | TransUNet-style | SegFormer-style |
|---|---:|---:|---:|
| Test Dice | **0.6697** | **0.6662** | 0.4550 |
| 95% case-level bootstrap interval | [0.637, 0.702] | [0.633, 0.700] | [0.432, 0.477] |
| Test IoU | 0.5436 | 0.5426 | 0.3062 |
| Precision | 0.8267 | 0.8403 | 0.4474 |
| Recall | 0.4067 | 0.3862 | 0.3285 |
| MCC | 0.5797 | 0.5695 | 0.3831 |
| Parameters | 1,948,500 | 3,506,465 | 2,792,769 |

The paired per-image Dice difference between Attention U-Net and the TransUNet-style model was approximately `0.0036`, with a 95% bootstrap interval of approximately `[-0.0052, 0.012]`. Because the interval includes zero, the experiment does not demonstrate that either model is better than the other.

All three models under-segmented the target regions. Recall was approximately 0.33–0.41, and the mean predicted mask area (16–25 pixels) was smaller than the mean ground-truth mask area (35 pixels). The high precision of the first two models therefore coexists with substantial missed target area.

### Protocol A: U-Net on the contract split

The U-Net reached a positive-case validation Dice of approximately `0.617`. Its indexed all-case test Dice was `0.132`. The large difference is consistent with the model predicting foreground on many negative images, which makes the all-case and positive-case views tell very different stories.

**Important verification limitation:** the Protocol A run did not retain a complete metrics file, report, or audit folder, so the test result cannot be independently recomputed from stored per-image predictions. The archived checkpoint also has one input channel while the configuration records three, and the surviving training log does not resolve the recorded best-epoch discrepancy. The U-Net score is therefore shown as an unverified historical index value, not as a fully verified benchmark result.

### Validation-test overlap

The audit found that **42 of the 174 Protocol B test images (24.1%) also appear in the 476-image validation set**. Since the checkpoint is selected using validation Dice, those 42 cases are not independent test cases.

| Model | Dice on all 174 test images | Dice on 132 validation-disjoint images |
|---|---:|---:|
| Attention U-Net | 0.6697 | 0.6723 |
| TransUNet-style | 0.6662 | 0.6676 |
| SegFormer-style | 0.4550 | 0.4567 |

Removing the overlapping cases changes Dice by at most about 0.003 in these runs, but the overlap remains a protocol fault and must be fixed before a definitive comparison.

## Metrics and uncertainty

### Pixel-level metrics

MedPlug reports Dice, IoU, precision, recall, specificity, Matthews correlation coefficient (MCC), false-positive rate (FPR), false-negative rate (FNR), pixel accuracy, and 95th-percentile Hausdorff distance (HD95).

For pixel-level counts TP, FP, FN, and TN:

\[
\mathrm{Dice} = \frac{2TP}{2TP + FP + FN}
\]

\[
\mathrm{IoU} = \frac{TP}{TP + FP + FN}
\]

\[
\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

Metrics are aggregated over all images and separately over positive images. Positive-case Dice and IoU are especially important for a dataset with many negative images, because aggregate scores can conceal whether a model is segmenting actual positive cases.

### Bootstrap uncertainty

Mean per-image test Dice is accompanied by a 95% percentile bootstrap interval using 5,000 resamples of the test cases. Paired model comparisons use the same cases and bootstrap the per-image difference. These intervals quantify test-case sampling uncertainty only; a single training seed does not estimate seed-to-seed variability.

### Empty-mask conventions

The scoring implementation makes the following conventions explicit:

- If both prediction and target are empty, Dice and IoU are set to 1.
- If the target is non-empty and the prediction is empty, HD95 receives a penalty equal to the image diagonal.

These conventions affect reported values and should be considered when comparing this implementation with other evaluation code.

### Image-level metrics: currently not suitable for comparison

The current image-level decision rule labels an image positive when at least `0.001` of its pixels are predicted as foreground. At `256 × 256`, this corresponds to about 65.5 pixels. The diagnostic test masks are very small (mean ground-truth area about 35 pixels), and the Protocol B test set contains only positive images. Under the current rule, all three Protocol B models were labeled negative on every test image; the resulting hard-label image-level metrics are not meaningful estimates of image-classification performance. Image-level ROC-AUC is undefined for a single-class test set, and PR-AUC is not informative in this setup.

For this reason, the image-level dashboard values are retained as diagnostic outputs but should **not** be used to rank models. A revised decision rule must be evaluated on a test set containing both positive and negative images before image-level metrics are suitable for comparison.

## What the audit layer detected

The audit layer examines stored experiment artifacts and independently checks whether the reported results are supported by them. The paper documents the following findings:

| Audit finding | Why it matters |
|---|---|
| 42 of 174 Protocol B test identifiers also occur in validation | The test set is not fully independent of checkpoint selection. |
| The image-level threshold corresponds to about 65.5 pixels, above the mean ground-truth mask area of about 35 pixels | The current decision rule degenerates on the positive-only diagnostic set. |
| Verification reports `FAILED` while the run report says `PASS` for two Protocol B runs | A run should not be represented as fully verified when the independent audit disagrees. |
| U-Net configuration records 3 input channels, but the archived checkpoint has a 1-channel first convolution | Configuration and checkpoint provenance are inconsistent. |
| U-Net best epoch is recorded as 4, while the surviving log covers epochs 14–23 and early-stopping history does not reconcile | The training history cannot be fully verified. |
| The U-Net run is missing a complete audit, report, and per-image artifact set | Its reported result cannot be independently reproduced from the archived run folder. |
| One validation image has an empty mask after resizing | The diagnostic validation set is not entirely positive after preprocessing. |
| Derived index files contain duplicate or placeholder-era records | Historical index values require manual cleaning before citation. |

The Protocol B audit reproduced the main segmentation metrics from the stored per-image records. However, image-level PR-AUC disagreed with the index for two runs; their verification status remained `FAILED` even though the run report said `PASS`. HD95 was verified as an aggregate of stored per-image values, but the original per-image values were not re-derived from archived pixel predictions against a reference implementation.

## Recommended workflow

### 1. Install dependencies

```bash
git clone https://github.com/pranay-3108/MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation.git
cd MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation
pip install -r requirements.txt
```

### 2. Run the test suite

```bash
pytest
```

### 3. Launch the control panel

```bash
python app.py
```

Configure the dataset, model, hyperparameters, and metrics in the browser interface.

### 4. Verify and prepare an experiment

Use **Verify & Prepare Run** in the control panel. Preflight checks are executed and the exact training command is written to disk.

### 5. Execute the saved command

Run the generated command in a separate terminal. `run_pipeline.py` owns the complete workflow:

```text
preflight → training → evaluation → audit → report generation
```

### Command-line alternative

```bash
python preflight.py --config configs/your_config.yaml
python run_pipeline.py --config configs/your_config.yaml
```

### Resume an interrupted training run

```bash
python train.py --config configs/your_config.yaml --resume
```

After a run, reopen the control panel to inspect metrics, validation history, bootstrap intervals, prediction overlays, and the artifact browser.

## Project structure

```text
app.py, run_pipeline.py, train.py, predict.py   # entry points
preflight.py, evaluate_checkpoint.py             # verification and re-evaluation
config.py, eval.py, data_loader.py, utils.py     # core logic
experiment_manager.py, research_trace.py        # run bookkeeping and audit
core/, plugins/, models/                         # plugin system and model components
configs/                                         # experiment configurations
experiments/                                     # run results and indexes
checkpoints/                                     # model weights
tools/                                           # maintenance utilities
tests/                                           # test suite
```

## Current limitations

The experiments demonstrate the framework and its audit capabilities, but they do not yet establish a definitive architecture ranking.

- Every configuration was run with a single seed (42); seed-level variance is unknown.
- Protocol A and Protocol B differ in data composition, input channels, early-stopping policy, and hardware. Results across the protocols are not directly comparable.
- Protocol B is positive-only and cannot measure false-positive behavior on healthy radiographs.
- The Protocol B test list overlaps validation by 42 images; the source of that overlap has not been established.
- The provenance and disjointness of the additional contract-split validation and test lists are not fully documented.
- Image-level metrics are uninterpretable under the current decision rule and positive-only test design.
- Pixel-level predictions were not archived for all runs, limiting qualitative comparison and full re-derivation of some metrics.
- The U-Net run has incomplete artifacts and unresolved configuration/checkpoint and training-history inconsistencies.
- The study uses one dataset, one threshold, and compact models trained from scratch; it does not establish clinical validity or performance of the original published architectures.
- Full experimental artifacts, including checkpoints and per-image tables, were not publicly released in the version of the repository inspected for the paper.

## Next steps

The paper identifies the following work as necessary before a definitive benchmark:

1. Add split-disjointness checks to preflight and rebuild a verified test split.
2. Train all four models on one shared split with the same channel count and early-stopping policy.
3. Repeat every configuration with at least three seeds (and five where compute permits).
4. Redesign the image-level decision rule and evaluate it on a test set containing both positive and negative cases.
5. Investigate the threshold, input resolution, and HD95 implementation.
6. Archive pixel-level predictions and add comparable qualitative visualizations.
7. Ensure verification status and run-report status cannot contradict each other.
8. Extend the evaluation to additional medical segmentation datasets.

## Citation

If you use MedPlug or refer to the experimental findings, please cite:

> Pranay Ugale, Manindra Dharani, Gaurav Paradhe, Mahamuni Kotangale, and Ashwini Pise. “MedPlug: Standardized and Reproducible Benchmarking of Medical Image Segmentation.” 2026.

## Contact

**Pranay Ugale**  
Department of CSE – Data Science  
Tulsiramji Gaikwad-Patil College of Engineering & Technology, Nagpur, India  
[pranayugale31@gmail.com](mailto:pranayugale31@gmail.com)  
[GitHub](https://github.com/pranay-3108) · [LinkedIn](https://linkedin.com/in/pranayugale)
