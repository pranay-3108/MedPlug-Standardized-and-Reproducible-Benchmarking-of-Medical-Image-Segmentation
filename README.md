MedPlug
Standardized & Reproducible Benchmarking of Medical Image SegmentationMedPlug is a modular, plugin-based framework that turns the common sources of irreproducibility in medical image segmentation into an enforced software contract. It does not propose a new network architecture. Its object of study is the benchmarking procedure itself.The ProblemReported Dice / IoU numbers for segmentation models are often incomparable. Two papers can claim different performance on the same dataset while differing in:
data splits and leakage
preprocessing and resolution
checkpoint selection rule
decision threshold
metric implementation details
handling of empty predictions
random seeds and environment
When these factors change together with the model, it becomes impossible to attribute performance differences to architecture. The dependence can be written as:$$R = f(D, P, M, L, O, K, T, E)$$A fair comparison varies only the model $M$. In practice, most research codebases change several arguments at once.MedPlug was built to make the surrounding protocol identical, auditable, and recoverable from stored files.What MedPlug Provides
Explicit interfaces
Dataset adaptation, preprocessing, model construction, training, evaluation, and reporting are decoupled. Adding a new model requires only a builder that satisfies the tensor contract:TextN × C × H × W  →  N × 1 × H × W
Dataset contract + preflight
Before any training begins, the framework validates annotation schema, split sizes, class counts, DICOM mapping, and RLE decoding. A pairwise leakage check for identifier disjointness is part of the planned hardening.
Fixed checkpoint rule
The best checkpoint is selected by positive-case validation Dice only. This avoids the common failure mode in which all-case Dice is dominated by the large number of negative images.
Run isolation & audit layer
Every run receives a unique identifier and a private directory. After completion the audit folder contains:
model identity and parameter count
configuration snapshot and environment record
dataset fingerprints and RNG state
per-image validation and test predictions + hashes
independently recomputed metrics
verification status (PASS / FAILED with listed mismatches)

Case-level uncertainty
Mean per-image Dice is reported together with 95 % percentile bootstrap intervals (5 000 resamples). Paired comparisons between models are also bootstrapped on the same images.
Current Demonstration (SIIM-ACR Pneumothorax)Four compact models trained from scratch under two deliberately separated protocols:






















ProtocolModelsSplitKey observationACompact U-NetFull contract split (with negatives)Positive-case val Dice ≈ 0.617, all-case test Dice 0.132 — model predicts foreground on almost every healthy imageBAttention U-Net, TransUNet-style, SegFormer-stylePositive-only diagnostic splitAttention U-Net 0.670, TransUNet-style 0.666, SegFormer-style 0.455. Paired bootstrap cannot separate the first two.Important: Because the protocols differ in data, channels, and early-stopping, the four runs must not be ranked against each other. They demonstrate the framework and the diagnostic power of the audit layer.The audit layer itself surfaced several protocol faults in the authors’ own runs (validation–test overlap of 42 images, degenerate image-level metrics, report vs verification disagreement, channel-count inconsistency, incomplete artifacts). These findings are documented in the accompanying paper.Architecture OverviewTextconfigs/*.yaml
       │
       ▼
core/plugin_loader.py
       │
       ├── dataset plugin
       ├── model plugin
       └── preprocess plugin
              │
              ▼
       stable training core
              │
       ┌──────┴──────┐
       ▼             ▼
  validation    final test
       │             │
       └──────┬──────┘
              ▼
     experiment / report / audit layerCore responsibilities remain separated:
plugins/datasets/ — path, DICOM, CSV/RLE semantics
plugins/models/ — model adapters and registry
plugins/preprocess/ — named preprocessing configurations
train.py / eval.py — stable training and evaluation loops
experiment_manager.py — run bookkeeping, checkpoints, artifacts
app.py — local control panel (configuration, verification, result exploration)
Metrics ReportedPixel-level (primary)
Dice, IoU, Precision, Recall, Specificity, MCC, FPR, FNR, Pixel Accuracy, HD95 (95th-percentile Hausdorff distance in pixels at the working resolution).Positive-case variants of Dice and IoU are always computed; on a positive-only test set they coincide with the all-case values.Image-level metrics are produced but currently flagged as unusable under the positive-only diagnostic protocol (the 0.001 foreground-fraction rule is larger than the mean ground-truth mask). A redesigned image-level decision rule is required before these numbers can be interpreted.Empty-prediction conventions are stated explicitly: both-empty images receive Dice/IoU = 1; non-empty target with empty prediction receives an HD95 penalty equal to the image diagonal.InstallationBashgit clone https://github.com/pranay-3108/MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation.git
cd MedPlug-Standardized-and-Reproducible-Benchmarking-of-Medical-Image-Segmentation
pip install -r requirements.txt
pytest   # recommendedDataset ConnectionThe dataset stays outside the repository. Point the YAML connector at a single root folder:Textyour_dataset_root/
├── data/
│   ├── train-rle.csv
│   ├── val-rle.csv
│   └── test-rle.csv
├── images/
│   ├── dicom-images-train/
│   └── dicom-images-test/
└── aligned_dataset/          # optional pre-aligned pairs
    ├── images/
    └── masks/In the configuration:YAMLdataset:
  plugin: siim_acr
  root: "C:/path/to/your_dataset_root"In source_mode: auto the framework uses the aligned image/mask pair when available and otherwise falls back to DICOM + RLE.Recommended Workflow
Start the control panel:Bashpython app.py
Configure model, training parameters, and metrics in the browser.
Click Verify & Prepare PowerShell Run. The app runs preflight, saves the configuration, and writes the exact training command.
Close the panel and execute the saved command in a separate terminal. run_pipeline.py owns the complete long-running workflow (preflight → training → evaluation → audit → report).
Re-open python app.py to explore the completed run: metric tables, validation history, bootstrap intervals, prediction overlays, and the full artifact browser.
Alternatively, from the command line:Bashpython preflight.py --config configs/your_config.yaml
python run_pipeline.py --config configs/your_config.yamlResume an interrupted run:Bashpython train.py --config configs/your_config.yaml --resumeResearch FiguresThe Research Figures view (and python research_figures.py) rebuilds the manuscript figures directly from saved audit artifacts:
mean test Dice with 95 % case-level bootstrap intervals (all shared cases and validation-disjoint subset)
positive-case validation Dice by epoch
qualitative grids (best / median / lowest Dice cases) after excluding validation overlap
Older runs that lack a qualitative cache can be back-filled with python tools/build_qualitative_cache.py when the original checkpoint and dataset are still available.Project Layout (high level)Textapp.py, run_pipeline.py, train.py, predict.py   entry points
preflight.py, evaluate_checkpoint.py            checks & re-evaluation
config.py, eval.py, data_loader.py, utils.py    core logic
experiment_manager.py, research_trace.py, ...   bookkeeping & audit
core/, plugins/, models/                        plugin system
configs/                                        experiment YAMLs
experiments/                                    run results & indexes
checkpoints/                                    model weights
tools/                                          maintenance scripts
tests/                                          pytest suiteCurrent Status & Known Limitations
Single-seed runs only; seed-level variance is not yet quantified.
Protocol A and Protocol B are not interchangeable; no four-model ranking is claimed.
The diagnostic (positive-only) split cannot measure false-positive behavior on healthy radiographs.
Image-level metrics under the current 0.001 rule are not interpretable on a positive-only test set.
Validation–test identifier overlap was discovered by the audit layer and must be removed in future runs.
The public repository currently contains the framework code; full run artifacts, checkpoints, and per-image tables from the paper experiments are not yet released.
A definitive multi-seed, single-protocol comparison of the four models is planned once the leakage check, image-level rule redesign, and shared-split infrastructure are complete.CitationIf you use MedPlug or refer to the experimental findings, please cite the accompanying paper:Pranay Ugale, Manindra Dharani, Gaurav Paradhe, Mahamuni Kotangale, Ashwini Pise.
“MedPlug: Standardized and Reproducible Benchmarking of Medical Image Segmentation.”
2026.(Full bibliographic details will be updated upon publication.)ContactPranay Ugale
Dept. of CSE – Data Science
Tulsiramji Gaikwad-Patil College of Engineering & Technology, Nagpur, India
pranayugale31@gmail.com


