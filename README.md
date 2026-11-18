# MedPlugBench — Plugin-Based Medical Segmentation Benchmark

## One-folder dataset connection

The dataset is **external** to the benchmark repository. Keep the complete dataset in one folder and point the YAML connector at that folder.

Example:

```text
C:\MedPlugBench\nothing\
├── data\
│   ├── train-rle.csv
│   ├── val-rle.csv
│   └── test-rle.csv
├── images\
│   ├── dicom-images-train\
│   └── dicom-images-test\
└── aligned_dataset\
    ├── images\
    └── masks\
```

Set only:

```yaml
dataset:
  plugin: siim_acr
  root: "C:/MedPlugBench/nothing"
```

The SIIM-ACR plugin resolves the CSV, DICOM, and aligned paths relative to that root. In `source_mode: auto`, each ImageId uses the aligned image/mask pair when available and otherwise falls back to DICOM + RLE. This keeps the original DICOM workflow intact while allowing a previously managed `aligned_dataset/` to be used without copying the training DICOM tree again.

## Architecture

```text
configs/*.yaml
       |
       v
core/plugin_loader.py
       |
       +-------------------+-------------------+
       |                   |                   |
       v                   v                   v
 dataset plugin       model plugin       preprocess plugin
       |                   |                   |
       +-------------------+-------------------+
                           |
                           v
                    stable training core
                           |
              +------------+-------------+
              |                          |
              v                          v
         validation                 final test
              |                          |
              +------------+-------------+
                           v
                  experiment/report layer
                           |
             graphs + visualizations + app.py
```

Core responsibilities stay separated:

- `plugins/datasets/`: dataset-specific path, DICOM, CSV/RLE semantics.
- `plugins/models/`: model adapter/registry.
- `plugins/preprocess/`: named preprocessing configuration.
- `core/plugin_loader.py`: connector and plugin discovery.
- `train.py`: stable training orchestration.
- `eval.py`: evaluation and metrics.
- `experiment_manager.py`: benchmark history, checkpoints, artifacts.
- `report_generator.py`: report formatting.
- `visualization.py`: visualizations and metric graphs.
- `app.py`: local control panel for configuring and verifying experiments, saving an isolated run, generating the exact PowerShell command, and viewing persisted results. It does not own the long-running training process.

Start it with:

```powershell
python app.py
```

The browser form controls the dataset root, model, batch size, epochs, learning rate, optimizer/scheduler, loss, device, checkpoint resume, and selected evaluation metrics. The UI writes a YAML experiment configuration and leaves the core/plugin implementation unchanged.


## Project layout

```text
app.py, run_pipeline.py, train.py, predict.py   entry points
preflight.py, evaluate_checkpoint.py            checks and re-evaluation
config.py, eval.py, data_loader.py, utils.py    core training/evaluation code
experiment_manager.py, research_trace.py, ...   run bookkeeping, audit, reports
core/, plugins/, models/                        plugin loader, dataset/model/preprocess plugins
configs/                                        YAML experiment configs
tools/                                          maintenance scripts
tests/, tests_plugin/                           pytest suites (run: pytest)
experiments/                                    run results and indexes
checkpoints/                                    trained model weights
predictions/                                    generated inference output (not tracked)
docs/                                           guides; docs/history = old reports, docs/training_logs = raw logs
```

Install dependencies with `pip install -r requirements.txt`, then run `pytest`.

## Results analysis and comparison

The control panel is also a results explorer. After a run completes, `app.py` reads the run's persisted metrics and training history and exposes:

- a full saved test-metric table
- an on-demand Metric Explorer for Recall, Precision, Specificity, Dice, IoU, MCC, HD95, ROC-AUC, PR-AUC, FPR/FNR, and image-level metrics
- validation-history inspection from `training_log.csv`
- compatible-run comparison with a selectable metric
- prediction/ground-truth/overlay views
- a run-local artifact browser for checkpoints, logs, metrics, graphs, comparisons, and reports

The UI does not recalculate the medical metrics or modify the training results. It reads the persisted run artifacts.

## No training is started by plugin/preflight verification

Use:

```powershell
python preflight.py --config configs/default.yaml
```

This checks the dataset, preprocessing shape, model forward path, and CUDA availability. It does not train.

Only after preflight passes should you run:

```powershell
python train.py --config configs/default.yaml
```

## Reproducible research runs

For a new run, `run_pipeline.py` can create an isolated run automatically; `--run-id` is optional. A new completed run contains the original benchmark artifacts plus an `audit/` directory with runtime model identity, parameter counts, configuration, environment, dataset fingerprints, RNG state, rich checkpoint metadata, per-image validation/test records, metric availability/status, independently recomputed metrics, prediction fingerprints, duplicate-run checks, and diagnostic graphs.

Example for the managed SIIM-ACR dataset:

```powershell
$env:MEDPLUG_DATASET_ROOT = "C:\Path\to\nothing"
python preflight.py --config configs/segformer_retrain.yaml
python run_pipeline.py --config configs/segformer_retrain.yaml
```

The best checkpoint is self-describing and stores the model state together with the model identity, training state, optimizer/scheduler state, configuration/provenance metadata, and checkpoint/state hashes. Missing or non-finite metrics are retained as `NOT_AVAILABLE` with a reason instead of being silently dropped.

## Resume

After an interrupted run:

```powershell
python train.py --config configs/default.yaml --resume
```

The latest completed-epoch checkpoint is used so training does not restart from epoch 1.


## Recommended workflow

1. Start the control panel with `python app.py` (or `start_medplugbench.ps1`).
2. Configure model, training parameters, and metrics.
3. Click **Verify & Prepare PowerShell Run**. The app runs the dataset and training-environment preflight and saves the run configuration plus `logs/training_command.txt`.
4. Close `app.py`. Copy the saved PowerShell command into a separate PowerShell window.
5. `run_pipeline.py` owns the complete long-running workflow independently of the browser: preflight -> training -> evaluation -> metrics -> visualization -> report.
6. After completion, run `python app.py` again to browse the persisted run and its artifacts.

The prepared command uses the exact Python interpreter running the control panel, so the UI and training environment stay aligned.


## Manuscript figures dashboard

The **Research Figures** view (`http://127.0.0.1:8765/?view=research`) builds manuscript-oriented figures directly from saved run artifacts:

- **Figure 2:** mean test Dice with 95% case-level bootstrap intervals, showing all shared diagnostic test cases and the subset disjoint from all recorded validation IDs.
- **Figure 3:** positive-case validation Dice by epoch for the Attention U-Net, TransUNet-style, and SegFormer-style diagnostic runs, read from `audit/epoch_diagnostics.jsonl`.
- **Figure 4:** a qualitative grid using the same best/median/lowest mean-Dice test cases across models, after excluding IDs found in validation. New training runs save an inference-time `audit/qualitative_cache.npz` so the dashboard can create this figure without rerunning test inference.

The main Dashboard also shows these research figures. The plots do not rank the official-split U-Net against the positive-only diagnostic runs, and a failed verification status stays visible. If an older run has only per-image metric rows and hashes, Figure 4 is deliberately marked as pending: raw pixels cannot be reconstructed from hashes. First try `python tools/build_qualitative_cache.py` to backfill the old runs. It writes a cache only when the saved checkpoint and dataset reproduce every archived prediction-mask hash and per-image Dice value. If the dataset/checkpoint is unavailable or verification fails, retrain the three diagnostic models with the updated code and configured dataset root.

To build or refresh the figure artifacts manually:

```powershell
python research_figures.py
```

Outputs are stored under `experiments/research_figures/` and are also served directly by `app.py`.

## Recovering the complete metric table

Older runs may have saved only the metrics selected in the training form even though the audit contains additional per-image measurements. The dashboard now merges the run metric file, audit ledger, and independent recomputation; a failed verification or positive-only image-classification caveat is displayed separately.

To rebuild the shared indexes from saved artifacts:

```powershell
python tools/rebuild_benchmark_index.py
```

The command is also called at the end of a new training run after the audit and completion status are saved. It only updates `experiments/benchmark_results.csv` and `experiments/leaderboard.csv`; it does not rewrite checkpoints, prediction files, metric ledgers, or verification evidence. Missing metrics are never replaced with zero, and an unavailable ROC-AUC remains blank. The index includes positive-case Dice/IoU, additional pixel metrics, protocol, verification status, test-case counts, validation/test overlap, and source notes.

**Research reporting note:** a positive-only test set cannot measure false-positive behavior on negative images. On such a set, image-level ROC-AUC is undefined and image-level PR-AUC is not informative. Review `Verification Status`, `Image Metrics Status`, and `Audit Mismatches` before citing values in the paper.
