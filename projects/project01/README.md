# Active-matter representations and latent forecasting

A research pipeline for a controlled comparison of ordinary autoencoders, physically constrained autoencoders and weighted POD on The Well `active_matter` dataset. It retains the students’ P02 global convolutional AE design and connects reconstruction, physical observables, frozen-latent probes and decoded forecasting.

**Start with the synthetic smoke test, then the real-data audit, then the first-seed pilot.** A successful software test does not establish a scientific advantage for physical constraints. Full-data training and GPU validation must run on the research server. See `VALIDATION.md` for exactly what was tested in the delivered version.

## 1. What this package does

- Reads the official train/valid/test HDF5 partitions with named fields and trajectory identities.
- Fits reversible, training-only normalization with balanced concentration/velocity/D/E blocks.
- Trains a shared AE warm-up and matched AE/PC-AE branches, with validation-only physical-weight selection.
- Fits an out-of-core weighted POD comparator at the same latent dimension.
- Evaluates reconstruction, cross-field consistency, three physical observables and simple AE output correction.
- Fits frozen-latent ridge probes and parameter/time-only controls; saves descriptive Pearson/Spearman latent–observable correlations.
- Trains the same residual latent MLP for every representation, plus linear autoregression and field persistence.
- Evaluates ten-step recursive forecasts and long rollouts from the first four frames. Saves an encoded-true-future reconstruction reference, explicitly labeled as using future information.
- Exports individual-seed tables, detailed measurements, conditional cluster-bootstrap comparisons and diagnostic figures.

The core release implements the official-split protocol. Activity extrapolation, defect tracking, spectra, additional constraint ablations and new architectures remain optional follow-up experiments; they are not silently included. Diagnostic plots are for checking experiments, not a claim that the final manuscript figures are complete.

## 2. Environment installation

Use a dedicated Conda environment named **`active_mattter`** (three consecutive `t` characters, as requested). The instructions below target the Linux NVIDIA GPU server, including the H100 setup discussed for the nuclei-classification project. Python **3.12** matches the CPU validation environment; GPU execution must still be verified on your server. On Windows, use WSL2 for the Bash runners; native Windows/GPU execution has not been tested here.

### Create and activate the Conda environment

Miniconda or Anaconda must already be available (`conda --version`). Open a terminal in the extracted `active_matter_repr/` project directory, which contains `pyproject.toml` and the shell runners:

```bash
conda create -n active_mattter python=3.12 pip -y
conda activate active_mattter
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
```

`PYTHONNOUSERSITE=1` prevents packages installed in your user Python directory from interfering with this environment. If the environment already exists, activate it instead of recreating it. If `conda activate` is unavailable in the current Bash shell, initialize that shell with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate active_mattter
```

### Install GPU-enabled PyTorch and the project

Use Conda to manage the environment and `python -m pip` inside it to install the official PyTorch **2.6.0 / CUDA 12.4** wheel:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[test]'
python -m pip check
amrepr --help
```

This is the CUDA wheel command listed in the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/). The server must have a compatible NVIDIA driver. `torchvision`, TensorFlow and The Well Python package are not needed by this pipeline. Do not install `requirements-tested-cpu.txt` into this GPU environment: it pins the CPU-only PyTorch build.

### Verify GPU access before the pilot

First inspect the GPUs and current processes:

```bash
nvidia-smi
```

Choose a GPU allocated to you. For example, to use physical GPU 1:

```bash
export CUDA_VISIBLE_DEVICES=1
python - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA unavailable: check the environment, driver and selected GPU"
print("Selected GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
x = torch.randn(128, 128, device="cuda:0", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all().item(), "Nonfinite GPU gradient"
print("GPU forward/backward check passed")
PY
```

Expect a CUDA-enabled version such as `2.6.0+cu124`, runtime `12.4`, and the selected NVIDIA GPU name. The Python executable should be inside the `active_mattter` environment. After `CUDA_VISIBLE_DEVICES=1`, physical GPU 1 is addressed as **`cuda:0`** within Python; therefore keep `device: cuda:0` in `configs/pilot.yaml`.

Then run the software check and a small GPU model profile:

```bash
bash run_smoke.sh
amrepr profile --config configs/pilot.yaml --batch-size 1
```

The synthetic smoke runner uses CPU by design, even in a GPU-enabled environment. The verification snippet and `profile` command explicitly exercise the GPU. After both pass, check the data destination in `configs/pilot.yaml` and start the pilot:

```bash
bash run_pilot.sh configs/pilot.yaml
```

Review the pilot results before launching the full configured three-seed comparison.

### Reuse the environment in a new terminal

```bash
conda activate active_mattter
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
```

Return to the project directory before running its shell scripts. The scripts use the active `python` by default. If you previously set a `PYTHON` variable for another project, point it at this environment:

```bash
export PYTHON="$CONDA_PREFIX/bin/python"
```

To record the actual server environment alongside your results:

```bash
mkdir -p environment_records
conda env export --no-builds > environment_records/active_mattter.yml
python -m pip freeze > environment_records/requirements-server.txt
nvidia-smi > environment_records/nvidia-smi.txt
```

### CPU-only alternative

For a machine without an NVIDIA GPU, use a separate environment and substitute this PyTorch installation command:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Install the project as above and set `device: cpu` in the research configuration. `requirements-tested-cpu.txt` records the exact original CPU test environment, including transitive dependencies; it is not a CUDA installation specification. The supplied GPU installation instructions were checked against official documentation, but creating this Conda environment and running CUDA must be done on your server.

## 3. Run the synthetic smoke test first

```bash
bash run_smoke.sh
```

Or choose a fresh output directory:

```bash
bash run_smoke.sh /absolute/path/to/smoke_check
```

The runner executes the tests, generates small explicitly synthetic periodic HDF5 files, runs the complete small pipeline, locks its choices, and exercises test evaluation. Its figures are marked **SYNTHETIC SOFTWARE TEST — NOT SCIENTIFIC RESULTS**. It uses a tiny 16×16 model and a two-dimensional latent to keep the software check affordable; those settings are not the research architecture. Existing fixture data are never overwritten. Use a new directory for another clean smoke test.

The test suite also exercises the actual 256×256, d=256 architecture with a forward/backward pass. That check establishes tensor/gradient functionality, not reconstruction quality or GPU memory requirements.

Run tests separately with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

The environment flag prevents unrelated globally installed pytest plugins from changing this suite. It does not skip the package’s tests. For a custom interpreter, set `PYTHON=/absolute/path/to/python` before any shell runner.

## 4. Prepare the real data and configuration

**Missing data are downloaded automatically during the audit**, from [The Well’s official Hugging Face repository](https://huggingface.co/datasets/polymathic-ai/active_matter). This happens when using `run_pilot.sh`, stage 1 of `run_pipeline.sh`, or `amrepr audit-data`. Existing complete local data work offline. The [dataset documentation](https://polymathic-ai.org/the_well/datasets/active_matter/) describes 256×256 spatial fields, 81 frames per trajectory, a length-10 periodic domain and stored interval 0.25.

Edit `configs/pilot.yaml`:

```yaml
data_root: /your/datasets/active_matter/data
auto_download: true
run_dir: ../runs/pilot_d256
device: cuda:0
workers: 4
threads: 4
batch_size: 8
effective_batch: 32
seeds: [1729, 2718, 31415]
```

`data_root` must directly contain `train/`, `valid/` and `test/`. Paths are resolved relative to the YAML file, not the terminal’s current directory. Scientific defaults are defined in `src/active_matter_repr/config.py`; `resolved_config.json` saves the complete effective configuration. Unknown keys are rejected rather than ignored.

If `data_root` does not exist, the downloader creates it and the three split folders. The supplied configurations default to `../data/active_matter/data`, relative to the configuration file. You may change this to a larger data disk before running. No new package dependency or separate manual download command is required.

The downloader saves the official repository revision in `download_manifest.json`, downloads only missing HDF5 files, displays progress, and checks available disk space. Newly transferred files must match the official size and SHA-256 checksum before becoming visible to the reader. Existing files are retained; a conflicting existing file size raises an error rather than overwriting your data. Existing datasets still undergo the normal scientific audit.

If the connection fails, rerun the same command. Revision-specific `.part` files preserve received bytes for an HTTP Range resume. If the server ignores Range, that partial transfer restarts safely. Keep `download_manifest.json` to resume the same revision. The full dataset requires tens of GB; the downloader prints the actual remaining size and checks for a 1 GiB reserve. Additional space is needed for training outputs.

Set `auto_download: false` to require manually provisioned local data. Synthetic smoke tests never contact the dataset repository. Starting at a later pipeline stage requires the already-completed audit; download checking belongs to stage 1.

Expected trajectory counts are 175/24/26. These are checked against the files, not assumed from the number of HDF5 files: one file can contain multiple trajectories. If counts differ, inspect the download and dataset version before altering expectations. The reader accepts `.h5` and `.hdf5` files directly inside each split directory. It requires full spatial fields, explicit x/y axes and periodic boundaries. Unsupported layouts fail with an error.

The default mapping is `concentration`, `velocity`, `D`, `E`, verified against public HDF5 metadata. Their canonical channel order is:

`c, vx, vy, Dxx, Dxy, Dyx, Dyy, Exx, Exy, Eyx, Eyy`.

### Important coordinate convention

The sampled public file stores endpoint-inclusive coordinate labels (`linspace(0, 10, 256)`) for periodic fields. Using their adjacent difference gives 10/255. The operators instead use the verified physical domain period divided by the number of grid points, **L/N = 10/256**. Both coordinate-label spacing and operator spacing are saved in the manifest. On the sampled real frame this choice substantially reduced the strain mismatch; the complete train/validation audit still has to verify it across the dataset. No last row/column is discarded.

## 5. Profile and audit before training

First check model memory and gradients on your selected device:

```bash
amrepr profile --config configs/pilot.yaml --batch-size 1
```

This writes `profile.json`. It measures one synthetic forward/backward pass, excluding optimizer state and I/O; it is not an epoch-time prediction. Increase the profile batch only when memory permits. The reference dense bottleneck is intentionally retained and is relatively large.

Then run the data audit:

```bash
amrepr audit-data --config configs/pilot.yaml
```

The audit hashes source files, identifies trajectories, checks duplicates and splits, validates grid/time/field metadata, fits training normalization, measures training observable scales, and checks physical identities on train/validation fields. Test fields are not used for fitting or model selection; the manifest does read first frames for duplicate detection. Historical test inspection is disclosed in the experiment lock.

Identical initial states at identical control parameters raise an error. Shared initial conditions at different parameter values can be valid experimental design and are instead recorded in `shared_initial_conditions`; they are not automatically deduplicated. Review these relationships when interpreting generalization.

Inspect `audit.json`, `manifest.json`, `transform.json` and `observable_stats.json`. Failure is intentional if a physical identity or data convention is inconsistent. Default residual tolerances are engineering gates, not universal physical constants. Do not simply loosen them until a run passes; determine whether the issue is field meaning, derivative convention, resolution, invalid data or an inappropriate identity. A failed gate prevents training.

The initial audit reads the full files for hashing and streams the training/validation frames. It is an I/O-heavy stage. Later commands verify saved file sizes/timestamps; rerunning the audit performs the full hash verification. No mechanism can protect against a user deliberately altering data while preserving metadata, so retain immutable source data.

## 6. First-seed scientific pilot

```bash
bash run_pilot.sh configs/pilot.yaml
```

This runs only the **first configured seed**, even though all three confirmation seeds are declared from the beginning. It runs audit, representations, probes/correlations, forecasters, validation, long-rollout diagnostics and reporting. It does not evaluate the real test set.

The pilot performs:

1. A 20-epoch data-only warm-up; the last warm-up state is the common branch point.
2. Ordinary AE and PC-AE branches at λ=0.01, 0.1, 1, for up to 60 continuation epochs each with identical budgets. Physical weight ramps over five epochs. Both arms become checkpoint-eligible after the ramp window.
3. Branch checkpoint selection by validation balanced reconstruction error. PC weight selection by Q_rec, with the guard `R_rec(PC) <= 1.05 * R_rec(AE)`; ties use R_rec then smaller λ.
4. Weighted POD, frozen-latent probes, matched MLP/linear temporal models and persistence comparison.

If no PC candidate passes the guard, the pipeline saves the candidates and stops. That is a scientific decision point, not a software crash to bypass. The selected AE/PC-AE snapshots may reconstruct poorly even when technically valid; inspect reconstruction errors, physical observables and latent diagnostics before expanding compute.

Keep these main pilot outputs for review:

- `penalty_candidates.json` and `selection.json`.
- `reports/valid/summary.csv` and `paired_statistics.json`.
- `analysis/1729/*/latent_correlations.json` and `probes.json`.
- `evaluation_valid.json` and `rollouts_valid.json`.
- Training histories and `profile.json`.

One seed is preliminary evidence. The bootstrap conditions on the fitted seeds and resamples parameter groups; it does not turn a one-seed pilot into a robust training-variability result.

## 7. Continue to the full configured comparison

After reviewing a satisfactory pilot, continue using **the same configuration and run directory**:

```bash
bash run_pipeline.sh configs/pilot.yaml
```

Completed compatible training resumes without additional epochs. This adds the remaining configured seeds, with the pilot-selected physical weight fixed, then regenerates analyses for all completed seeds. This avoids retraining the first seed solely to expand the study.

For an unchanged three-seed protocol there are three common warm-ups and eight continuation branches in total: four pilot branches and two branches for each additional seed. There are nine nonlinear temporal fits, one for each representation/seed combination; POD fitting is reused. Ordinary-AE output correction reuses its representation and predictor.

`configs/confirmation.yaml` is an alternative independent three-seed run template. It is not needed to extend an existing pilot. Do not switch to a new run directory if your aim is to reuse a compatible pilot.

### Continue from stage X

```bash
bash run_pipeline.sh configs/pilot.yaml START_STAGE END_STAGE
```

| Stage | Operation | Prerequisite |
|---|---|---|
| 1 | Data/physics audit | Source HDF5 files |
| 2 | AE/PC-AE/POD fitting | Passed audit |
| 3 | Latent caches, probes, correlations, reconstruction analysis | Selected representations |
| 4 | MLP and linear autoregression | Representation analysis |
| 5 | Validation evaluation and long-rollout diagnostics | Fitted forecasters |
| 6 | Tables, statistical comparisons and diagnostic figures | Saved evaluation |

For example, `bash run_pipeline.sh configs/pilot.yaml 4 6` continues from temporal fitting. Omitted endpoints default to stages 1–6. These are pipeline stage numbers, not the manuscript project’s Step 1–6.

Checkpoint resume is at completed epoch boundaries. An interrupted partial epoch is replayed from the preceding consistent checkpoint. Both Adam state and Torch RNG states are restored; loader ordering is derived from seed and epoch. The tests compare a simulated interrupted run against an uninterrupted CPU run. Do not expect bitwise identity across different GPU models, PyTorch builds or numerical backends.

Do not run two processes writing to the same `run_dir`. Use separate run directories for independent protocols. Changing scientific settings or source code requires a new run; changing `device`, `workers`, `worker_transport` or CPU thread count does not change the scientific configuration identity. Keep such resource changes in your execution notes. Changing microbatch size is treated conservatively as a protocol change.

### Background execution and GPU selection

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 nohup bash run_pilot.sh configs/pilot.yaml > logs/pilot.log 2>&1 &
tail -f logs/pilot.log
```

With `CUDA_VISIBLE_DEVICES=1`, that physical GPU becomes `cuda:0` inside the process, so keep `device: cuda:0` in the YAML. The runners set a deterministic cuBLAS workspace configuration before Python starts. They never occupy both GPUs automatically. Select a free GPU according to your shared-server rules.

## 8. Freeze choices and evaluate the test set

Once all configured seeds are complete and the validation results have been reviewed:

```bash
amrepr lock --config configs/pilot.yaml
amrepr evaluate --config configs/pilot.yaml --split test
amrepr diagnose-rollouts --config configs/pilot.yaml --split test
amrepr report --config configs/pilot.yaml --split test
```

The lock fingerprints selected artifacts, code, configuration and endpoint definitions. Test evaluation rejects a stale or missing lock. It is an auditable workflow guard, not an access-control system or a claim that the historical test set was pristine. Do not use test results to choose λ, seeds, checkpoints or example cases, or present a later test-driven change as prespecified.

## 9. Scientific definitions

All physical quantities use inverse-transformed fields. Mean subtraction is fixed per component; scales are uncentered training RMS values shared within the c/v/D/E blocks. Near-constant concentration is therefore not inflated by per-frame min–max or tiny-variance standardization.

| Quantity | Definition/use |
|---|---|
| R_rec | Mean of four block MSEs, each divided by its training RMS squared and averaged over that block’s components |
| Physics loss | Mean of nondimensional D symmetry, E symmetry, velocity divergence and E−sym(∇v) squared residual families |
| K_v | Half the spatial mean of vx²+vy²; velocity intensity, not a conserved energy claim |
| Enstrophy | Half the spatial mean of (∂x vy−∂y vx)² |
| Mean local order | Spatial mean of sqrt((Dxx−Dyy)²+4 Dxy_sym²)/c, conditional on verified D convention |
| Q_rec | Mean absolute error of those observables, divided by each observable’s training standard deviation |
| Q_probe | Same standardized errors from a frozen-latent ridge probe |
| F_10 | Mean over leads 1–10 of equally weighted normalized velocity and D RMSE |

Effectively constant observables are excluded before model comparison and recorded. Invalid concentration/order or nonfinite forecasts are disclosed; incomplete evaluations are not averaged as if failed samples were absent. Physical admissibility diagnostics do not clamp or repair predictions silently.

Primary aggregation averages observations within trajectories, trajectories within parameter groups, then groups equally. `aggregation_sensitivity.json` also reports equal-trajectory averages. Detailed CSV/JSON records permit further checks. R² in probe output is separately labeled as pooled-frame R²; it is not the hierarchical primary endpoint. Descriptive correlations have no frame-independence p-values and do not establish that an individual latent coordinate has a unique physical meaning.

The temporal model receives four consecutive latents and known α/ζ. Those parameters do not enter the AE. Encoder and decoder parameters remain frozen during temporal fitting, but the decoder remains in the autograd graph so decoded-field loss trains the temporal model. Prediction recursively advances latent states without re-encoding them. MLP and linear AR selection use validation F_10. No true future state enters a forecast; the encoded-true-future curve is a separately labeled reconstruction diagnostic.

The AE uses circular padding in encoder convolutions and preserves the original transposed-convolution decoder, with a single linear output head. The decoder architecture itself is not claimed to enforce periodic equivariance; both AE arms share it. Physical derivatives use periodic operators with an explicit zero Nyquist multiplier for an even-grid real first derivative. Finite-difference sensitivity requires its own configuration and run.

## 10. Outputs and code map

| File/module | Purpose |
|---|---|
| `data.py`, `config.py` | HDF5 reader, split manifest and strict configuration |
| `transforms.py`, `physics.py` | Reversible scales, balanced losses, derivatives and observables |
| `models.py` | Common AE, weighted POD and residual latent MLP |
| `training.py`, `representations.py` | Matched training/resume, POD, caches, probes and correlations |
| `forecasting.py`, `evaluation.py` | Recursive dynamics, long diagnostics and common measurements |
| `pipeline.py`, `artifacts.py` | Stage orchestration, selections, fingerprints and atomic writes |
| `statistics.py`, `reporting.py` | Group aggregation, cluster bootstrap, tables and diagnostic plots |
| `synthetic.py`, `tests/` | Analytic fixtures and actual execution tests |

Within a run, `best.pt` and `last.pt` are distinct. Warm-up branches use `last.pt`; continuation and temporal evaluation use selected `best.pt`. Caches contain latent arrays, observables and explicit trajectory/frame identities. They are invalidated when the representation/configuration changes. New code does not import the archived notebooks or load their missing/ambiguous checkpoints.

`evaluation_*.json` retains reconstruction block errors, physical residuals, admissibility flags, probe records and lead-wise errors for MLP/linear/persistence controls. `reports/*/measurements.csv` is a flat export of primary endpoints; consult the JSON for all controls and component diagnostics. `rollouts_*.json` stores physical descriptors and errors throughout the first-origin rollout. Nothing in report generation retrains or selects models.

## 11. Compute, limitations and troubleshooting

- **POD RAM:** Incremental PCA avoids a feature-by-feature covariance matrix, but its SVD is still substantial. At 256×256, one float32 flattened snapshot is approximately 2.75 MiB; 512 rows alone require about 1.38 GiB, with additional basis and SVD workspace. Leave substantially more free RAM, preferably 16–32 GiB or more, and profile it. `pod_batch` must allow at least d rows per update. The implementation reports an incremental approximation, not an exact global SVD guarantee.
- **Checkpoints:** AE weights plus Adam states occupy much more space than a weight-only file. Allow many GiB for three warm-ups and eight branches, with separate best/last states. Retain them until the study is finalized.
- **Data workers:** Each spawned worker opens its own bounded read-only HDF5 handle cache. `workers: 0` is simplest for initial debugging. `worker_transport: numpy` uses pipe serialization instead of PyTorch tensor-sharing sockets on restricted hosts; it can add transfer overhead. Standard `torch` mode is the server default. A worker timeout raises an error rather than hanging indefinitely.
- **Precision:** This first release uses float32 training and higher-precision pooled statistics. AMP/compilation are not silently enabled before GPU/operator validation. CPU/GPU throughput and optimization equivalence must be checked before adding either.
- **CUDA unavailable:** Check that the active interpreter has the CUDA wheel and sees the selected GPU. The package does not silently fall back to CPU when CUDA was requested.
- **Out of memory:** Reduce resource use before launching the scientific run. If changing an already-started microbatch/scientific configuration, use a new run directory and document it. Avoid concurrent jobs competing for RAM during POD.
- **Missing cache/checkpoint or stale configuration:** Resume the prerequisite stage using the original configuration. Never rename unrelated weights to satisfy a filename or delete fingerprints to bypass the check.
- **No eligible PC candidate:** Review `penalty_candidates.json`. A failed guard is a result requiring diagnosis, not permission for an unrestricted hyperparameter search.

The code supplies a reproducible experimental instrument. It does not establish novelty, interpretability, physical closure, publication readiness or a positive PC-AE effect without the real controlled results.

## 12. Provenance and references

The AE architecture is derived from the students’ P02 `1Training_final.py`; the physical-constraint comparison is retained from their later P02 branches. Normalization, physical operators, evaluation, lifecycle management and latent forecasting have been reimplemented to address the audit. The original archive is preserved separately. This distribution does not assign a new license to the students’ original work; agree authorship/licensing before public release.

- [The Well dataset documentation](https://polymathic-ai.org/the_well/datasets/active_matter/).
- [The Well HDF5 format](https://polymathic-ai.org/the_well/data_format/).
- [The Well benchmark paper](https://arxiv.org/abs/2412.00568).
- Maddu, Weady & Shelley, *Learning fast, accurate, and stable closures of a kinetic theory of an active fluid*, Journal of Computational Physics 504, 112869 (2024), as cited by the dataset documentation.

Use the audited experimental strategy when interpreting these outputs and planning subsequent work.
