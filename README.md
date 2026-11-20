# Bladder EIS Reproducibility Repository

**Code and supporting results for:**  
**Reference-Induced Leakage in Subject-Referenced Classification of Repeated Measurements: Theory, a Pre-Training Index, and Evidence from Three Datasets**

**Authors:** Syaiful Anam (Universitas Brawijaya), Ahmad Zarkasi (Universitas Mulawarman), and Ahmad Hakiim Jamaluddin (Universiti Putra Malaysia).

This repository contains a full Jupyter/Google Colab reproduction notebook, the companion Python scripts, precomputed results and manuscript figures for the study. The analysis investigates **pseudo-replication, tissue-sample-wise versus spectrum-wise referencing, frequency-selection leakage, fully nested validation, patient-level uncertainty and simulations** in bladder-tissue electrical impedance spectroscopy (EIS) classification.

> **Reproducibility note:** Precomputed outputs are included from the supplied research package. Its `results/consistency_check.csv` records **464 PASS entries out of 464 manuscript comparisons**. This is a *reported, saved check*, not a claim that the complete computation has been rerun independently in this repository preparation process.

## 1. Repository structure

```text
bladder-eis-reproducibility/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── .github/workflows/validate.yml
├── Bladder_EIS_IJS_Full_Reproduction.ipynb  # complete reproduction notebook
├── eis_pipeline.py, eis_groups.py            # data / referencing / classification
├── cole_analysis.py                          # Cole-model representations
├── ijs_analysis.py, ijs_revision*.py         # core and extended analyses
├── ijs_permutation.py                        # fixed-configuration permutation null
├── ijs_selection_permutation.py              # selection-aware permutation test
├── ijs_simulation*.py, sim_summary.py        # simulation studies
├── nested_core.py, ijs_nested_sensitivity.py # fully nested pipeline selection
├── article_tables.py, make_supp_tables.py    # tables (generated from notebook)
├── ijs_figures.py, make_fig*.py               # figures
├── prepare_cached_results.py                # stage tracked results for notebook
├── verify_repository.py                     # fast package/content validation
├── figs/                                     # five manuscript figures
└── results/                                  # saved tables, CSV, JSON, figures
    └── revision/                             # saved long-running analyses
```

Other source scripts, including the supplementary checks and reviewer analyses, are also stored at the repository root. This **flat Python-module layout is intentional**: the supplied notebook uses `%%writefile` and invokes the scripts by their relative filenames.

## 2. Requirements

- Python 3.10+ (the original authors' recorded verified environment was **Python 3.13.16**).
- `git` for retrieving the openly available source dataset.
- JupyterLab, Jupyter Notebook, or Google Colab.
- Python dependencies are listed in `requirements.txt`.

**Environment reported in the original code README:** Python 3.13.16; NumPy 2.5.3; pandas 3.0.5; SciPy 1.18.1; scikit-learn 1.9.1; XGBoost 3.4.1; Matplotlib 3.11.2. These are provenance details from the supplied package, **not a fresh environment verification**. Dependency ranges are provided for installation convenience; exact numerical agreement may be environment-dependent.

## 3. Quick start — recommended (use saved long-running results)

Clone your GitHub repository or extract the ZIP, then open a terminal **inside this repository's root directory**.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_cached_results.py
jupyter lab Bladder_EIS_IJS_Full_Reproduction.ipynb
```

In Jupyter, select the Python environment used for installation, then choose **Run All Cells**. The notebook will download the source dataset from the public GitHub repository if it is absent. Sections 1–15 still perform computations; long-running Section 16 steps are skipped if their expected result files already exist under `outputs_ijs/revision/`.

**Why run `prepare_cached_results.py`?** It copies the supplied `results/` into `outputs_ijs/` (the output directory used by the notebook/scripts). Files already in `outputs_ijs/` are left unchanged by default, protecting any new calculations. To restore supplied results intentionally, run `python prepare_cached_results.py --overwrite`.

You can review the included results directly in `results/` without installing Python packages or downloading the dataset.

### Google Colab

Open the notebook from your GitHub repository in Colab. To use the saved outputs, start by cloning your repository in a *setup cell*, then switch to its folder, and run the cache preparation script **before** executing the reproduction cells. For example (replace the two placeholders):

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
%cd /content/YOUR_REPOSITORY
!python prepare_cached_results.py
```

Run the other notebook cells in order afterward. If you open the notebook from GitHub but run it directly in `/content` without checking out the repository, it can still write scripts and download the dataset, **but its saved `results/` are not automatically available**, so time-consuming analyses may be recomputed.

## 4. Full recomputation from raw data

The raw EIS measurements are **not bundled in this repository**. The notebook automatically clones:

- Dataset repository: https://github.com/cayobro/electrical-impedance-data-bladder
- Original dataset/article: Veil et al. (2025), *Scientific Reports*, **15**, 825. https://doi.org/10.1038/s41598-024-84844-9

For running standalone scripts without the notebook, first obtain the dataset:

```bash
git clone https://github.com/cayobro/electrical-impedance-data-bladder.git
python ijs_analysis.py
```

To **recompute rather than reuse** a cached experiment, delete its corresponding file(s) in `outputs_ijs/` or `outputs_ijs/revision/` before running the notebook. Avoid restoring the cache afterward until your rerun has completed. The notebook contains the detailed execution order and generates its Python modules from the embedded code.

**Runtime warning (as reported by the authors on two CPU cores):** The main sections take roughly 15–25 minutes. A full run of all additional analyses is substantially longer; the **selection-aware 100-permutation test alone can take around 10–12 hours**, and a larger 1,000-permutation test can take several days. Scripts implement checkpoints so longer experiments can resume. The provided selection-aware result is based on **100 permutations**, not 1,000 (`results/revision/selection_permutation_results.json`, `n_perm = 100`). The separate fixed-configuration permutation analysis uses 1,000 permutations.

For a larger selection-aware permutation test (requires substantial compute):

```bash
python ijs_selection_permutation.py 1000
```

The original paper-associated checks and displayed figures reflect the supplied 100-permutation configuration; changing the permutation count will require adjusting related report/check values before treating the original consistency checks as applicable.

## 5. Validate saved outputs and regenerate tables

To check repository completeness, JSON/CSV readability, Python syntax and the **saved** consistency-check summary:

```bash
python verify_repository.py
```

To rebuild Tables 1–4 from supplied cached outputs without rerunning all experiments:

```bash
python prepare_cached_results.py
python article_tables.py
```

This saves `outputs_ijs/article_tables.xlsx`, `outputs_ijs/article_tables.tex`, and `outputs_ijs/article_tables_check.csv`. It requires `pandas` and `openpyxl` from `requirements.txt`.

For the bladder and simulation supplementary tables (Tables S2–S4, S6–S8, and S10–S12 of Online Resource 1):

```bash
python make_supp_tables.py
```

The full notebook generates Figures 1–4 and S3 and contains the manuscript-wide consistency checks. Output files produced by new runs go into `outputs_ijs/` and are ignored by Git. The *supplied reference results* remain tracked under `results/`.


## 6. Replication on the head-and-neck bioimpedance dataset

Scripts `hn_core.py`, `hn_icc.py`, `hn_permutation.py` and `hn_splits.py` reproduce the replication section of the article on the public **Electric_Bioimpedance_H&N_Dataset** (Missale 2022, Mendeley Data, https://doi.org/10.17632/b9t274zxgz.2, CC BY 4.0). The dataset is not bundled; download `Dataset_EBI_3.csv` into the repository root (or set the environment variable `HN_DATA` to its path).

```bash
python hn_icc.py                       # ICC, design effect, effective N per feature
python hn_splits.py lr                 # random vs unit-grouped vs leave-one-patient-out validation
python hn_splits.py hgb
python hn_permutation.py lr 1000       # within-pair permutation null, references rebuilt
python hn_index.py                     # reference-leakage index (Corollary 1) vs empirical null AUC
OMP_NUM_THREADS=1 python hn_permutation.py hgb 50 11 _a   # boosting null, run in two halves
OMP_NUM_THREADS=1 python hn_permutation.py hgb 50 22 _b
```

The dataset has no specimen or puncture identifier, so the unit is the patient x tissue type x status cell. Boosting uses scikit-learn `HistGradientBoostingClassifier`. Saved outputs are in `results/hn/` (`perm_hgb.json` combines the two halves `perm_hgb_a.json` and `perm_hgb_b.json`). Runtime on two CPU cores: about 6 minutes for the logistic-regression null and about 45 minutes for the boosting null.

## 7. Validation of the reference-leakage index (simulation grid)

`sim_index_grid.py` compares the reference-leakage index of Corollary 1, AUC0 = F_{d,d}(rho), with the Monte Carlo null AUC in 1200 conditions (Grid A: ICC x repeats x reference units x features x design, mean and median references; Grid B: correlated features with equal or unequal ICCs). `make_fig_index.py` draws Fig. S1. Outputs are in `results/index_grid/` (`grid_A.csv`, `grid_B.csv`, `summary.json`, `table_S10.csv`, `FigS1_Index_Validation.png`). Runtime: about 5 minutes on two CPU cores.

```bash
python sim_index_grid.py
python make_fig_index.py
```

## 7a. Reference-leakage index as a stand-alone tool

`leakage_index.py` implements the reference-leakage index (Corollary 1; Algorithm 1 of the article) for any repeated-measures data set with subject, unit and class labels:

```python
from leakage_index import reference_leakage_index
res = reference_leakage_index(X, subject, unit, y, design="LSO")   # "LSO", "UW" or "SM"
print(res["AUC0"])   # 0.5: the referencing design hands no null discrimination to a classifier
```

It reproduces the index reported for the wearable stress dataset (0.603 for leave-self-out referencing). `python leakage_index.py` runs a small self-check on simulated data.

## 8. Replication on the wearable stress dataset (PhysioNet Non-EEG)

`noneeg_io.py` reads the WFDB files of the PhysioNet *Non-EEG Dataset for Assessment of Neurological Status* v1.0.0 (Birjandtalab et al., SiPS 2016; https://physionet.org/content/noneeg/1.0.0/, ODC-By 1.0) without the `wfdb` package; `noneeg_analysis.py` runs the replication (units: four relaxation and three stress stages per subject; repeats: 30-s windows). Download and unzip the dataset, then:

```bash
D=path/to/non-eeg-dataset-for-assessment-of-neurological-status-1.0.0
python noneeg_analysis.py $D main none            # ICC, granularity, observed BA, index (also: noacc, nophysical)
python noneeg_analysis.py $D main lr 1000         # logistic-regression permutation null (also: noacc, nophysical)
OMP_NUM_THREADS=1 python noneeg_analysis.py $D main hgb 50 11 _a   # boosting null, two halves
OMP_NUM_THREADS=1 python noneeg_analysis.py $D main hgb 50 22 _b
```

Outputs are in `results/noneeg/` (`noneeg_windows.csv` holds the window features; `perm_main_hgb.json` combines the two halves; `permutation_table.csv` summarizes all nulls). Runtime on two CPU cores: about 10 minutes per logistic-regression null and 45 minutes for the boosting null.

## 8a. Index calibration and semi-synthetic known-effect experiment

`sim_index_calibration.py` shows that the index predicts the null balanced accuracy of a trained classifier (140 simulated conditions; about 1 minute). `semisynthetic_noneeg.py` injects a known effect into real relaxation units of the wearable stress dataset and compares each design's estimate with the deployment accuracy on fresh units (200 replicates per cell; about 3 minutes per value of n_h). `make_fig_s2.py` draws Fig. S2.

```bash
python sim_index_calibration.py
python semisynthetic_noneeg.py 1 200; python semisynthetic_noneeg.py 2 200; python semisynthetic_noneeg.py 3 200
python make_fig_s2.py
```

Outputs: `results/index_grid/calibration.csv`, `calibration_summary.json`; `results/noneeg/semisynthetic_nh{1,2,3}.csv`, `semisynthetic_summary.csv`, `FigS2_Index_Calibration_Semisynthetic.png`. The semi-synthetic script reads `results/noneeg/noneeg_windows.csv`, so run `noneeg_analysis.py` first.

## 8b. Common normalization practices and screening values of the index

`noneeg_practices.py` compares six per-subject z-scoring practices on the wearable stress data (no normalization; all windows of the subject; relaxation baseline with the query included; and the leave-self-out, unit-wise, and size-matched baselines). Part A uses the real labels (leave-one-subject-out, logistic regression and boosting, 1000-permutation null with every normalization recomputed; about 14 minutes on two cores). Part B repeats the semi-synthetic known-effect experiment for n_h = 2, 3 and effect 0 or 1 (200 replicates; about 5 minutes). `make_lookup.py` writes the screening values of the index (Table 5 of the article, Table S18). `zscore_index.py` validates the index for z-scored references (`leakage_index.reference_leakage_index(..., zscore=True)`; designs INC, LSO, UW, SM) in 96 simulated conditions and on the wearable data.

```bash
python noneeg_practices.py A 1000
python noneeg_practices.py B 200
python make_lookup.py
python zscore_index.py both      # index for z-scored references (Remark 1); about 30 s
python noneeg_practices_hgb_null.py 100   # boosting null of the practices; about 50 min, resumable
python sim_zscore_calibration.py 20       # classifier calibration on signed z-scores; about 30 min
```

Outputs: `results/noneeg/practices_real.csv`, `practices_semisynthetic.csv`, `practices_semisynthetic_summary.csv`; `results/index_grid/lookup_LSO.csv`, `zscore_grid.csv`; `results/noneeg/zscore_index_real.csv`, `practices_hgb_null.csv`; `results/index_grid/zscore_calibration.csv`, `zscore_calibration_summary.json`. Baseline z-scoring (query included or leave-self-out) shifts the null to 0.54-0.55 for logistic regression and 0.66-0.69 for boosting and overstates deployment accuracy by 0.05-0.09; size-matched normalization stays within 0.01 of the truth. Full-session z-scoring keeps the null at chance but overstates deployment accuracy by 0.10-0.17 when deployment normalizes against a baseline only (transductive optimism, not detectable by the permutation null).

## 9. Replication on WESAD (optional; wearable stress detection; run on Google Colab)

`WESAD_Reference_Leakage.ipynb` downloads WESAD (Schmidt et al., ICMI 2018; https://doi.org/10.24432/C57K5T), extracts chest-sensor features from non-overlapping 60-s windows, and repeats the analyses of the article with the five protocol segments of each subject as units (baseline, amusement, meditation 1 and 2 as reference class; stress as target class): ICC and design effect, validation granularity, leave-self-out, unit-wise and size-matched referencing with leave-one-subject-out validation, per-subject baseline z-scoring, within-subject permutation nulls, and the reference-leakage index. Runtime on Colab: about 60-75 minutes. It writes `wesad_outputs.zip`.

`wesad_synthetic_test.py` creates synthetic data in the WESAD pickle format for a smoke test (`python wesad_synthetic_test.py /tmp/syn null`, then run the notebook code with `WESAD_TEST=1 WESAD_DIR=/tmp/syn`). Synthetic results are not results.

## 10. What is and is not included

**Included:** full updated notebook; original Python analysis modules; `article_tables.py` recovered directly from the newer supplied notebook; the submitted package's precomputed CSV/JSON outputs; figures; `LICENSE` and `CITATION.cff`.

**Excluded intentionally:** the manuscript PDFs/LaTeX drafts, cover letter, editorial email drafts and internal revision correspondence from the original submission ZIP. These are not necessary to reproduce the code and should not be uploaded to a public repository by accident. The public upstream dataset itself is also not repackaged.

**Privacy/data caution:** The source data are public research data, but they include biological specimen metadata. Review the upstream dataset's access and citation conditions before reuse. Loading externally downloaded Python pickle files should only be done for sources you trust.

## 11. Citation and license

Software license: **MIT** (`LICENSE`), as specified in the supplied code package. Please cite the scientific article and this code repository using `CITATION.cff`.

The MIT license covers the code as supplied. **No additional licensing rights over the separately hosted dataset or journal manuscript are asserted here.**

## 12. Publish to GitHub

1. Extract `Bladder_EIS_GitHub_Ready.zip`.
2. Create a new GitHub repository (preferably empty, without an autogenerated README or license).
3. Upload **the contents of `bladder-eis-reproducibility/`** (including the hidden `.gitignore` and `.github/`), or use:

   ```bash
   cd bladder-eis-reproducibility
   git init
   git add .
   git commit -m "Initial release: bladder EIS reproducibility package"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

4. Check the public repository preview (notebook, figures and README), then create a versioned GitHub release. If desired, connect GitHub to Zenodo and archive that release to obtain a DOI.

---

### Ringkasan penggunaan (Bahasa Indonesia)

Paket ini sudah memiliki notebook utama, seluruh kode Python, hasil eksperimen yang disimpan, lima gambar, berkas sitasi, lisensi, dan README. **Setelah mengekstrak ZIP:** instal dependensi (`pip install -r requirements.txt`), siapkan cache (`python prepare_cached_results.py`), lalu buka `Bladder_EIS_IJS_Full_Reproduction.ipynb` melalui Jupyter. Untuk pemeriksaan cepat tanpa mengulang eksperimen, jalankan `python verify_repository.py`.
