# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

iDILI-Predict — a Python package for predicting drug-induced liver injury (DILI) from Cell Painting morphological features. Takes well-level CellProfiler data, normalizes via MAD robustize, and trains AutoGluon models with compound-grouped cross-validation.

## Running

```bash
# CLI
python -m idili_predict --input example_data/63D_well_median_fixed.csv --dtypes example_data/column_dtypes.csv --output-dir results/

# Install as package
pip install -e .
pip install -e ".[notebooks]"  # for UMAP/scanpy notebooks
```

## Package Structure

- `idili_predict/cli.py` — argparse entry point (`--input`, `--dtypes`, `--output-dir`, `--time-limit`, `--presets`, `--num-bag-folds`)
- `idili_predict/pipeline.py` — orchestrator: load -> impute -> normalize -> train AutoGluon -> evaluate OOF
- `idili_predict/normalize.py` — MAD robustize via pycytominer, applied per plate
- `idili_predict/utils.py` — `load_column_types()`, `encode_labels()`, `impute_plate_medians()`, `validate_required_columns()`
- `notebooks/walkthrough.ipynb` — interactive step-by-step demo
- `notebooks/umap_visualization.ipynb` — UMAP with StandardScaler (not MAD)

## Key Decisions

- **Normalization:** MAD robustize for ML pipeline; StandardScaler for UMAP visualization
- **Group-aware splitting:** `Metadata_CMPD` prevents compound leakage across CV folds
- **Class balancing:** Sample weights (not SMOTE) computed as `n_samples / (n_classes * n_per_class)`
- **Column types:** `column_dtypes.csv` with `ColumnName`/`ColumnType` columns defines feature vs metadata

## Dependencies

Core: `autogluon[tabular]`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `pycytominer`, `tqdm`
Notebooks: `umap-learn`, `scanpy`, `harmonypy`, `matplotlib`, `anndata`
