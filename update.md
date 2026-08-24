# 2026-08-15 20:50 Asia/Dhaka — MedSentiX notebook/code implementation

## Completed

- Implemented `utils/device.py` with the required CUDA -> MPS -> CPU device selection and random seed utility using seed `42`.
- Implemented `models/baselines.py` with all seven specified baselines:
  - SVM + TF-IDF
  - Double BiGRU + GloVe
  - BiLSTM + GloVe
  - BiLSTM-CNN + GloVe
  - BERT + BiLSTM
  - RoBERTa + BiLSTM
  - BioBERT standalone
- Implemented `models/medsentix.py` with:
  - BioBERT encoder
  - 2-layer BiLSTM
  - guided 4-head aspect-aware attention
  - auxiliary aspect heads
  - class-weighted training
  - validation-accuracy checkpointing
  - early stopping
  - ECE, cross-dataset generalization, ablation, ABSA validation, SHAP, and attention-entropy helpers
- Generated all 13 required modular notebooks:
  - `01_preprocessing_drugscom.ipynb`
  - `02_preprocessing_druglib.ipynb`
  - `03_preprocessing_webmd.ipynb`
  - `04_splitting.ipynb`
  - `05_baselines.ipynb`
  - `06_train_medsentix_D.ipynb`
  - `07_train_medsentix_DW.ipynb`
  - `08_train_medsentix_DDL.ipynb`
  - `09_train_medsentix_Full.ipynb`
  - `10_evaluation.ipynb`
  - `11_ablation.ipynb`
  - `12_absa_validation.ipynb`
  - `13_shap_analysis.ipynb`
- Added the requested consolidated paper notebook:
  - `notebooks/MedSentiX_full_pipeline.ipynb`
- Added `scripts/generate_notebooks.py` so the notebook set can be regenerated consistently.
- Added `models/__init__.py` and `utils/__init__.py` for stable package imports from notebooks.

## Verification

- Python syntax check passed for:
  - `utils/device.py`
  - `models/baselines.py`
  - `models/medsentix.py`
  - `scripts/generate_notebooks.py`
- Jupyter notebook validation passed for all 14 notebooks.
- All notebook code cells parse as valid Python.
- Scans found no hardcoded `.cuda()` / `.mps()` calls, no `localStorage`, and no absolute `/Users/...` project paths.
- The combined notebook does not import or execute the 13 modular notebooks and does not import from `models` or `utils`; the implementation code is inlined directly in notebook cells.

## Not Run

- Full runtime execution/training was not run because the active local Python environment is missing `torch`.
- No accuracy, F1, SHAP, or ablation result values were fabricated. Result tables and figures are produced only when the notebooks are executed against the real datasets and trained checkpoints.
