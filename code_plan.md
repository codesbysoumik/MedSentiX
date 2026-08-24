# MedSentiX — Full Coding Specification

## For Codex Code — Incremental File-by-File Implementation

---

## PROJECT OVERVIEW

**Model Name:** MedSentiX
**Task:** Aspect-Based Sentiment Analysis (ABSA) of patient drug reviews
**Domain:** Healthcare NLP / Pharmacovigilance
**Paper Title:** MedSentiX: Hybrid BioBERT-BiLSTM with Guided Multi-Head
Aspect-Aware Attention and SHAP Explainability for Drug Review
Sentiment Classification

---

## CRITICAL RULES FOR ALL CODE

1. Every file must include comments explaining what each block does
2. Every file must use the device utility — never hardcode cuda or mps
3. Every file must handle Mac MPS and NVIDIA CUDA automatically
4. Random seed is always 42 — set at the top of every file
5. All paths are relative to the project root /Users/shadiptopranto/Desktop/MedSentiX
6. Never use localStorage or browser storage APIs
7. All results save to results/tables/ or results/figures/
8. All model checkpoints save to checkpoints/

---

## DEVICE UTILITY — MUST READ BEFORE ANY CODE

File: utils/device.py

This is the first file to create. Every other file imports from here.
The device function must:

- Check torch.cuda.is_available() first — NVIDIA GPU
- Then check torch.backends.mps.is_available() — Apple Silicon
- Fall back to CPU if neither available
- Print which device was selected on call
- Return a torch.device object

Usage in every other file:

```python
from utils.device import get_device
DEVICE = get_device()
```

---

## FILE ARCHITECTURE

MedSentiX/
│
├── data/
│ ├── raw/
│ │ ├── drugsComTrain_raw.csv
│ │ ├── drugsComTest_raw.csv
│ │ ├── drugLibTrain_raw.tsv
│ │ ├── drugLibTest_raw.tsv
│ │ └── webmd_raw.csv
│ │
│ ├── processed/
│ │ ├── drugs_com_clean.csv
│ │ ├── druglib_clean.csv
│ │ └── webmd_clean.csv
│ │
│ └── splits/
│ ├── drugs_com_train.csv
│ ├── drugs_com_val.csv
│ ├── drugs_com_test.csv
│ ├── webmd_train.csv
│ ├── webmd_val.csv
│ ├── webmd_test.csv
│ ├── druglib_train.csv
│ ├── druglib_val.csv
│ └── druglib_test.csv
│
├── glove/
│ └── glove.6B.100d.txt
│
├── models/
│ ├── baselines.py
│ └── medsentix.py
│
├── notebooks/
│ ├── 01_preprocessing_drugscom.ipynb
│ ├── 02_preprocessing_druglib.ipynb
│ ├── 03_preprocessing_webmd.ipynb
│ ├── 04_splitting.ipynb
│ ├── 05_baselines.ipynb
│ ├── 06_train_medsentix_D.ipynb
│ ├── 07_train_medsentix_DW.ipynb
│ ├── 08_train_medsentix_DDL.ipynb
│ ├── 09_train_medsentix_Full.ipynb
│ ├── 10_evaluation.ipynb
│ ├── 11_ablation.ipynb
│ ├── 12_absa_validation.ipynb
│ └── 13_shap_analysis.ipynb
│
├── checkpoints/
│ ├── baselines/
│ │ ├── svm.pkl
│ │ ├── bilstm.pt
│ │ ├── bilstm_cnn.pt
│ │ ├── bert_bilstm.pt
│ │ ├── roberta_bilstm.pt
│ │ ├── biobert_standalone.pt
│ │ └── double_bigru.pt
│ │
│ └── medsentix/
│ ├── medsentix_D.pt
│ ├── medsentix_DW.pt
│ ├── medsentix_DDL.pt
│ └── medsentix_Full.pt
│
├── results/
│ ├── figures/
│ │ ├── confusion_matrices/
│ │ ├── training_curves/
│ │ ├── shap_plots/
│ │ └── attention_heatmaps/
│ │
│ └── tables/
│ ├── baseline_results.csv
│ ├── variant_comparison.csv
│ ├── absa_validation.csv
│ └── ablation_results.csv
│
├── utils/
│ └── device.py
│
├── code_plan.md
│
├── update.md
│
├── requirements.txt
└── README.md

---

## DATASETS — FULL SPECIFICATION

### Dataset 1 — Drugs.com (Primary)

- Raw files: drugsComTrain_raw.csv and drugsComTest_raw.csv
- Combine both before processing
- Total: ~215,063 reviews
- Columns: uniqueID, drugName, condition, review, rating, date, usefulCount
- Columns to KEEP: review, rating only
- Columns to DROP: uniqueID, drugName, condition, date, usefulCount
- Rating scale: 1-10
- Label construction:
  - rating 1-4 → 0 (Negative) — 25% of data
  - rating 5-6 → 1 (Neutral) — 9% of data
  - rating 7-10 → 2 (Positive) — 66% of data
- Split: 80% train / 10% val / 10% test — STRATIFIED
- Use for: All 7 baselines + all 4 MedSentiX variants (primary benchmark)

### Dataset 2 — Druglib.com (ABSA Validation)

- Raw files: drugLibTrain_raw.tsv and drugLibTest_raw.tsv
- TAB separated — use sep='\t' when loading
- Total: ~4,143 reviews
- Columns: urlDrugName, rating, effectiveness, sideEffects, condition,
  benefitsReview, sideEffectsReview, commentsReview
- Columns to KEEP: rating, effectiveness, sideEffects,
  benefitsReview, sideEffectsReview, commentsReview
- Columns to DROP: urlDrugName, condition
- Rating scale: 1-5 (overall)
- Label construction for overall rating:
  - rating 1-2 → 0 (Negative)
  - rating 3 → 1 (Neutral)
  - rating 4-5 → 2 (Positive)
- Effectiveness column label construction (categorical text):
  - "Ineffective" → 0 (Negative)
  - "Marginally Effective" OR
    "Moderately Effective" → 1 (Neutral)
  - "Considerably Effective" OR
    "Highly Effective" → 2 (Positive)
- SideEffects column label construction (categorical text):
  - "Severe Side Effects" OR
    "Extremely Severe Side Effects" → 0 (Negative)
  - "Mild Side Effects" OR
    "Moderate Side Effects" → 1 (Neutral)
  - "No Side Effects" → 2 (Positive)
- Text columns to clean: benefitsReview, sideEffectsReview, commentsReview
  — clean each separately with same pipeline as Drugs.com
- Split: 80% train / 10% val / 10% test — STRATIFIED
- Use for: MedSentiX-DDL and MedSentiX-Full training
  ABSA validation for all MedSentiX variants
  Aspect head supervision — Efficacy and Side Effects heads

### Dataset 3 — WebMD (Aspect Supervision + Additional Training)

- Raw file: webmd_raw.csv
- Total: ~362,806 reviews
- Columns: drugName, condition, review, rating, effectiveness,
  easeOfUse, satisfaction, sideEffectsReview, age, sex,
  date, usefulCount
- Columns to KEEP: review, rating, effectiveness, easeOfUse,
  satisfaction, sideEffectsReview
- Columns to DROP: drugName, condition, age, sex, date, usefulCount
- Rating scale: 1-5 (all numerical ratings)
- Label construction for overall rating:
  - rating 1-2 → 0 (Negative)
  - rating 3 → 1 (Neutral)
  - rating 4-5 → 2 (Positive)
- Aspect label construction from numerical ratings:
  - effectiveness → same 1-2/3/4-5 boundaries → Efficacy head label
  - easeOfUse → same 1-2/3/4-5 boundaries → Ease of Use head label
  - satisfaction → same 1-2/3/4-5 boundaries → Overall Satisfaction label
- Side effects label: derived from sideEffectsReview text column
  (same text cleaning pipeline, label from sentiment of text)
- Split: 80% train / 10% val / 10% test — STRATIFIED
- Use for: MedSentiX-DW and MedSentiX-Full training
  Aspect head supervision — Efficacy, Ease of Use, Overall heads

---

## TEXT CLEANING PIPELINE

Apply identically to all datasets and all text columns.
Order of operations is strict:

1. Check if value is string — if not return empty string
2. Convert to lowercase
3. Decode HTML entities — use html.unescape()
4. Remove HTML tags — regex <.\*?>
5. Remove URLs — regex http\S+
6. Remove special characters — keep only letters numbers spaces
   apostrophes and hyphens
7. Remove extra whitespace — collapse multiple spaces to one
8. Strip leading and trailing whitespace
9. Return cleaned string

DO NOT remove:

- Negations — not, never, no, didn't, doesn't, wasn't, wouldn't
  These are critical for sentiment — removing them destroys meaning
- Medical terms — these are exactly what BioBERT handles
- Numbers — "took 2 weeks to work" is contextually useful

---

## LABEL MAPPING — CONSISTENT ACROSS ALL FILES

```python
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL  = {0: "Negative", 1: "Neutral", 2: "Positive"}
NUM_CLASSES = 3
```

---

## GLOVE CONFIGURATION

- File: glove/glove.6B.100d.txt
- Dimension: 100
- Used by: BiLSTM, BiLSTM-CNN, Double BiGRU baselines
- Special tokens:
  - index 0 → PAD → zero vector
  - index 1 → UNK → random normal vector
- Max sequence length for GloVe models: 256 tokens
- Do NOT remove stop words before GloVe tokenization
  — negations must be preserved

---

## MODEL SPECIFICATIONS

### BioBERT Configuration

- Model: dmis-lab/biobert-base-cased-v1.2
- Tokenizer: same as above from HuggingFace
- Hidden size: 768
- Max sequence length: 512 tokens
- Pretrained on: 29 million PubMed biomedical papers
- Fine-tuned during training — NOT frozen

### MedSentiX Architecture — Four Components

**Component 1 — BioBERTEncoder**

- Input: input_ids [batch, seq_len], attention_mask [batch, seq_len]
- Output: last_hidden_state [batch, seq_len, 768]
- Dropout: 0.3 applied to output

**Component 2 — BiLSTMLayer**

- Input: BioBERT output [batch, seq_len, 768]
- hidden_size: 256 per direction
- num_layers: 2
- bidirectional: True
- dropout: 0.3 between layers
- Layer normalization applied after BiLSTM
- Output: [batch, seq_len, 512]

**Component 3 — GuidedMultiHeadAspectAttention**

- Input: BiLSTM output [batch, seq_len, 512]
- embed_dim: 512
- num_heads: 4
- Head assignment:
  - Head 0 → Efficacy
  - Head 1 → Side Effects
  - Head 2 → Ease of Use
  - Head 3 → Overall Satisfaction
- dropout: 0.1
- batch_first: True
- need_weights: True — required for visualization
- average_attn_weights: False — keep per-head weights
- Residual connection applied
- Layer norm after attention
- Key padding mask applied from attention_mask
- Output: pooled [batch, 512], attn_weights [batch, 4, seq_len, seq_len]
- Stores attention_weights as class attribute for SHAP and visualization

**Guided Training — How Each Head Is Supervised**
During training on datasets with aspect labels:

- A secondary auxiliary loss is computed per head
- Head 0 auxiliary loss: cross entropy against efficacy label
- Head 1 auxiliary loss: cross entropy against side effects label
- Head 2 auxiliary loss: cross entropy against ease of use label
- Head 3 auxiliary loss: cross entropy against satisfaction label
- Auxiliary head output: linear projection from head output to 3 classes
- Total loss = main_loss + (lambda \* sum(auxiliary_losses))
- lambda: 0.3 — controls how strongly aspect supervision guides heads
- When no aspect labels available (Drugs.com only):
  lambda = 0 — auxiliary loss is disabled automatically

**Component 4 — ClassificationHead**

- Input: pooled representation [batch, 512]
- Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 3)
- Output: logits [batch, 3]

### MedSentiX Forward Pass

input_ids + attention_mask
↓
BioBERTEncoder → [batch, seq_len, 768]
↓
BiLSTMLayer → [batch, seq_len, 512]
↓
GuidedMultiHeadAspectAttention
→ pooled: [batch, 512]
→ attn_weights: [batch, 4, seq, seq]
↓
ClassificationHead → logits: [batch, 3]

Returns: logits, attn_weights, auxiliary_logits (per head)

---

## FOUR MEDSENTIX VARIANTS

### MedSentiX-D

- Training data: Drugs.com only
- Aspect supervision: NONE — lambda = 0
- Heads: random initialization, no guidance
- Competes against: Colón-Ruiz 2020 (Micro-F1 90.46%)
- Checkpoint: checkpoints/medsentix/medsentix_D.pt

### MedSentiX-DW

- Training data: Drugs.com + WebMD combined
- Aspect supervision: WebMD numerical ratings
  - Head 0 (Efficacy) ← effectiveness column
  - Head 2 (Ease of Use) ← easeOfUse column
  - Head 3 (Overall) ← satisfaction column
  - Head 1 (Side Effects) ← no label — lambda for this head = 0
- Competes against: Al-Hadhrami 2024 (96% accuracy)
- Checkpoint: checkpoints/medsentix/medsentix_DW.pt

### MedSentiX-DDL

- Training data: Drugs.com + Druglib.com combined
- Aspect supervision: Druglib categorical ratings
  - Head 0 (Efficacy) ← effectiveness column
  - Head 1 (Side Effects) ← sideEffects column
  - Head 2 (Ease of Use) ← no label — lambda for this head = 0
  - Head 3 (Overall) ← no label — lambda for this head = 0
- Competes against: Gräßer 2018, Rani & Jain 2024, Durga 2024
- Checkpoint: checkpoints/medsentix/medsentix_DDL.pt

### MedSentiX-Full

- Training data: Drugs.com + WebMD + Druglib.com combined
- Aspect supervision: All four heads guided
  - Head 0 (Efficacy) ← WebMD effectiveness + Druglib effectiveness
  - Head 1 (Side Effects) ← Druglib sideEffects column
  - Head 2 (Ease of Use) ← WebMD easeOfUse column
  - Head 3 (Overall) ← WebMD satisfaction column
- This is the HEADLINE result — best expected performance
- Competes against: everyone simultaneously
- Checkpoint: checkpoints/medsentix/medsentix_Full.pt

---

## TRAINING CONFIGURATION — SAME FOR ALL VARIANTS

```python
BATCH_SIZE      = 16   # fits 12GB VRAM and Mac 16GB unified memory
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01
EPOCHS          = 10
WARMUP_RATIO    = 0.1  # first 10% of steps are warmup
DROPOUT         = 0.3
RANDOM_SEED     = 42
MAX_LEN         = 512
LAMBDA_ASPECT   = 0.3  # auxiliary aspect loss weight
GRADIENT_CLIP   = 1.0
EARLY_STOPPING_PATIENCE = 3  # stop if val accuracy does not improve
```

Optimizer: AdamW
Scheduler: get_linear_schedule_with_warmup from HuggingFace
Loss function: CrossEntropyLoss with class weights
Class weights: [1.5, 2.0, 1.0] for [Negative, Neutral, Positive]
— Neutral gets highest weight because it is underrepresented at 9%

Save best model: based on validation accuracy
— save state dict whenever val accuracy improves

---

## SEVEN BASELINE MODELS

### Baseline 1 — SVM + TF-IDF

- Library: scikit-learn
- Vectorizer: TfidfVectorizer
  - max_features: 50000
  - ngram_range: (1, 2)
  - sublinear_tf: True
- Classifier: LinearSVC
  - C: 1.0
  - max_iter: 2000
  - random_state: 42
- No GPU required — CPU only
- Checkpoint: checkpoints/baselines/svm.pkl (use joblib.dump)

### Baseline 2 — Double BiGRU + GloVe

- Represents: Han et al. 2020 approach
- Two stacked BiGRU layers
- Layer 1: GRU(embed_dim=100, hidden=256, bidirectional=True)
- Layer 2: GRU(input=512, hidden=256, bidirectional=True)
- Mean pooling → Linear(512,128) → ReLU → Dropout → Linear(128,3)
- GloVe.6B.100d embeddings — fine-tuned during training
- Optimizer: Adam lr=1e-3
- Checkpoint: checkpoints/baselines/double_bigru.pt

### Baseline 3 — BiLSTM + GloVe

- Single BiLSTM layer
- LSTM(embed_dim=100, hidden=256, num_layers=2, bidirectional=True)
- Mean pooling → Linear(512,128) → ReLU → Dropout → Linear(128,3)
- GloVe.6B.100d embeddings — fine-tuned during training
- Optimizer: Adam lr=1e-3
- Checkpoint: checkpoints/baselines/bilstm.pt

### Baseline 4 — BiLSTM-CNN + GloVe

- Represents: Al-Hadhrami 2024 approach
- CNN then BiLSTM
- CNN: Conv1d(100, 64, kernel_size=5) → ReLU → MaxPool1d(4)
- BiLSTM: LSTM(64, 64, bidirectional=True, dropout=0.3)
- Mean pooling → Linear(128,3)
- GloVe.6B.100d embeddings
- Optimizer: Adam lr=1e-3
- Checkpoint: checkpoints/baselines/bilstm_cnn.pt

### Baseline 5 — BERT + BiLSTM

- Represents: Colón-Ruiz 2020 approach
- Backbone: bert-base-uncased from HuggingFace
- BiLSTM on top: LSTM(768, 256, num_layers=2, bidirectional=True)
- Layer norm → Mean pooling → Linear(512,128) → Dropout → Linear(128,3)
- Optimizer: AdamW lr=2e-5
- Checkpoint: checkpoints/baselines/bert_bilstm.pt

### Baseline 6 — RoBERTa + BiLSTM

- Represents: Durga 2024 approach without ACO
- Backbone: roberta-base from HuggingFace
- BiLSTM on top: same as BERT+BiLSTM
- Optimizer: AdamW lr=2e-5
- Checkpoint: checkpoints/baselines/roberta_bilstm.pt

### Baseline 7 — BioBERT Standalone

- Most critical baseline — isolates BioBERT contribution
- Backbone: dmis-lab/biobert-base-cased-v1.2
- CLS token pooling only — no BiLSTM on top
- Linear(768, 256) → ReLU → Dropout → Linear(256, 3)
- Optimizer: AdamW lr=2e-5
- Checkpoint: checkpoints/baselines/biobert_standalone.pt

---

## METRICS TO COLLECT

### For ALL models (7 baselines + 4 MedSentiX variants)

- Accuracy
- Precision (weighted)
- Recall (weighted)
- Macro F1
- Weighted F1
- Per-class F1: Negative F1, Neutral F1, Positive F1
- Matthews Correlation Coefficient (MCC)
- Cohen Kappa
- Confusion matrix (normalized)
- Inference time per sample (milliseconds)
- Total parameter count

### For MedSentiX variants only

- Training loss curve (per epoch)
- Validation accuracy curve (per epoch)
- Expected Calibration Error (ECE)
- Cross-dataset generalization score

### For ABSA validation (MedSentiX-DDL and MedSentiX-Full on Druglib)

- Per-aspect Accuracy: Efficacy, Side Effects, Ease of Use, Overall
- Per-aspect F1: same four dimensions
- Average Aspect F1

### For SHAP analysis (MedSentiX-Full only)

- Global SHAP beeswarm — top 20 words
- Per-class SHAP bar chart
- Waterfall plots — 3 selected reviews
- Aspect-level SHAP per head
- Attention entropy per head

---

## CLASS WEIGHTS

```python
# Negative=1.5, Neutral=2.0, Positive=1.0
# Neutral gets highest weight — only 9% of Drugs.com data
class_weights = torch.tensor([1.5, 2.0, 1.0], dtype=torch.float)
```

---

## SCORES TO BEAT — REFERENCE TARGETS

### On Drugs.com

| Paper            | Model       | Accuracy          | Macro-F1 |
| ---------------- | ----------- | ----------------- | -------- |
| Colón-Ruiz 2020  | BERT+BiLSTM | 90.46% (Micro-F1) | —        |
| Al-Hadhrami 2024 | BiLSTM-CNN  | 96-97.1%          | —        |
| Rani & Jain 2024 | MLDBM       | 87.71%            | 85.97%   |

### On Druglib.com (ABSA)

| Paper            | Model              | Accuracy               | Macro-F1 |
| ---------------- | ------------------ | ---------------------- | -------- |
| Gräßer 2018      | Logistic Reg       | 77.70% (effectiveness) | —        |
| Han 2020         | PM-DBiGRU          | 78.26%                 | 77.75%   |
| Rani & Jain 2024 | MLDBM              | 78.97%                 | 76.23%   |
| Durga 2024       | RoBERTa+BiLSTM+ACO | 96.78%\*               | —        |

\*Durga 2024 result is cited with caveat — evaluation conditions
unverifiable from available paper access

---

## NOTEBOOK IMPLEMENTATION GUIDE

### 01_preprocessing_drugscom.ipynb

Sections in order:

1. Imports and device setup
2. Load drugsComTrain_raw.csv and drugsComTest_raw.csv
3. Combine into one DataFrame
4. Shape inspection — print rows, columns, dtypes
5. Null value check — print counts per column
6. Duplicate check — print count
7. Rating distribution plot — bar chart
8. Drop unnecessary columns — keep review and rating only
9. Drop nulls and duplicates
10. Apply text cleaning pipeline to review column
11. Assign sentiment labels from rating
12. Class distribution plot after labeling
13. Sample 5 cleaned reviews — visual check
14. Save to data/processed/drugs_com_clean.csv
15. Print final shape and class distribution

### 02_preprocessing_druglib.ipynb

Sections in order:

1. Imports and device setup
2. Load drugLibTrain_raw.tsv and drugLibTest_raw.tsv with sep='\t'
3. Combine into one DataFrame
4. Shape inspection
5. Print all column names and sample rows
6. Null check
7. Keep relevant columns only
8. Apply text cleaning to benefitsReview, sideEffectsReview,
   commentsReview separately
9. Assign overall sentiment label from rating (1-5 scale)
10. Assign effectiveness aspect label from categorical text
11. Assign sideEffects aspect label from categorical text
12. Print label distributions for all four label columns
13. Save to data/processed/druglib_clean.csv
14. Print final shape

### 03_preprocessing_webmd.ipynb

Sections in order:

1. Imports and device setup
2. Load webmd_raw.csv
3. Shape inspection — expect ~362,806 rows
4. Print all 12 column names
5. Keep: review, rating, effectiveness, easeOfUse, satisfaction,
   sideEffectsReview
6. Drop nulls
7. Apply text cleaning to review and sideEffectsReview
8. Assign overall label from rating (1-5)
9. Assign effectiveness aspect label from numerical rating
10. Assign easeOfUse aspect label from numerical rating
11. Assign satisfaction aspect label from numerical rating
12. Note: sideEffects text column — label assignment from text
    will be done during training — too complex for preprocessing
13. Print all label distributions
14. Save to data/processed/webmd_clean.csv

### 04_splitting.ipynb

Sections in order:

1. Imports — sklearn train_test_split
2. Load all three processed CSVs
3. Split Drugs.com — stratify on label column — seed 42
   80% train, 10% val, 10% test
4. Verify class balance in each Drugs.com split
5. Split WebMD — same parameters
6. Split Druglib.com — same parameters
7. Print sizes of all 9 splits
8. Save all 9 CSVs to data/splits/
9. Final verification — load each saved file and print shape

---

## IMPLEMENTATION ORDER — DO IN THIS SEQUENCE

1. utils/device.py
2. notebooks/01_preprocessing_drugscom.ipynb
3. notebooks/02_preprocessing_druglib.ipynb
4. notebooks/03_preprocessing_webmd.ipynb
5. notebooks/04_splitting.ipynb
6. models/baselines.py
7. notebooks/05_baselines.ipynb
8. models/medsentix.py
9. notebooks/06_train_medsentix_D.ipynb
10. notebooks/07_train_medsentix_DW.ipynb
11. notebooks/08_train_medsentix_DDL.ipynb
12. notebooks/09_train_medsentix_Full.ipynb
13. notebooks/10_evaluation.ipynb
14. notebooks/11_ablation.ipynb
15. notebooks/12_absa_validation.ipynb
16. notebooks/13_shap_analysis.ipynb

---

## TESTING ON MAC BEFORE DGX

Before running on full data add this cell at the top of every
notebook during development:

```python
# DEV MODE — remove before DGX run
DEV_MODE = True
SAMPLE_SIZE = 1000  # rows to use during testing
```

Load data with:

```python
if DEV_MODE:
    df = df.head(SAMPLE_SIZE)
```

Remove DEV_MODE flag before DGX session. Everything else stays
identical — the device utility handles CUDA vs MPS automatically.

---

## IMPORTANT PAPER CONTEXT — READ BEFORE CODING

This model is for a conference paper targeting IEEE QPAIN 2026
or ICCIT 2026. Supervised by Prof. Kamruddin Nur at AIUB.

Key findings from literature that inform coding decisions:

- Neutral class is always the hardest — only 9% of Drugs.com
  Use class weights [1.5, 2.0, 1.0] for [Neg, Neu, Pos]
- BioBERT (biomedical) outperforms BERT (general) on medical text
- Guided attention heads must genuinely specialize per aspect
  Verify via attention entropy — low entropy = focused head
- SHAP is the first explainability layer in drug review ABSA
  This is the strongest novelty claim — treat it carefully
- Primary competitor: Rani & Jain 2024 — BERT+MHSA+Dual BiLSTM
  They got 87.71% on Drugs.com — we must beat this clearly
- Do NOT remove negations during text cleaning — critical for
  sentiment — "no side effects" is positive not negative

## The update.md is a markdown file and all the necessary information regarding what has been done in the coding and what has been completed and what needs to be done is ledgered in details with date and time stamp

---

## WHEN ASKING CODE AGENT TO IMPLEMENT A FILE

Always say:
"Implement [filename] according to the MedSentiX specification.
Read the spec carefully before writing any code.
Follow the implementation order and all critical rules."

Work one file at a time. Verify each file runs without errors
on small data before moving to the next file.
