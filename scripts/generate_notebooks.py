"""Generate the MedSentiX modular notebooks and full-pipeline notebook.

The notebooks are emitted with nbformat so their JSON is valid and repeatable.
Each notebook keeps DEV_MODE enabled for small-sample development runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def md(text: str):
    """Create a Markdown notebook cell."""
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    """Create a code notebook cell."""
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_notebook(filename: str, cells) -> None:
    """Write one notebook with a Python 3 kernelspec."""
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, NOTEBOOK_DIR / filename)


ENV_CELL = r"""
# DEV MODE keeps notebook runs small during development; set to False for full DGX experiments.
DEV_MODE = True
SAMPLE_SIZE = 1000

# Resolve the project root from either the repository root or notebooks/ directory.
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the shared device and seed utilities required by the specification.
from utils.device import RANDOM_SEED, get_device, set_seed

set_seed(RANDOM_SEED)
DEVICE = get_device()
"""


COMBINED_ENV_CELL = r"""
# DEV MODE keeps notebook runs small during development; set to False for full DGX experiments.
DEV_MODE = True
SAMPLE_SIZE = 1000

# Resolve the project root from either the repository root or notebooks/ directory.
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""


PREPROCESSING_UTILS = r"""
# Shared preprocessing utilities used identically across datasets.
import html
import re

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
NUM_CLASSES = 3


def clean_text(value):
    # Apply the exact text cleaning pipeline from the specification.
    if not isinstance(value, str):
        return ""
    text = value.lower()
    text = html.unescape(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_from_rating_10(rating):
    # Map Drugs.com 1-10 ratings to Negative/Neutral/Positive labels.
    rating = float(rating)
    if 1 <= rating <= 4:
        return 0
    if 5 <= rating <= 6:
        return 1
    if 7 <= rating <= 10:
        return 2
    return None


def label_from_rating_5(rating):
    # Map 1-5 ratings to Negative/Neutral/Positive labels.
    rating = float(rating)
    if 1 <= rating <= 2:
        return 0
    if rating == 3:
        return 1
    if 4 <= rating <= 5:
        return 2
    return None


def label_from_effectiveness(value):
    # Map Druglib effectiveness text to the three-class label space.
    mapping = {
        "Ineffective": 0,
        "Marginally Effective": 1,
        "Moderately Effective": 1,
        "Considerably Effective": 2,
        "Highly Effective": 2,
    }
    return mapping.get(value, None)


def label_from_side_effects(value):
    # Map Druglib side-effect severity text to the three-class label space.
    mapping = {
        "Severe Side Effects": 0,
        "Extremely Severe Side Effects": 0,
        "Mild Side Effects": 1,
        "Moderate Side Effects": 1,
        "No Side Effects": 2,
    }
    return mapping.get(value, None)
"""


def preprocessing_drugscom():
    return [
        md("# Drugs.com Preprocessing\nPrepare the primary Drugs.com benchmark exactly as specified."),
        code(ENV_CELL),
        code("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n" + PREPROCESSING_UTILS),
        md("## Load Raw Files\nThe train and test raw files are combined before any cleaning."),
        code(r"""
raw_dir = PROJECT_ROOT / "data/raw"
train_raw = pd.read_csv(raw_dir / "drugsComTrain_raw.csv")
test_raw = pd.read_csv(raw_dir / "drugsComTest_raw.csv")
df = pd.concat([train_raw, test_raw], ignore_index=True)
if DEV_MODE:
    df = df.head(SAMPLE_SIZE)
print("Combined shape:", df.shape)
print(df.dtypes)
"""),
        md("## Inspect Missing Values, Duplicates, and Ratings"),
        code(r"""
print("Null counts:")
print(df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="rating", color="steelblue")
plt.title("Drugs.com Rating Distribution")
plt.tight_layout()
plt.show()
"""),
        md("## Clean Text and Assign Labels\nOnly `review` and `rating` are kept. Negations, numbers, and medical terms are preserved."),
        code(r"""
df = df[["review", "rating"]].dropna().drop_duplicates().copy()
df["review"] = df["review"].apply(clean_text)
df = df[df["review"].str.len() > 0].copy()
df["label"] = df["rating"].apply(label_from_rating_10)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)
print(df.head(5))
"""),
        md("## Verify Distribution and Save"),
        code(r"""
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", hue="label", palette="Set2", legend=False)
plt.xticks([0, 1, 2], ["Negative", "Neutral", "Positive"])
plt.title("Drugs.com Label Distribution")
plt.tight_layout()
plt.show()

print("Sample cleaned reviews:")
display(df[["review", "rating", "label"]].sample(min(5, len(df)), random_state=RANDOM_SEED))

output_path = PROJECT_ROOT / "data/processed/drugs_com_clean.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Final shape:", df.shape)
print(df["label"].value_counts(normalize=True).sort_index())
"""),
    ]


def preprocessing_druglib():
    return [
        md("# Druglib.com Preprocessing\nPrepare Druglib for ABSA validation and aspect-head supervision."),
        code(ENV_CELL),
        code("import pandas as pd\n\n" + PREPROCESSING_UTILS),
        md("## Load Raw TSV Files"),
        code(r"""
raw_dir = PROJECT_ROOT / "data/raw"
train_raw = pd.read_csv(raw_dir / "drugLibTrain_raw.tsv", sep="\t")
test_raw = pd.read_csv(raw_dir / "drugLibTest_raw.tsv", sep="\t")
df = pd.concat([train_raw, test_raw], ignore_index=True)
if DEV_MODE:
    df = df.head(SAMPLE_SIZE)
print("Combined shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())
print("Null counts:")
print(df.isna().sum())
"""),
        md("## Clean Review Fields and Labels"),
        code(r"""
keep_cols = ["rating", "effectiveness", "sideEffects", "benefitsReview", "sideEffectsReview", "commentsReview"]
df = df[keep_cols].dropna().drop_duplicates().copy()
for column in ["benefitsReview", "sideEffectsReview", "commentsReview"]:
    df[column] = df[column].apply(clean_text)

# Concatenate the three Druglib text fields into one review for model training.
df["review"] = (
    df["benefitsReview"].astype(str) + " " +
    df["sideEffectsReview"].astype(str) + " " +
    df["commentsReview"].astype(str)
).str.strip()
df["label"] = df["rating"].apply(label_from_rating_5)
df["efficacy_label"] = df["effectiveness"].apply(label_from_effectiveness)
df["side_effects_label"] = df["sideEffects"].apply(label_from_side_effects)
df["ease_label"] = -100
df["satisfaction_label"] = -100
df = df.dropna(subset=["label", "efficacy_label", "side_effects_label"]).copy()
for column in ["label", "efficacy_label", "side_effects_label", "ease_label", "satisfaction_label"]:
    df[column] = df[column].astype(int)
"""),
        md("## Verify and Save"),
        code(r"""
for column in ["label", "efficacy_label", "side_effects_label"]:
    print(f"\n{column} distribution")
    print(df[column].value_counts(normalize=True).sort_index())

output_path = PROJECT_ROOT / "data/processed/druglib_clean.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Final shape:", df.shape)
display(df.head())
"""),
    ]


def preprocessing_webmd():
    return [
        md("# WebMD Preprocessing\nPrepare WebMD for additional training and numerical aspect supervision."),
        code(ENV_CELL),
        code("import pandas as pd\n\n" + PREPROCESSING_UTILS),
        md("## Load and Normalize Columns"),
        code(r"""
raw_path = PROJECT_ROOT / "data/raw/webmd_raw.csv"
df = pd.read_csv(raw_path)
if DEV_MODE:
    df = df.head(SAMPLE_SIZE)
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# The downloaded WebMD file uses capitalized names; normalize them to the specification.
rename_map = {
    "Reviews": "review",
    "Effectiveness": "effectiveness",
    "EaseofUse": "easeOfUse",
    "Satisfaction": "satisfaction",
    "Sides": "sideEffectsReview",
}
df = df.rename(columns=rename_map)
if "rating" not in df.columns:
    df["rating"] = df["satisfaction"]
keep_cols = ["review", "rating", "effectiveness", "easeOfUse", "satisfaction", "sideEffectsReview"]
df = df[keep_cols].dropna().drop_duplicates().copy()
"""),
        md("## Clean Text and Assign Labels"),
        code(r"""
df["review"] = df["review"].apply(clean_text)
df["sideEffectsReview"] = df["sideEffectsReview"].apply(clean_text)
df = df[(df["review"].str.len() > 0) & (df["sideEffectsReview"].str.len() > 0)].copy()
df["label"] = df["rating"].apply(label_from_rating_5)
df["efficacy_label"] = df["effectiveness"].apply(label_from_rating_5)
df["ease_label"] = df["easeOfUse"].apply(label_from_rating_5)
df["satisfaction_label"] = df["satisfaction"].apply(label_from_rating_5)

# WebMD side-effect text is retained, but no supervised side-effect label is assigned here.
df["side_effects_label"] = -100
df = df.dropna(subset=["label", "efficacy_label", "ease_label", "satisfaction_label"]).copy()
for column in ["label", "efficacy_label", "ease_label", "satisfaction_label", "side_effects_label"]:
    df[column] = df[column].astype(int)
"""),
        md("## Verify and Save"),
        code(r"""
for column in ["label", "efficacy_label", "ease_label", "satisfaction_label"]:
    print(f"\n{column} distribution")
    print(df[column].value_counts(normalize=True).sort_index())

output_path = PROJECT_ROOT / "data/processed/webmd_clean.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Final shape:", df.shape)
display(df.head())
"""),
    ]


def splitting_notebook():
    return [
        md("# Dataset Splitting\nCreate stratified 80/10/10 train, validation, and test splits for all datasets."),
        code(ENV_CELL),
        code("import pandas as pd\nfrom sklearn.model_selection import train_test_split"),
        md("## Load Processed Datasets"),
        code(r"""
processed_dir = PROJECT_ROOT / "data/processed"
datasets = {
    "drugs_com": pd.read_csv(processed_dir / "drugs_com_clean.csv"),
    "webmd": pd.read_csv(processed_dir / "webmd_clean.csv"),
    "druglib": pd.read_csv(processed_dir / "druglib_clean.csv"),
}
if DEV_MODE:
    datasets = {name: frame.head(SAMPLE_SIZE).copy() for name, frame in datasets.items()}
for name, frame in datasets.items():
    print(name, frame.shape, frame["label"].value_counts(normalize=True).sort_index().to_dict())
"""),
        md("## Stratified Split Helper"),
        code(r"""
def stratified_80_10_10(frame):
    # Split a dataframe with stratification on the sentiment label.
    train_df, temp_df = train_test_split(
        frame,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=frame["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


split_dir = PROJECT_ROOT / "data/splits"
split_dir.mkdir(parents=True, exist_ok=True)
saved_paths = []
for name, frame in datasets.items():
    train_df, val_df, test_df = stratified_80_10_10(frame)
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = split_dir / f"{name}_{split_name}.csv"
        split_df.to_csv(path, index=False)
        saved_paths.append(path)
        print(path.name, split_df.shape, split_df["label"].value_counts(normalize=True).sort_index().to_dict())
"""),
        md("## Reload Verification"),
        code(r"""
for path in saved_paths:
    print(path.name, pd.read_csv(path).shape)
"""),
    ]


def baselines_notebook():
    return [
        md("# Baseline Models\nTrain the seven paper baselines on the Drugs.com primary benchmark."),
        code(ENV_CELL),
        code(r"""
import pandas as pd
from models.baselines import train_all_baselines

train_df = pd.read_csv(PROJECT_ROOT / "data/splits/drugs_com_train.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data/splits/drugs_com_val.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data/splits/drugs_com_test.csv")
if DEV_MODE:
    train_df = train_df.head(SAMPLE_SIZE)
    val_df = val_df.head(max(100, SAMPLE_SIZE // 5))
    test_df = test_df.head(max(100, SAMPLE_SIZE // 5))

results = train_all_baselines(train_df, val_df, test_df, project_root=PROJECT_ROOT, epochs=10)
display(results)
"""),
    ]


def train_variant_notebook(variant: str, title: str, description: str):
    return [
        md(f"# {title}\n{description}"),
        code(ENV_CELL),
        code(f"""
from models.medsentix import train_medsentix_variant

model, results, history = train_medsentix_variant(
    "{variant}",
    project_root=PROJECT_ROOT,
    dev_mode=DEV_MODE,
    sample_size=SAMPLE_SIZE,
    epochs=10,
)
display(results)
print(history)
"""),
    ]


def evaluation_notebook():
    return [
        md("# Model Evaluation\nEvaluate all saved MedSentiX variants and consolidate results with baseline tables."),
        code(ENV_CELL),
        code(r"""
import pandas as pd
from models.medsentix import evaluate_all_medsentix_variants

variant_results = evaluate_all_medsentix_variants(PROJECT_ROOT, dev_mode=DEV_MODE, sample_size=SAMPLE_SIZE)
baseline_path = PROJECT_ROOT / "results/tables/baseline_results.csv"
if baseline_path.exists():
    baseline_results = pd.read_csv(baseline_path)
else:
    baseline_results = pd.DataFrame()
    print("Baseline results are missing; run 05_baselines.ipynb first.")

all_results = pd.concat([baseline_results, variant_results], ignore_index=True, sort=False)
output_path = PROJECT_ROOT / "results/tables/all_model_results.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
all_results.to_csv(output_path, index=False)
display(all_results)
print("Saved:", output_path)
"""),
    ]


def ablation_notebook():
    return [
        md("# Ablation Study\nMeasure the contribution of guided attention, auxiliary supervision, and the BiLSTM layer."),
        code(ENV_CELL),
        code(r"""
from models.medsentix import run_ablation_study

ablation_results = run_ablation_study(
    PROJECT_ROOT,
    dev_mode=DEV_MODE,
    sample_size=SAMPLE_SIZE,
    epochs=10,
)
display(ablation_results)
"""),
    ]


def absa_notebook():
    return [
        md("# ABSA Validation\nValidate MedSentiX aspect heads on Druglib aspect labels."),
        code(ENV_CELL),
        code(r"""
from models.medsentix import validate_absa_on_druglib

absa_results = validate_absa_on_druglib(
    variants=("D", "DW", "DDL", "Full"),
    project_root=PROJECT_ROOT,
    dev_mode=DEV_MODE,
    sample_size=SAMPLE_SIZE,
)
display(absa_results)
"""),
    ]


def shap_notebook():
    return [
        md("# SHAP Explainability\nRun SHAP analysis and attention-entropy analysis for MedSentiX-Full."),
        code(ENV_CELL),
        code(r"""
from models.medsentix import run_shap_analysis

entropy_summary = run_shap_analysis(
    PROJECT_ROOT,
    dev_mode=DEV_MODE,
    sample_size=min(100, SAMPLE_SIZE),
)
display(entropy_summary)
"""),
    ]


def notebook_safe_source(path: Path) -> str:
    """Inline project source into the combined notebook without notebook-unsafe imports."""
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"from __future__ import annotations\n", "", source)
    source = source.replace("PROJECT_ROOT = Path(__file__).resolve().parents[1]", "# PROJECT_ROOT is supplied by the notebook environment")
    source = re.sub(r"from utils\.device import .*?\n", "# Device utilities are defined in this notebook.\n", source)
    source = re.sub(r"from models\.baselines import \(\n.*?\)\n", "# Baseline utilities are defined above in this notebook.\n", source, flags=re.S)
    return source.strip()


def full_pipeline_notebook():
    """Create the single consolidated paper notebook requested by the user."""
    device_source = notebook_safe_source(PROJECT_ROOT / "utils/device.py")
    baseline_source = notebook_safe_source(PROJECT_ROOT / "models/baselines.py")
    medsentix_source = notebook_safe_source(PROJECT_ROOT / "models/medsentix.py")
    return [
        md("# MedSentiX: Full Experimental Pipeline\nA consolidated, paper-friendly notebook for the complete end-to-end MedSentiX workflow."),
        md("## 1. Environment and Reproducibility\nThis section configures relative project paths, random seed 42, and automatic CUDA/MPS/CPU device detection."),
        code(COMBINED_ENV_CELL),
        md("### Device Utility Implementation\nThe utility is included directly so this notebook remains self-contained."),
        code(device_source),
        code("set_seed(RANDOM_SEED)\nDEVICE = get_device()"),
        md("## 2. Dataset Preparation\nThe three datasets are loaded, inspected, cleaned, labeled, and split using the exact rules in the specification."),
        code("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\n\n" + PREPROCESSING_UTILS),
        md("### 2.1 Dataset Loading and Inspection"),
        code(r"""
raw_dir = PROJECT_ROOT / "data/raw"
drugs_train_raw = pd.read_csv(raw_dir / "drugsComTrain_raw.csv")
drugs_test_raw = pd.read_csv(raw_dir / "drugsComTest_raw.csv")
druglib_train_raw = pd.read_csv(raw_dir / "drugLibTrain_raw.tsv", sep="\t")
druglib_test_raw = pd.read_csv(raw_dir / "drugLibTest_raw.tsv", sep="\t")
webmd_raw = pd.read_csv(raw_dir / "webmd_raw.csv")
for name, frame in {
    "drugs_train_raw": drugs_train_raw,
    "drugs_test_raw": drugs_test_raw,
    "druglib_train_raw": druglib_train_raw,
    "druglib_test_raw": druglib_test_raw,
    "webmd_raw": webmd_raw,
}.items():
    print(name, frame.shape)
    print(frame.columns.tolist())
"""),
        md("### 2.2 Drugs.com Preprocessing"),
        code(r"""
drugs_df = pd.concat([drugs_train_raw, drugs_test_raw], ignore_index=True)
if DEV_MODE:
    drugs_df = drugs_df.head(SAMPLE_SIZE)
print("Null counts:")
print(drugs_df.isna().sum())
print("Duplicate rows:", drugs_df.duplicated().sum())
drugs_df = drugs_df[["review", "rating"]].dropna().drop_duplicates().copy()
drugs_df["review"] = drugs_df["review"].apply(clean_text)
drugs_df = drugs_df[drugs_df["review"].str.len() > 0].copy()
drugs_df["label"] = drugs_df["rating"].apply(label_from_rating_10)
drugs_df = drugs_df.dropna(subset=["label"]).copy()
drugs_df["label"] = drugs_df["label"].astype(int)
drugs_df.to_csv(PROJECT_ROOT / "data/processed/drugs_com_clean.csv", index=False)
print(drugs_df.shape)
print(drugs_df["label"].value_counts(normalize=True).sort_index())
"""),
        md("### 2.3 Druglib.com Preprocessing"),
        code(r"""
druglib_df = pd.concat([druglib_train_raw, druglib_test_raw], ignore_index=True)
if DEV_MODE:
    druglib_df = druglib_df.head(SAMPLE_SIZE)
druglib_df = druglib_df[["rating", "effectiveness", "sideEffects", "benefitsReview", "sideEffectsReview", "commentsReview"]].dropna().drop_duplicates().copy()
for column in ["benefitsReview", "sideEffectsReview", "commentsReview"]:
    druglib_df[column] = druglib_df[column].apply(clean_text)
druglib_df["review"] = (
    druglib_df["benefitsReview"].astype(str) + " " +
    druglib_df["sideEffectsReview"].astype(str) + " " +
    druglib_df["commentsReview"].astype(str)
).str.strip()
druglib_df["label"] = druglib_df["rating"].apply(label_from_rating_5)
druglib_df["efficacy_label"] = druglib_df["effectiveness"].apply(label_from_effectiveness)
druglib_df["side_effects_label"] = druglib_df["sideEffects"].apply(label_from_side_effects)
druglib_df["ease_label"] = -100
druglib_df["satisfaction_label"] = -100
druglib_df = druglib_df.dropna(subset=["label", "efficacy_label", "side_effects_label"]).copy()
for column in ["label", "efficacy_label", "side_effects_label", "ease_label", "satisfaction_label"]:
    druglib_df[column] = druglib_df[column].astype(int)
druglib_df.to_csv(PROJECT_ROOT / "data/processed/druglib_clean.csv", index=False)
print(druglib_df.shape)
"""),
        md("### 2.4 WebMD Preprocessing"),
        code(r"""
webmd_df = webmd_raw.rename(columns={
    "Reviews": "review",
    "Effectiveness": "effectiveness",
    "EaseofUse": "easeOfUse",
    "Satisfaction": "satisfaction",
    "Sides": "sideEffectsReview",
})
if "rating" not in webmd_df.columns:
    webmd_df["rating"] = webmd_df["satisfaction"]
if DEV_MODE:
    webmd_df = webmd_df.head(SAMPLE_SIZE)
webmd_df = webmd_df[["review", "rating", "effectiveness", "easeOfUse", "satisfaction", "sideEffectsReview"]].dropna().drop_duplicates().copy()
webmd_df["review"] = webmd_df["review"].apply(clean_text)
webmd_df["sideEffectsReview"] = webmd_df["sideEffectsReview"].apply(clean_text)
webmd_df = webmd_df[(webmd_df["review"].str.len() > 0) & (webmd_df["sideEffectsReview"].str.len() > 0)].copy()
webmd_df["label"] = webmd_df["rating"].apply(label_from_rating_5)
webmd_df["efficacy_label"] = webmd_df["effectiveness"].apply(label_from_rating_5)
webmd_df["ease_label"] = webmd_df["easeOfUse"].apply(label_from_rating_5)
webmd_df["satisfaction_label"] = webmd_df["satisfaction"].apply(label_from_rating_5)
webmd_df["side_effects_label"] = -100
webmd_df = webmd_df.dropna(subset=["label", "efficacy_label", "ease_label", "satisfaction_label"]).copy()
for column in ["label", "efficacy_label", "ease_label", "satisfaction_label", "side_effects_label"]:
    webmd_df[column] = webmd_df[column].astype(int)
webmd_df.to_csv(PROJECT_ROOT / "data/processed/webmd_clean.csv", index=False)
print(webmd_df.shape)
"""),
        md("### 2.5 Dataset Splitting"),
        code(r"""
def stratified_80_10_10(frame):
    train_df, temp_df = train_test_split(frame, test_size=0.20, random_state=RANDOM_SEED, stratify=frame["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["label"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


split_dir = PROJECT_ROOT / "data/splits"
split_dir.mkdir(parents=True, exist_ok=True)
for name, frame in {"drugs_com": drugs_df, "webmd": webmd_df, "druglib": druglib_df}.items():
    for split_name, split_df in zip(["train", "val", "test"], stratified_80_10_10(frame)):
        split_df.to_csv(split_dir / f"{name}_{split_name}.csv", index=False)
        print(name, split_name, split_df.shape, split_df["label"].value_counts(normalize=True).sort_index().to_dict())
"""),
        md("## 3. Baseline Models\nThe following cell contains the actual baseline model implementations used by the modular notebooks."),
        code(baseline_source),
        md("### 3.1 Train Baselines"),
        code(r"""
train_df = pd.read_csv(PROJECT_ROOT / "data/splits/drugs_com_train.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data/splits/drugs_com_val.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data/splits/drugs_com_test.csv")
if DEV_MODE:
    train_df = train_df.head(SAMPLE_SIZE)
    val_df = val_df.head(max(100, SAMPLE_SIZE // 5))
    test_df = test_df.head(max(100, SAMPLE_SIZE // 5))
baseline_results = train_all_baselines(train_df, val_df, test_df, project_root=PROJECT_ROOT, epochs=10)
display(baseline_results)
"""),
        md("## 4. MedSentiX Architecture\nThe actual MedSentiX implementation is included below: BioBERT, BiLSTM, guided attention, classification head, losses, and experiment utilities."),
        code(medsentix_source),
        md("## 5. MedSentiX-D"),
        code('model_D, results_D, history_D = train_medsentix_variant("D", PROJECT_ROOT, DEV_MODE, SAMPLE_SIZE, epochs=10)\ndisplay(results_D)'),
        md("## 6. MedSentiX-DW"),
        code('model_DW, results_DW, history_DW = train_medsentix_variant("DW", PROJECT_ROOT, DEV_MODE, SAMPLE_SIZE, epochs=10)\ndisplay(results_DW)'),
        md("## 7. MedSentiX-DDL"),
        code('model_DDL, results_DDL, history_DDL = train_medsentix_variant("DDL", PROJECT_ROOT, DEV_MODE, SAMPLE_SIZE, epochs=10)\ndisplay(results_DDL)'),
        md("## 8. MedSentiX-Full"),
        code('model_Full, results_Full, history_Full = train_medsentix_variant("Full", PROJECT_ROOT, DEV_MODE, SAMPLE_SIZE, epochs=10)\ndisplay(results_Full)'),
        md("## 9. Model Evaluation"),
        code(r"""
variant_results = evaluate_all_medsentix_variants(PROJECT_ROOT, dev_mode=DEV_MODE, sample_size=SAMPLE_SIZE)
all_model_results = pd.concat([baseline_results, variant_results], ignore_index=True, sort=False)
all_model_results.to_csv(PROJECT_ROOT / "results/tables/all_model_results.csv", index=False)
display(all_model_results)
"""),
        md("## 10. Ablation Study"),
        code(r"""
ablation_results = run_ablation_study(PROJECT_ROOT, dev_mode=DEV_MODE, sample_size=SAMPLE_SIZE, epochs=10)
display(ablation_results)
"""),
        md("## 11. ABSA Validation"),
        code(r"""
absa_results = validate_absa_on_druglib(("D", "DW", "DDL", "Full"), PROJECT_ROOT, DEV_MODE, SAMPLE_SIZE)
display(absa_results)
"""),
        md("## 12. SHAP Explainability"),
        code(r"""
entropy_summary = run_shap_analysis(PROJECT_ROOT, dev_mode=DEV_MODE, sample_size=min(100, SAMPLE_SIZE))
display(entropy_summary)
"""),
        md("## 13. Final Results\nThis section loads the calculated result tables and points to the generated figures. It does not invent or hardcode scores."),
        code(r"""
for table_name in [
    "baseline_results.csv",
    "variant_comparison.csv",
    "all_model_results.csv",
    "ablation_results.csv",
    "absa_validation.csv",
    "shap_attention_entropy.csv",
]:
    path = PROJECT_ROOT / "results/tables" / table_name
    if path.exists():
        print("\n", table_name)
        display(pd.read_csv(path))
    else:
        print("Missing:", path)

figure_root = PROJECT_ROOT / "results/figures"
print("Generated figure files:")
for path in sorted(figure_root.rglob("*.png")):
    print(path.relative_to(PROJECT_ROOT))
"""),
    ]


def main() -> None:
    """Generate every required notebook."""
    write_notebook("01_preprocessing_drugscom.ipynb", preprocessing_drugscom())
    write_notebook("02_preprocessing_druglib.ipynb", preprocessing_druglib())
    write_notebook("03_preprocessing_webmd.ipynb", preprocessing_webmd())
    write_notebook("04_splitting.ipynb", splitting_notebook())
    write_notebook("05_baselines.ipynb", baselines_notebook())
    write_notebook("06_train_medsentix_D.ipynb", train_variant_notebook("D", "MedSentiX-D Training", "Train the Drugs.com-only variant with auxiliary aspect loss disabled."))
    write_notebook("07_train_medsentix_DW.ipynb", train_variant_notebook("DW", "MedSentiX-DW Training", "Train on Drugs.com plus WebMD with WebMD numerical aspect supervision."))
    write_notebook("08_train_medsentix_DDL.ipynb", train_variant_notebook("DDL", "MedSentiX-DDL Training", "Train on Drugs.com plus Druglib with Druglib efficacy and side-effect supervision."))
    write_notebook("09_train_medsentix_Full.ipynb", train_variant_notebook("Full", "MedSentiX-Full Training", "Train the headline variant on Drugs.com, WebMD, and Druglib."))
    write_notebook("10_evaluation.ipynb", evaluation_notebook())
    write_notebook("11_ablation.ipynb", ablation_notebook())
    write_notebook("12_absa_validation.ipynb", absa_notebook())
    write_notebook("13_shap_analysis.ipynb", shap_notebook())
    write_notebook("MedSentiX_full_pipeline.ipynb", full_pipeline_notebook())


if __name__ == "__main__":
    main()
