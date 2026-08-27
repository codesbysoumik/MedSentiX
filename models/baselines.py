"""Baseline models and training utilities for MedSentiX.

The code in this module follows the paper specification for all seven
baselines. Notebook cells call these helpers so the implementation remains
reproducible while keeping each notebook readable.
"""

from __future__ import annotations

import json
import math
import re
import time
from functools import partial
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except Exception:  # pragma: no cover - notebooks surface dependency errors clearly.
    AutoModel = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None

from utils.device import RANDOM_SEED, default_num_workers, get_amp_dtype, get_device, set_seed
from utils.memory import cleanup_memory


# Project-level constants mirror the paper specification.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
NUM_CLASSES = 3
GLOVE_DIM = 100
GLOVE_MAX_LEN = 256
TRANSFORMER_MAX_LEN = 512
BATCH_SIZE = 24
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
DROPOUT = 0.3
CLASS_WEIGHTS = torch.tensor([1.5, 2.0, 1.0], dtype=torch.float)


def ensure_output_dirs(project_root: Path = PROJECT_ROOT) -> None:
    """Create checkpoint and result directories used by the baseline workflow."""
    for rel in [
        "checkpoints/baselines",
        "results/tables",
        "results/figures/confusion_matrices",
        "results/figures/training_curves",
    ]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)


def parameter_count(model: nn.Module) -> int:
    """Count trainable and frozen parameters for reporting in result tables."""
    return int(sum(param.numel() for param in model.parameters()))


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    inference_ms_per_sample: float = math.nan,
    params: float = math.nan,
) -> Dict[str, float]:
    """Compute every classification metric required by the paper specification."""
    labels = [0, 1, 2]
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "negative_f1": per_class_f1[0],
        "neutral_f1": per_class_f1[1],
        "positive_f1": per_class_f1[2],
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "inference_ms_per_sample": inference_ms_per_sample,
        "parameter_count": params,
    }


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model_name: str,
    project_root: Path = PROJECT_ROOT,
) -> np.ndarray:
    """Save a normalized confusion matrix figure and return the matrix values."""
    output_dir = project_root / "results/figures/confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        yticklabels=[ID2LABEL[i] for i in range(NUM_CLASSES)],
    )
    plt.title(f"{model_name} Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()
    return matrix


def save_training_curve(history: Dict[str, List[float]], model_name: str, project_root: Path = PROJECT_ROOT) -> None:
    """Save loss and validation accuracy curves for neural baselines."""
    if not history:
        return
    output_dir = project_root / "results/figures/training_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if history.get("train_loss"):
        plt.plot(history["train_loss"], label="Train Loss")
    if history.get("val_accuracy"):
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Training Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_training_curve.png", dpi=300)
    plt.close()


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text for GloVe baselines while preserving negations and numbers."""
    return re.findall(r"[a-z0-9'-]+", str(text).lower())


def build_vocab(texts: Iterable[str], max_vocab: int = 50000) -> Dict[str, int]:
    """Build a capped vocabulary with PAD=0 and UNK=1 special tokens."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def load_glove_embeddings(
    glove_path: Path,
    vocab: Dict[str, int],
    embedding_dim: int = GLOVE_DIM,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Load only the GloVe vectors needed for the current vocabulary."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.05, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)

    with glove_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            pieces = line.rstrip().split(" ")
            token = pieces[0]
            if token in vocab and len(pieces) == embedding_dim + 1:
                matrix[vocab[token]] = np.asarray(pieces[1:], dtype=np.float32)
    return matrix


class GloveReviewDataset(Dataset):
    """Numericalize review text for BiLSTM, BiLSTM-CNN, and Double-BiGRU baselines."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Dict[str, int], max_len: int = GLOVE_MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Truncated but NOT padded here — padding happens per-batch in
        # glove_collate so short reviews don't pay for GLOVE_MAX_LEN-length
        # compute. Numerically identical to fixed padding, just faster.
        tokens = simple_tokenize(self.texts[index])[: self.max_len]
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens] or [self.vocab["<PAD>"]]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(ids), dtype=torch.float),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def glove_collate(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad each batch to its own longest sequence instead of a fixed max length."""
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0.0)
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TransformerReviewDataset(Dataset):
    """Tokenize review text for BERT, RoBERTa, and BioBERT baselines."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int = TRANSFORMER_MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # No fixed-length padding here — see transformer_collate. Reviews are
        # rarely near TRANSFORMER_MAX_LEN tokens, so per-sample max_length
        # padding was burning most of the compute on PAD tokens.
        encoded = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def transformer_collate(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad each batch to its own longest sequence instead of TRANSFORMER_MAX_LEN."""
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool sequence states over non-padding tokens."""
    mask = mask.unsqueeze(-1).float()
    return (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class DoubleBiGRUClassifier(nn.Module):
    """Two stacked BiGRU layers with GloVe embeddings."""

    def __init__(self, embedding_matrix: np.ndarray, dropout: float = DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.gru1 = nn.GRU(GLOVE_DIM, 256, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, NUM_CLASSES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        out, _ = self.gru1(embedded)
        out, _ = self.gru2(out)
        return self.classifier(masked_mean(out, attention_mask))


class BiLSTMGloveClassifier(nn.Module):
    """Two-layer bidirectional LSTM with fine-tuned GloVe embeddings."""

    def __init__(self, embedding_matrix: np.ndarray, dropout: float = DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(GLOVE_DIM, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, NUM_CLASSES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.embedding(input_ids))
        return self.classifier(masked_mean(out, attention_mask))


class BiLSTMCNNGloveClassifier(nn.Module):
    """CNN feature extractor followed by a BiLSTM classifier."""

    def __init__(self, embedding_matrix: np.ndarray, dropout: float = DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False, padding_idx=0)
        self.conv = nn.Conv1d(GLOVE_DIM, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(4)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        convolved = torch.relu(self.conv(embedded))
        pooled = self.pool(convolved).transpose(1, 2)
        pooled_mask = attention_mask[:, ::4][:, : pooled.size(1)]
        if pooled_mask.size(1) < pooled.size(1):
            pad = pooled.size(1) - pooled_mask.size(1)
            pooled_mask = torch.nn.functional.pad(pooled_mask, (0, pad))
        out, _ = self.lstm(pooled)
        return self.classifier(masked_mean(out, pooled_mask))


class TransformerBiLSTMClassifier(nn.Module):
    """BERT/RoBERTa encoder with a two-layer BiLSTM classification head."""

    def __init__(self, model_name: str, dropout: float = DROPOUT):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for transformer baselines.")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(hidden_size, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.Dropout(dropout), nn.Linear(128, NUM_CLASSES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out, _ = self.lstm(hidden)
        out = self.norm(out)
        return self.classifier(masked_mean(out, attention_mask))


class BioBERTStandaloneClassifier(nn.Module):
    """BioBERT baseline using CLS pooling without BiLSTM or guided attention."""

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2", dropout: float = DROPOUT):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for BioBERT baseline.")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(cls_state)


def train_svm_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    checkpoint_path: Path,
    model_name: str = "svm",
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, float]:
    """Train and evaluate the TF-IDF + LinearSVC baseline on Drugs.com."""
    ensure_output_dirs(project_root)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
            ("svm", LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED)),
        ]
    )
    pipeline.fit(train_df["review"].astype(str), train_df["label"].astype(int))

    start = time.perf_counter()
    predictions = pipeline.predict(test_df["review"].astype(str))
    elapsed = time.perf_counter() - start
    inference_ms = 1000.0 * elapsed / max(len(test_df), 1)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, checkpoint_path)
    metrics = compute_classification_metrics(test_df["label"].astype(int), predictions, inference_ms, params=0)
    save_confusion_matrix(test_df["label"].astype(int), predictions, model_name, project_root)
    return {"model": model_name, **metrics}


def evaluate_neural_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a neural baseline and return metrics, labels, and predictions."""
    model.eval()
    labels: List[int] = []
    predictions: List[int] = []
    batch = input_ids = attention_mask = logits = None
    amp_dtype = get_amp_dtype(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
    elapsed = time.perf_counter() - start
    inference_ms = 1000.0 * elapsed / max(len(labels), 1)
    metrics = compute_classification_metrics(labels, predictions, inference_ms, parameter_count(model))
    y_true, y_pred = np.asarray(labels), np.asarray(predictions)
    del batch, input_ids, attention_mask, logits
    return metrics, y_true, y_pred


def train_neural_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: Path,
    model_name: str,
    lr: float,
    use_scheduler: bool = False,
    project_root: Path = PROJECT_ROOT,
    epochs: int = EPOCHS,
) -> Dict[str, float]:
    """Train a neural baseline with class-weighted cross entropy and early checkpointing."""
    set_seed(RANDOM_SEED)
    ensure_output_dirs(project_root)
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) if use_scheduler else torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler and get_linear_schedule_with_warmup is not None:
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

    amp_dtype = get_amp_dtype(device)
    # fp16 (used on Turing GPUs like Kaggle's T4, which lack bf16 tensor
    # cores) has a narrow exponent range and can underflow small gradients,
    # so it needs loss scaling. bf16 matches fp32's exponent range and
    # doesn't — the scaler is a no-op passthrough in that case.
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_accuracy": []}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    batch = input_ids = attention_mask = labels = loss = None

    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        for step, batch in enumerate(tqdm(train_loader, desc=f"{model_name} epoch {epoch + 1}/{epochs}"), start=1):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                loss = criterion(model(input_ids=input_ids, attention_mask=attention_mask), labels)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
            # Plain print(), not tqdm's \r — see medsentix.py for why this
            # matters on Kaggle's background/commit log export.
            if step % log_every == 0 or step == total_batches:
                print(
                    f"[{model_name}] epoch {epoch + 1}/{epochs} step {step}/{total_batches} "
                    f"loss={float(np.mean(losses[-log_every:])):.4f}",
                    flush=True,
                )

        val_metrics, _, _ = evaluate_neural_model(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        print(
            f"[{model_name}] epoch {epoch + 1}/{epochs} done — "
            f"train_loss={history['train_loss'][-1]:.4f} val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"[{model_name}] early stopping at epoch {epoch + 1} (patience={EARLY_STOPPING_PATIENCE})", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics, y_true, y_pred = evaluate_neural_model(model, test_loader, device)
    save_confusion_matrix(y_true, y_pred, model_name, project_root)
    save_training_curve(history, model_name, project_root)
    with (checkpoint_path.with_suffix(".history.json")).open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    # The caller owns the model and loaders. Release only objects local to this
    # completed training run before it advances to the next baseline.
    del optimizer, scheduler, criterion, scaler, batch, input_ids, attention_mask, labels, loss
    cleanup_memory()
    return {"model": model_name, **test_metrics}


def make_glove_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    project_root: Path = PROJECT_ROOT,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Build vocabulary, embedding matrix, and dataloaders for GloVe baselines."""
    vocab = build_vocab(train_df["review"].astype(str))
    embeddings = load_glove_embeddings(project_root / "glove/glove.6B.100d.txt", vocab)
    train_ds = GloveReviewDataset(train_df["review"], train_df["label"].astype(int), vocab)
    val_ds = GloveReviewDataset(val_df["review"], val_df["label"].astype(int), vocab)
    test_ds = GloveReviewDataset(test_df["review"], test_df["label"].astype(int), vocab)
    pad_id = vocab["<PAD>"]
    collate = partial(glove_collate, pad_id=pad_id)
    workers = default_num_workers(4)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, **common),
        DataLoader(val_ds, batch_size=batch_size, collate_fn=collate, **common),
        DataLoader(test_ds, batch_size=batch_size, collate_fn=collate, **common),
        embeddings,
    )


def make_transformer_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create tokenizer-backed dataloaders for transformer baselines."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required for transformer baselines.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = TransformerReviewDataset(train_df["review"], train_df["label"].astype(int), tokenizer)
    val_ds = TransformerReviewDataset(val_df["review"], val_df["label"].astype(int), tokenizer)
    test_ds = TransformerReviewDataset(test_df["review"], test_df["label"].astype(int), tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = partial(transformer_collate, pad_id=pad_id)
    workers = default_num_workers(4)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, **common),
        DataLoader(val_ds, batch_size=batch_size, collate_fn=collate, **common),
        DataLoader(test_ds, batch_size=batch_size, collate_fn=collate, **common),
    )


def train_all_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
    epochs: int = EPOCHS,
) -> pd.DataFrame:
    """Train all seven baselines and save the consolidated baseline result table."""
    ensure_output_dirs(project_root)
    results: List[Dict[str, float]] = []
    checkpoints = project_root / "checkpoints/baselines"

    results_path = project_root / "results/tables/baseline_results.csv"

    def _flush() -> None:
        pd.DataFrame(results).to_csv(results_path, index=False)

    results.append(train_svm_baseline(train_df, test_df, checkpoints / "svm.pkl", project_root=project_root))
    _flush()
    cleanup_memory()

    glove_train, glove_val, glove_test, embeddings = make_glove_loaders(train_df, val_df, test_df, project_root=project_root)
    glove_jobs = [
        ("double_bigru", DoubleBiGRUClassifier),
        ("bilstm", BiLSTMGloveClassifier),
        ("bilstm_cnn", BiLSTMCNNGloveClassifier),
    ]
    for name, model_class in glove_jobs:
        model = model_class(embeddings)
        try:
            results.append(
                train_neural_classifier(
                    model, glove_train, glove_val, glove_test, checkpoints / f"{name}.pt", name, lr=1e-3, epochs=epochs
                )
            )
            _flush()
        finally:
            del model
            cleanup_memory()
    del glove_train, glove_val, glove_test, embeddings
    cleanup_memory()

    transformer_jobs = [
        ("bert_bilstm", "bert-base-uncased", lambda: TransformerBiLSTMClassifier("bert-base-uncased")),
        ("roberta_bilstm", "roberta-base", lambda: TransformerBiLSTMClassifier("roberta-base")),
        ("biobert_standalone", "dmis-lab/biobert-base-cased-v1.2", BioBERTStandaloneClassifier),
    ]
    for name, tokenizer_name, model_factory in transformer_jobs:
        train_loader, val_loader, test_loader = make_transformer_loaders(train_df, val_df, test_df, tokenizer_name)
        model = model_factory()
        try:
            results.append(
                train_neural_classifier(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    checkpoints / f"{name}.pt",
                    name,
                    lr=2e-5,
                    use_scheduler=True,
                    epochs=epochs,
                )
            )
            _flush()
        finally:
            del model, train_loader, val_loader, test_loader
            cleanup_memory()

    result_df = pd.DataFrame(results)
    _flush()
    return result_df
