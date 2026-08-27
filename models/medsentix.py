"""MedSentiX architecture, training, evaluation, ablation, and explainability.

This module implements the four-component model described in the specification:
BioBERT, BiLSTM, guided multi-head aspect attention, and the classification
head. It also contains reusable experiment helpers used by the notebooks.
"""

from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import shap
except Exception:  # pragma: no cover - SHAP notebooks report this directly.
    shap = None

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except Exception:  # pragma: no cover - notebooks surface dependency errors clearly.
    AutoModel = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None

from models.baselines import (
    CLASS_WEIGHTS,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    compute_classification_metrics,
    parameter_count,
    save_confusion_matrix,
    save_training_curve,
)
from utils.device import RANDOM_SEED, default_num_workers, get_amp_dtype, get_device, set_seed
from utils.memory import cleanup_memory


# Training constants are kept identical to the specification.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
BATCH_SIZE = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 10
WARMUP_RATIO = 0.1
DROPOUT = 0.3
MAX_LEN = 512
LAMBDA_ASPECT = 0.3
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 3
MISSING_ASPECT_LABEL = -100
ASPECT_NAMES = ["efficacy", "side_effects", "ease_of_use", "overall_satisfaction"]
ASPECT_COLUMNS = ["efficacy_label", "side_effects_label", "ease_label", "satisfaction_label"]


def ensure_medsentix_dirs(project_root: Path = PROJECT_ROOT) -> None:
    """Create all output directories used by MedSentiX experiments."""
    for rel in [
        "checkpoints/medsentix",
        "results/tables",
        "results/figures/confusion_matrices",
        "results/figures/training_curves",
        "results/figures/attention_heatmaps",
        "results/figures/shap_plots",
    ]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a sequence over valid, non-padding tokens."""
    mask = mask.unsqueeze(-1).float()
    return (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class BioBERTEncoder(nn.Module):
    """Fine-tuned BioBERT encoder with dropout applied to token states."""

    def __init__(self, model_name: str = BIOBERT_MODEL_NAME, dropout: float = DROPOUT):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required to instantiate BioBERTEncoder.")
        self.biobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)


class BiLSTMLayer(nn.Module):
    """Two-layer bidirectional LSTM that maps BioBERT states to 512 dimensions."""

    def __init__(self, input_size: int = 768, hidden_size: int = 256, num_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequence)
        return self.layer_norm(output)


class GuidedMultiHeadAspectAttention(nn.Module):
    """Four-head attention layer with explicit aspect head assignments."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        key_padding_mask = attention_mask.eq(0)
        attended, attn_weights = self.attention(
            sequence,
            sequence,
            sequence,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = self.layer_norm(sequence + self.dropout(attended))
        pooled = masked_mean(attended, attention_mask)

        # Split the attended representation into four head-specific projections
        # for auxiliary aspect supervision.
        head_sequences = torch.chunk(attended, self.num_heads, dim=-1)
        head_outputs = [masked_mean(head_sequence, attention_mask) for head_sequence in head_sequences]
        return pooled, attn_weights, head_outputs


class ClassificationHead(nn.Module):
    """Main sentiment classification head for the pooled MedSentiX state."""

    def __init__(self, input_dim: int = 512, dropout: float = DROPOUT):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


class MedSentiX(nn.Module):
    """Hybrid BioBERT-BiLSTM model with guided aspect-aware attention."""

    def __init__(
        self,
        model_name: str = BIOBERT_MODEL_NAME,
        dropout: float = DROPOUT,
        use_bilstm: bool = True,
        use_attention: bool = True,
        use_auxiliary_heads: bool = True,
    ):
        super().__init__()
        self.use_bilstm = use_bilstm
        self.use_attention = use_attention
        self.use_auxiliary_heads = use_auxiliary_heads
        self.encoder = BioBERTEncoder(model_name=model_name, dropout=dropout)
        encoder_dim = self.encoder.biobert.config.hidden_size
        self.bilstm = BiLSTMLayer(input_size=encoder_dim, dropout=dropout) if use_bilstm else nn.Identity()
        feature_dim = 512 if use_bilstm else encoder_dim
        self.attention = GuidedMultiHeadAspectAttention(embed_dim=feature_dim, num_heads=4) if use_attention else None
        self.classifier = ClassificationHead(input_dim=feature_dim, dropout=dropout)
        head_dim = feature_dim // 4
        self.auxiliary_classifiers = nn.ModuleList([nn.Linear(head_dim, NUM_CLASSES) for _ in range(4)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        sequence = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.bilstm(sequence)
        if self.use_attention and self.attention is not None:
            pooled, attn_weights, head_outputs = self.attention(sequence, attention_mask, return_attention=return_attention)
        else:
            pooled = masked_mean(sequence, attention_mask)
            attn_weights = None
            head_outputs = [chunk for chunk in torch.chunk(pooled, 4, dim=-1)]
        logits = self.classifier(pooled)
        auxiliary_logits = []
        if self.use_auxiliary_heads:
            auxiliary_logits = [classifier(head_output) for classifier, head_output in zip(self.auxiliary_classifiers, head_outputs)]
        return logits, attn_weights, auxiliary_logits


class MedSentiXDataset(Dataset):
    """Tokenizer-backed dataset with optional labels for four aspect heads."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        # No fixed-length padding here — see medsentix_collate. Most reviews
        # are far shorter than MAX_LEN, so per-sample padding to 512 was
        # wasting the bulk of every forward/backward pass on PAD tokens.
        encoded = self.tokenizer(
            str(row["review"]),
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
        for column in ASPECT_COLUMNS:
            item[column] = torch.tensor(int(row.get(column, MISSING_ASPECT_LABEL)), dtype=torch.long)
        return item


def medsentix_collate(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad each batch to its own longest sequence instead of MAX_LEN (512)."""
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["labels"] for b in batch])
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    for column in ASPECT_COLUMNS:
        out[column] = torch.stack([b[column] for b in batch])
    return out


def _coerce_aspect_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return an integer aspect column or a missing-label placeholder."""
    if column not in frame.columns:
        return pd.Series([MISSING_ASPECT_LABEL] * len(frame), index=frame.index, dtype="int64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(MISSING_ASPECT_LABEL).astype("int64")


def standardize_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Align each processed split to the shared MedSentiX training schema."""
    output = pd.DataFrame(index=frame.index)
    output["review"] = frame["review"].astype(str)
    output["label"] = pd.to_numeric(frame["label"], errors="coerce").astype("int64")
    for column in ASPECT_COLUMNS:
        output[column] = _coerce_aspect_column(frame, column)
    output["source"] = source
    return output.dropna(subset=["review", "label"]).reset_index(drop=True)


def load_variant_splits(variant: str, project_root: Path = PROJECT_ROOT, dev_mode: bool = False, sample_size: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine dataset splits for one MedSentiX variant."""
    split_dir = project_root / "data/splits"
    datasets_by_variant = {
        "D": ["drugs_com"],
        "DW": ["drugs_com", "webmd"],
        "DDL": ["drugs_com", "druglib"],
        "Full": ["drugs_com", "webmd", "druglib"],
    }
    if variant not in datasets_by_variant:
        raise ValueError(f"Unknown variant: {variant}")

    combined = {}
    for split in ["train", "val", "test"]:
        pieces = []
        for dataset_name in datasets_by_variant[variant]:
            path = split_dir / f"{dataset_name}_{split}.csv"
            frame = pd.read_csv(path)
            if dev_mode:
                frame = frame.head(sample_size)
            pieces.append(standardize_frame(frame, dataset_name))
        combined[split] = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return combined["train"], combined["val"], combined["test"]


def make_medsentix_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """Create BioBERT tokenizer and dataloaders for MedSentiX."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required for MedSentiX dataloaders.")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = partial(medsentix_collate, pad_id=pad_id)
    workers = default_num_workers(4)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    train_loader = DataLoader(MedSentiXDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True, collate_fn=collate, **common)
    val_loader = DataLoader(MedSentiXDataset(val_df, tokenizer), batch_size=batch_size, collate_fn=collate, **common)
    test_loader = DataLoader(MedSentiXDataset(test_df, tokenizer), batch_size=batch_size, collate_fn=collate, **common)
    return train_loader, val_loader, test_loader, tokenizer


def compute_medsentix_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    auxiliary_logits: Sequence[torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    criterion: nn.Module,
    lambda_aspect: float = LAMBDA_ASPECT,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine main sentiment loss with valid per-head auxiliary aspect losses."""
    main_loss = criterion(logits, labels)
    aux_total = torch.tensor(0.0, device=logits.device)
    aux_values: Dict[str, float] = {}
    for index, (name, column) in enumerate(zip(ASPECT_NAMES, ASPECT_COLUMNS)):
        if index >= len(auxiliary_logits):
            continue
        aspect_labels = batch[column].to(logits.device)
        valid_mask = aspect_labels.ne(MISSING_ASPECT_LABEL)
        if valid_mask.any():
            aux_loss = criterion(auxiliary_logits[index][valid_mask], aspect_labels[valid_mask])
            aux_total = aux_total + aux_loss
            aux_values[f"{name}_loss"] = float(aux_loss.detach().cpu())
    total_loss = main_loss + (lambda_aspect * aux_total if aux_values else 0.0)
    return total_loss, {"main_loss": float(main_loss.detach().cpu()), **aux_values}


def evaluate_medsentix(model: MedSentiX, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a MedSentiX model and return metrics, labels, predictions, probabilities."""
    model.eval()
    labels: List[int] = []
    predictions: List[int] = []
    probabilities: List[np.ndarray] = []
    batch = input_ids = attention_mask = logits = probs = None
    amp_dtype = get_amp_dtype(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            probabilities.extend(probs.detach().cpu().numpy())
            predictions.extend(torch.argmax(probs, dim=1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
    elapsed = time.perf_counter() - start
    inference_ms = 1000.0 * elapsed / max(len(labels), 1)
    metrics = compute_classification_metrics(labels, predictions, inference_ms, parameter_count(model))
    metrics["ece"] = expected_calibration_error(np.asarray(labels), np.asarray(probabilities))
    y_true, y_pred, probability_array = np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)
    del batch, input_ids, attention_mask, logits, probs
    return metrics, y_true, y_pred, probability_array


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error from predicted probabilities."""
    if len(y_true) == 0:
        return math.nan
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == y_true
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def train_medsentix_variant(
    variant: str,
    project_root: Path = PROJECT_ROOT,
    dev_mode: bool = False,
    sample_size: int = 1000,
    epochs: int = EPOCHS,
    lambda_aspect: Optional[float] = None,
    use_bilstm: bool = True,
    use_attention: bool = True,
    use_auxiliary_heads: bool = True,
    checkpoint_name: Optional[str] = None,
    save_variant_results: bool = True,
) -> Tuple[MedSentiX, pd.DataFrame, Dict[str, List[float]]]:
    """Train one MedSentiX variant and save its checkpoint and result row."""
    set_seed(RANDOM_SEED)
    ensure_medsentix_dirs(project_root)
    effective_lambda = 0.0 if variant == "D" else (LAMBDA_ASPECT if lambda_aspect is None else lambda_aspect)
    train_df, val_df, test_df = load_variant_splits(variant, project_root, dev_mode, sample_size)
    train_loader, val_loader, test_loader, tokenizer = make_medsentix_loaders(train_df, val_df, test_df)

    device = get_device()
    model = MedSentiX(use_bilstm=use_bilstm, use_attention=use_attention, use_auxiliary_heads=use_auxiliary_heads).to(device)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    checkpoint_stem = checkpoint_name or f"medsentix_{variant}"
    checkpoint_path = project_root / f"checkpoints/medsentix/{checkpoint_stem}.pt"
    history = {"train_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    batch = input_ids = attention_mask = labels = logits = auxiliary_logits = loss = None

    amp_dtype = get_amp_dtype(device)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)  # ~10 plain-text checkpoints per epoch
        for step, batch in enumerate(tqdm(train_loader, desc=f"MedSentiX-{variant} epoch {epoch + 1}/{epochs}"), start=1):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits, _, auxiliary_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, _ = compute_medsentix_loss(logits, labels, auxiliary_logits, batch, criterion, effective_lambda)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            # Plain print(), not tqdm's \r-based update — Kaggle's log
            # viewer/exporter (especially in background/commit mode) doesn't
            # reliably capture carriage-return overwrites, so without this a
            # long run can look completely silent even while it's healthy.
            if step % log_every == 0 or step == total_batches:
                print(
                    f"[MedSentiX-{variant}] epoch {epoch + 1}/{epochs} "
                    f"step {step}/{total_batches} loss={float(np.mean(losses[-log_every:])):.4f}",
                    flush=True,
                )

        val_metrics, _, _, _ = evaluate_medsentix(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        print(
            f"[MedSentiX-{variant}] epoch {epoch + 1}/{epochs} done — "
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
                print(f"[MedSentiX-{variant}] early stopping at epoch {epoch + 1} (patience={EARLY_STOPPING_PATIENCE})", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics, y_true, y_pred, probabilities = evaluate_medsentix(model, test_loader, device)
    metrics["model"] = f"medsentix_{variant}"
    metrics["cross_dataset_generalization"] = cross_dataset_generalization(model, variant, project_root, device, dev_mode, sample_size)
    save_confusion_matrix(y_true, y_pred, f"medsentix_{variant}", project_root)
    save_training_curve(history, f"medsentix_{variant}", project_root)
    with checkpoint_path.with_suffix(".history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    new_row = pd.DataFrame([metrics])
    result_df = new_row
    if save_variant_results:
        table_path = project_root / "results/tables/variant_comparison.csv"
        if table_path.exists():
            previous = pd.read_csv(table_path)
            previous = previous[previous["model"] != metrics["model"]]
            result_df = pd.concat([previous, new_row], ignore_index=True)
        result_df.to_csv(table_path, index=False)
    # Keep only the explicitly returned trained model. All other resources are
    # completed and can be released before the caller starts another variant.
    del optimizer, scheduler, criterion
    del train_loader, val_loader, test_loader, tokenizer
    del train_df, val_df, test_df, batch, input_ids, attention_mask, labels, logits, auxiliary_logits, loss
    del y_true, y_pred, probabilities, new_row
    cleanup_memory()
    return model, result_df, history


def load_medsentix_checkpoint(
    variant: str,
    project_root: Path = PROJECT_ROOT,
    device: Optional[torch.device] = None,
    **model_kwargs,
) -> MedSentiX:
    """Load a saved MedSentiX checkpoint for evaluation or explainability."""
    device = device or get_device()
    model = MedSentiX(**model_kwargs).to(device)
    checkpoint_path = project_root / f"checkpoints/medsentix/medsentix_{variant}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def cross_dataset_generalization(
    model: MedSentiX,
    variant: str,
    project_root: Path,
    device: torch.device,
    dev_mode: bool = False,
    sample_size: int = 1000,
) -> float:
    """Average accuracy on held-out datasets not used directly by a variant."""
    heldout_by_variant = {
        "D": ["webmd", "druglib"],
        "DW": ["druglib"],
        "DDL": ["webmd"],
        "Full": [],
    }
    heldouts = heldout_by_variant.get(variant, [])
    if not heldouts:
        return math.nan
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
    accuracies: List[float] = []
    for dataset_name in heldouts:
        path = project_root / f"data/splits/{dataset_name}_test.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if dev_mode:
            frame = frame.head(sample_size)
        standardized = standardize_frame(frame, dataset_name)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        loader = DataLoader(
            MedSentiXDataset(standardized, tokenizer),
            batch_size=BATCH_SIZE,
            collate_fn=partial(medsentix_collate, pad_id=pad_id),
            num_workers=default_num_workers(2),
            pin_memory=True,
        )
        metrics, labels, predictions, probabilities = evaluate_medsentix(model, loader, device)
        accuracies.append(float(metrics["accuracy"]))
        del loader, standardized, frame, metrics, labels, predictions, probabilities
        cleanup_memory()
    del tokenizer
    cleanup_memory()
    return float(np.mean(accuracies)) if accuracies else math.nan


def evaluate_all_medsentix_variants(
    project_root: Path = PROJECT_ROOT,
    dev_mode: bool = False,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """Evaluate every saved MedSentiX checkpoint on its configured test split."""
    ensure_medsentix_dirs(project_root)
    device = get_device()
    rows = []
    for variant in ["D", "DW", "DDL", "Full"]:
        checkpoint_path = project_root / f"checkpoints/medsentix/medsentix_{variant}.pt"
        if not checkpoint_path.exists():
            print(f"Skipping MedSentiX-{variant}: missing checkpoint {checkpoint_path}")
            continue
        _, _, test_df = load_variant_splits(variant, project_root, dev_mode, sample_size)
        unused_train_loader, unused_val_loader, test_loader, tokenizer = make_medsentix_loaders(test_df, test_df, test_df)
        model = load_medsentix_checkpoint(variant, project_root, device)
        try:
            metrics, y_true, y_pred, probabilities = evaluate_medsentix(model, test_loader, device)
            metrics["model"] = f"medsentix_{variant}"
            metrics["cross_dataset_generalization"] = cross_dataset_generalization(model, variant, project_root, device, dev_mode, sample_size)
            save_confusion_matrix(y_true, y_pred, f"medsentix_{variant}", project_root)
            rows.append(metrics)
        finally:
            del model, unused_train_loader, unused_val_loader, test_loader, tokenizer, test_df
            cleanup_memory()
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df.to_csv(project_root / "results/tables/variant_comparison.csv", index=False)
    return result_df


def run_ablation_study(
    project_root: Path = PROJECT_ROOT,
    dev_mode: bool = True,
    sample_size: int = 1000,
    epochs: int = EPOCHS,
) -> pd.DataFrame:
    """Run focused ablations for the Full variant without changing headline settings."""
    rows = []
    configs = [
        {"ablation": "full_model", "use_bilstm": True, "use_attention": True, "use_auxiliary_heads": True, "lambda_aspect": LAMBDA_ASPECT},
        {"ablation": "no_auxiliary_guidance", "use_bilstm": True, "use_attention": True, "use_auxiliary_heads": True, "lambda_aspect": 0.0},
        {"ablation": "no_guided_attention", "use_bilstm": True, "use_attention": False, "use_auxiliary_heads": False, "lambda_aspect": 0.0},
        {"ablation": "no_bilstm", "use_bilstm": False, "use_attention": True, "use_auxiliary_heads": True, "lambda_aspect": LAMBDA_ASPECT},
    ]
    for config in configs:
        model, train_result, history = train_medsentix_variant(
            "Full",
            project_root=project_root,
            dev_mode=dev_mode,
            sample_size=sample_size,
            epochs=epochs,
            lambda_aspect=config["lambda_aspect"],
            use_bilstm=config["use_bilstm"],
            use_attention=config["use_attention"],
            use_auxiliary_heads=config["use_auxiliary_heads"],
            checkpoint_name=f"ablation_{config['ablation']}",
            save_variant_results=False,
        )
        device = next(model.parameters()).device
        _, _, test_df = load_variant_splits("Full", project_root, dev_mode, sample_size)
        unused_train_loader, unused_val_loader, test_loader, tokenizer = make_medsentix_loaders(test_df, test_df, test_df)
        try:
            metrics, labels, predictions, probabilities = evaluate_medsentix(model, test_loader, device)
            rows.append({"ablation": config["ablation"], **metrics})
        finally:
            del model, train_result, history
            del unused_train_loader, unused_val_loader, test_loader, tokenizer, test_df
            cleanup_memory()
    result_df = pd.DataFrame(rows)
    result_df.to_csv(project_root / "results/tables/ablation_results.csv", index=False)
    return result_df


def validate_absa_on_druglib(
    variants: Sequence[str] = ("DDL", "Full"),
    project_root: Path = PROJECT_ROOT,
    dev_mode: bool = False,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """Evaluate aspect heads on Druglib labels and save ABSA validation metrics."""
    ensure_medsentix_dirs(project_root)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
    frame = pd.read_csv(project_root / "data/splits/druglib_test.csv")
    if dev_mode:
        frame = frame.head(sample_size)
    dataset = MedSentiXDataset(standardize_frame(frame, "druglib"), tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=partial(medsentix_collate, pad_id=pad_id),
        num_workers=default_num_workers(2),
        pin_memory=True,
    )
    rows = []
    for variant in variants:
        checkpoint_path = project_root / f"checkpoints/medsentix/medsentix_{variant}.pt"
        if not checkpoint_path.exists():
            print(f"Skipping ABSA validation for MedSentiX-{variant}: missing checkpoint.")
            continue
        model = load_medsentix_checkpoint(variant, project_root, device)
        try:
            aspect_true: Dict[str, List[int]] = {name: [] for name in ASPECT_NAMES}
            aspect_pred: Dict[str, List[int]] = {name: [] for name in ASPECT_NAMES}
            main_true: List[int] = []
            main_pred: List[int] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    logits, _, auxiliary_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    main_true.extend(batch["labels"].tolist())
                    main_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    for index, (name, column) in enumerate(zip(ASPECT_NAMES, ASPECT_COLUMNS)):
                        labels = batch[column]
                        valid = labels.ne(MISSING_ASPECT_LABEL)
                        if valid.any() and index < len(auxiliary_logits):
                            aspect_true[name].extend(labels[valid].tolist())
                            aspect_pred[name].extend(torch.argmax(auxiliary_logits[index].detach().cpu()[valid], dim=1).tolist())
            row = {"model": f"medsentix_{variant}"}
            f1_values = []
            for name in ASPECT_NAMES:
                if aspect_true[name]:
                    row[f"{name}_accuracy"] = accuracy_score(aspect_true[name], aspect_pred[name])
                    row[f"{name}_f1"] = f1_score(aspect_true[name], aspect_pred[name], average="macro", zero_division=0)
                    f1_values.append(row[f"{name}_f1"])
                elif name == "overall_satisfaction":
                    row[f"{name}_accuracy"] = accuracy_score(main_true, main_pred)
                    row[f"{name}_f1"] = f1_score(main_true, main_pred, average="macro", zero_division=0)
                    f1_values.append(row[f"{name}_f1"])
                else:
                    row[f"{name}_accuracy"] = math.nan
                    row[f"{name}_f1"] = math.nan
            row["average_aspect_f1"] = float(np.nanmean(f1_values)) if f1_values else math.nan
            rows.append(row)
        finally:
            del model
            cleanup_memory()
    result_df = pd.DataFrame(rows)
    result_df.to_csv(project_root / "results/tables/absa_validation.csv", index=False)
    del loader, dataset, frame, tokenizer
    cleanup_memory()
    return result_df


def attention_entropy(attention_weights: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Compute normalized entropy for each attention head."""
    weights = attention_weights.detach().cpu()
    mask = attention_mask.detach().cpu().bool()
    entropies = []
    for head in range(weights.size(1)):
        head_weights = weights[:, head]
        valid_values = []
        for batch_index in range(head_weights.size(0)):
            valid_len = int(mask[batch_index].sum().item())
            if valid_len > 1:
                values = head_weights[batch_index, :valid_len, :valid_len].clamp(min=1e-12)
                entropy = -(values * values.log()).sum(dim=-1) / math.log(valid_len)
                valid_values.append(float(entropy.mean().item()))
        entropies.append(float(np.mean(valid_values)) if valid_values else math.nan)
    return np.asarray(entropies)


def save_attention_heatmap(
    model: MedSentiX,
    tokenizer,
    review: str,
    project_root: Path = PROJECT_ROOT,
    filename: str = "medsentix_full_attention_heatmap.png",
) -> None:
    """Save an aspect-head attention heatmap for one review."""
    ensure_medsentix_dirs(project_root)
    device = next(model.parameters()).device
    encoded = tokenizer(review, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))[: int(encoded["attention_mask"].sum())]
    with torch.inference_mode():
        _, attention, _ = model(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device), return_attention=True
        )
    if attention is None:
        del encoded
        cleanup_memory()
        return
    attention = attention.detach().cpu()[0, :, : len(tokens), : len(tokens)].mean(dim=1).numpy()
    plt.figure(figsize=(max(8, len(tokens) * 0.25), 4))
    sns.heatmap(attention, cmap="viridis", xticklabels=tokens, yticklabels=ASPECT_NAMES)
    plt.xticks(rotation=90)
    plt.title("MedSentiX Aspect Attention")
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/attention_heatmaps/{filename}", dpi=300)
    plt.close()
    del attention, encoded
    cleanup_memory()


def run_shap_analysis(
    project_root: Path = PROJECT_ROOT,
    dev_mode: bool = True,
    sample_size: int = 100,
) -> pd.DataFrame:
    """Run SHAP explainability for MedSentiX-Full and save paper figures."""
    ensure_medsentix_dirs(project_root)
    if shap is None:
        raise ImportError("shap is required for SHAP analysis.")
    device = get_device()
    model = load_medsentix_checkpoint("Full", project_root, device)
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
    frame = pd.read_csv(project_root / "data/splits/drugs_com_test.csv")
    if dev_mode:
        frame = frame.head(sample_size)
    texts = frame["review"].astype(str).tolist()

    def predict_proba(batch_texts: Sequence[str]) -> np.ndarray:
        encoded = tokenizer(
            list(batch_texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits, _, _ = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, output_names=[ID2LABEL[i] for i in range(NUM_CLASSES)])
    shap_values = explainer(texts)

    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/shap_plots/global_top20_bar.png", dpi=300)
    plt.close()

    for class_index, class_name in ID2LABEL.items():
        shap.plots.bar(shap_values[:, :, class_index], max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(project_root / f"results/figures/shap_plots/{class_name.lower()}_class_bar.png", dpi=300)
        plt.close()

    for index in range(min(3, len(texts))):
        shap.plots.waterfall(shap_values[index, :, int(frame.iloc[index]["label"])], max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(project_root / f"results/figures/shap_plots/waterfall_review_{index + 1}.png", dpi=300)
        plt.close()

    entropy_rows = []
    for text in texts[: min(32, len(texts))]:
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        with torch.inference_mode():
            _, attention, _ = model(
                encoded["input_ids"].to(device), encoded["attention_mask"].to(device), return_attention=True
            )
        if attention is not None:
            entropy_rows.append(attention_entropy(attention, encoded["attention_mask"]))
    entropy_df = pd.DataFrame(entropy_rows, columns=ASPECT_NAMES)
    entropy_summary = entropy_df.mean().reset_index()
    entropy_summary.columns = ["aspect_head", "attention_entropy"]
    entropy_summary.to_csv(project_root / "results/tables/shap_attention_entropy.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=entropy_summary, x="aspect_head", y="attention_entropy")
    plt.xticks(rotation=20, ha="right")
    plt.title("Attention Entropy by Aspect Head")
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/shap_plots/attention_entropy_by_head.png", dpi=300)
    plt.close()
    del model, tokenizer, frame, texts, masker, explainer, shap_values, entropy_rows, entropy_df, encoded, attention
    cleanup_memory()
    return entropy_summary
