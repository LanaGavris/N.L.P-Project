"""
Module: BERT Fine-Tuner
═══════════════════════
Fine-tunes bert-base-uncased on the labeled contradiction pairs.

Input : labeled_pairs.json  (from auto_labeler.py)
Output: ./bert_contradiction_model/  (saved model)

Architecture:
  BERT [CLS] claim_a [SEP] claim_b [SEP] → Linear(768 → 2) → softmax

Split: 70% train / 15% val / 15% test
The test split is HELD OUT — only used in evaluation.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ─── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME  = "bert-base-uncased"
MAX_LENGTH  = 256       # tokens per pair — enough for two claims
BATCH_SIZE  = 16        # safe for Colab T4 GPU
EPOCHS      = 4
LR          = 2e-5
WARMUP_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────

class ContradictionDataset(Dataset):
    def __init__(self, texts_a: list, texts_b: list, labels: list, tokenizer):
        self.encodings = tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids",
                              torch.zeros_like(self.encodings["input_ids"]))[idx],
            "labels":         self.labels[idx],
        }


# ─── Data loading & splitting ──────────────────────────────────────────────

def load_and_split(
    json_path: str,
    test_size: float = 0.15,
    val_size: float  = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load labeled pairs and split into train / val / test.
    Test set is HELD OUT — saved separately for evaluation.
    """
    with open(json_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Keep only confident labels (0 or 1, not -1)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    print(f"[Split] Total pairs: {len(df)}")
    print(f"  Contradictions (1): {df['label'].sum()}")
    print(f"  Non-contradictions (0): {(df['label'] == 0).sum()}")
    print(f"  Balance: {df['label'].mean():.2%} positive")

    # Stratified split to preserve class balance
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=SEED
    )
    val_ratio_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio_adjusted, stratify=train_val["label"],
        random_state=SEED
    )

    print(f"\n[Split] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Train pos: {train['label'].mean():.2%} | "
          f"Val pos: {val['label'].mean():.2%} | "
          f"Test pos: {test['label'].mean():.2%}")

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ─── Training ─────────────────────────────────────────────────────────────

class BERTFineTuner:
    """Fine-tunes BERT for binary contradiction classification."""

    def __init__(self, model_name: str = MODEL_NAME, output_dir: str = "bert_contradiction_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer  = BertTokenizer.from_pretrained(model_name)
        self.model      = None
        print(f"[BERT] Device: {DEVICE}")
        print(f"[BERT] Model: {model_name}")

    def train(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        epochs:   int   = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float       = LR,
    ) -> dict:
        """Fine-tune BERT. Returns training history."""

        # Build datasets
        train_ds = ContradictionDataset(
            train_df["text_a"].tolist(),
            train_df["text_b"].tolist(),
            train_df["label"].tolist(),
            self.tokenizer,
        )
        val_ds = ContradictionDataset(
            val_df["text_a"].tolist(),
            val_df["text_b"].tolist(),
            val_df["label"].tolist(),
            self.tokenizer,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        # Class weights to handle imbalance
        class_weights = compute_class_weight(
            "balanced",
            classes=np.array([0, 1]),
            y=train_df["label"].values,
        )
        weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        # Model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(DEVICE)

        # Optimizer & scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_f1 = 0.0
        best_model_path = os.path.join(self.output_dir, "best_model")

        print(f"\n[BERT] Starting training: {epochs} epochs, {len(train_loader)} steps/epoch")

        for epoch in range(epochs):
            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            train_losses = []

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                token_type_ids = batch["token_type_ids"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                # Apply class weights to loss
                import torch.nn.functional as F
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels, weight=weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

                if (step + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} "
                          f"| Loss: {np.mean(train_losses[-10:]):.4f}", end="\r")

            avg_train_loss = np.mean(train_losses)

            # ── Validate ───────────────────────────────────────────────────
            val_metrics = self._evaluate_loader(val_loader, weights)
            val_f1 = val_metrics["f1"]

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_f1)

            print(f"\n  Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save(best_model_path)
                print(f"  ✓ New best model saved (F1={val_f1:.4f})")

        print(f"\n[BERT] Training complete. Best Val F1: {best_val_f1:.4f}")
        print(f"[BERT] Best model saved to: {best_model_path}")
        return history

    def _evaluate_loader(self, loader: DataLoader, class_weights=None) -> dict:
        """Evaluate on a DataLoader, return metrics dict."""
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        import torch.nn.functional as F

        self.model.eval()
        all_preds, all_labels, losses = [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                token_type_ids = batch["token_type_ids"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits
                preds  = torch.argmax(logits, dim=1)

                if class_weights is not None:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
                    losses.append(loss.item())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return {
            "accuracy":  accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall":    recall_score(all_labels, all_preds, zero_division=0),
            "f1":        f1_score(all_labels, all_preds, zero_division=0),
            "loss":      np.mean(losses) if losses else 0.0,
            "preds":     all_preds,
            "labels":    all_labels,
        }

    def predict(self, texts_a: list[str], texts_b: list[str]) -> tuple[list[int], list[float]]:
        """Predict labels and probabilities for new pairs."""
        assert self.model is not None, "Model not trained. Call train() first."
        self.model.eval()

        ds = ContradictionDataset(texts_a, texts_b, [0] * len(texts_a), self.tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE)

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    token_type_ids=batch["token_type_ids"].to(DEVICE),
                )
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        return all_preds, all_probs

    def _save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save(self, path: str | None = None) -> None:
        save_path = path or os.path.join(self.output_dir, "final_model")
        self._save(save_path)
        print(f"[BERT] Model saved to: {save_path}")

    @classmethod
    def load(cls, model_path: str) -> "BERTFineTuner":
        """Load a saved fine-tuned model."""
        instance = cls.__new__(cls)
        instance.model_name  = model_path
        instance.output_dir  = str(Path(model_path).parent)
        instance.tokenizer   = BertTokenizer.from_pretrained(model_path)
        instance.model       = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(DEVICE)
        print(f"[BERT] Loaded model from: {model_path}")
        return instance


# ─── Training history plot ─────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = "training_history.png") -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Val Loss",   color="#3498db")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_f1"], "^-", color="#2ecc71", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 (contradiction class)")
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)

    plt.suptitle("BERT Fine-Tuning — bert-base-uncased", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Training history saved: {save_path}")


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load splits (assumes labeled_pairs.json exists)
    train_df, val_df, test_df = load_and_split("labeled_pairs.json")

    # Save test split for evaluation
    test_df.to_json("test_split.json", orient="records", indent=2)
    print(f"✓ Test split saved: test_split.json ({len(test_df)} pairs)")

    # Fine-tune
    trainer = BERTFineTuner(output_dir="bert_contradiction_model")
    history = trainer.train(train_df, val_df)
    trainer.save()
    plot_training_history(history)
