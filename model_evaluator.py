"""
Module: Model Evaluator (updated for dissent-based dataset)
════════════════════════════════════════════════════════════
Compares Rule-Based Detector vs Fine-Tuned BERT on held-out test set.

Compatible with ALL data sources:
  - hf_dissent_fetcher  (majority/dissent pairs, label_source=structural)
  - auto_labeler_fix    (prosecution/defense pairs, label_source=gemini/rule_based)
  - mixed               (any combination)

Metrics:
  - Accuracy, Precision, Recall, F1, ROC AUC
  - Confusion matrix
  - Per-crime-type / per-topic F1 breakdown
  - Label source breakdown (structural vs gemini vs rule_based)

Outputs (PNG + JSON):
  - model_comparison_table.png
  - confusion_matrices.png
  - metric_comparison.png
  - roc_curves.png
  - full_model_eval.json
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, classification_report,
)


# ─── Negation patterns (self-contained, no external import needed) ─────────

NEGATION_PAIRS = [
    (r"\bknew\b|\baware\b|\bdirected\b",
     r"\bunaware\b|\bdid\s+not\s+know\b|\bdelegated\b"),
    (r"\bintended?\b|\bintentional\b|\bdeliberate\b",
     r"\bgood\s+faith\b|\bno\s+intent\b|\bbelieved\b"),
    (r"\bfraud\b|\bstole\b|\bmisappropriat\b",
     r"\blawful\b|\bauthorized\b|\bpermitted\b"),
    (r"\bconcealed?\b|\bhid\b|\bdeceived?\b",
     r"\bdisclosed?\b|\btransparent\b|\baccurate\b"),
    (r"\bguilty\b|\bconvicted\b|\bliable\b",
     r"\binnocent\b|\bnot\s+guilty\b|\bacquitted\b"),
    (r"\bknowingly\b|\bwillfully\b|\bintentionally\b",
     r"\bnegligent\b|\bunintentional\b|\bmistake\b"),
    # Dissent-specific patterns
    (r"\bwe\s+hold\b|\bwe\s+affirm\b|\bwe\s+find\b",
     r"\bi\s+dissent\b|\bi\s+disagree\b|\bi\s+would\s+reverse\b"),
    (r"\bmajority\s+correctly\b|\bcourt\s+properly\b",
     r"\bmajority\s+errs\b|\bmajority\s+is\s+wrong\b|\bcannot\s+agree\b"),
]


# ─── Rule-based predictor ─────────────────────────────────────────────────

class RuleBasedPredictor:
    """
    Rule-based contradiction detector.
    Works for both prosecution/defense AND majority/dissent pairs.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def predict(
        self,
        texts_a: list[str],
        texts_b: list[str],
        party_a: list[str] | None = None,
        party_b: list[str] | None = None,
    ) -> tuple[list[int], list[float]]:

        preds, scores = [], []

        for i, (ta, tb) in enumerate(zip(texts_a, texts_b)):
            ta_l = ta.lower()
            tb_l = tb.lower()

            # Negation pattern matches
            neg_hits = 0
            for pos_pat, neg_pat in NEGATION_PAIRS:
                if ((re.search(pos_pat, ta_l) and re.search(neg_pat, tb_l)) or
                    (re.search(neg_pat, ta_l) and re.search(pos_pat, tb_l))):
                    neg_hits += 1

            # Word overlap (low overlap + negation = stronger contradiction)
            words_a = set(ta_l.split())
            words_b = set(tb_l.split())
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

            # Party bonus: opposing parties → more likely to contradict
            party_bonus = 0.0
            if party_a and party_b:
                pa = (party_a[i] or "").lower()
                pb = (party_b[i] or "").lower()
                opposing = {
                    ("prosecution", "defense"), ("defense", "prosecution"),
                    ("majority", "dissent"),     ("dissent", "majority"),
                    ("court", "defense"),        ("defense", "court"),
                }
                if (pa, pb) in opposing:
                    party_bonus = 0.1

            score = min(neg_hits * 0.25 + overlap * 0.05 + party_bonus, 1.0)
            label = 1 if score >= self.threshold else 0

            preds.append(label)
            scores.append(score)

        return preds, scores


# ─── Metrics ──────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float],
    model_name: str,
) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fpr, tpr = fpr.tolist(), tpr.tolist()
    except ValueError:
        auc, fpr, tpr = 0.0, [0, 1], [0, 1]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "model":                 model_name,
        "accuracy":              round(float(acc),  4),
        "precision":             round(float(prec), 4),
        "recall":                round(float(rec),  4),
        "f1":                    round(float(f1),   4),
        "roc_auc":               round(float(auc),  4),
        "confusion_matrix":      cm.tolist(),
        "fpr":                   fpr,
        "tpr":                   tpr,
        "classification_report": report,
        "n_test":                len(y_true),
        "n_positive":            int(sum(y_true)),
        "n_negative":            int(len(y_true) - sum(y_true)),
    }


def compute_per_group(
    df: pd.DataFrame,
    y_pred: list[int],
    group_col: str,
) -> dict:
    """F1 per group (crime_type, topic, label_source, etc.)"""
    df = df.copy()
    df["_pred"] = y_pred

    if group_col not in df.columns:
        return {}

    result = {}
    for group_val in df[group_col].unique():
        mask = df[group_col] == group_val
        if mask.sum() < 3:
            continue
        f = f1_score(
            df.loc[mask, "label"],
            df.loc[mask, "_pred"],
            zero_division=0,
        )
        result[str(group_val)] = round(float(f), 4)

    return result


# ─── Main evaluator ───────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates Rule-Based and Fine-Tuned BERT on the held-out test set.
    Auto-detects data format (dissent-based or prosecution/defense).
    """

    def __init__(
        self,
        test_json: str = "test_split.json",
        bert_model_path: str = "bert_model/best_model",
        rule_threshold: float = 0.3,
    ):
        print("[Evaluator] Loading test set...")
        with open(test_json) as f:
            data = json.load(f)
        self.test_df = pd.DataFrame(data)

        # Detect data format
        self._detect_format()

        print(f"[Evaluator] {len(self.test_df)} pairs | "
              f"label=1: {self.test_df['label'].sum()} | "
              f"label=0: {(self.test_df['label']==0).sum()}")
        print(f"[Evaluator] Data format: {self.data_format}")

        self.rule_model = RuleBasedPredictor(threshold=rule_threshold)

        self.bert_model = None
        # Try both common model path conventions
        candidates = [
            bert_model_path,
            bert_model_path.replace("bert_model", "bert_contradiction_model"),
            bert_model_path.replace("bert_contradiction_model", "bert_model"),
        ]
        for path in candidates:
            if os.path.exists(path):
                from bert_finetuner import BERTFineTuner
                self.bert_model = BERTFineTuner.load(path)
                print(f"[Evaluator] BERT loaded from: {path}")
                break

        if not self.bert_model:
            print(f"[Evaluator] ⚠ BERT model not found — tried:")
            for p in candidates:
                print(f"    {p}")
            print("  Rule-based only evaluation will run.")

    def _detect_format(self):
        """Detect whether data is dissent-based or prosecution/defense."""
        parties = set()
        if "party_a" in self.test_df.columns:
            parties.update(self.test_df["party_a"].dropna().unique())
        if "party_b" in self.test_df.columns:
            parties.update(self.test_df["party_b"].dropna().unique())

        if "majority" in parties or "dissent" in parties:
            self.data_format = "dissent"
        elif "prosecution" in parties or "defense" in parties:
            self.data_format = "prosecution_defense"
        else:
            self.data_format = "unknown"

        # Pick the best grouping column for per-group analysis
        for col in ["crime_type", "topic", "label_source"]:
            if col in self.test_df.columns:
                self.group_col = col
                break
        else:
            self.group_col = None

    def evaluate(self) -> dict:
        y_true  = self.test_df["label"].tolist()
        texts_a = self.test_df["text_a"].tolist()
        texts_b = self.test_df["text_b"].tolist()
        party_a = self.test_df.get("party_a", pd.Series([None]*len(y_true))).tolist()
        party_b = self.test_df.get("party_b", pd.Series([None]*len(y_true))).tolist()

        results = {}

        # ── Rule-based ────────────────────────────────────────────────────
        print("\n[Evaluator] Rule-Based Detector...")
        rb_preds, rb_scores = self.rule_model.predict(
            texts_a, texts_b, party_a, party_b
        )
        results["rule_based"] = compute_all_metrics(
            y_true, rb_preds, rb_scores, "Rule-Based Detector"
        )
        if self.group_col:
            results["rule_based"]["per_group"] = compute_per_group(
                self.test_df, rb_preds, self.group_col
            )

        # ── Fine-tuned BERT ───────────────────────────────────────────────
        if self.bert_model:
            print("[Evaluator] Fine-Tuned BERT...")
            bert_preds, bert_scores = self.bert_model.predict(texts_a, texts_b)
            results["bert_finetuned"] = compute_all_metrics(
                y_true, bert_preds, bert_scores, "Fine-Tuned BERT"
            )
            if self.group_col:
                results["bert_finetuned"]["per_group"] = compute_per_group(
                    self.test_df, bert_preds, self.group_col
                )
        else:
            print("[Evaluator] Skipping BERT — model not loaded")

        # ── Dataset info ──────────────────────────────────────────────────
        results["_meta"] = {
            "data_format":    self.data_format,
            "group_col":      self.group_col,
            "n_test":         len(y_true),
            "n_positive":     int(sum(y_true)),
            "n_negative":     int(len(y_true) - sum(y_true)),
            "label_sources":  (
                self.test_df["label_source"].value_counts().to_dict()
                if "label_source" in self.test_df.columns else {}
            ),
        }

        return results

    def print_comparison(self, results: dict) -> None:
        w = 64
        models = [k for k in results.keys() if not k.startswith("_")]
        metrics = [
            ("Accuracy",  "accuracy"),
            ("Precision", "precision"),
            ("Recall",    "recall"),
            ("F1 Score",  "f1"),
            ("ROC AUC",   "roc_auc"),
        ]

        meta = results.get("_meta", {})
        print(f"\n{'═'*w}")
        print(f" MODEL EVALUATION — Test Set")
        print(f" n={meta.get('n_test','?')} | "
              f"pos={meta.get('n_positive','?')} | "
              f"neg={meta.get('n_negative','?')} | "
              f"format={meta.get('data_format','?')}")
        if meta.get("label_sources"):
            src = " | ".join(f"{k}={v}" for k, v in meta["label_sources"].items())
            print(f" Label sources: {src}")
        print(f"{'═'*w}")

        # Main metrics table
        header = f"{'Metric':<22}"
        for m in models:
            header += f" {results[m]['model']:>20}"
        print(f"\n{header}")
        print("─" * w)

        for label, key in metrics:
            row = f"{label:<22}"
            for m in models:
                row += f" {results[m].get(key, 0.0):>20.4f}"
            print(row)

        # Confusion matrices
        print(f"\n{'─'*w}")
        for m in models:
            cm = np.array(results[m]["confusion_matrix"])
            print(f"\n {results[m]['model']}:")
            print(f"               Pred=0    Pred=1")
            print(f"  Actual=0:   {cm[0,0]:>6}    {cm[0,1]:>6}   (TN / FP)")
            print(f"  Actual=1:   {cm[1,0]:>6}    {cm[1,1]:>6}   (FN / TP)")

        # Per-group F1
        group_col = meta.get("group_col")
        if group_col:
            print(f"\n{'─'*w}")
            print(f" F1 per {group_col}:")
            all_groups = set()
            for m in models:
                all_groups.update(results[m].get("per_group", {}).keys())

            gh = f"  {group_col[:24]:<24}"
            for m in models:
                gh += f" {results[m]['model'][:18]:>18}"
            print(gh)

            for g in sorted(all_groups):
                row = f"  {g[:24]:<24}"
                for m in models:
                    val = results[m].get("per_group", {}).get(g, float("nan"))
                    if np.isnan(val):
                        row += f" {'—':>18}"
                    else:
                        row += f" {val:>18.4f}"
                print(row)

        print(f"\n{'═'*w}")


# ─── Plots ────────────────────────────────────────────────────────────────

def _model_keys(results: dict) -> list[str]:
    return [k for k in results.keys() if not k.startswith("_")]


def plot_metric_comparison(
    results: dict, save_path: str = "metric_comparison.png"
) -> None:
    models      = _model_keys(results)
    model_names = [results[m]["model"] for m in models]
    metrics     = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels      = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    colors      = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    x     = np.arange(len(metrics))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (mk, color) in enumerate(zip(models, colors)):
        values = [results[mk].get(m, 0.0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=model_names[i], color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Rule-Based vs Fine-Tuned BERT — All Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ {save_path}")


def plot_confusion_matrices(
    results: dict, save_path: str = "confusion_matrices.png"
) -> None:
    models = _model_keys(results)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, mk in zip(axes, models):
        cm = np.array(results[mk]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Contradiction", "Contradiction"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(results[mk]["model"], fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=15)

    plt.suptitle("Confusion Matrices — Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ {save_path}")


def plot_roc_curves(
    results: dict, save_path: str = "roc_curves.png"
) -> None:
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for (mk, color) in zip(_model_keys(results), colors):
        fpr = results[mk].get("fpr", [0, 1])
        tpr = results[mk].get("tpr", [0, 1])
        auc = results[mk].get("roc_auc", 0.0)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{results[mk]['model']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Contradiction Detection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ {save_path}")


def plot_summary_table(
    results: dict, save_path: str = "model_comparison_table.png"
) -> None:
    models = _model_keys(results)
    metric_rows = [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1 Score",  "f1"),
        ("ROC AUC",   "roc_auc"),
    ]
    col_labels = ["Metric"] + [results[m]["model"] for m in models]
    table_data = []

    for label, key in metric_rows:
        row = [label]
        for m in models:
            row.append(f"{results[m].get(key, 0.0):.4f}")
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(4 + 3 * len(models), len(metric_rows) * 0.7 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight F1 row
    f1_row = next(i for i, (l, _) in enumerate(metric_rows) if l == "F1 Score") + 1
    for j in range(len(col_labels)):
        table[f1_row, j].set_facecolor("#ffeeba")

    # Highlight best per row in green
    for i, (_, key) in enumerate(metric_rows):
        vals = [results[m].get(key, 0.0) for m in models]
        best_col = int(np.argmax(vals)) + 1
        table[i + 1, best_col].set_facecolor("#d5f5e3")

    plt.title("Model Comparison — Test Set Results",
              fontsize=13, fontweight="bold", pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ {save_path}")


def save_results(results: dict, path: str = "full_model_eval.json") -> None:
    def _convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    # Remove internal temp keys before saving
    clean = {k: v for k, v in results.items() if not k.startswith("_cm_")}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=_convert)
    print(f"✓ {path}")


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    ev = ModelEvaluator(
        test_json="test_split.json",
        bert_model_path="bert_model/best_model",
        rule_threshold=0.3,
    )
    results = ev.evaluate()
    ev.print_comparison(results)
    save_results(results)
    plot_metric_comparison(results,  "metric_comparison.png")
    plot_confusion_matrices(results, "confusion_matrices.png")
    plot_roc_curves(results,         "roc_curves.png")
    plot_summary_table(results,      "model_comparison_table.png")


# ─── Human gold set evaluation ────────────────────────────────────────────

def evaluate_on_human_gold(
    human_json: str = "human_pairs.json",
    bert_model_path: str = "bert_model/best_model",
    rule_threshold: float = 0.3,
) -> dict:
    """
    Evaluate both models on the human-annotated gold set.

    This is the most reliable evaluation because:
    - Labels come from human expert (not structural heuristics)
    - Perfect 50/50 balance (no class imbalance artifacts)
    - Diverse cases: FTX, Enron, Theranos, Silk Road, Madoff, etc.

    Returns results dict compatible with print_comparison() and plot functions.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score, roc_curve,
        classification_report,
    )

    with open(human_json) as f:
        data = json.load(f)

    df      = pd.DataFrame(data)
    y_true  = df["label"].tolist()
    texts_a = df["text_a"].tolist()
    texts_b = df["text_b"].tolist()
    party_a = df.get("party_a", pd.Series([""]*len(y_true))).tolist()
    party_b = df.get("party_b", pd.Series([""]*len(y_true))).tolist()

    print(f"\n[Human Gold] {len(df)} pairs | "
          f"label=1: {sum(y_true)} | label=0: {len(y_true)-sum(y_true)}")
    print(f"  Cases: {df['case_id'].nunique()} | "
          f"Label source: {df['label_source'].unique().tolist()}")

    results = {}

    # Rule-based
    rb = RuleBasedPredictor(threshold=rule_threshold)
    rb_preds, rb_scores = rb.predict(texts_a, texts_b, party_a, party_b)
    results["rule_based"] = compute_all_metrics(
        y_true, rb_preds, rb_scores, "Rule-Based Detector"
    )
    results["rule_based"]["per_group"] = compute_per_group(
        df, rb_preds, "topic"
    )

    # Fine-tuned BERT
    if os.path.exists(bert_model_path):
        from bert_finetuner import BERTFineTuner
        bert = BERTFineTuner.load(bert_model_path)
        bert_preds, bert_scores = bert.predict(texts_a, texts_b)
        results["bert_finetuned"] = compute_all_metrics(
            y_true, bert_preds, bert_scores, "Fine-Tuned BERT"
        )
        results["bert_finetuned"]["per_group"] = compute_per_group(
            df, bert_preds, "topic"
        )
    else:
        print(f"  BERT model not found at {bert_model_path}")

    results["_meta"] = {
        "data_format":   "human_gold",
        "group_col":     "topic",
        "n_test":        len(y_true),
        "n_positive":    int(sum(y_true)),
        "n_negative":    int(len(y_true) - sum(y_true)),
        "label_sources": df["label_source"].value_counts().to_dict(),
        "cases":         df["case_id"].unique().tolist(),
    }

    return results


def print_human_vs_structural(
    human_results: dict,
    structural_results: dict,
) -> None:
    """
    Side-by-side comparison: human gold vs structural gold.
    Shows how much label noise affects reported metrics.
    """
    w = 72
    print(f"\n{'═'*w}")
    print(f" EVALUATION COMPARISON")
    print(f" Structural gold (n={structural_results.get('_meta',{}).get('n_test','?')},"
          f" majority/dissent labels)"
          f" vs Human gold (n={human_results.get('_meta',{}).get('n_test','?')},"
          f" expert labels)")
    print(f"{'═'*w}")

    metrics = [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1 Score",  "f1"),
        ("ROC AUC",   "roc_auc"),
    ]
    models = [k for k in human_results if not k.startswith("_")]

    header = f"{'Metric':<22} {'Model':<24} {'Structural':>12} {'Human Gold':>12} {'Δ':>8}"
    print(header)
    print("─" * w)

    for _, key in metrics:
        first = True
        for m in models:
            s_val = structural_results.get(m, {}).get(key, float("nan"))
            h_val = human_results.get(m, {}).get(key, float("nan"))
            delta = h_val - s_val if not (
                isinstance(s_val, float) and isinstance(h_val, float)
                and (s_val != s_val or h_val != h_val)
            ) else float("nan")
            label = _ if first else ""
            model_name = human_results[m].get("model", m)[:22]
            delta_str = f"{delta:+.4f}" if delta == delta else "  N/A"
            print(f"{label:<22} {model_name:<24} {s_val:>12.4f} {h_val:>12.4f} {delta_str:>8}")
            first = False
        print()

    print(f"{'─'*w}")
    print(f" Per-topic F1 (Fine-Tuned BERT on human gold):")
    per_topic = human_results.get("bert_finetuned", {}).get("per_group", {})
    for topic, f1 in sorted(per_topic.items(), key=lambda x: -x[1]):
        bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
        print(f"  {topic:<20} {bar} {f1:.4f}")
    print(f"{'═'*w}")
