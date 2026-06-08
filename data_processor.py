"""
data_processor.py
══════════════════
Combines real (dissent) + synthetic cases → labeled pairs.
Implements CASE-LEVEL split and hybrid validation.
"""

import os
import json
import random
import re
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_client import LLMClient, get_client


# ─── Data structures ──────────────────────────────────────────────────────

@dataclass
class LegalCase:
    case_id:            str
    topic:              str
    source:             str
    prosecution_claims: list[str] = field(default_factory=list)
    defense_claims:     list[str] = field(default_factory=list)
    court_ruling:       str       = ""
    evidence:           list[str] = field(default_factory=list)
    majority_text:      str       = ""
    dissent_text:       str       = ""


@dataclass
class ClaimPair:
    pair_id:      str
    case_id:      str
    text_a:       str
    text_b:       str
    party_a:      str
    party_b:      str
    topic:        str
    label:        int
    confidence:   float
    label_source: str
    split:        str = ""

    def to_dict(self) -> dict:
        return {
            "pair_id":       self.pair_id,
            "case_id":       self.case_id,
            "text_a":        self.text_a,
            "text_b":        self.text_b,
            "party_a":       self.party_a,
            "party_b":       self.party_b,
            "topic":         self.topic,
            "label":         self.label,
            "confidence":    self.confidence,
            "label_source":  self.label_source,
            "split":         self.split,
            "combined_text": f"{self.text_a} [SEP] {self.text_b}",
        }


# ─── Helpers ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = 400) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        cur.append(s); cur_len += len(s)
        if cur_len >= size:
            c = " ".join(cur).strip()
            if len(c) >= 80:
                chunks.append(c[:500])
            cur = cur[-1:]; cur_len = len(cur[0]) if cur else 0
    if cur:
        c = " ".join(cur).strip()
        if len(c) >= 80:
            chunks.append(c[:500])
    return chunks


def _get_majority_text(d: dict) -> str:
    """Handles both old and new field name conventions."""
    return (d.get("majority_text") or
            d.get("majority_opinion") or
            d.get("majority") or "")


def _get_dissent_text(d: dict) -> str:
    """Handles both old and new field name conventions."""
    return (d.get("dissent_text") or
            d.get("dissenting_opinion") or
            d.get("dissent") or "")


# ─── Pair generator ───────────────────────────────────────────────────────

def generate_pairs_from_case(case: LegalCase, max_pairs: int = 30) -> list[ClaimPair]:
    pairs   = []
    counter = 0

    # From structured synthetic claims
    if case.prosecution_claims and case.defense_claims:
        for pc in case.prosecution_claims[:3]:
            for dc in case.defense_claims[:3]:
                pairs.append(ClaimPair(
                    pair_id=f"{case.case_id}_p{counter}",
                    case_id=case.case_id,
                    text_a=pc[:400], text_b=dc[:400],
                    party_a="prosecution", party_b="defense",
                    topic=case.topic, label=1, confidence=0.9,
                    label_source="synthetic" if "synthetic" in case.source else "structural",
                ))
                counter += 1

        for i in range(len(case.prosecution_claims) - 1):
            pairs.append(ClaimPair(
                pair_id=f"{case.case_id}_p{counter}",
                case_id=case.case_id,
                text_a=case.prosecution_claims[i][:400],
                text_b=case.prosecution_claims[i+1][:400],
                party_a="prosecution", party_b="prosecution",
                topic=case.topic, label=0, confidence=0.85,
                label_source="structural",
            ))
            counter += 1

        for i in range(len(case.defense_claims) - 1):
            pairs.append(ClaimPair(
                pair_id=f"{case.case_id}_p{counter}",
                case_id=case.case_id,
                text_a=case.defense_claims[i][:400],
                text_b=case.defense_claims[i+1][:400],
                party_a="defense", party_b="defense",
                topic=case.topic, label=0, confidence=0.85,
                label_source="structural",
            ))
            counter += 1

        if case.court_ruling:
            for dc in case.defense_claims[:2]:
                pairs.append(ClaimPair(
                    pair_id=f"{case.case_id}_p{counter}",
                    case_id=case.case_id,
                    text_a=case.court_ruling[:400], text_b=dc[:400],
                    party_a="court", party_b="defense",
                    topic=case.topic, label=1, confidence=0.85,
                    label_source="structural",
                ))
                counter += 1

    # From raw majority/dissent text
    if case.majority_text and case.dissent_text:
        maj_chunks = _chunk_text(case.majority_text)[:4]
        dis_chunks = _chunk_text(case.dissent_text)[:3]

        for mc in maj_chunks:
            for dc in dis_chunks[:2]:
                pairs.append(ClaimPair(
                    pair_id=f"{case.case_id}_p{counter}",
                    case_id=case.case_id,
                    text_a=mc, text_b=dc,
                    party_a="majority", party_b="dissent",
                    topic=case.topic, label=1, confidence=1.0,
                    label_source="structural",
                ))
                counter += 1

        for i in range(min(len(maj_chunks) - 1, 2)):
            pairs.append(ClaimPair(
                pair_id=f"{case.case_id}_p{counter}",
                case_id=case.case_id,
                text_a=maj_chunks[i], text_b=maj_chunks[i+1],
                party_a="majority", party_b="majority",
                topic=case.topic, label=0, confidence=0.95,
                label_source="structural",
            ))
            counter += 1

    return pairs[:max_pairs]


# ─── Case-level split ─────────────────────────────────────────────────────

def case_level_split(
    all_cases:   list[LegalCase],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    seed:        int   = 42,
) -> tuple[list[LegalCase], list[LegalCase], list[LegalCase]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    real      = [c for c in all_cases if c.source == "real_dissent"]
    synthetic = [c for c in all_cases if c.source != "real_dissent"]

    random.seed(seed)
    random.shuffle(real)

    n_val  = max(1, int(len(real) * val_ratio))
    n_test = max(1, int(len(real) * test_ratio))

    val_cases   = real[:n_val]
    test_cases  = real[n_val:n_val + n_test]
    train_cases = real[n_val + n_test:] + synthetic

    print(f"[Split] Case-level split:")
    print(f"  Train : {len(train_cases)} cases "
          f"({len(train_cases)-len(synthetic)} real + {len(synthetic)} synthetic)")
    print(f"  Val   : {len(val_cases)} cases (real only)")
    print(f"  Test  : {len(test_cases)} cases (real only — GOLD SET)")

    return train_cases, val_cases, test_cases


# ─── Hybrid validation ────────────────────────────────────────────────────

def hybrid_validation(
    pairs:           list[ClaimPair],
    bert_model_path: str | None   = None,
    google_api_key:  str | None   = None,   # kept for compat
    bert_threshold:  float        = 0.8,
    llm_sample_rate: float        = 0.25,
    split:           str          = "train",
) -> list[ClaimPair]:
    if split != "train":
        print(f"[Validation] {split}: skipping (gold set preserved)")
        return pairs

    print(f"\n[Validation] Hybrid validation on {len(pairs)} train pairs...")

    # Step 1: BERT confidence filter
    if bert_model_path and os.path.exists(bert_model_path):
        print(f"[Validation] Step 1: BERT filter (threshold={bert_threshold})")
        try:
            from bert_finetuner import BERTFineTuner
            bert   = BERTFineTuner.load(bert_model_path)
            ta     = [p.text_a for p in pairs]
            tb     = [p.text_b for p in pairs]
            _, probs = bert.predict(ta, tb)

            kept = []
            for pair, prob in zip(pairs, probs):
                bert_agrees = (
                    (prob >= bert_threshold and pair.label == 1) or
                    ((1 - prob) >= bert_threshold and pair.label == 0)
                )
                if bert_agrees:
                    pair.confidence  = round(max(prob, 1 - prob), 4)
                    pair.label_source = "bert_validated"
                    kept.append(pair)

            print(f"  BERT filter: {len(pairs)} → {len(kept)} "
                  f"({len(kept)/max(len(pairs),1):.0%} kept)")
            pairs = kept
        except Exception as e:
            print(f"  BERT filter skipped: {e}")
    else:
        print("[Validation] Step 1: BERT model not found — skipping")

    # Step 2: LLM spot-check
    try:
        client = get_client()
    except Exception:
        client = None

    if client and client.is_configured:
        sample_size = max(1, int(len(pairs) * llm_sample_rate))
        sample      = random.sample(pairs, min(sample_size, len(pairs)))
        print(f"[Validation] Step 2: LLM spot-check on {len(sample)} pairs "
              f"({llm_sample_rate:.0%} sample) via {client.provider}")

        agree = disagree = 0
        for pair in sample[:20]:
            verdict = _llm_check_pair(pair, client)
            if verdict == pair.label:
                agree += 1
            else:
                disagree += 1
            client.sleep()

        if agree + disagree > 0:
            print(f"  LLM agreement with structural labels: "
                  f"{agree/(agree+disagree):.0%}")
    else:
        print("[Validation] Step 2: No LLM configured — skipping spot-check")

    return pairs


def _llm_check_pair(pair: ClaimPair, client: LLMClient) -> int:
    result = client.call_json(
        prompt=(
            f'[{pair.party_a.upper()}]: "{pair.text_a[:250]}"\n\n'
            f'[{pair.party_b.upper()}]: "{pair.text_b[:250]}"\n\n'
            'Do these genuinely contradict? Return JSON: {"label": 0 or 1}'
        ),
        system='Legal contradiction checker. Return only JSON: {"label": 0 or 1}',
        temperature=0.0,
    )
    return int(bool(result.get("label", 0)))


# ─── Main processor ───────────────────────────────────────────────────────

class DataProcessor:

    def __init__(
        self,
        google_api_key:      str | None = None,
        bert_model_path:     str | None = None,
        max_pairs_per_case:  int        = 30,
        seed:                int        = 42,
    ):
        self.api_key   = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.bert_path = bert_model_path
        self.max_pairs = max_pairs_per_case
        self.seed      = seed

    def process(
        self,
        dissent_cases_json:   str = "dissent_cases.json",
        synthetic_cases_json: str = "synthetic_cases.json",
        human_pairs_json:     str = "",   # path to human_pairs.json; added to TRAIN only
        output_dir:           str = ".",
        drive_path:           str = "",
    ) -> dict[str, pd.DataFrame]:
        import shutil

        real_cases      = self._load_real_cases(dissent_cases_json)
        synthetic_cases = self._load_synthetic_cases(synthetic_cases_json)
        all_cases       = real_cases + synthetic_cases

        print(f"\n[Processor] Total: {len(all_cases)} cases "
              f"({len(real_cases)} real + {len(synthetic_cases)} synthetic)")

        # Load human pairs for augmenting train set
        human_train_pairs = self._load_human_pairs(human_pairs_json)

        train_c, val_c, test_c = case_level_split(all_cases, seed=self.seed)

        splits = {}
        for split_name, cases in [("train", train_c),
                                   ("val",   val_c),
                                   ("test",  test_c)]:
            pairs = []
            for case in cases:
                for p in generate_pairs_from_case(case, self.max_pairs):
                    p.split = split_name
                    pairs.append(p)

            # Add human pairs to TRAIN only — keep val/test clean
            if split_name == "train" and human_train_pairs:
                pairs.extend(human_train_pairs)
                print(f"  + {len(human_train_pairs)} human expert pairs added to train")

            n_pos = sum(p.label for p in pairs)
            n_neg = sum(1 - p.label for p in pairs)
            print(f"\n[Processor] {split_name}: {len(pairs)} pairs "
                  f"(pos={n_pos}, neg={n_neg})")

            if split_name == "train":
                pairs = hybrid_validation(
                    pairs,
                    bert_model_path=self.bert_path,
                    bert_threshold=0.8,
                    split="train",
                )

            df   = pd.DataFrame([p.to_dict() for p in pairs])
            path = os.path.join(output_dir, f"{split_name}_split.json")
            df.to_json(path, orient="records", indent=2)
            if drive_path:
                shutil.copy(path, f"{drive_path}/{split_name}_split.json")
            print(f"  Saved: {path}")
            splits[split_name] = df

        self._print_summary(splits)
        return splits

    def _load_human_pairs(self, json_path: str) -> list[ClaimPair]:
        """
        Load human-annotated pairs and convert to ClaimPair objects for train.
        Human pairs: already labeled, high quality, no case-level split needed —
        they go directly into train (same cases as evaluation, but BERT hasn't
        seen them during its current training run).
        """
        if not json_path or not os.path.exists(json_path):
            return []
        with open(json_path) as f:
            raw = json.load(f)
        pairs = []
        for i, d in enumerate(raw):
            pairs.append(ClaimPair(
                pair_id=d.get("pair_id", f"HUMAN_{i:03d}"),
                case_id=d.get("case_id", "human"),
                text_a=d.get("text_a", ""),
                text_b=d.get("text_b", ""),
                party_a=d.get("party_a", "prosecution"),
                party_b=d.get("party_b", "defense"),
                topic=d.get("topic", "criminal"),
                label=int(d.get("label", 0)),
                confidence=float(d.get("confidence", 1.0)),
                label_source="human_expert",
                split="train",
            ))
        n1 = sum(p.label for p in pairs)
        n0 = sum(1 - p.label for p in pairs)
        print(f"[Processor] Loaded {len(pairs)} human expert pairs "
              f"(pos={n1}, neg={n0}) → train only")
        return pairs

    def _load_real_cases(self, json_path: str) -> list[LegalCase]:
        if not os.path.exists(json_path):
            print(f"[Processor] No file: {json_path}")
            return []
        with open(json_path) as f:
            raw = json.load(f)

        cases = []
        for d in raw:
            # Support both old and new field names
            majority = _get_majority_text(d)
            dissent  = _get_dissent_text(d)

            if not majority or not dissent:
                continue   # skip incomplete records

            cases.append(LegalCase(
                case_id=d.get("case_id", d.get("filename", "UNKNOWN")),
                topic=d.get("crime_type", d.get("topic", "criminal")),
                source="real_dissent",
                majority_text=majority,
                dissent_text=dissent,
            ))

        print(f"[Processor] Loaded {len(cases)} real dissent cases "
              f"({len(raw)-len(cases)} skipped — missing text)")
        return cases

    def _load_synthetic_cases(self, json_path: str) -> list[LegalCase]:
        if not os.path.exists(json_path):
            print(f"[Processor] No file: {json_path}")
            return []
        with open(json_path) as f:
            raw = json.load(f)

        cases = []
        for d in raw:
            cases.append(LegalCase(
                case_id=d["case_id"],
                topic=d["topic"],
                source=d.get("source", "synthetic"),
                prosecution_claims=d.get("prosecution_claims", []),
                defense_claims=d.get("defense_claims", []),
                court_ruling=d.get("court_ruling", ""),
                evidence=d.get("evidence", []),
            ))

        print(f"[Processor] Loaded {len(cases)} synthetic cases")
        return cases

    def _print_summary(self, splits: dict[str, pd.DataFrame]) -> None:
        print(f"\n{'═'*55}")
        print(f" DATASET SUMMARY (case-level split)")
        print(f"{'─'*55}")
        for name, df in splits.items():
            n1   = df["label"].sum()
            n0   = (df["label"] == 0).sum()
            srcs = df["label_source"].value_counts().to_dict()
            print(f" {name:5s}: {len(df):4d} pairs | "
                  f"pos={n1:3d} neg={n0:3d} | {srcs}")
        print(f"{'═'*55}")
