"""
Module: Dissent Fetcher via HuggingFace (no rate limits)
══════════════════════════════════════════════════════════
Streams common-pile/caselaw_access_project and finds cases
that contain BOTH majority AND dissent opinions.

Why HuggingFace instead of CourtListener API:
  - No rate limits (streaming, not API calls)
  - Same data source (CourtListener bulk export)
  - Faster for bulk collection
  - No token needed

Dissent detection strategy:
  The dataset `id` field encodes the source file path.
  Cases with multiple opinions share the same base path prefix.
  We also detect dissents from text signals:
    "I dissent", "I respectfully dissent", "dissenting opinion",
    "Justice X, dissenting"

Output: list of DissentCase objects, each with:
  majority_text : text of the majority opinion
  dissent_text  : text of the dissenting opinion
  label=1 guaranteed — no Gemini needed
"""

import os
import re
import json
import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DissentCase:
    case_id: str
    case_name: str
    majority_text: str
    dissent_text: str
    majority_author: str
    dissent_author: str
    court: str
    date: str
    crime_type: str
    quality_score: int
    source_ids: list[str] = field(default_factory=list)


# ─── Dissent detection signals ────────────────────────────────────────────

DISSENT_OPENERS = [
    r"i\s+(?:respectfully\s+)?dissent\b",
    r"dissenting\s+opinion",
    r"(?:justice|judge|chief\s+justice)\s+\w+[,\s]+dissenting",
    r"i\s+would\s+(?:reverse|affirm|vacate|remand)\b",
    r"i\s+cannot\s+agree\s+with\s+the\s+(?:majority|court)",
    r"the\s+majority\s+(?:errs|is\s+wrong|misreads|overlooks)",
    r"i\s+write\s+separately\s+to\s+(?:dissent|disagree)",
    r"with\s+respect[,\s]+i\s+dissent",
]

MAJORITY_OPENERS = [
    r"(?:per\s+curiam|delivered\s+the\s+opinion\s+of\s+the\s+court)",
    r"opinion\s+of\s+the\s+court\b",
    r"(?:justice|judge|chief\s+justice)\s+\w+\s+delivered",
    r"this\s+(?:appeal|case|matter)\s+(?:comes?|arises?|involves?)\b",
    r"we\s+(?:affirm|reverse|hold|find|conclude)\b",
    r"the\s+(?:district|trial)\s+court\s+(?:held|found|concluded)\b",
    r"defendant\s+(?:was\s+convicted|appeals|contends)\b",
    r"(?:government|prosecution)\s+(?:argues|contends|maintains)\b",
]

CRIMINAL_KEYWORDS = [
    "fraud", "conspiracy", "wire fraud", "criminal", "felony",
    "conviction", "indictment", "beyond reasonable doubt",
    "drug trafficking", "bribery", "murder", "robbery",
    "obstruction", "mens rea", "sentence", "guilty",
]


def is_dissent(text: str) -> bool:
    t = text[:2000].lower()
    return any(re.search(p, t) for p in DISSENT_OPENERS)


def is_majority(text: str) -> bool:
    t = text[:2000].lower()
    return any(re.search(p, t) for p in MAJORITY_OPENERS)


def is_criminal(text: str) -> bool:
    t = text.lower()
    return sum(1 for k in CRIMINAL_KEYWORDS if k in t) >= 3


def infer_crime_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["wire fraud", "mail fraud", "securities fraud"]): return "fraud"
    if "drug" in t or "narcotics" in t:   return "drug trafficking"
    if "conspiracy" in t:                  return "conspiracy"
    if "bribery" in t or "corrupt" in t:  return "bribery"
    if "murder" in t or "homicide" in t:  return "homicide"
    if "robbery" in t or "theft" in t:    return "theft"
    if "tax" in t and "fraud" in t:       return "tax fraud"
    return "criminal"


def extract_author(text: str) -> str:
    """Extract judge name from opinion opening."""
    patterns = [
        r"(?:Justice|Judge|Chief\s+Justice)\s+([A-Z][A-Za-z\-]+)",
        r"([A-Z][A-Za-z\-]+),\s+(?:J\.|C\.J\.|Circuit\s+Judge)",
    ]
    for p in patterns:
        m = re.search(p, text[:500])
        if m:
            return m.group(1)
    return "Unknown"


def score_pair(majority: str, dissent: str) -> int:
    """Quality score for a majority+dissent pair."""
    score = 0
    mt, dt = majority.lower(), dissent.lower()

    # Both texts have enough content
    if len(majority) >= 1000: score += 20
    if len(dissent)  >= 500:  score += 20

    # Criminal content
    if is_criminal(majority): score += 25

    # Explicit adversarial language in majority
    adv = mt.count("defendant") + mt.count("government") + mt.count("argues")
    if adv >= 5: score += 20

    # Dissent explicitly disagrees
    if any(re.search(p, dt[:1000]) for p in DISSENT_OPENERS[:4]):
        score += 15

    return min(score, 100)


# ─── Case grouper ─────────────────────────────────────────────────────────

def _get_base_id(row_id: str) -> str:
    """
    Extract base case identifier from row id.
    Example: "f2d_474/html/0001-01.html" → "f2d_474/html/0001"
    Multiple opinions of the same case share the same base.
    """
    # Remove trailing "-NN.html" suffix
    base = re.sub(r'-\d+\.html?$', '', row_id)
    return base


# ─── Main fetcher ─────────────────────────────────────────────────────────

@dataclass
class HFFetchConfig:
    min_quality_score: int = 40
    max_cases: int = 50
    max_scan: int = 200000           # rows to scan in dataset
    checkpoint_every: int = 5
    checkpoint_path: str = "hf_dissent_checkpoint.json"
    output_path: str = "dissent_cases.json"
    drive_output_path: str = ""
    group_window: int = 10           # rows to buffer for grouping


def fetch_dissent_cases(
    config: HFFetchConfig = HFFetchConfig(),
    verbose: bool = True,
) -> list[DissentCase]:
    """
    Stream caselaw_access_project and find majority+dissent pairs.
    Groups rows by base case ID to match opinions from the same case.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: !pip install datasets")

    # Load checkpoint
    accepted: list[dict] = []
    scanned = 0

    if os.path.exists(config.checkpoint_path):
        with open(config.checkpoint_path) as f:
            ckpt = json.load(f)
        accepted = ckpt.get("cases", [])
        scanned  = ckpt.get("scanned", 0)
        print(f"[HF] Resuming: {len(accepted)} cases, {scanned:,} rows scanned")
        if len(accepted) >= config.max_cases:
            print(f"[HF] Already have {len(accepted)} cases ✓")
            return [_dict_to_case(c) for c in accepted]
    else:
        print(f"[HF] Starting fresh")

    print(f"[HF] Target: {config.max_cases} majority+dissent pairs")
    print(f"[HF] Max scan: {config.max_scan:,} rows\n")

    dataset = load_dataset(
        "common-pile/caselaw_access_project",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    if scanned > 0:
        print(f"[HF] Skipping {scanned:,} already-scanned rows...")
        dataset = dataset.skip(scanned)

    # Buffer: group rows by base case ID
    # { base_id: [(text, row_id), ...] }
    buffer: dict[str, list] = defaultdict(list)
    pre_filtered = 0
    grouped = 0

    for row in dataset:
        if len(accepted) >= config.max_cases:
            break
        if scanned >= config.max_scan:
            print(f"\n[HF] Max scan reached ({config.max_scan:,} rows)")
            break

        scanned += 1
        text   = row.get("text", "")
        row_id = row.get("id", "")

        if len(text) < 300:
            continue

        pre_filtered += 1
        base_id = _get_base_id(row_id)
        buffer[base_id].append((text, row_id))

        # Try to match majority+dissent within buffered group
        group = buffer[base_id]
        if len(group) >= 2:
            grouped += 1
            pair = _find_dissent_pair(group, base_id)
            if pair and pair.quality_score >= config.min_quality_score:
                if is_criminal(pair.majority_text):
                    accepted.append(_case_to_dict(pair))
                    if verbose:
                        print(f"  ✓ [{len(accepted):3d}/{config.max_cases}] "
                              f"{pair.case_name[:50]} "
                              f"(score={pair.quality_score})")

                    if len(accepted) % config.checkpoint_every == 0:
                        _save_checkpoint(config, accepted, scanned)
                        print(f"  💾 Checkpoint ({len(accepted)} cases)")

                del buffer[base_id]   # free memory

        # Also check single-document dissents (one file with both opinions)
        elif len(group) == 1:
            text_only = group[0][0]
            if _has_embedded_dissent(text_only):
                pair = _extract_embedded_dissent(text_only, base_id, row_id)
                if pair and pair.quality_score >= config.min_quality_score:
                    if is_criminal(pair.majority_text):
                        accepted.append(_case_to_dict(pair))
                        if verbose:
                            print(f"  ✓ [{len(accepted):3d}/{config.max_cases}] "
                                  f"{pair.case_name[:50]} "
                                  f"[embedded] (score={pair.quality_score})")
                        del buffer[base_id]

                        if len(accepted) % config.checkpoint_every == 0:
                            _save_checkpoint(config, accepted, scanned)
                            print(f"  💾 Checkpoint ({len(accepted)} cases)")

        # Progress
        if scanned % 5000 == 0 and verbose:
            print(f"  ... {scanned:,} rows | accepted: {len(accepted)} | "
                  f"buffered groups: {len(buffer)}")

        # Flush old buffer entries to save memory
        if len(buffer) > 500:
            old_keys = list(buffer.keys())[:100]
            for k in old_keys:
                del buffer[k]

    # Final save
    _save_checkpoint(config, accepted, scanned)

    # Save output
    with open(config.output_path, "w") as f:
        json.dump(accepted, f, indent=2)
    if config.drive_output_path:
        _copy_to_drive(config.output_path, config.drive_output_path)

    print(f"\n[HF] Done!")
    print(f"  Rows scanned  : {scanned:,}")
    print(f"  Pre-filtered  : {pre_filtered:,}")
    print(f"  Groups found  : {grouped}")
    print(f"  Cases accepted: {len(accepted)}")

    return [_dict_to_case(c) for c in accepted]


def _find_dissent_pair(
    group: list[tuple[str, str]],
    base_id: str,
) -> DissentCase | None:
    """Match majority and dissent from a group of texts."""
    majorities = [(t, rid) for t, rid in group if is_majority(t) and not is_dissent(t)]
    dissents   = [(t, rid) for t, rid in group if is_dissent(t)]

    if not majorities or not dissents:
        return None

    maj_text, maj_id = majorities[0]
    dis_text, dis_id = dissents[0]

    score = score_pair(maj_text, dis_text)
    if score < 30:
        return None

    name = _extract_case_name(maj_text, base_id)
    return DissentCase(
        case_id=base_id,
        case_name=name,
        majority_text=maj_text,
        dissent_text=dis_text,
        majority_author=extract_author(maj_text),
        dissent_author=extract_author(dis_text),
        court=_extract_court(maj_text),
        date=_extract_date(maj_text),
        crime_type=infer_crime_type(maj_text),
        quality_score=score,
        source_ids=[maj_id, dis_id],
    )


def _has_embedded_dissent(text: str) -> bool:
    """Check if a single document contains both majority and dissent."""
    t = text.lower()
    has_maj = any(re.search(p, t[:3000]) for p in MAJORITY_OPENERS[:4])
    has_dis = any(re.search(p, t) for p in DISSENT_OPENERS[:4])
    return has_maj and has_dis and len(text) > 2000


def _extract_embedded_dissent(
    text: str, base_id: str, row_id: str
) -> DissentCase | None:
    """Split a single document into majority and dissent sections."""
    # Find dissent separator
    for pattern in DISSENT_OPENERS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m and m.start() > 500:
            majority_text = text[:m.start()].strip()
            dissent_text  = text[m.start():].strip()

            if len(majority_text) < 200 or len(dissent_text) < 100:
                continue

            score = score_pair(majority_text, dissent_text)
            if score < 30:
                continue

            name = _extract_case_name(majority_text, base_id)
            return DissentCase(
                case_id=base_id,
                case_name=name,
                majority_text=majority_text,
                dissent_text=dissent_text,
                majority_author=extract_author(majority_text),
                dissent_author=extract_author(dissent_text),
                court=_extract_court(majority_text),
                date=_extract_date(majority_text),
                crime_type=infer_crime_type(majority_text),
                quality_score=score,
                source_ids=[row_id],
            )
    return None


def _extract_case_name(text: str, fallback: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:10]:
        if re.search(r'\bv\.?\s+[A-Z]', line) and len(line) < 200:
            return line
    return fallback


def _extract_court(text: str) -> str:
    for p in [
        r"United States Court of Appeals[^\n,]*",
        r"United States District Court[^\n,]*",
        r"Supreme Court[^\n,]*",
    ]:
        m = re.search(p, text[:2000], re.IGNORECASE)
        if m:
            return m.group()[:100]
    return "Unknown"


def _extract_date(text: str) -> str:
    m = re.search(
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'\.?\s+\d{1,2},?\s+\d{4})\b',
        text[:3000]
    )
    return m.group(1) if m else ""


def _save_checkpoint(config, cases, scanned):
    with open(config.checkpoint_path, "w") as f:
        json.dump({"cases": cases, "scanned": scanned}, f)
    if config.drive_output_path:
        _copy_to_drive(config.checkpoint_path, config.drive_output_path)


def _copy_to_drive(src, drive_dir):
    import shutil
    os.makedirs(drive_dir, exist_ok=True)
    shutil.copy(src, f"{drive_dir}/{os.path.basename(src)}")


def _case_to_dict(c: DissentCase) -> dict:
    return {k: v for k, v in vars(c).items()}


def _dict_to_case(d: dict) -> DissentCase:
    return DissentCase(**d)


# ─── Labeled dataset builder ──────────────────────────────────────────────

def build_labeled_dataset(
    cases: list[DissentCase],
    output_path: str = "labeled_pairs.json",
    drive_path: str = "",
    chunk_size: int = 400,
) -> list[dict]:
    """
    Build labeled pairs from dissent cases.
    label=1: majority_chunk vs dissent_chunk (structural guarantee)
    label=0: majority_chunk_A vs majority_chunk_B (same side)
    """
    import random, re
    random.seed(42)

    pairs = _load_existing(output_path)
    done  = {p["case_id"] for p in pairs}
    print(f"\n[Builder] {len(cases)} cases → labeled pairs")
    print(f"  Already done: {len(done)}\n")

    for i, case in enumerate(cases):
        if case.case_id in done:
            continue

        maj_chunks = _chunk(case.majority_text, chunk_size)
        dis_chunks = _chunk(case.dissent_text,  chunk_size)
        if not maj_chunks or not dis_chunks:
            continue

        # label=1
        for mc in maj_chunks[:3]:
            for dc in dis_chunks[:2]:
                pairs.append({
                    "text_a": mc, "text_b": dc,
                    "party_a": "majority", "party_b": "dissent",
                    "label": 1, "confidence": 1.0,
                    "topic": case.crime_type, "case": case.case_name[:80],
                    "case_id": case.case_id, "label_source": "structural",
                    "combined_text": f"{mc} [SEP] {dc}",
                })

        # label=0
        for j in range(min(len(maj_chunks)-1, 2)):
            pairs.append({
                "text_a": maj_chunks[j], "text_b": maj_chunks[j+1],
                "party_a": "majority", "party_b": "majority",
                "label": 0, "confidence": 0.95,
                "topic": case.crime_type, "case": case.case_name[:80],
                "case_id": case.case_id, "label_source": "structural",
                "combined_text": f"{maj_chunks[j]} [SEP] {maj_chunks[j+1]}",
            })

        done.add(case.case_id)

        if (i + 1) % 5 == 0:
            _save_pairs(pairs, output_path, drive_path)
            n1 = sum(1 for p in pairs if p["label"]==1)
            n0 = sum(1 for p in pairs if p["label"]==0)
            print(f"  [{i+1}/{len(cases)}] {len(pairs)} pairs "
                  f"(pos={n1}, neg={n0}) → saved")
        else:
            print(f"  [{i+1}/{len(cases)}] {case.case_name[:45]}")

    _save_pairs(pairs, output_path, drive_path)
    n1 = sum(1 for p in pairs if p["label"]==1)
    n0 = sum(1 for p in pairs if p["label"]==0)
    print(f"\n✓ {len(pairs)} пар | label=1: {n1} | label=0: {n0}")
    return pairs


def _chunk(text: str, size: int) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        cur.append(s); cur_len += len(s)
        if cur_len >= size:
            c = " ".join(cur).strip()
            if len(c) >= 80:
                chunks.append(c[:500])
            cur = cur[-1:]; cur_len = len(cur[0])
    if cur:
        c = " ".join(cur).strip()
        if len(c) >= 80:
            chunks.append(c[:500])
    return chunks


def _load_existing(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return []


def _save_pairs(pairs, path, drive_path):
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    if drive_path:
        _copy_to_drive(path, drive_path)


if __name__ == "__main__":
    config = HFFetchConfig(max_cases=30, max_scan=100000)
    cases  = fetch_dissent_cases(config, verbose=True)
    pairs  = build_labeled_dataset(cases)
    print(f"Ready: {len(pairs)} labeled pairs")
