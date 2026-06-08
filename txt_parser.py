"""
Module: TXT Parser
══════════════════
Reads CourtListener TXT files from Google Drive /nlp/data,
extracts clean text from each case.

Drop-in replacement for html_parser.py —
same RawCase output, same interface.

Usage in Colab:
  from txt_parser import load_cases_from_drive, print_dataset_stats
  cases = load_cases_from_drive("/content/drive/MyDrive/nlp/data")
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawCase:
    filename: str
    title: str
    court: str
    date: str
    text: str
    html_length: int          # kept for compatibility (stores original file size)
    parties: dict[str, str] = field(default_factory=dict)


# ─── Text cleaner ──────────────────────────────────────────────────────────

def clean_txt(text: str) -> str:
    """
    Normalize plain text from CourtListener:
    - Remove excessive whitespace
    - Remove page headers/footers (common in court docs)
    - Normalize line endings
    """
    # Remove common court document artifacts
    # Page numbers like "- 12 -" or "Page 12"
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'(?i)^page\s+\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove lines that are just underscores or dashes (separators)
    text = re.sub(r'^[_\-=]{5,}\s*$', '', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_title(text: str, filename: str) -> str:
    """
    Extract case title from the first lines of text.
    CourtListener TXT files usually start with the case name.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Try to find "X v. Y" pattern in first 10 lines
    for line in lines[:10]:
        if re.search(r'\bv\.?\s+[A-Z]', line):
            return line[:200]

    # Try first non-empty line
    if lines:
        return lines[0][:200]

    return Path(filename).stem.replace('_', ' ').replace('-', ' ')


def extract_meta(text: str) -> tuple[str, str]:
    """Extract court name and date from text."""
    court = "Unknown Court"
    date  = ""

    # Court patterns
    for pattern in [
        r'(United States (?:District|Court of Appeals|Bankruptcy) Court[^\n,]*)',
        r'(U\.S\. (?:District|Court of Appeals)[^\n,]*)',
        r'(Supreme Court of[^\n,]*)',
        r'([\w\s]+ Court of [\w\s]+)',
    ]:
        m = re.search(pattern, text[:3000], re.IGNORECASE)
        if m:
            court = m.group(1).strip()[:120]
            break

    # Date patterns
    m = re.search(
        r'\b((?:January|February|March|April|May|June|July|'
        r'August|September|October|November|December)'
        r'\s+\d{1,2},?\s+\d{4})\b',
        text[:5000]
    )
    if m:
        date = m.group(1)
    else:
        m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text[:3000])
        if m:
            date = m.group(1)
        else:
            m = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', text[:3000])
            if m:
                date = m.group(1)

    return court, date


def infer_parties(text: str) -> dict[str, str]:
    """Infer prosecution and defense party names."""
    parties = {"prosecution": "Government / Prosecution", "defense": "Defendant"}

    m = re.search(
        r'(United States|U\.S\.|People|State|Government)\s+v\.?\s+'
        r'([A-Z][a-zA-Z\s\-\.]+?)(?:\n|,|\.|$)',
        text[:2000], re.IGNORECASE
    )
    if m:
        parties["prosecution"] = m.group(1).strip()
        parties["defense"]     = m.group(2).strip()[:80]

    return parties


# ─── Main loader ──────────────────────────────────────────────────────────

def load_cases_from_drive(
    drive_path: str,
    min_text_length: int = 500,
    verbose: bool = True,
) -> list[RawCase]:
    """
    Load all TXT files from the given Google Drive folder.

    Args:
        drive_path      : e.g. "/content/drive/MyDrive/nlp/data"
        min_text_length : skip files shorter than this
        verbose         : print progress

    Returns:
        list[RawCase] with clean text
    """
    drive_path = Path(drive_path)

    if not drive_path.exists():
        raise FileNotFoundError(
            f"Path not found: {drive_path}\n"
            "Make sure Google Drive is mounted:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')"
        )

    # Find TXT files (recursive)
    txt_files = sorted(
        list(drive_path.rglob("*.txt")) +
        list(drive_path.rglob("*.TXT"))
    )

    if not txt_files:
        raise ValueError(
            f"No TXT files found in {drive_path}\n"
            f"Contents: {[p.name for p in drive_path.iterdir()][:15]}"
        )

    if verbose:
        print(f"[Parser] Found {len(txt_files)} TXT files in {drive_path}")

    cases   = []
    skipped = 0

    for i, filepath in enumerate(txt_files):
        try:
            # Try multiple encodings
            raw_text = None
            for enc in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
                try:
                    raw_text = filepath.read_text(encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if raw_text is None:
                if verbose:
                    print(f"  [{i+1}] ⚠ Encoding error: {filepath.name}")
                skipped += 1
                continue

            text = clean_txt(raw_text)

            if len(text) < min_text_length:
                if verbose:
                    print(f"  [{i+1}] ⚠ Too short ({len(text)} chars): {filepath.name}")
                skipped += 1
                continue

            title        = extract_title(text, filepath.name)
            court, date  = extract_meta(text)
            parties      = infer_parties(text)

            case = RawCase(
                filename=filepath.name,
                title=title,
                court=court,
                date=date,
                text=text,
                html_length=len(raw_text),   # original file size
                parties=parties,
            )
            cases.append(case)

            if verbose:
                print(f"  [{i+1}/{len(txt_files)}] ✓ "
                      f"{filepath.name[:35]:<35} | "
                      f"{len(text):>7,} chars | "
                      f"{title[:30]}")

        except Exception as e:
            if verbose:
                print(f"  [{i+1}] ✗ {filepath.name}: {e}")
            skipped += 1

    print(f"\n[Parser] Loaded {len(cases)} cases "
          f"({skipped} skipped, {len(txt_files)} total)")
    return cases


def print_dataset_stats(cases: list[RawCase]) -> None:
    if not cases:
        print("No cases loaded.")
        return

    lengths = [len(c.text) for c in cases]
    print(f"\n{'═'*52}")
    print(f" Loaded cases   : {len(cases)}")
    print(f" Avg text length: {sum(lengths)//len(lengths):,} chars")
    print(f" Min            : {min(lengths):,} chars")
    print(f" Max            : {max(lengths):,} chars")
    print(f"{'─'*52}")
    for c in cases[:8]:
        print(f" {c.filename[:30]:<30} | {len(c.text):>7,} chars")
    if len(cases) > 8:
        print(f" ... and {len(cases)-8} more")
    print(f"{'═'*52}")


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    cases = load_cases_from_drive(path, verbose=True)
    print_dataset_stats(cases)
