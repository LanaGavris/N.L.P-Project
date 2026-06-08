"""
synthetic_augmenter.py
═══════════════════════
Generates exactly 10 structured synthetic legal cases via Gemini.
Each case has explicit prosecution/defense/court claims.

Output format per case:
{
  "case_id": "SYN_001",
  "topic": "wire fraud",
  "prosecution_claims": ["...", "..."],
  "defense_claims":     ["...", "..."],
  "court_ruling":       "...",
  "evidence":           ["...", "..."],
  "source": "synthetic"
}
"""

import os
import json
import time
from llm_client import LLMClient, get_client

SYNTHETIC_CASES_TEMPLATE = [
    {"topic": "wire fraud",         "defendant": "Investment firm CEO",     "charge": "misrepresenting returns to investors"},
    {"topic": "securities fraud",   "defendant": "CFO",                     "charge": "falsifying financial statements"},
    {"topic": "conspiracy",         "defendant": "Organized crime ring",    "charge": "coordinated money laundering"},
    {"topic": "tax evasion",        "defendant": "Real estate developer",   "charge": "concealing offshore accounts"},
    {"topic": "bribery",            "defendant": "Public official",         "charge": "accepting payments for contracts"},
    {"topic": "insider trading",    "defendant": "Fund manager",            "charge": "trading on material non-public info"},
    {"topic": "drug trafficking",   "defendant": "Cartel lieutenant",       "charge": "directing narcotics distribution"},
    {"topic": "obstruction",        "defendant": "Corporate executive",     "charge": "destroying evidence under subpoena"},
    {"topic": "embezzlement",       "defendant": "Non-profit treasurer",    "charge": "diverting charitable funds"},
    {"topic": "healthcare fraud",   "defendant": "Medical clinic owner",    "charge": "billing for unprovided services"},
]

CASE_GENERATION_PROMPT = """Generate a structured legal case for a US criminal court.

Case details:
  Topic: {topic}
  Defendant type: {defendant}
  Charge: {charge}

Generate realistic adversarial content. Return ONLY valid JSON:
{{
  "case_id": "{case_id}",
  "topic": "{topic}",
  "prosecution_claims": [
    "3-4 specific prosecution claims (2-3 sentences each)",
    "Each claim should assert guilt, knowledge, or intent",
    "Use realistic legal language"
  ],
  "defense_claims": [
    "3-4 specific defense claims (2-3 sentences each)",
    "Each claim should directly contradict a prosecution claim",
    "Use realistic legal language"
  ],
  "court_ruling": "2-3 sentence court finding that resolves the key disputes",
  "evidence": [
    "2-3 pieces of evidence mentioned in the case"
  ],
  "source": "synthetic"
}}

Important: prosecution and defense claims must explicitly contradict each other on intent, knowledge, or facts."""


def generate_synthetic_cases(
    llm: "LLMClient | None" = None,
    google_api_key: str | None = None,   # kept for backwards compat
    output_path: str = "synthetic_cases.json",
    drive_path: str = "",
) -> list[dict]:
    """
    Generate exactly 10 structured synthetic cases.
    Uses unified LLMClient (Gemini or OpenRouter).
    """
    client = llm or get_client()

    # Load existing if available
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        if len(existing) >= 10:
            print(f"[SyntheticAug] Loaded {len(existing)} existing cases")
            return existing
        print(f"[SyntheticAug] Found {len(existing)}, generating more...")
    else:
        existing = []

    done_ids = {c["case_id"] for c in existing}
    cases    = list(existing)

    print(f"[SyntheticAug] Generating {10 - len(cases)} synthetic cases via Gemini...")

    for tmpl in SYNTHETIC_CASES_TEMPLATE:
        case_id = f"SYN_{SYNTHETIC_CASES_TEMPLATE.index(tmpl)+1:03d}"
        if case_id in done_ids:
            continue
        if len(cases) >= 10:
            break

        if client.is_configured:
            prompt = CASE_GENERATION_PROMPT.format(
                case_id=case_id, **tmpl
            )
            result = client.call_json(
                prompt=prompt,
                system="You are a legal case generator. Return only valid JSON.",
                use_smart_model=True,
                temperature=0.7,
            )
            client.sleep()
            if "error" not in result and "prosecution_claims" in result:
                result["case_id"] = case_id
                result["source"]  = "synthetic"
                cases.append(result)
                print(f"  ✓ {case_id}: {tmpl['topic']}")
            else:
                print(f"  ⚠ {case_id} failed, using template")
                cases.append(_template_case(case_id, tmpl))

        else:
            cases.append(_template_case(case_id, tmpl))
            print(f"  ✓ {case_id}: {tmpl['topic']} (template, no LLM)")

    _save(cases, output_path, drive_path)
    print(f"\n[SyntheticAug] Done: {len(cases)} synthetic cases")
    return cases[:10]


def _template_case(case_id: str, tmpl: dict) -> dict:
    """Built-in fallback cases when no API key available."""
    topic = tmpl["topic"]
    defendant = tmpl["defendant"]

    templates = {
        "wire fraud": {
            "prosecution_claims": [
                f"The government contends the {defendant} knowingly transmitted false information via wire communications to defraud investors of millions of dollars.",
                "The prosecution argues defendant was fully aware that the financial projections shared with clients were fabricated and had no basis in reality.",
                "The government maintains defendant deliberately concealed material risks and used investor funds for personal enrichment.",
            ],
            "defense_claims": [
                f"The defense argues the {defendant} acted in good faith and genuinely believed all representations made to investors were accurate.",
                "Defense counsel contends any inaccuracies in financial projections were honest mistakes based on reasonable market assumptions, not intentional fraud.",
                "The defense maintains defendant had no personal financial motive to commit fraud given existing legitimate wealth.",
            ],
            "court_ruling": "The Court finds the evidence of knowing misrepresentation overwhelming. Defendant's sophistication and direct involvement in preparing false statements establishes fraudulent intent beyond reasonable doubt.",
            "evidence": ["Wire transfer records", "Email communications", "Investor testimony"],
        },
        "securities fraud": {
            "prosecution_claims": [
                "The prosecution argues defendant knowingly filed materially false financial statements with the SEC to inflate stock price.",
                "The government contends defendant sold personal shares at artificially inflated prices while withholding negative information from investors.",
                "The prosecution maintains defendant directed subordinates to manipulate accounting records to conceal losses.",
            ],
            "defense_claims": [
                "The defense contends all financial statements were prepared by independent auditors and defendant relied on their professional judgment.",
                "Defense argues defendant had no knowledge that accounting practices were improper under GAAP at the time.",
                "The defense maintains defendant did not sell shares during the relevant period and had no motive to inflate stock price.",
            ],
            "court_ruling": "The jury finds defendant guilty. Documentary evidence and cooperating witness testimony establish knowing participation in the fraudulent scheme.",
            "evidence": ["SEC filings", "Audit records", "Stock transaction history"],
        },
    }

    tdata = templates.get(topic, templates["wire fraud"])
    return {
        "case_id":             case_id,
        "topic":               topic,
        "prosecution_claims":  tdata["prosecution_claims"],
        "defense_claims":      tdata["defense_claims"],
        "court_ruling":        tdata["court_ruling"],
        "evidence":            tdata["evidence"],
        "source":              "synthetic_template",
    }


def _save(cases, path, drive_path):
    with open(path, "w") as f:
        json.dump(cases, f, indent=2)
    if drive_path:
        import shutil, os
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(path, f"{drive_path}/{os.path.basename(path)}")


if __name__ == "__main__":
    cases = generate_synthetic_cases()
    print(f"\nSample case:")
    c = cases[0]
    print(f"  ID: {c['case_id']} | Topic: {c['topic']}")
    print(f"  Prosecution[0]: {c['prosecution_claims'][0][:80]}...")
    print(f"  Defense[0]:     {c['defense_claims'][0][:80]}...")
    print(f"  Court:          {c['court_ruling'][:80]}...")
