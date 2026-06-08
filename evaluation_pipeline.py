"""
evaluation_pipeline.py
═══════════════════════
Blind LLM judge + ablation evaluation.

Fixes in this version:
  - Judge handles partial JSON (missing systems → NaN, not 0.0)
  - Aggregation skips NaN verdicts
  - Standard RAG receives full case texts, not pair fragments
  - Rate limit handling: failed queries are retried once
"""

import os, json, time, random, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import defaultdict

from llm_client import LLMClient, get_client
from graph_rag_pipeline import RAGAnswer


JUDGE_SYSTEM = """You are a blind legal research evaluator.
You receive a legal question and multiple anonymized answers (System A, B, C...).
You do NOT know which system generated each answer.

Evaluate EACH system on these 4 criteria (0.0 to 1.0):
  1. contradiction_coverage: Does the answer explicitly present BOTH prosecution AND defense positions AND name their contradiction?
  2. party_coverage: Are BOTH sides of the dispute clearly represented?
  3. answer_completeness: Does the answer fully address what was asked?
  4. faithfulness: Does the answer avoid inventing facts not grounded in legal record?

IMPORTANT: You MUST return scores for ALL systems listed. If an answer is empty or an error, score it 0.0 on all criteria.

Return ONLY valid JSON with exactly these keys (one per system):
{
  "system_A": {"contradiction_coverage": 0.8, "party_coverage": 1.0, "answer_completeness": 0.7, "faithfulness": 0.9},
  "system_B": {"contradiction_coverage": 0.3, "party_coverage": 0.5, "answer_completeness": 0.4, "faithfulness": 0.8}
}"""

DEFAULT_TEST_QUERIES = [
    "Did the defendant knowingly participate in the fraudulent scheme?",
    "Was the defendant aware that investor funds were being misused?",
    "Did the defendant act with criminal intent or in good faith?",
    "What evidence supports the prosecution's theory of the case?",
    "How did the defense explain the defendant's actions?",
    "What was the court's final ruling on the key disputed facts?",
    "Were the defendant's financial representations materially false?",
    "Did the defendant have a motive to commit the alleged crime?",
]


@dataclass
class JudgeVerdict:
    query:                  str
    system_name:            str
    anon_label:             str
    contradiction_coverage: float
    party_coverage:         float
    answer_completeness:    float
    faithfulness:           float

    @property
    def avg_score(self) -> float:
        vals = [self.contradiction_coverage, self.party_coverage,
                self.answer_completeness, self.faithfulness]
        valid = [v for v in vals if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float('nan')


@dataclass
class SystemEvalResult:
    system_name:                str
    accuracy:                   float = 0.0
    precision:                  float = 0.0
    recall:                     float = 0.0
    f1:                         float = 0.0
    avg_contradiction_coverage: float = 0.0
    avg_party_coverage:         float = 0.0
    avg_completeness:           float = 0.0
    avg_faithfulness:           float = 0.0
    avg_judge_score:            float = 0.0
    n_queries:                  int   = 0


# ─── Safe score extractor ─────────────────────────────────────────────────

def _safe_score(scores: dict, key: str) -> float:
    """Return score or NaN if missing/invalid."""
    v = scores.get(key)
    try:
        f = float(v)
        return f if 0.0 <= f <= 1.0 else float('nan')
    except (TypeError, ValueError):
        return float('nan')


# ─── Blind judge ──────────────────────────────────────────────────────────

class BlindLLMJudge:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_client()
        if not self.llm.is_configured:
            print("[Judge] No LLM — mock verdicts will be used")
        else:
            print(f"[Judge] Blind judge ready | provider: {self.llm.provider}")

    def judge_all(
        self,
        systems: dict[str, object],
        queries: list[str],
        verbose:  bool = True,
    ) -> list[JudgeVerdict]:
        all_verdicts: list[JudgeVerdict] = []

        for q_idx, query in enumerate(queries):
            if verbose:
                print(f"\n[Judge] Query {q_idx+1}/{len(queries)}: {query[:60]}...")

            # Step 1: Collect answers
            answers: dict[str, str] = {}
            for sys_name, system in systems.items():
                try:
                    result = system.answer(query, verbose=False)
                    ans = result.answer.strip()
                    # Mark empty/error answers clearly
                    answers[sys_name] = ans if ans and not ans.startswith("[") else \
                                        "[No answer generated]"
                except Exception as e:
                    answers[sys_name] = f"[Error: {str(e)[:80]}]"
                time.sleep(0.3)

            # Step 2: Anonymize + shuffle
            sys_names = list(answers.keys())
            random.shuffle(sys_names)
            labels    = [f"system_{c}" for c in
                         string.ascii_uppercase[:len(sys_names)]]
            anon_map  = dict(zip(labels, sys_names))   # label → real name

            anon_text = "\n\n".join(
                f"[{lbl.upper().replace('_', ' ')}]:\n{answers[anon_map[lbl]][:600]}"
                for lbl in labels
            )
            prompt = (
                f"LEGAL QUESTION: {query}\n\n"
                f"ANONYMIZED ANSWERS:\n{anon_text}\n\n"
                f"Evaluate ALL {len(labels)} systems. "
                f"Return JSON with keys: {', '.join(labels)}"
            )

            # Step 3: Call judge (with retry on failure)
            verdicts_raw = self._call_judge(prompt, labels)

            # Step 4: Parse — missing systems get NaN (not 0)
            for lbl in labels:
                real_name = anon_map[lbl]
                scores    = verdicts_raw.get(lbl)

                if scores and isinstance(scores, dict):
                    cc = _safe_score(scores, "contradiction_coverage")
                    pc = _safe_score(scores, "party_coverage")
                    ac = _safe_score(scores, "answer_completeness")
                    fa = _safe_score(scores, "faithfulness")
                else:
                    # System not in judge response — skip this verdict
                    if verbose:
                        print(f"  ⚠ {real_name}: not in judge response — skipped")
                    continue

                all_verdicts.append(JudgeVerdict(
                    query=query,
                    system_name=real_name,
                    anon_label=lbl,
                    contradiction_coverage=cc,
                    party_coverage=pc,
                    answer_completeness=ac,
                    faithfulness=fa,
                ))

            if verbose:
                q_verdicts = [v for v in all_verdicts if v.query == query]
                for v in sorted(q_verdicts, key=lambda x: -x.avg_score):
                    score_str = f"{v.avg_score:.2f}" if not np.isnan(v.avg_score) else "N/A"
                    print(f"  {v.system_name[:40]:<40} avg={score_str}")

        return all_verdicts

    def _call_judge(self, prompt: str, expected_labels: list[str]) -> dict:
        """Call judge LLM with one retry if response is incomplete."""
        if not self.llm.is_configured:
            return {
                lbl: {
                    "contradiction_coverage": random.uniform(0.3, 0.9),
                    "party_coverage":         random.uniform(0.4, 1.0),
                    "answer_completeness":    random.uniform(0.4, 0.9),
                    "faithfulness":           random.uniform(0.6, 1.0),
                }
                for lbl in expected_labels
            }

        for attempt in range(2):
            result = self.llm.call_json(
                prompt=prompt,
                system=JUDGE_SYSTEM,
                use_smart_model=True,
                temperature=0.0,
            )
            self.llm.sleep()

            if "error" in result:
                if attempt == 0:
                    print(f"  ⚠ Judge error, retrying... ({result['error'][:60]})")
                    time.sleep(10)
                    continue
                return {}

            # Check completeness
            found = sum(1 for lbl in expected_labels if lbl in result)
            if found >= len(expected_labels) * 0.8:  # 80% threshold
                return result
            elif attempt == 0:
                print(f"  ⚠ Judge returned {found}/{len(expected_labels)} systems, retrying...")
                time.sleep(5)

        return result if result else {}


# ─── Build Standard RAG document index from full case texts ───────────────

def build_rag_documents(
    dissent_cases_json: str = "dissent_cases.json",
    synthetic_cases_json: str = "synthetic_cases.json",
    train_case_ids: set | None = None,
) -> list[dict]:
    """
    Build document index from full case texts (not pair fragments).
    This gives Standard RAG fair access to the same information as Graph RAG.
    
    Each document = one chunk of majority or dissent text with metadata.
    """
    docs = []

    def _get_maj(d):
        return d.get("majority_text") or d.get("majority_opinion") or ""
    def _get_dis(d):
        return d.get("dissent_text") or d.get("dissenting_opinion") or ""

    def _chunk(text, size=400):
        import re
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

    # Real cases
    if os.path.exists(dissent_cases_json):
        with open(dissent_cases_json) as f:
            real_cases = json.load(f)
        for d in real_cases:
            cid = d.get("case_id", d.get("filename", ""))
            if train_case_ids and cid not in train_case_ids:
                continue
            topic = d.get("crime_type", d.get("topic", "criminal"))
            for chunk in _chunk(_get_maj(d))[:4]:
                docs.append({"text": chunk, "metadata": {
                    "case": cid, "topic": topic, "party": "majority"}})
            for chunk in _chunk(_get_dis(d))[:3]:
                docs.append({"text": chunk, "metadata": {
                    "case": cid, "topic": topic, "party": "dissent"}})

    # Synthetic cases
    if os.path.exists(synthetic_cases_json):
        with open(synthetic_cases_json) as f:
            syn_cases = json.load(f)
        for d in syn_cases:
            cid   = d.get("case_id", "")
            if train_case_ids and cid not in train_case_ids:
                continue
            topic = d.get("topic", "criminal")
            for claim in d.get("prosecution_claims", []):
                docs.append({"text": claim[:400], "metadata": {
                    "case": cid, "topic": topic, "party": "prosecution"}})
            for claim in d.get("defense_claims", []):
                docs.append({"text": claim[:400], "metadata": {
                    "case": cid, "topic": topic, "party": "defense"}})

    print(f"[RAG Docs] Built {len(docs)} documents from case texts "
          f"(majority/dissent/claims)")
    return docs


# ─── Classification evaluation ────────────────────────────────────────────

def evaluate_classification(
    test_json:       str,
    bert_model_path: str,
    rule_threshold:  float = 0.3,
) -> dict[str, SystemEvalResult]:
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)
    from model_evaluator import RuleBasedPredictor

    with open(test_json) as f:
        data = json.load(f)
    df      = pd.DataFrame(data)
    y_true  = df["label"].tolist()
    texts_a = df["text_a"].tolist()
    texts_b = df["text_b"].tolist()
    party_a = df.get("party_a", pd.Series([""]*len(y_true))).tolist()
    party_b = df.get("party_b", pd.Series([""]*len(y_true))).tolist()

    results = {}

    rb = RuleBasedPredictor(threshold=rule_threshold)
    rb_preds, _ = rb.predict(texts_a, texts_b, party_a, party_b)
    results["rule_based"] = SystemEvalResult(
        system_name="Rule-Based",
        accuracy =round(float(accuracy_score(y_true, rb_preds)), 4),
        precision=round(float(precision_score(y_true, rb_preds, zero_division=0)), 4),
        recall   =round(float(recall_score(y_true, rb_preds, zero_division=0)), 4),
        f1       =round(float(f1_score(y_true, rb_preds, zero_division=0)), 4),
    )

    if os.path.exists(bert_model_path):
        from bert_finetuner import BERTFineTuner
        bert = BERTFineTuner.load(bert_model_path)
        bert_preds, _ = bert.predict(texts_a, texts_b)
        results["bert"] = SystemEvalResult(
            system_name="Fine-Tuned BERT",
            accuracy =round(float(accuracy_score(y_true, bert_preds)), 4),
            precision=round(float(precision_score(y_true, bert_preds, zero_division=0)), 4),
            recall   =round(float(recall_score(y_true, bert_preds, zero_division=0)), 4),
            f1       =round(float(f1_score(y_true, bert_preds, zero_division=0)), 4),
        )

    return results


# ─── Aggregate verdicts ───────────────────────────────────────────────────

def aggregate_verdicts(verdicts: list[JudgeVerdict]) -> dict[str, SystemEvalResult]:
    by_system: dict[str, list[JudgeVerdict]] = defaultdict(list)
    for v in verdicts:
        if not np.isnan(v.avg_score):  # skip NaN verdicts
            by_system[v.system_name].append(v)

    results = {}
    for sys_name, vs in by_system.items():
        def safe_mean(vals):
            clean = [x for x in vals if not np.isnan(x)]
            return round(float(np.mean(clean)), 4) if clean else 0.0

        results[sys_name] = SystemEvalResult(
            system_name=sys_name,
            avg_contradiction_coverage=safe_mean([v.contradiction_coverage for v in vs]),
            avg_party_coverage=safe_mean([v.party_coverage for v in vs]),
            avg_completeness=safe_mean([v.answer_completeness for v in vs]),
            avg_faithfulness=safe_mean([v.faithfulness for v in vs]),
            avg_judge_score=safe_mean([v.avg_score for v in vs]),
            n_queries=len(vs),
        )
    return results


# ─── Reporting ────────────────────────────────────────────────────────────

def print_ablation_table(
    rag_results: dict[str, SystemEvalResult],
    clf_results: dict[str, SystemEvalResult] | None = None,
) -> None:
    w = 90
    print(f"\n{'═'*w}")
    print(f" ABLATION STUDY — Full Results")
    print(f"{'─'*w}")
    print(f" {'System':<42} {'Contra.Cov':>10} {'Party.Cov':>10} "
          f"{'Complete':>9} {'Faithful':>9} {'Avg':>6} {'N':>4}")
    print(f"{'─'*w}")

    order = ["1_full_graph_rag", "2_no_contradicts", "3_rule_based_edges",
             "4_no_resolved_by", "5_standard_rag",   "6_standard_rag_topk"]
    for key in order:
        if key not in rag_results:
            continue
        r    = rag_results[key]
        flag = "  ◀ OUR SYSTEM" if key == "1_full_graph_rag" else ""
        print(f" {r.system_name:<42} {r.avg_contradiction_coverage:>10.4f} "
              f"{r.avg_party_coverage:>10.4f} {r.avg_completeness:>9.4f} "
              f"{r.avg_faithfulness:>9.4f} {r.avg_judge_score:>6.4f} "
              f"{r.n_queries:>4}{flag}")

    if clf_results:
        print(f"\n{'─'*w}")
        print(f" CLASSIFICATION (BERT on gold test set):")
        print(f" {'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>9} {'F1':>9}")
        print(f"{'─'*w}")
        for r in clf_results.values():
            print(f" {r.system_name:<30} {r.accuracy:>10.4f} "
                  f"{r.precision:>10.4f} {r.recall:>9.4f} {r.f1:>9.4f}")
    print(f"{'═'*w}")


def plot_ablation(
    rag_results: dict[str, SystemEvalResult],
    save_path:   str = "ablation_results.png",
) -> None:
    order  = ["1_full_graph_rag", "2_no_contradicts", "3_rule_based_edges",
              "4_no_resolved_by", "5_standard_rag",   "6_standard_rag_topk"]
    avail  = [k for k in order if k in rag_results]
    if not avail:
        print("No results to plot")
        return

    labels = [rag_results[k].system_name.replace(". ", ".\n", 1) for k in avail]
    metric_data = {
        "Contradiction\nCoverage": [rag_results[k].avg_contradiction_coverage for k in avail],
        "Party\nCoverage":         [rag_results[k].avg_party_coverage         for k in avail],
        "Completeness":            [rag_results[k].avg_completeness           for k in avail],
        "Faithfulness":            [rag_results[k].avg_faithfulness           for k in avail],
    }
    x      = np.arange(len(labels))
    w      = 0.18
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    for i, (mname, vals) in enumerate(metric_data.items()):
        offset = (i - len(metric_data)/2 + 0.5) * w
        ax_bar.bar(x + offset, vals, w, label=mname,
                   color=colors[i], alpha=0.85, edgecolor="white")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=7)
    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_ylabel("Score (LLM Judge)")
    ax_bar.set_title("Ablation Study — RAG System Comparison\n(Blind LLM Judge, Anonymized)")
    ax_bar.legend(loc="upper right", fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    avgs   = [rag_results[k].avg_judge_score for k in avail]
    best   = int(np.argmax(avgs)) if avgs else 0
    ax_bar.axvline(x=best, color="gold", linewidth=3, alpha=0.4, zorder=0)

    matrix = np.array([[rag_results[k].avg_contradiction_coverage,
                         rag_results[k].avg_party_coverage,
                         rag_results[k].avg_completeness,
                         rag_results[k].avg_faithfulness] for k in avail])
    im = ax_heat.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_xticks(range(4))
    ax_heat.set_xticklabels(["Contra.", "Party", "Compl.", "Faith."], fontsize=9)
    ax_heat.set_yticks(range(len(avail)))
    ax_heat.set_yticklabels([rag_results[k].system_name[:22] for k in avail], fontsize=7)
    ax_heat.set_title("Score Heatmap")
    for i in range(len(avail)):
        for j in range(4):
            ax_heat.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if matrix[i,j] < 0.4 else "black")
    plt.colorbar(im, ax=ax_heat, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Ablation plot saved: {save_path}")


def save_all_results(
    rag_results: dict[str, SystemEvalResult],
    clf_results: dict[str, SystemEvalResult],
    verdicts:    list[JudgeVerdict],
    path:        str = "full_evaluation_results.json",
) -> None:
    def clean(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    data = {
        "rag_results": {k: {kk: clean(vv) for kk, vv in vars(v).items()}
                        for k, v in rag_results.items()},
        "clf_results": {k: vars(v) for k, v in clf_results.items()},
        "verdicts": [{kk: clean(vv) for kk, vv in vars(v).items()}
                     for v in verdicts],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Results saved: {path}")


# ─── Query generator from graph ───────────────────────────────────────────

def generate_queries_from_graph(
    kg,
    n_queries: int = 8,
    llm: "LLMClient | None" = None,
) -> list[dict]:
    """
    Generate evaluation queries grounded in actual contradictions found by BERT.

    For each top-scored CONTRADICTS edge in the graph:
      - Takes the majority and dissent text
      - Asks LLM to generate one natural question whose answer
        REQUIRES reading both sides
      - Falls back to a template question if no LLM

    Returns list of {"query": str, "case_id": str, "topic": str,
                     "majority_hint": str, "dissent_hint": str}
    """
    import random

    # Collect top CONTRADICTS edges sorted by weight
    edges = []
    for u, v, data in kg.G.edges(data=True):
        if data.get("edge_type") == "CONTRADICTS":
            u_data = kg.get_node_data(u)
            v_data = kg.get_node_data(v)
            if not u_data or not v_data:
                continue
            # Only majority→dissent direction (skip reverse)
            if u_data.get("party") in ("majority","prosecution") and \
               v_data.get("party") in ("dissent","defense"):
                edges.append({
                    "weight":   data.get("weight", 0.0),
                    "u_id":     u,
                    "v_id":     v,
                    "u_text":   u_data.get("text",""),
                    "v_text":   v_data.get("text",""),
                    "case_id":  u_data.get("case_id",""),
                    "topic":    u_data.get("topic","criminal"),
                })

    if not edges:
        print("[QueryGen] No CONTRADICTS edges found — using default queries")
        return [{"query": q, "case_id": "generic", "topic": "criminal",
                 "majority_hint": "", "dissent_hint": ""}
                for q in DEFAULT_TEST_QUERIES[:n_queries]]

    # Sort by weight, pick diverse cases
    edges.sort(key=lambda x: -x["weight"])

    # Pick top edges, one per case_id for diversity
    selected = []
    seen_cases = set()
    for e in edges:
        if e["case_id"] not in seen_cases:
            selected.append(e)
            seen_cases.add(e["case_id"])
        if len(selected) >= n_queries:
            break

    # If fewer than n_queries unique cases, fill with top remaining
    if len(selected) < n_queries:
        for e in edges:
            if e not in selected:
                selected.append(e)
            if len(selected) >= n_queries:
                break

    print(f"[QueryGen] Generating {len(selected)} queries from BERT-predicted contradictions")

    queries = []
    client  = llm or get_client()

    for i, edge in enumerate(selected):
        query = _generate_one_query(edge, client, i)
        queries.append({
            "query":         query,
            "case_id":       edge["case_id"],
            "topic":         edge["topic"],
            "majority_hint": edge["u_text"][:120],
            "dissent_hint":  edge["v_text"][:120],
        })
        if client.is_configured:
            client.sleep()
        print(f"  [{i+1}/{len(selected)}] {edge['topic']} | {query[:65]}")

    return queries


def _generate_one_query(edge: dict, client, idx: int) -> str:
    """Generate one question from a majority/dissent contradiction."""
    if not client.is_configured:
        return _template_query(edge)

    prompt = (
        f"A court opinion contains these two conflicting positions:\n\n"
        f"[MAJORITY]: {edge['u_text'][:250]}\n\n"
        f"[DISSENT]: {edge['v_text'][:250]}\n\n"
        f"Write ONE short, specific legal question (max 15 words) that:\n"
        f"- Can only be answered by reading BOTH positions\n"
        f"- Is factual, not rhetorical\n"
        f"- Starts with 'Did', 'Was', 'How', or 'What'\n"
        f"Return ONLY the question, nothing else."
    )
    response = client.call(
        prompt=prompt,
        system="You are a legal question generator. Return only the question text.",
        temperature=0.3,
        max_tokens=40,
    )
    # Clean up
    q = response.strip().strip('"').strip("'")
    if not q or q.startswith("["):
        return _template_query(edge)
    if not q.endswith("?"):
        q += "?"
    return q


def _template_query(edge: dict) -> str:
    """Rule-based fallback query from topic."""
    templates = {
        "fraud":           "Did the defendant knowingly misrepresent material facts to investors?",
        "conspiracy":      "Did the defendant agree to participate in the criminal conspiracy?",
        "drug trafficking":"Did the defendant knowingly direct the distribution of narcotics?",
        "bribery":         "Did the defendant corruptly offer payments to government officials?",
        "tax fraud":       "Did the defendant intentionally conceal income from tax authorities?",
        "homicide":        "Did the defendant act with premeditated intent to cause death?",
        "theft":           "Did the defendant knowingly take property without authorization?",
        "criminal":        "Did the defendant act with the requisite criminal intent?",
    }
    topic = edge.get("topic", "criminal").lower()
    return templates.get(topic, templates["criminal"])
