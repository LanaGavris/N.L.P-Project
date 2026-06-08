# Contradiction-Aware Legal Graph RAG
### Adversarial Claim Detection and Graph-Enhanced Retrieval for Court Documents
*NLP Capstone Project — HIT*

---

## Research Question

> Can a contradiction-aware graph retrieval strategy quantifiably improve **legal argument balance**, **factual faithfulness**, and **answer completeness** compared to standard semantic RAG and zero-shot reasoning?

---

## Key Idea

Standard vector RAG suffers from **semantic blindness** in adversarial domains: it retrieves chunks by similarity but cannot model dialectical conflicts between prosecution and defense claims. This project addresses this by:

1. Using **majority vs. dissent opinions** as weak supervision signals for contradiction detection
2. Training a **fine-tuned BERT classifier** to detect genuine contradictions in claim pairs
3. Building a **Legal Knowledge Graph** whose edges come from BERT predictions — not raw weak labels
4. Using this graph for **contradiction-aware retrieval** that explicitly surfaces both sides of every dispute

---

## Architecture

```
Real Cases (HuggingFace)          Synthetic Cases (Gemini)
  majority + dissent                 10 structured cases
        |                                   |
        └──────────┬────────────────────────┘
                   ↓
         [data_processor.py]
         CASE-LEVEL SPLIT (no leakage)
                   |
        ┌──────────┼──────────┐
        ↓          ↓          ↓
      Train       Val        Test
  real+synthetic real only  real only
    (70%)        (15%)      (15%) ← GOLD SET
        |
        ↓
  [bert_finetuner.py]
  Fine-tune bert-base-uncased
  [CLS] claim_a [SEP] claim_b [SEP] → contradiction / not
        |
        ↓
  [knowledge_graph.py]
  Graph built from BERT predictions
  Nodes: claim, evidence, court_ruling
  Edges: CONTRADICTS (BERT), SUPPORTS, RESOLVED_BY, HAS_EVIDENCE
        |
        ↓
  [evaluation_pipeline.py]
  6 Ablation configs + Blind LLM Judge
  Evaluated on GOLD test set
```

---

## Project Structure

```
project/
├── google_llm.py             # Google Gemini API client (free tier)
├── hf_dissent_fetcher.py     # Streams caselaw_access_project, finds dissent pairs
├── synthetic_augmenter.py    # Generates 10 structured synthetic cases via Gemini
├── data_processor.py         # Case-level split + hybrid validation pipeline
├── bert_finetuner.py         # Fine-tunes bert-base-uncased for contradiction classification
├── knowledge_graph.py        # Builds Legal KG from BERT predictions
├── graph_rag_pipeline.py     # All 6 ablation RAG systems + Standard RAG + Zero-Shot
├── evaluation_pipeline.py    # Blind LLM judge + ablation evaluation
├── model_evaluator.py        # Classification metrics (Accuracy/P/R/F1/CM/ROC)
├── results_analysis.py       # Bootstrap CI, error analysis, report paragraph
├── txt_parser.py             # Parses CourtListener TXT files (alternative input)
└── Main_Pipeline.ipynb       # Single notebook — run cells in order
```

---

## Setup

### 1. Google Colab
- Open `Main_Pipeline.ipynb` in Colab
- **Runtime → Change runtime type → T4 GPU** (required for Cell 7)
- Upload all `.py` files to Colab (📁 button → Upload)

### 2. API Keys

| Key | Required? | Where to get |
|---|---|---|
| `GOOGLE_API_KEY` | Optional | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free |

Without the Gemini key: synthetic generation uses built-in templates, zero/few-shot baselines use rule-based fallback, LLM judge uses mock scores. **The core pipeline (data collection → BERT fine-tuning → graph → ablation) works completely without any API key.**

### 3. Install
```python
!pip install transformers torch scikit-learn datasets
!pip install sentence-transformers spacy pandas matplotlib networkx pyvis
!python -m spacy download en_core_web_sm
```

---

## Data Pipeline

### Real Cases — No API Key Needed
Streams `common-pile/caselaw_access_project` from HuggingFace.
Finds cases with both majority AND dissent opinions.
- `majority_chunk vs dissent_chunk` → **label=1** (structural guarantee)
- `majority_chunk_A vs majority_chunk_B` → **label=0**

### Synthetic Cases — Optional (Gemini)
10 structured cases with explicit prosecution/defense/court claims.
Gemini generates realistic adversarial claim pairs per topic:
wire fraud, securities fraud, conspiracy, bribery, insider trading, etc.

### Case-Level Split
```
All pairs from ONE case → ONE split only
Train: real + synthetic cases (70%)
Val:   real cases only      (15%)
Test:  real cases only      (15%) ← GOLD SET, never seen during training
```

### Hybrid Validation (Train Only)
1. **BERT confidence filter** — keep pairs where fine-tuned BERT is confident (p > 0.8)
2. **LLM spot-check** — Gemini validates ~25% sample; agreement rate reported as quality metric

---

## ML Task

**Primary: Binary Classification of Legal Claim Pairs**
```
Input:  claim_a + claim_b
Output: contradiction (1) / not contradiction (0)
```

**Secondary: Graph-Enhanced Legal RAG**
Using contradiction graph for retrieval and answer generation.

---

## Models Compared

| Model | Type | Description |
|---|---|---|
| Rule-Based | Heuristic | Negation pattern matching |
| Gemini Zero-Shot | Prompting | No examples |
| Gemini Few-Shot | Prompting | 4 labeled examples in prompt |
| **BERT-base (fine-tuned)** | Fine-tuned | Our primary model |

---

## Ablation Study (6 Configurations)

All systems use the same LLM (Gemini), same prompt structure, same token budget (600 tokens), same test queries.

| # | System | Description |
|---|---|---|
| 1 | **Full Graph RAG** ← **ours** | All edge types (CONTRADICTS + SUPPORTS + RESOLVED_BY) |
| 2 | Graph RAG w/o CONTRADICTS | Remove contradiction edges |
| 3 | Graph RAG rule-based edges | Replace BERT edges with negation heuristics |
| 4 | Graph RAG w/o RESOLVED_BY | Remove court resolution traversal |
| 5 | Standard Vector RAG | Cosine similarity, top-5, no graph |
| 6 | Standard Vector RAG Top-10 | Cosine similarity, top-10, no graph |

---

## Evaluation Metrics

### Classification (BERT on GOLD test set)
| Metric | Description |
|---|---|
| Accuracy | Overall correctness |
| Precision | Of predicted contradictions, how many are real |
| Recall | Of real contradictions, how many were found |
| F1 | Harmonic mean |
| ROC AUC | Area under ROC curve |
| Confusion Matrix | TP / FP / FN / TN breakdown |

### RAG Quality (Blind LLM Judge)
Answers are **anonymized** (System A, B, C...) and **order-randomized** before judging.

| Metric | Description |
|---|---|
| Contradiction Coverage | Did the answer surface both sides and note conflicts? |
| Party Coverage | Are both prosecution and defense represented? |
| Answer Completeness | Does the answer fully address the question? |
| Faithfulness | No hallucinated facts? |

---

## Results

> **Fill in after running experiments**

### Classification — GOLD Test Set

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Rule-Based | | | | | |
| Gemini Zero-Shot | | | | | |
| Gemini Few-Shot | | | | | |
| **BERT-base (fine-tuned)** | | | | | |

### Ablation Study — RAG Quality (LLM Judge)

| System | Contra. Coverage | Party Coverage | Completeness | Faithfulness | Avg |
|---|---|---|---|---|---|
| 1. Full Graph RAG (**ours**) | | | | | |
| 2. No CONTRADICTS edges | | | | | |
| 3. Rule-based edges | | | | | |
| 4. No RESOLVED_BY | | | | | |
| 5. Standard Vector RAG | | | | | |
| 6. Standard RAG Top-10 | | | | | |

### Key Observations
<!-- Write 3-5 bullet points after running experiments -->
- <!-- e.g. "Full Graph RAG achieves X% higher contradiction coverage than Standard RAG" -->
- <!-- e.g. "Removing CONTRADICTS edges drops coverage by X% — confirming their importance" -->
- <!-- e.g. "Rule-based edges score Y% lower than BERT edges — showing fine-tuning adds value" -->
- <!-- e.g. "BERT achieves F1=X on gold test set vs rule-based F1=Y" -->

---

## Limitations

- **Weak supervision**: majority/dissent labels are noisy signals, not all are true contradictions
- **Small test set**: limited statistical power; bootstrap CI reported for all metrics
- **Judge bias**: Gemini used as both generator and judge; partially mitigated by blind evaluation
- **Domain shift**: BERT trained on dissent language may generalize poorly to prosecution/defense
- **Dataset size**: ~50 real cases limits diversity; synthetic augmentation partially compensates

---

## Novelty

1. **Adversarial claim pairs** from majority/dissent opinions as weak supervision — not contract review
2. **BERT-predicted graph edges** — graph topology learned from data, not hard-coded rules
3. **Contradiction-aware retrieval** — explicit dialectical reasoning in RAG
4. **Structured ablation** — 6 configurations isolating each component's contribution
5. **Blind evaluation** — anonymized, randomized judge protocol for fair comparison


---

## Switching LLM Provider

Set keys and provider in **Cell 2** of `Main_Pipeline.ipynb`:

```python
# Google Gemini (free: 1500 fast req/day, 500 smart req/day)
GOOGLE_API_KEY    = "AIza..."

# OpenRouter (free: 200 req/day, auto-selects best model)
OPENROUTER_API_KEY = "sk-or-..."

# Which to use: "auto" | "gemini" | "openrouter"
LLM_PROVIDER = "auto"    # auto = OpenRouter if set, else Gemini

# OpenRouter model (if using OpenRouter):
OR_MODEL = "openrouter/free"   # auto-selects best available free model
```

### When to use each

| Situation | Recommendation |
|---|---|
| Gemini quota not exhausted | `provider="gemini"` — higher daily limit |
| Gemini quota exhausted | `provider="openrouter"` — switch immediately |
| Running long experiments | `provider="openrouter"` — 200 req/day, auto-recovers |
| Need best reasoning quality | `provider="openrouter"`, `OR_MODEL="deepseek/deepseek-r1:free"` |
| Need large context (1M tokens) | `provider="openrouter"`, `OR_MODEL="meta-llama/llama-4-maverick:free"` |
| Both keys set | `provider="auto"` — prefers OpenRouter |

### Rate limits comparison

| Provider | Model | Req/Day | Req/Min | Reset |
|---|---|---|---|---|
| Google Gemini | gemini-2.5-flash-lite | 1500 | 30 | Daily |
| Google Gemini | gemini-2.5-flash | 500 | 15 | Daily |
| OpenRouter | any :free model | 200 | 20 | Daily |

### Free OpenRouter models (May 2026)

```python
OR_FREE_MODELS = [
    "openrouter/free",                       # auto-select (recommended)
    "meta-llama/llama-4-maverick:free",      # best quality, 1M context
    "deepseek/deepseek-r1:free",             # strong reasoning
    "deepseek/deepseek-v3:free",             # general purpose
    "qwen/qwen3-235b-a22b:free",             # large model
    "google/gemma-3-27b-it:free",            # reliable fallback
]
```

### Getting OpenRouter API key (30 seconds)
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign in (Google/GitHub)
3. Keys → Create Key
4. Copy `sk-or-...` key
5. Paste in Cell 2: `OPENROUTER_API_KEY = "sk-or-..."`

No credit card needed for free models.

---

## References

- Lewis et al. (2020). *RAG for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Edge et al. (2024). *From Local to Global: Graph RAG*. Microsoft Research.
- Pile of Law (2022). *256GB Open-Source Legal Dataset*. NeurIPS.
- Zheng et al. (2023). *Judging LLM-as-a-Judge*. NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.

---

*NLP Capstone — HIT | [Your Name] | [Semester]*
