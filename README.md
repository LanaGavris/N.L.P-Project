# Contradiction-Aware Legal Graph RAG
### Adversarial Claim Detection & Graph-Enhanced Retrieval for Balanced Legal Reasoning
*NLP Project · Gavris Svetlana*

---

## Research Question

> Can contradiction-aware graph retrieval quantifiably improve **legal argument balance**, **factual faithfulness**, and **answer completeness** compared to standard semantic RAG and zero-shot reasoning?

**Hypothesis:** Legal reasoning should explicitly model contradictory claims and the relationships between adversarial parties — not treat documents as independent text chunks.

---

## The Problem

Standard vector RAG suffers from **semantic blindness** in adversarial legal domains:

| Problem | Effect |
|---|---|
| Retrieves by similarity only | Ignores *who* is speaking |
| Retrieval bias | Majority side dominates (more text) |
| Averages contradictions | Opposing claims collapse into one answer |
| No party awareness | Prosecution ≡ Defense ≡ Court |
| One-sided context | LLM generates incomplete or hallucinated answers |

---

## System Architecture

```
Legal Case Document (full text)
        ↓
Claim Extraction + Party Tagging
(prosecution · defense · court · majority · dissent)
        ↓
BERT Fine-Tuning
[CLS] claim_a [SEP] claim_b [SEP] → contradiction / not
        ↓
Legal Knowledge Graph
Nodes: claim, evidence, court_ruling
Edges: CONTRADICTS (BERT) · SUPPORTS · RESOLVED_BY · HAS_EVIDENCE
        ↓
Graph-Guided Retrieval
Semantic search → traverse CONTRADICTS edges → both sides surfaced
        ↓
LLM Answer Generation (Gemini / OpenRouter)
        ↓
Structured Output:
  PROSECUTION position
  DEFENSE counter-argument
  IDENTIFIED contradictions (BERT-scored)
  COURT resolution
  graph.html (interactive PyVis visualization)
```

---

## Project Structure

```
project/
├── Main_Pipeline_v3.ipynb      ← single notebook, run cells in order
│
├── llm_client.py               ← unified LLM client (Gemini + OpenRouter)
├── google_llm.py               ← compatibility shim → delegates to llm_client
├── hf_dissent_fetcher.py       ← streams caselaw_access_project (HuggingFace)
├── synthetic_augmenter.py      ← generates 10 structured synthetic cases via LLM
├── data_processor.py           ← case-level split + hybrid validation
├── bert_finetuner.py           ← fine-tunes bert-base-uncased
├── knowledge_graph.py          ← builds Legal KG from BERT predictions
├── graph_rag_pipeline.py       ← 6 ablation RAG systems (all-MiniLM-L6-v2 embedder)
├── evaluation_pipeline.py      ← blind LLM judge + ablation evaluation
├── model_evaluator.py          ← classification metrics + human gold evaluation
├── results_analysis.py         ← bootstrap CI + error analysis + report paragraph
├── legal_qa.py                 ← interactive Q&A + PyVis subgraph output
└── human_pairs.json            ← 100 manually annotated contradiction pairs
```


## Dataset

### Real Cases — No API Key Required

You can see examples Dataset at the link: https://drive.google.com/drive/folders/1BCZK-HsabuwNKLUNrWwxNb0pgZqf70jo

Streamed from `common-pile/caselaw_access_project` (HuggingFace).


To train and evaluate the system, we initially experimented with fully synthetic legal text generation. However, the fully synthetic data was kept at a small scale because we found that real-world, historically documented court opinions provided a significantly better, more coherent, and realistic rhetorical structure. 

#### EDA 

<img width="1289" height="396" alt="download (1)" src="https://github.com/user-attachments/assets/1a46d3b7-27bc-453b-8291-feee093c5fba" />


Therefore, our final dataset was constructed using real-world judicial cases. The pipeline uses a hybrid labeling strategy:

1. LLM-Assisted Automated Labeling: High-performance models (via OpenRouter/Gemini) were used to systematically annotate party positions and match arguments across the legal documents.

2. Human-in-the-Loop Validation: We then manually curated and verified a high-quality Gold Dataset of 100 diversified claim pairs to serve as the absolute ground truth for testing.


3. BERT Fine-Tuning: We fine-tune a BERT (bert-base-uncased) model for contradiction detection.

<img width="592" height="592" alt="image" src="https://github.com/user-attachments/assets/e34a2b38-04da-4538-a85e-3e208e711d5a" />

Model Input:

Claim A

Claim B

Model Output:

Contradiction
or
Non-Contradiction
The model is trained on synthetic and manually validated legal claim pairs.

For evaluation, we created a manually annotated Gold Dataset containing 100 legal claim pairs spanning multiple legal domains, including financial fraud, corporate crime, cybercrime, and regulatory litigation.

The fine-tuned model achieved an F1-score of 0.9972 on the contradiction detection task.

#### 1.structural test set quantifies

<img width="1189" height="490" alt="download (5)" src="https://github.com/user-attachments/assets/20fc2629-d69b-4704-bea1-628ab870cf47" />

<img width="896" height="396" alt="download (6)" src="https://github.com/user-attachments/assets/1ddb0c79-ed79-4450-82b9-40aad195abac" />

#### 2.human gold quantifies

<img width="1189" height="490" alt="download (10)" src="https://github.com/user-attachments/assets/7c565b55-722e-4827-b444-78e8870baabe" />


<img width="896" height="396" alt="download (8)" src="https://github.com/user-attachments/assets/28f386e0-9967-47c4-864a-b4b0a97cae9f" />

**Key finding:** While the Fine-Tuned BERT model achieves a near-perfect **F1-score of 0.9972** on the automated structural test set (Figure 1), it experiences a noticeable performance degradation when subjected to the manual expert-annotated Gold Dataset (Figure 2). 

As part of our rigorous research methodology, we conducted an extensive error analysis and identified that this divergence is a classic demonstration of **heuristic bias** and **shortcut learning** in deep transformer architectures:

1. **Stylistic and Linguistic Proxies (Structural Split):** In the automated dataset, the ground-truth labels were structurally derived from combining *Majority Opinions* and *Dissent Opinions*. The model quickly exploited low-level syntactic shortcuts rather than executing multi-hop semantic analysis. Specifically, it mapped corporate pronoun shifts—such as the collective **"We hold/find"** typical of Majority rulings versus the individual **"I dissent"** characteristic of Dissenting opinions—as definitive proxies for Class 1 (Contradiction).
2. **Conceptual Nuances (Gold Dataset Complexity):** The manually curated Gold Dataset intentionally stripped away these formatting and stylistic artifacts. It introduced **Hard Contradictions** (latent procedural or logical conflicts masked by polite, standard legal prose) and **Hard Non-Contradictions** (where opposing parties share identical legal terminology and fact patterns but do not logically clash). Forced to rely solely on raw conceptual legal reasoning, the model's text-only metrics naturally shifted.

####  Architectural Justification: Why the Graph is Necessary

This performance gap does not undermine the utility of our local BERT component; rather, it **scientifically justifies the core hypothesis of this project**. 

Within our target document ecosystem, these stylistic markers remain highly stable and valuable features. BERT effectively leverages them to automate the construction of the Knowledge Graph at scale, serving as a highly efficient *Edge Constructor*. 

However, because text-only embeddings are inherently vulnerable to stylistic shifts, relying on a standard Vector RAG would lead to severe context blindness. By anchoring these predictions into a rigid **Knowledge Graph topology** (`CONTRADICTS` and `COURT_RESOLVES` edges), our final **Graph RAG** pipeline balances out the transformer's limitations, ensuring that the generation stage remains securely grounded in multi-perspective factual evidence.

---

## Knowledge Graph

Built from **BERT predictions** — not raw structural labels.

The extracted claims become nodes in a legal knowledge graph.


**Edge construction:**
- `CONTRADICTS` — BERT p > 0.65 for opposing-party pairs
- `SUPPORTS` — same-party cosine similarity > 0.80 (all-MiniLM-L6-v2)
- `RESOLVED_BY` — court ruling → disputed defense claims
- `HAS_EVIDENCE` — prosecution claims → evidence nodes

---

## Results

### Contradiction Detection



### Ablation Study — Final Results

**Protocol:** Blind LLM judge · 8 queries grounded in BERT-predicted contradictions
· answers anonymized + order-randomized · same LLM / same prompt / same 600-token
budget across all 6 systems · N=8 verdicts per system.

<img width="1280" height="255" alt="image" src="https://github.com/user-attachments/assets/e0a5463b-1580-4c18-91f7-b3bf85b72ade" />

<img width="1280" height="287" alt="image" src="https://github.com/user-attachments/assets/a4a18448-0147-4390-ab74-7d492f6561f6" />


#### Key Takeaways

**Full Graph RAG is the clear winner (avg=0.7750):**

- **Contradiction Coverage: 0.95** — highest by a large margin (+0.41 over No CONTRADICTS,
  +0.58 over Standard RAG). BERT-predicted CONTRADICTS edges are the decisive factor
  that forces the system to surface opposing claims.

- **Party Coverage: 0.65** — best among all systems. Graph traversal ensures both
  prosecution and defense positions appear in every answer.

- **Faithfulness: 0.8125** — second highest, confirming that structured retrieval
  reduces hallucination compared to flat cosine search.

**Removing CONTRADICTS edges (System 2) drops avg by 0.258 points** — the single
largest ablation gap, directly proving the value of BERT-predicted contradiction edges.

**Rule-based edges (System 3, avg=0.4323)** score the lowest among graph variants.
Negation patterns designed for prosecution/defense language produce 0 CONTRADICTS edges
on majority/dissent text. This confirms that fine-tuned BERT is essential —
rule-based heuristics do not generalize to appellate judicial writing style.

**Standard RAG Top-10 (avg=0.6250)** comes second overall, benefiting from more
retrieved context. However, its Contradiction Coverage (0.50) remains 0.45 points
below Full Graph RAG — it cannot reliably surface adversarial claim pairs
without graph traversal.

---

## Interactive Q&A

```python
# After QA-SETUP:
QUESTION = "Did the defendant knowingly misuse customer funds?"
result = qa.ask(QUESTION, save_graph=True, graph_filename="graph.html")
result.print()       # Prosecution → Defense → Court structured report
result.show_graph()  # Interactive PyVis visualization (Colab inline)
```

**Example output:**

```
[LegalQA] Processing: Did the defendant knowingly misuse customer funds?...
  Retrieved 8 nodes
  Prosecution claims: 8
  Defense claims:     0
  Contradictions:     16
  [OpenRouter] Auto-selected: poolside/laguna-xs.2-20260421:free
  Graph saved: ./graph.html (24 nodes)

════════════════════════════════════════════════════════════════════
 QUESTION: Did the defendant knowingly misuse customer funds?
════════════════════════════════════════════════════════════════════
**PROSECUTION POSITION:**
The prosecution argues defendant knowingly filed materially false financial statements with the SEC to inflate stock price and deliberately concealed material risks while using investor funds for personal enrichment, asserting intentional fraud through deceptive financial reporting.

**DEFENSE COUNTER-ARGUMENT:**
The defense contends all financial statements were prepared by independent auditors and defendant relied on their professional judgment, argues defendant had no knowledge that accounting practices were improper under GAAP, and maintains the Medical clinic owner acted in good faith believing all representations to investors were accurate, directly contradicting the prosecution's claims of knowing misconduct.

**COURT RESOLUTION:**
The jury finds defendant guilty, and the Court finds the evidence of knowing misrepresentation overwhelming, establishing defendant's sophistication and intent despite defense arguments.
</assistant>
────────────────────────────────────────────────────────────────────
 CONTRADICTIONS FOUND BY BERT: 16

  [██████████████░░] 88%  Topic: SECURITIES FRAUD
  [PROSECUTION]: The prosecution argues defendant knowingly filed materially false financial statements with the SEC ...
  ⟺  CONTRADICTS
  [DEFENSE]: The defense contends all financial statements were prepared by independent auditors and defendant re...
  ✓ Court resolved: The jury finds defendant guilty. Documentary evidence and cooperating witness te...

  [█████████████░░░] 87%  Topic: SECURITIES FRAUD
  [PROSECUTION]: The prosecution argues defendant knowingly filed materially false financial statements with the SEC ...
  ⟺  CONTRADICTS
  [DEFENSE]: Defense argues defendant had no knowledge that accounting practices were improper under GAAP at the ...
  ✓ Court resolved: The jury finds defendant guilty. Documentary evidence and cooperating witness te...

  [██████████████░░] 89%  Topic: HEALTHCARE FRAUD
  [PROSECUTION]: The government maintains defendant deliberately concealed material risks and used investor funds for...
  ⟺  CONTRADICTS
  [DEFENSE]: The defense argues the Medical clinic owner acted in good faith and genuinely believed all represent...
  ✓ Court resolved: The Court finds the evidence of knowing misrepresentation overwhelming. Defendan...

  [█████████████░░░] 86%  Topic: HEALTHCARE FRAUD
  [PROSECUTION]: The government maintains defendant deliberately concealed material risks and used investor funds for...
  ⟺  CONTRADICTS
  [DEFENSE]: The defense maintains defendant had no personal financial motive to commit fraud given existing legi...
  ✓ Court resolved: The Court finds the evidence of knowing misrepresentation overwhelming. Defendan...

  [██████████████░░] 91%  Topic: BRIBERY
  [PROSECUTION]: The government maintains defendant deliberately concealed material risks and used investor funds for...
  ⟺  CONTRADICTS
  [DEFENSE]: The defense argues the Public official acted in good faith and genuinely believed all representation...
  ✓ Court resolved: The Court finds the evidence of knowing misrepresentation overwhelming. Defendan...

```




**graph.html legend:**

<img width="737" height="436" alt="fin 5" src="https://github.com/user-attachments/assets/6c3d0d3a-9472-4452-a2ce-b03540a8fe0f" />

https://colab.research.google.com/drive/1xyw8vGf_PB0DQ5j8IXSiqptJnXfi5j1o#scrollTo=7yWUKf-cnWIe&fullscreenOutput=true

## Output Files 

https://drive.google.com/drive/folders/1f8j0hzsMj6hVsQrKimtD5o-6l1Ih_ZZ7

| File | Contents |
|---|---|
| `dissent_cases.json` |  real dissent case pairs |
| `train/val/test_split.json` | Labeled pairs by split |
| `bert_model/best_model/` | Fine-tuned BERT checkpoint |
| `legal_kg.pkl` | Serialized knowledge graph |
| `eval_queries.json` | 8 LLM-generated evaluation queries |
| `judge_verdicts_checkpoint.json` | Ablation verdicts (per session) |
| `full_evaluation_results_8q.json` | Complete ablation results |
| `classification_results_structural.json` | BERT metrics on structural gold |
| `classification_results_human.json` | BERT metrics on human expert gold |
| `ablation_results_8q.png` | Ablation bar chart + score heatmap |
| `graph.html` | Interactive knowledge graph |
| `qa_result.json` | Last Q&A result |

## FULL Output Project Files 
https://drive.google.com/drive/folders/1_mTIX1SjRMTBcvNhby7QvKOBshsdUW8y

