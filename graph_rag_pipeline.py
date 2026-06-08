"""
graph_rag_pipeline.py
══════════════════════
Graph RAG + all 6 ablation systems.

IMPORTANT: StandardVectorRAG uses all-MiniLM-L6-v2 for embeddings,
NOT the fine-tuned BERT. This ensures fair comparison — BERT is only
used for building CONTRADICTS edges in the graph.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_graph import LegalKnowledgeGraph
from llm_client import LLMClient, get_client

MAX_TOKENS    = 600
EMBEDDER_NAME = "all-MiniLM-L6-v2"   # used by ALL retrieval systems for fairness

SYSTEM_PROMPT = (
    "You are a legal research assistant analyzing adversarial court documents. "
    "Answer based ONLY on the provided context. "
    "Always represent BOTH sides of any legal dispute. "
    "If contradictions are shown, explicitly acknowledge them. "
    "Do not invent facts."
)


@dataclass
class RAGAnswer:
    query:            str
    answer:           str
    system_name:      str
    retrieved_nodes:  list[dict] = field(default_factory=list)
    contradictions:   list[dict] = field(default_factory=list)
    court_resolution: str        = ""
    evidence:         list[str]  = field(default_factory=list)
    context_used:     str        = ""


def _build_graph_prompt(query: str, context: dict) -> str:
    parts = [f"QUESTION: {query}\n"]
    if context.get("prosecution"):
        parts.append(f"PROSECUTION POSITION:\n{context['prosecution']}\n")
    if context.get("defense"):
        parts.append(f"DEFENSE POSITION:\n{context['defense']}\n")
    if context.get("contradictions"):
        parts.append("IDENTIFIED CONTRADICTIONS:")
        for c in context["contradictions"]:
            parts.append(f"  [PROSECUTION]: {c['prosecution'][:200]}")
            parts.append(f"  CONTRADICTS")
            parts.append(f"  [DEFENSE]:     {c['defense'][:200]}")
        parts.append("")
    if context.get("court_resolution"):
        parts.append(f"COURT RESOLUTION:\n{context['court_resolution']}\n")
    if context.get("evidence"):
        parts.append("EVIDENCE:\n" + "\n".join(f"  - {e}" for e in context["evidence"]))
    parts.append(
        "\nStructured answer:\n"
        "1. Prosecution's position\n"
        "2. Defense's position\n"
        "3. Key contradictions\n"
        "4. Court's resolution (if available)"
    )
    return "\n".join(parts)


def _build_standard_prompt(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(
        f"[Document {i+1}]: {c[:300]}" for i, c in enumerate(chunks)
    )
    return (
        f"QUESTION: {query}\n\n"
        f"RETRIEVED DOCUMENTS:\n{context}\n\n"
        "Answer the question based on the retrieved documents."
    )


# ─── Shared semantic embedder (loaded once, shared across all systems) ────

_shared_embedder: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _shared_embedder
    if _shared_embedder is None:
        print(f"[RAG] Loading semantic embedder: {EMBEDDER_NAME}")
        _shared_embedder = SentenceTransformer(EMBEDDER_NAME)
    return _shared_embedder


# ─── Graph RAG Pipeline ───────────────────────────────────────────────────

class GraphRAGPipeline:
    """
    Graph-based RAG. Retrieval uses all-MiniLM-L6-v2 (same as Standard RAG).
    Advantage comes from traversing CONTRADICTS / RESOLVED_BY edges after retrieval.
    """

    def __init__(
        self,
        kg: LegalKnowledgeGraph,
        llm: LLMClient | None = None,
        top_k: int = 5,
        system_name: str = "Graph RAG",
    ):
        self.kg          = kg
        self.llm         = llm or get_client()
        self.top_k       = top_k
        self.system_name = system_name

    def answer(self, query: str, verbose: bool = False) -> RAGAnswer:
        relevant = self.kg.find_relevant_nodes(query, top_k=self.top_k)

        prosecution_texts, defense_texts = [], []
        contradictions, evidence = [], []
        court_resolution = ""
        retrieved_nodes  = []

        for node_id, score in relevant:
            data  = self.kg.get_node_data(node_id)
            if not data:
                continue
            party = data.get("party", "")
            text  = data.get("text", "")
            retrieved_nodes.append({
                "node_id": node_id, "party": party,
                "text": text, "score": score,
            })

            if party in ("prosecution", "majority"):
                prosecution_texts.append(text)
            elif party in ("defense", "dissent"):
                defense_texts.append(text)

            for contra_id, weight in self.kg.get_contradictions(node_id)[:2]:
                contra_data  = self.kg.get_node_data(contra_id)
                if not contra_data:
                    continue
                contra_party = contra_data.get("party", "")
                contra_text  = contra_data.get("text", "")
                if party in ("prosecution", "majority") and \
                   contra_party in ("defense", "dissent"):
                    contradictions.append({
                        "prosecution": text,
                        "defense": contra_text,
                        "score": weight,
                    })
                elif party in ("defense", "dissent") and \
                     contra_party in ("prosecution", "majority"):
                    contradictions.append({
                        "prosecution": contra_text,
                        "defense": text,
                        "score": weight,
                    })

            res = self.kg.get_court_resolution(node_id)
            if res and not court_resolution:
                court_resolution = res
            evidence.extend(self.kg.get_supporting_evidence(node_id)[:2])

        context = {
            "prosecution":      "\n".join(prosecution_texts[:3]),
            "defense":          "\n".join(defense_texts[:3]),
            "contradictions":   contradictions[:3],
            "court_resolution": court_resolution,
            "evidence":         list(set(evidence))[:3],
        }
        prompt = _build_graph_prompt(query, context)

        if self.llm.is_configured:
            answer = self.llm.call(
                prompt, SYSTEM_PROMPT, max_tokens=MAX_TOKENS
            )
            self.llm.sleep()
        else:
            answer = f"[{self.system_name}] MOCK — configure LLM in Cell 2"

        return RAGAnswer(
            query=query, answer=answer,
            system_name=self.system_name,
            retrieved_nodes=retrieved_nodes,
            contradictions=contradictions,
            court_resolution=court_resolution,
            evidence=evidence,
            context_used=prompt,
        )


# ─── Standard Vector RAG ──────────────────────────────────────────────────

class StandardVectorRAG:
    """
    Baseline: flat cosine-similarity retrieval, no graph structure.
    Uses all-MiniLM-L6-v2 — same embedder as GraphRAGPipeline.
    This ensures fair comparison: only difference is graph traversal.
    """

    def __init__(
        self,
        documents: list[dict],
        llm: LLMClient | None = None,
        top_k: int = 5,
        system_name: str = "Standard Vector RAG",
    ):
        self.llm         = llm or get_client()
        self.top_k       = top_k
        self.system_name = system_name
        self.docs        = documents

        # Use shared all-MiniLM embedder — NOT fine-tuned BERT
        embedder     = _get_embedder()
        self.embs    = embedder.encode(
            [d["text"][:400] for d in documents],
            show_progress_bar=False,
        )
        self._embedder = embedder
        print(f"[{system_name}] Index: {len(documents)} docs | "
              f"Embedder: {EMBEDDER_NAME}")

    def answer(self, query: str, verbose: bool = False) -> RAGAnswer:
        q_emb  = self._embedder.encode([query[:400]])[0]
        sims   = cosine_similarity(q_emb.reshape(1, -1), self.embs)[0]
        top    = np.argsort(sims)[::-1][:self.top_k]
        chunks = [self.docs[i]["text"] for i in top]
        prompt = _build_standard_prompt(query, chunks)

        if self.llm.is_configured:
            answer = self.llm.call(
                prompt, SYSTEM_PROMPT, max_tokens=MAX_TOKENS
            )
            self.llm.sleep()
        else:
            answer = f"[{self.system_name}] MOCK — configure LLM in Cell 2"

        return RAGAnswer(
            query=query, answer=answer,
            system_name=self.system_name,
            retrieved_nodes=[
                {"text": c, "score": float(sims[top[i]])}
                for i, c in enumerate(chunks)
            ],
            context_used=prompt,
        )


# ─── Zero-Shot Baseline ───────────────────────────────────────────────────

class ZeroShotBaseline:
    """No retrieval — LLM answers from training knowledge only."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_client()

    def answer(self, query: str, verbose: bool = False) -> RAGAnswer:
        if self.llm.is_configured:
            answer = self.llm.call(
                query,
                system=(
                    "You are a legal expert. Answer from your knowledge. "
                    "Be specific about legal arguments from both sides."
                ),
                temperature=0.3,
                max_tokens=MAX_TOKENS,
            )
            self.llm.sleep()
        else:
            answer = "[Zero-Shot] MOCK — configure LLM in Cell 2"
        return RAGAnswer(query=query, answer=answer, system_name="Zero-Shot")


# ─── Ablation factory ─────────────────────────────────────────────────────

def build_ablation_systems(
    kg_full: LegalKnowledgeGraph,
    documents: list[dict],
    cases_data: list[dict],
    bert_model_path: str,
    llm: LLMClient | None = None,
) -> dict[str, object]:
    """
    Build all 6 ablation configurations.
    All share: same LLM, same prompt structure, same token budget,
    same semantic embedder (all-MiniLM-L6-v2).
    """
    client = llm or get_client()

    print(f"\n[Ablation] Building 6 systems | LLM: {client.provider}")
    print(f"[Ablation] Semantic embedder: {EMBEDDER_NAME} (shared, fair comparison)")

    systems = {}

    # 1. Full Graph RAG
    systems["1_full_graph_rag"] = GraphRAGPipeline(
        kg=kg_full, llm=client,
        system_name="1. Full Graph RAG"
    )

    # 2. Graph RAG without CONTRADICTS edges
    print("  Building system 2/6 (no CONTRADICTS)...")
    kg2 = LegalKnowledgeGraph()
    kg2.build(
        cases_data=cases_data, bert_model_path=bert_model_path,
        add_contradicts=False, add_resolved_by=True, verbose=False,
    )
    systems["2_no_contradicts"] = GraphRAGPipeline(
        kg=kg2, llm=client,
        system_name="2. Graph RAG (no CONTRADICTS)"
    )

    # 3. Graph RAG with rule-based edges (no BERT)
    print("  Building system 3/6 (rule-based edges)...")
    kg3 = LegalKnowledgeGraph()
    kg3.build(
        cases_data=cases_data, bert_model_path=bert_model_path,
        edge_source="rule_based", add_contradicts=True,
        add_resolved_by=True, verbose=False,
    )
    systems["3_rule_based_edges"] = GraphRAGPipeline(
        kg=kg3, llm=client,
        system_name="3. Graph RAG (rule-based edges)"
    )

    # 4. Graph RAG without RESOLVED_BY
    print("  Building system 4/6 (no RESOLVED_BY)...")
    kg4 = LegalKnowledgeGraph()
    kg4.build(
        cases_data=cases_data, bert_model_path=bert_model_path,
        add_contradicts=True, add_resolved_by=False, verbose=False,
    )
    systems["4_no_resolved_by"] = GraphRAGPipeline(
        kg=kg4, llm=client,
        system_name="4. Graph RAG (no RESOLVED_BY)"
    )

    # 5. Standard Vector RAG top-5
    print("  Building system 5/6 (Standard RAG top-5)...")
    systems["5_standard_rag"] = StandardVectorRAG(
        documents=documents, llm=client, top_k=5,
        system_name="5. Standard Vector RAG"
    )

    # 6. Standard Vector RAG top-10
    print("  Building system 6/6 (Standard RAG top-10)...")
    systems["6_standard_rag_topk"] = StandardVectorRAG(
        documents=documents, llm=client, top_k=10,
        system_name="6. Standard Vector RAG (Top-10)"
    )

    print(f"\n[Ablation] All 6 systems ready ✓")
    return systems
