"""
knowledge_graph.py
═══════════════════
Builds a Legal Knowledge Graph from BERT predictions.

IMPORTANT: Graph is NOT built from weak structural labels directly.
It is built using LEARNED relationships from the fine-tuned BERT classifier.
This is the key architectural distinction from naive graph construction.

Node types:
  - claim      (tagged: prosecution / defense / court / majority / dissent)
  - evidence
  - court_ruling

Edge types:
  - CONTRADICTS   (predicted by BERT, p > threshold)
  - SUPPORTS      (same-party claims with high similarity)
  - RESOLVED_BY   (court_ruling → disputed claim)
  - HAS_EVIDENCE  (claim → evidence)

The graph enables 4 retrieval modes used in ablation study:
  1. Full Graph RAG (all edge types)
  2. Without CONTRADICTS edges
  3. Rule-based CONTRADICTS (no BERT)
  4. Without RESOLVED_BY traversal
"""

import os
import re
import json
import pickle
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GraphNode:
    node_id:   str
    text:      str
    node_type: str   # "claim" | "evidence" | "court_ruling"
    party:     str   # "prosecution" | "defense" | "court" | "majority" | "dissent"
    case_id:   str
    topic:     str


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str  # "CONTRADICTS" | "SUPPORTS" | "RESOLVED_BY" | "HAS_EVIDENCE"
    weight: float
    origin_datasource: str = "structural"  # "bert" | "rule_based" | "structural"

class LegalKnowledgeGraph:
    """
    Knowledge graph built from BERT-predicted contradiction edges.

    Construction pipeline:
      1. Add claim nodes from all train cases
      2. Use fine-tuned BERT to score all cross-party pairs
      3. Add CONTRADICTS edges where BERT predicts p > threshold
      4. Add SUPPORTS edges (same-party, high semantic similarity)
      5. Add RESOLVED_BY edges from court rulings
      6. Add HAS_EVIDENCE edges from structured cases
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._embeddings: dict[str, np.ndarray] = {}
        self._embedder: SentenceTransformer | None = None

    # ─── Construction ─────────────────────────────────────────────────────

    def build(
        self,
        cases_data: list[dict],        # from data_processor output
        bert_model_path: str | None = None,
        embedder_name: str = "all-MiniLM-L6-v2",
        bert_threshold: float = 0.65,  # lower than validation filter
        support_threshold: float = 0.80,
        edge_source: str = "bert",     # "bert" | "rule_based" | "structural"
        add_contradicts: bool = True,
        add_resolved_by: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Build the full knowledge graph.

        edge_source controls ablation:
          "bert"       → use fine-tuned BERT predictions (full system)
          "rule_based" → use negation pattern heuristics only
          "none"       → no CONTRADICTS edges (ablation)
        """
        print(f"\n[Graph] Building knowledge graph...")
        print(f"  Edge source: {edge_source} | "
              f"add_contradicts={add_contradicts} | "
              f"add_resolved_by={add_resolved_by}")

        # Load embedder
        self._embedder = SentenceTransformer(embedder_name)

        # Load BERT if needed
        bert_model = None
        if edge_source == "bert" and bert_model_path and os.path.exists(bert_model_path):
            from bert_finetuner import BERTFineTuner
            bert_model = BERTFineTuner.load(bert_model_path)
            print(f"  BERT loaded from: {bert_model_path}")

        # 1. Add nodes
        all_nodes: list[GraphNode] = []
        for case in cases_data:
            nodes = self._extract_nodes(case)
            for node in nodes:
                self.G.add_node(
                    node.node_id,
                    text=node.text,
                    node_type=node.node_type,
                    party=node.party,
                    case_id=node.case_id,
                    topic=node.topic,
                )
                # Compute embedding
                emb = self._embedder.encode([node.text[:400]])[0]
                self._embeddings[node.node_id] = emb
            all_nodes.extend(nodes)

        print(f"  Nodes added: {self.G.number_of_nodes()}")

        # 2. CONTRADICTS edges
        if add_contradicts and edge_source != "none":
            n_contra = self._add_contradiction_edges(
                all_nodes, bert_model, bert_threshold, edge_source
            )
            print(f"  CONTRADICTS edges: {n_contra}")

        # 3. SUPPORTS edges (same party, high similarity)
        n_support = self._add_support_edges(all_nodes, support_threshold)
        print(f"  SUPPORTS edges: {n_support}")

        # 4. RESOLVED_BY edges
        if add_resolved_by:
            n_resolved = self._add_resolved_by_edges(all_nodes)
            print(f"  RESOLVED_BY edges: {n_resolved}")

        # 5. HAS_EVIDENCE edges
        n_evidence = self._add_evidence_edges(cases_data)
        print(f"  HAS_EVIDENCE edges: {n_evidence}")

        print(f"\n[Graph] Built: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    def _extract_nodes(self, case: dict) -> list[GraphNode]:
        nodes = []
        case_id = case.get("case_id", "UNK")
        topic   = case.get("topic", "unknown")

        def add(text, ntype, party):
            if not text or len(text.strip()) < 20:
                return
            nid = f"{case_id}_{party}_{len(nodes)}"
            nodes.append(GraphNode(nid, text[:400], ntype, party, case_id, topic))

        for c in case.get("prosecution_claims", []):
            add(c, "claim", "prosecution")
        for c in case.get("defense_claims", []):
            add(c, "claim", "defense")
        if case.get("court_ruling"):
            add(case["court_ruling"], "court_ruling", "court")
        for e in case.get("evidence", []):
            add(e, "evidence", "court")

        # Raw text cases
        for chunk in self._chunk(case.get("majority_text", ""), 400)[:3]:
            add(chunk, "claim", "majority")
        for chunk in self._chunk(case.get("dissent_text", ""), 400)[:2]:
            add(chunk, "claim", "dissent")

        return nodes

    def _add_contradiction_edges(
        self,
        nodes: list[GraphNode],
        bert_model,
        threshold: float,
        source: str,
    ) -> int:
        """Add CONTRADICTS edges based on BERT predictions or rules."""
        OPPOSING = {
            ("prosecution", "defense"), ("defense", "prosecution"),
            ("majority", "dissent"),    ("dissent", "majority"),
            ("court", "defense"),
        }

        claim_nodes = [n for n in nodes if n.node_type == "claim"]
        count = 0

        # Group by case to avoid cross-case edges
        by_case: dict[str, list[GraphNode]] = {}
        for n in claim_nodes:
            by_case.setdefault(n.case_id, []).append(n)

        for case_id, case_nodes in by_case.items():
            # Generate candidate pairs (opposing parties only)
            candidates = [
                (a, b) for i, a in enumerate(case_nodes)
                for b in case_nodes[i+1:]
                if (a.party, b.party) in OPPOSING
            ]

            if not candidates:
                continue

            if source == "bert" and bert_model:
                texts_a = [p[0].text for p in candidates]
                texts_b = [p[1].text for p in candidates]
                _, probs = bert_model.predict(texts_a, texts_b)
                for (a, b), prob in zip(candidates, probs):
                    if prob >= threshold:
                        self._add_edge(a.node_id, b.node_id,
                                       "CONTRADICTS", prob, "bert")
                        self._add_edge(b.node_id, a.node_id,
                                       "CONTRADICTS", prob, "bert")
                        count += 1

            elif source == "rule_based":
                for a, b in candidates:
                    score = self._rule_score(a.text, b.text)
                    if score >= threshold:
                        self._add_edge(a.node_id, b.node_id,
                                       "CONTRADICTS", score, "rule_based")
                        self._add_edge(b.node_id, a.node_id,
                                       "CONTRADICTS", score, "rule_based")
                        count += 1

        return count

    def _rule_score(self, text_a: str, text_b: str) -> float:
        """Rule-based contradiction score via negation patterns."""
        NEGATION_PAIRS = [
            (r"\bknew\b|\baware\b|\bdirected\b",
             r"\bunaware\b|\bdid\s+not\s+know\b|\bdelegated\b"),
            (r"\bintended?\b|\bdeliberate\b",
             r"\bgood\s+faith\b|\bno\s+intent\b|\bbelieved\b"),
            (r"\bguilty\b|\bconvicted\b",
             r"\binnocent\b|\bnot\s+guilty\b|\bacquitted\b"),
            (r"\bfraud\b|\bmisappropriat\b",
             r"\blawful\b|\bauthorized\b|\bpermitted\b"),
        ]
        ta, tb = text_a.lower(), text_b.lower()
        hits = sum(
            1 for pp, np_ in NEGATION_PAIRS
            if (re.search(pp, ta) and re.search(np_, tb)) or
               (re.search(np_, ta) and re.search(pp, tb))
        )
        return min(hits * 0.3, 1.0)

    def _add_support_edges(
        self, nodes: list[GraphNode], threshold: float
    ) -> int:
        """Add SUPPORTS edges between same-party nodes with high similarity."""
        by_party_case: dict[tuple, list[GraphNode]] = {}
        for n in nodes:
            if n.node_type == "claim":
                key = (n.case_id, n.party)
                by_party_case.setdefault(key, []).append(n)

        count = 0
        for (case_id, party), group in by_party_case.items():
            if len(group) < 2:
                continue
            embs = np.array([self._embeddings[n.node_id] for n in group])
            sims = cosine_similarity(embs)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if sims[i, j] >= threshold:
                        self._add_edge(group[i].node_id, group[j].node_id,
                                       "SUPPORTS", float(sims[i, j]), "similarity")
                        count += 1
        return count

    def _add_resolved_by_edges(self, nodes: list[GraphNode]) -> int:
        """Add RESOLVED_BY edges from court rulings to contradicted claims."""
        court_nodes = [n for n in nodes if n.node_type == "court_ruling"]
        claim_nodes = [n for n in nodes if n.node_type == "claim"
                       and n.party == "defense"]
        count = 0
        for cn in court_nodes:
            for claim in claim_nodes:
                if cn.case_id == claim.case_id:
                    self._add_edge(cn.node_id, claim.node_id,
                                   "RESOLVED_BY", 1.0, "structural")
                    count += 1
        return count

    def _add_evidence_edges(self, cases_data: list[dict]) -> int:
        """Add HAS_EVIDENCE edges."""
        count = 0
        for case in cases_data:
            case_id = case.get("case_id", "")
            pros_nodes = [n for n in self.G.nodes()
                          if self.G.nodes[n].get("case_id") == case_id
                          and self.G.nodes[n].get("party") == "prosecution"]
            evi_nodes  = [n for n in self.G.nodes()
                          if self.G.nodes[n].get("case_id") == case_id
                          and self.G.nodes[n].get("node_type") == "evidence"]
            for pn in pros_nodes[:2]:
                for en in evi_nodes:
                    self._add_edge(pn, en, "HAS_EVIDENCE", 1.0, "structural")
                    count += 1
        return count

    def _add_edge(self, src, tgt, etype, weight, esource):
        self.G.add_edge(src, tgt,
                        edge_type=etype,
                        weight=weight,
                        edge_source=esource)

    # ─── Retrieval ─────────────────────────────────────────────────────────

    def find_relevant_nodes(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Semantic search over all nodes."""
        if not self._embedder or not self._embeddings:
            return []
        q_emb = self._embedder.encode([query[:400]])[0]
        ids   = list(self._embeddings.keys())
        embs  = np.array([self._embeddings[i] for i in ids])
        sims  = cosine_similarity(q_emb.reshape(1, -1), embs)[0]
        top   = np.argsort(sims)[::-1][:top_k]
        return [(ids[i], float(sims[i])) for i in top]

    def get_contradictions(self, node_id: str) -> list[tuple[str, float]]:
        """Return all nodes connected by CONTRADICTS edges."""
        result = []
        for _, tgt, data in self.G.out_edges(node_id, data=True):
            if data.get("edge_type") == "CONTRADICTS":
                result.append((tgt, data.get("weight", 0.0)))
        return sorted(result, key=lambda x: -x[1])

    def get_court_resolution(self, node_id: str) -> str | None:
        """Return court ruling that resolves a claim."""
        for src, _, data in self.G.in_edges(node_id, data=True):
            if data.get("edge_type") == "RESOLVED_BY":
                return self.G.nodes[src].get("text", "")
        return None

    def get_supporting_evidence(self, node_id: str) -> list[str]:
        """Return evidence nodes linked to a claim."""
        result = []
        for _, tgt, data in self.G.out_edges(node_id, data=True):
            if data.get("edge_type") == "HAS_EVIDENCE":
                result.append(self.G.nodes[tgt].get("text", ""))
        return result

    def get_node_data(self, node_id: str) -> dict:
        return dict(self.G.nodes.get(node_id, {}))

    def stats(self) -> dict:
        edge_types = {}
        for _, _, d in self.G.edges(data=True):
            et = d.get("edge_type", "?")
            edge_types[et] = edge_types.get(et, 0) + 1
        return {
            "nodes":      self.G.number_of_nodes(),
            "edges":      self.G.number_of_edges(),
            "edge_types": edge_types,
        }

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "legal_kg.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Graph] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LegalKnowledgeGraph":
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _chunk(text: str, size: int) -> list[str]:
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
