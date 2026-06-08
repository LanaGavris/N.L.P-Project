"""
legal_qa.py
════════════
Interactive Legal Q&A with Adversarial-Aware Response.

Input:  a natural language question about a case
Output:
  1. Structured analytical report (Prosecution → Defense → Court)
  2. graph.html — interactive PyVis subgraph of relevant contradictions

Usage:
    from legal_qa import LegalQA
    qa = LegalQA(kg=kg, llm=llm)
    result = qa.ask("Did Bankman-Fried knowingly misuse customer funds?")
    result.print()
    result.show_graph()
"""

import os
import re
import json
import textwrap
from dataclasses import dataclass, field

from knowledge_graph import LegalKnowledgeGraph
from llm_client import LLMClient, get_client


# ─── Output container ─────────────────────────────────────────────────────

@dataclass
class QAResult:
    question:        str
    prosecution:     str          # prosecution position
    defense:         str          # defense counter-argument
    court:           str          # court resolution
    full_report:     str          # full LLM-generated report
    contradictions:  list[dict]   # BERT-predicted contradiction pairs used
    retrieved_nodes: list[dict]   # all nodes retrieved for context
    graph_html:      str = ""     # PyVis HTML content
    graph_path:      str = ""     # saved HTML file path

    def print(self) -> None:
        w = 68
        print(f"\n{'═'*w}")
        print(f" QUESTION: {self.question}")
        print(f"{'═'*w}")
        print(self.full_report)
        print(f"\n{'─'*w}")
        print(f" CONTRADICTIONS FOUND BY BERT: {len(self.contradictions)}")
        for i, c in enumerate(self.contradictions, 1):
            score = c.get('score', 0)
            bar   = "█" * int(score * 16) + "░" * (16 - int(score * 16))
            print(f"\n  [{bar}] {score:.0%}  Topic: {c.get('topic','?').upper()}")
            print(f"  [{c.get('party_a','?').upper()}]: {c.get('text_a','')[:100]}...")
            print(f"  ⟺  CONTRADICTS")
            print(f"  [{c.get('party_b','?').upper()}]: {c.get('text_b','')[:100]}...")
            if c.get('court_resolution'):
                print(f"  ✓ Court resolved: {c['court_resolution'][:80]}...")
        if self.graph_path:
            print(f"\n{'─'*w}")
            print(f" Graph saved: {self.graph_path}")
            print(f" Open in browser to explore interactively.")
        print(f"{'═'*w}")

    def show_graph(self) -> None:
        """Display graph in Jupyter/Colab."""
        if not self.graph_path:
            print("No graph generated.")
            return
        try:
            from IPython.display import HTML, display
            display(HTML(self.graph_html))
        except ImportError:
            print(f"Open {self.graph_path} in your browser.")

    def to_dict(self) -> dict:
        return {
            "question":       self.question,
            "prosecution":    self.prosecution,
            "defense":        self.defense,
            "court":          self.court,
            "full_report":    self.full_report,
            "contradictions": self.contradictions,
            "n_nodes":        len(self.retrieved_nodes),
            "graph_path":     self.graph_path,
        }


# ─── Prompt ───────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a senior legal analyst producing adversarial-aware case reports. "
    "You have access to claims from both sides of a court case, "
    "explicitly identified contradictions between them, and the court's resolution. "
    "Always present ALL three positions: prosecution, defense, and court. "
    "Be precise and cite the specific claims provided. "
    "Do not invent facts not present in the context."
)

PROMPT_TEMPLATE = """QUESTION: {question}

═══ PROSECUTION CLAIMS ═══
{prosecution_text}

═══ DEFENSE CLAIMS ═══
{defense_text}

═══ BERT-IDENTIFIED CONTRADICTIONS ═══
{contradictions_text}

═══ COURT RESOLUTION ═══
{court_text}

Write a structured analytical report with exactly these three sections:

**PROSECUTION POSITION:**
[2-3 sentences summarizing prosecution's argument with specific claims]

**DEFENSE COUNTER-ARGUMENT:**
[2-3 sentences summarizing defense's position and how it contradicts prosecution]

**COURT RESOLUTION:**
[1-2 sentences on how the court resolved the dispute, or "Not yet resolved" if unavailable]

Keep the report factual, balanced, and under 300 words."""


# ─── Main Q&A class ───────────────────────────────────────────────────────

class LegalQA:
    """
    Interactive legal Q&A with adversarial-aware response and graph output.
    """

    def __init__(
        self,
        kg: LegalKnowledgeGraph,
        llm: LLMClient | None = None,
        top_k: int = 8,
        graph_output_dir: str = ".",
    ):
        self.kg               = kg
        self.llm              = llm or get_client()
        self.top_k            = top_k
        self.graph_output_dir = graph_output_dir

        if not self.llm.is_configured:
            print("[LegalQA] No LLM configured — reports will be template-based")
        else:
            print(f"[LegalQA] Ready | LLM: {self.llm.provider} | top_k: {top_k}")

    def ask(
        self,
        question: str,
        save_graph: bool = True,
        graph_filename: str = "graph.html",
        verbose: bool = True,
    ) -> QAResult:
        """
        Main entry point.

        Args:
            question:       natural language question about the case
            save_graph:     whether to save the PyVis HTML file
            graph_filename: output filename for the graph
            verbose:        print progress

        Returns:
            QAResult with full report + graph
        """
        if verbose:
            print(f"\n[LegalQA] Processing: {question[:60]}...")

        # 1. Retrieve relevant nodes
        relevant = self.kg.find_relevant_nodes(question, top_k=self.top_k)
        if verbose:
            print(f"  Retrieved {len(relevant)} nodes")

        # 2. Collect claims by party + contradictions
        prosecution, defense, court_resolutions = [], [], []
        contradictions = []
        seen_contra    = set()
        retrieved_nodes = []

        for node_id, score in relevant:
            data  = self.kg.get_node_data(node_id)
            if not data:
                continue
            party = data.get("party", "")
            text  = data.get("text", "")
            topic = data.get("topic", "")
            retrieved_nodes.append({
                "node_id": node_id, "party": party,
                "text": text, "score": score, "topic": topic,
            })

            if party in ("prosecution", "majority"):
                prosecution.append(text)
            elif party in ("defense", "dissent"):
                defense.append(text)
            elif party == "court":
                court_resolutions.append(text)

            # Walk CONTRADICTS edges from this node
            for contra_id, weight in self.kg.get_contradictions(node_id)[:3]:
                pair_key = tuple(sorted([node_id, contra_id]))
                if pair_key in seen_contra:
                    continue
                seen_contra.add(pair_key)

                contra_data = self.kg.get_node_data(contra_id)
                if not contra_data:
                    continue

                contra_party = contra_data.get("party", "")
                contra_text  = contra_data.get("text", "")

                # Ensure prosecution→defense ordering
                if party in ("prosecution", "majority"):
                    pa, ta = party, text
                    pb, tb = contra_party, contra_text
                else:
                    pa, ta = contra_party, contra_text
                    pb, tb = party, text

                court_res = (
                    self.kg.get_court_resolution(node_id) or
                    self.kg.get_court_resolution(contra_id) or ""
                )

                contradictions.append({
                    "party_a":         pa,
                    "text_a":          ta,
                    "party_b":         pb,
                    "text_b":          tb,
                    "score":           round(weight, 3),
                    "topic":           topic,
                    "court_resolution": court_res,
                    "node_a":          node_id,
                    "node_b":          contra_id,
                })

        # Also grab court resolution from retrieved court nodes
        for node_id, _ in relevant:
            data = self.kg.get_node_data(node_id)
            if data and data.get("node_type") == "court_ruling":
                court_resolutions.append(data.get("text", ""))

        if verbose:
            print(f"  Prosecution claims: {len(prosecution)}")
            print(f"  Defense claims:     {len(defense)}")
            print(f"  Contradictions:     {len(contradictions)}")

        # 3. Build context strings
        pros_text  = _join(prosecution[:3])
        def_text   = _join(defense[:3])
        court_text = _join(court_resolutions[:2]) or "No explicit court resolution found."

        contra_text = ""
        if contradictions:
            parts = []
            for i, c in enumerate(contradictions[:4], 1):
                parts.append(
                    f"[Contradiction {i} | score={c['score']:.2f}]\n"
                    f"  [{c['party_a'].upper()}]: {c['text_a'][:180]}\n"
                    f"  ⟺ CONTRADICTS\n"
                    f"  [{c['party_b'].upper()}]: {c['text_b'][:180]}"
                )
                if c.get("court_resolution"):
                    parts[-1] += f"\n  ✓ Court: {c['court_resolution'][:100]}"
            contra_text = "\n\n".join(parts)
        else:
            contra_text = "No direct contradictions detected for this query."

        # 4. Generate report
        report, pros_section, def_section, court_section = self._generate_report(
            question, pros_text, def_text, contra_text, court_text
        )

        # 5. Build PyVis subgraph
        graph_html = ""
        graph_path = ""
        if save_graph:
            subgraph_node_ids = (
                {n["node_id"] for n in retrieved_nodes} |
                {c["node_a"] for c in contradictions} |
                {c["node_b"] for c in contradictions}
            )
            graph_html = _build_pyvis_html(
                self.kg, subgraph_node_ids, question
            )
            graph_path = os.path.join(self.graph_output_dir, graph_filename)
            with open(graph_path, "w", encoding="utf-8") as f:
                f.write(graph_html)
            if verbose:
                print(f"  Graph saved: {graph_path} "
                      f"({len(subgraph_node_ids)} nodes)")

        return QAResult(
            question=question,
            prosecution=pros_section,
            defense=def_section,
            court=court_section,
            full_report=report,
            contradictions=contradictions,
            retrieved_nodes=retrieved_nodes,
            graph_html=graph_html,
            graph_path=graph_path,
        )

    def _generate_report(
        self,
        question: str,
        pros_text: str,
        def_text: str,
        contra_text: str,
        court_text: str,
    ) -> tuple[str, str, str, str]:
        """Generate report and extract the three sections."""

        prompt = PROMPT_TEMPLATE.format(
            question=question,
            prosecution_text=pros_text or "No prosecution claims retrieved.",
            defense_text=def_text or "No defense claims retrieved.",
            contradictions_text=contra_text,
            court_text=court_text,
        )

        if self.llm.is_configured:
            report = self.llm.call(
                prompt=prompt,
                system=SYSTEM,
                use_smart_model=False,
                temperature=0.1,
                max_tokens=500,
            )
            self.llm.sleep()
        else:
            report = _template_report(question, pros_text, def_text, court_text)

        # Extract sections
        pros_s  = _extract_section(report, "PROSECUTION POSITION")
        def_s   = _extract_section(report, "DEFENSE COUNTER-ARGUMENT")
        court_s = _extract_section(report, "COURT RESOLUTION")

        return report, pros_s, def_s, court_s


# ─── PyVis subgraph builder ───────────────────────────────────────────────

PARTY_COLORS = {
    "prosecution": "#e74c3c",
    "majority":    "#e74c3c",
    "defense":     "#3498db",
    "dissent":     "#9b59b6",
    "court":       "#2ecc71",
    "evidence":    "#95a5a6",
}

EDGE_COLORS = {
    "CONTRADICTS":   "#e74c3c",
    "SUPPORTS":      "#2ecc71",
    "RESOLVED_BY":   "#f39c12",
    "HAS_EVIDENCE":  "#95a5a6",
}


def _build_pyvis_html(
    kg: LegalKnowledgeGraph,
    node_ids: set[str],
    question: str,
) -> str:
    """
    Build a self-contained PyVis HTML file for the subgraph.
    Works without a running server — all JS is inline.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<html><body>pyvis not installed. Run: pip install pyvis</body></html>"

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=True,
        cdn_resources="in_line",
    )
    net.set_options("""{
      "nodes": {
        "font": {"size": 12, "face": "arial"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "shadow": true
      },
      "physics": {
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 120},
        "stabilization": {"iterations": 150}
      },
      "interaction": {"hover": true, "tooltipDelay": 100}
    }""")

    added = set()
    for nid in node_ids:
        data  = kg.get_node_data(nid)
        if not data or nid in added:
            continue
        party     = data.get("party", "unknown")
        text      = data.get("text", "")
        topic     = data.get("topic", "")
        node_type = data.get("node_type", "claim")
        color     = PARTY_COLORS.get(party, "#95a5a6")
        label     = text[:22].strip() + "…" if len(text) > 22 else text
        title     = (
            f"<b>[{party.upper()}]</b><br>"
            f"Topic: {topic}<br><br>"
            f"{text[:300]}"
        )
        size = 18 if node_type == "court_ruling" else 14
        net.add_node(
            nid, label=label, color=color,
            title=title, size=size,
            shape="diamond" if node_type == "court_ruling" else "dot",
        )
        added.add(nid)

    # Add edges between subgraph nodes
    for u, v, edata in kg.G.edges(data=True):
        if u not in added or v not in added:
            continue
        etype = edata.get("edge_type", "")
        w     = edata.get("weight", 0.5)
        color = EDGE_COLORS.get(etype, "#555555")
        width = 4 if etype == "CONTRADICTS" else 1.5
        label = f"{etype}\n{w:.2f}" if etype == "CONTRADICTS" else etype
        net.add_edge(u, v, label=label, color=color,
                     width=width, title=f"{etype} (weight={w:.2f})")

    # Legend via a title annotation
    net.html = net.html if hasattr(net, "html") else ""

    # Generate HTML
    html = net.generate_html()

    # Inject question banner at top
    banner = (
        f'<div style="position:fixed;top:0;left:0;right:0;z-index:9999;'
        f'background:#0d1117;color:#58a6ff;padding:8px 16px;'
        f'font-family:monospace;font-size:13px;border-bottom:1px solid #30363d;">'
        f'<b>Query:</b> {question[:100]}'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:#e74c3c">● Prosecution/Majority</span>&nbsp;'
        f'<span style="color:#3498db">● Defense</span>&nbsp;'
        f'<span style="color:#9b59b6">● Dissent</span>&nbsp;'
        f'<span style="color:#2ecc71">● Court</span>&nbsp;'
        f'<span style="color:#95a5a6">● Evidence</span>'
        f'</div>'
        f'<div style="margin-top:42px;">'
    )
    html = html.replace("<body>", f"<body>{banner}", 1)
    html = html + "</div>"

    return html


# ─── Helpers ──────────────────────────────────────────────────────────────

def _join(texts: list[str], sep: str = "\n\n") -> str:
    return sep.join(f"• {t.strip()}" for t in texts if t.strip())


def _extract_section(report: str, header: str) -> str:
    """Extract a named section from the report."""
    pattern = rf"\*\*{re.escape(header)}[:\*]*\*\*(.*?)(?=\*\*[A-Z]|\Z)"
    m = re.search(pattern, report, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: look for the header without markdown
    pattern2 = rf"{re.escape(header)}[:\s]+(.*?)(?=(?:PROSECUTION|DEFENSE|COURT)[:\s]|\Z)"
    m2 = re.search(pattern2, report, re.DOTALL | re.IGNORECASE)
    return m2.group(1).strip() if m2 else ""


def _template_report(
    question: str,
    pros: str,
    defense: str,
    court: str,
) -> str:
    """Fallback report when no LLM is configured."""
    return (
        f"**PROSECUTION POSITION:**\n"
        f"{pros[:400] if pros else 'No prosecution claims retrieved for this query.'}\n\n"
        f"**DEFENSE COUNTER-ARGUMENT:**\n"
        f"{defense[:400] if defense else 'No defense claims retrieved for this query.'}\n\n"
        f"**COURT RESOLUTION:**\n"
        f"{court[:300] if court else 'No court resolution found.'}\n\n"
        f"[Template report — configure LLM in Cell 2 for full analysis]"
    )
