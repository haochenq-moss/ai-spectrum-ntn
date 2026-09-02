"""Small dependency-free retrieval memory for spectrum-management guidance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class KnowledgeDocument:
    text: str
    category: str


class MemoryRAG:
    """Rank short knowledge documents by token overlap without API credentials."""

    def __init__(self, documents: Iterable[KnowledgeDocument] | None = None) -> None:
        self.documents = list(documents or [])

    def add_knowledge(self, text: str, category: str = "general") -> None:
        self.documents.append(KnowledgeDocument(text=text, category=category))

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeDocument]:
        query_terms = Counter(re.findall(r"[a-z]+", query.lower()))
        def score(document: KnowledgeDocument) -> int:
            return sum((query_terms & Counter(re.findall(r"[a-z]+", document.text.lower()))).values())
        return sorted(self.documents, key=score, reverse=True)[:top_k]


def default_knowledge() -> MemoryRAG:
    """Create core operational guidance for offline agent decisions."""
    memory = MemoryRAG()
    memory.add_knowledge("Strong jammer energy requires adversarial water filling to protect spectral efficiency.", "jamming")
    memory.add_knowledge("Elevated interference without a jammer favors differentiable allocation for rapid adaptation.", "interference")
    memory.add_knowledge("Stable links and ordinary load can use classical water filling as a conservative fallback.", "allocation")
    return memory