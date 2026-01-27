"""
Document retrieval for assistant.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from signforge.core.config import get_project_root
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class DocRetriever:
    """Simple keyword-based document retrieval."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}
        self._load_docs()

    def _load_docs(self) -> None:
        """Load documentation files."""
        root = get_project_root()
        
        # Load README
        readme = root / "README.md"
        if readme.exists():
            self.docs["readme"] = readme.read_text(encoding="utf-8")
        
        # Load config files
        configs_dir = root / "configs"
        if configs_dir.exists():
            for cfg in configs_dir.rglob("*.yaml"):
                key = f"config_{cfg.stem}"
                self.docs[key] = cfg.read_text(encoding="utf-8")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        Search documents for relevant content.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of relevant text snippets
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for doc_name, content in self.docs.items():
            # Simple keyword matching
            score = sum(1 for word in query_words if word in content.lower())
            if score > 0:
                # Extract relevant section
                snippet = self._extract_snippet(content, query_words)
                results.append((score, snippet))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [r[1] for r in results[:top_k]]

    def _extract_snippet(self, content: str, query_words: set[str], context_lines: int = 5) -> str:
        """Extract relevant snippet from content."""
        lines = content.split("\n")
        
        # Find line with most matches
        best_idx = 0
        best_score = 0
        
        for i, line in enumerate(lines):
            score = sum(1 for word in query_words if word in line.lower())
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Extract context
        start = max(0, best_idx - context_lines)
        end = min(len(lines), best_idx + context_lines + 1)
        
        return "\n".join(lines[start:end])
