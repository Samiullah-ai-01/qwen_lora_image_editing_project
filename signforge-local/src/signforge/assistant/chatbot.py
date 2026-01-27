"""
Personal assistant chatbot for SignForge.
Rule-based with retrieval over docs and logs.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from signforge.assistant.retrieval import DocRetriever
from signforge.assistant.tools import DiagnosticTools
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class Chatbot:
    """SignForge personal assistant."""

    def __init__(self) -> None:
        self.retriever = DocRetriever()
        self.tools = DiagnosticTools()
        
        # Command patterns
        self.commands = {
            r"^diagnose": self._cmd_diagnose,
            r"^explain weights?": self._cmd_explain_weights,
            r"^help": self._cmd_help,
            r"^status": self._cmd_status,
            r"^how (to|do I)": self._cmd_how_to,
            r"^what (is|are)": self._cmd_what_is,
            r"^why": self._cmd_why,
        }
        
        # Common questions and answers
        self.faq = {
            "install": "Run `make setup` to install dependencies. Ensure Python 3.10+ and Node 18+ are available.",
            "cuda": "CUDA errors usually mean GPU drivers are outdated or VRAM is insufficient. Try reducing resolution.",
            "memory": "For OOM errors: enable VAE tiling, use attention slicing, or reduce image resolution.",
            "xformers": "Install xformers with: `pip install xformers --index-url https://download.pytorch.org/whl/cu121`",
            "training": "Place images + .txt caption files in data/raw/<domain>/<concept>/, then run `make preprocess` and `make train-one CONCEPT=<name>`.",
            "adapters": "Adapters are LoRA weights. Lower weights = less influence. Default weights are tuned for minimal interference.",
        }

    def chat(self, message: str) -> str:
        """Process a chat message and return response."""
        message = message.strip()
        if not message:
            return "Please enter a message."
        
        # Check for commands
        message_lower = message.lower()
        for pattern, handler in self.commands.items():
            if re.match(pattern, message_lower):
                return handler(message)
        
        # Try retrieval
        docs = self.retriever.search(message)
        if docs:
            return f"Based on documentation:\n\n{docs[0]}"
        
        # Check FAQ keywords
        for keyword, answer in self.faq.items():
            if keyword in message_lower:
                return answer
        
        return (
            "I'm not sure how to help with that. Try:\n"
            "- `diagnose` - Check system health\n"
            "- `explain weights` - Explain adapter weights\n"
            "- `status` - Get current status\n"
            "- `help` - Show all commands"
        )

    def _cmd_diagnose(self, message: str) -> str:
        """Run diagnostics."""
        return self.tools.diagnose()

    def _cmd_explain_weights(self, message: str) -> str:
        """Explain adapter weights."""
        return self.tools.explain_weights()

    def _cmd_help(self, message: str) -> str:
        """Show help."""
        return """Available commands:
- `diagnose` - Check system health and suggest fixes
- `explain weights` - Explain current adapter weights
- `status` - Get server and queue status
- `help` - Show this help message

You can also ask questions like:
- "How do I install dependencies?"
- "Why did generation fail?"
- "What are adapters?"
- "How to train a LoRA?"
"""

    def _cmd_status(self, message: str) -> str:
        """Get status."""
        return self.tools.get_status()

    def _cmd_how_to(self, message: str) -> str:
        """Handle how-to questions."""
        message_lower = message.lower()
        
        if "install" in message_lower or "setup" in message_lower:
            return self.faq["install"]
        elif "train" in message_lower:
            return self.faq["training"]
        elif "adapter" in message_lower or "lora" in message_lower:
            return "Load adapters via the UI or API. Each adapter controls one aspect (sign type, mounting, etc.). Adjust weights to control influence."
        
        docs = self.retriever.search(message)
        if docs:
            return docs[0]
        
        return "I don't have specific instructions for that. Check the README or docs/ folder."

    def _cmd_what_is(self, message: str) -> str:
        """Handle what-is questions."""
        message_lower = message.lower()
        
        if "adapter" in message_lower or "lora" in message_lower:
            return self.faq["adapters"]
        elif "signforge" in message_lower:
            return "SignForge is a local-first signage mockup generator using Qwen-Image with multi-LoRA composition."
        
        docs = self.retriever.search(message)
        return docs[0] if docs else "I don't have a definition for that."

    def _cmd_why(self, message: str) -> str:
        """Handle why questions."""
        message_lower = message.lower()
        
        if "fail" in message_lower or "error" in message_lower:
            diag = self.tools.diagnose()
            return f"Let me check...\n\n{diag}"
        elif "slow" in message_lower:
            return "Generation can be slow due to: high resolution, many steps, or limited GPU VRAM. Try the 'fast' profile."
        
        return "Could you be more specific about what you're asking about?"
