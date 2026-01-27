"""
Prompt optimization and rewriting for SignForge.

Provides local prompt enhancement without external services.
"""

from __future__ import annotations

import re
from typing import Optional

from signforge.core.logging import get_logger

logger = get_logger(__name__)


class PromptOptimizer:
    """
    Optimizes prompts for better signage generation.
    
    Uses rule-based enhancement:
    - Adds quality boosters
    - Fixes common issues
    - Expands abbreviations
    - Ensures consistency
    """
    
    def __init__(self) -> None:
        """Initialize the optimizer."""
        # Quality boosters to append
        self.quality_boosters = [
            "high quality",
            "professional photography",
            "sharp focus",
            "detailed",
        ]
        
        # Signage-specific boosters
        self.signage_boosters = [
            "commercial signage",
            "readable text",
            "proper perspective",
        ]
        
        # Common abbreviations to expand
        self.abbreviations = {
            "LED": "LED illuminated",
            "RGB": "RGB LED",
            "SS": "stainless steel",
            "AL": "aluminum",
        }
        
        # Words that indicate signage context
        self.signage_keywords = [
            "sign", "signage", "letters", "logo", "storefront",
            "banner", "billboard", "marquee", "awning",
        ]
    
    def optimize(
        self,
        prompt: str,
        add_quality: bool = True,
        add_signage: bool = True,
        expand_abbreviations: bool = True,
    ) -> str:
        """
        Optimize a prompt.
        
        Args:
            prompt: Input prompt
            add_quality: Add quality boosters
            add_signage: Add signage-specific terms
            expand_abbreviations: Expand common abbreviations
            
        Returns:
            Optimized prompt
        """
        result = prompt.strip()
        
        # Expand abbreviations
        if expand_abbreviations:
            result = self._expand_abbreviations(result)
        
        # Check if signage-related
        is_signage = any(kw in result.lower() for kw in self.signage_keywords)
        
        # Add boosters
        boosters = []
        
        if add_quality:
            # Only add boosters not already present
            for booster in self.quality_boosters:
                if booster.lower() not in result.lower():
                    boosters.append(booster)
                    break  # Just add one
        
        if add_signage and is_signage:
            for booster in self.signage_boosters:
                if booster.lower() not in result.lower():
                    boosters.append(booster)
                    break
        
        if boosters:
            result = f"{result}, {', '.join(boosters)}"
        
        return result
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in text."""
        result = text
        for abbr, expansion in self.abbreviations.items():
            # Only expand if it's a standalone word
            pattern = rf"\b{abbr}\b"
            result = re.sub(pattern, expansion, result)
        return result
    
    def enhance_for_readability(
        self,
        prompt: str,
        sign_text: str,
    ) -> str:
        """
        Enhance prompt for text readability.
        
        Args:
            prompt: Base prompt
            sign_text: The text that should appear on the sign
            
        Returns:
            Enhanced prompt
        """
        # Emphasize the sign text
        enhanced = prompt
        
        # Add clarity emphasis
        readability_terms = [
            f"clearly displaying '{sign_text}'",
            "legible text",
            "crisp lettering",
        ]
        
        # Check if text is quoted in prompt
        if f"'{sign_text}'" not in enhanced and f'"{sign_text}"' not in enhanced:
            enhanced = enhanced.replace(
                sign_text,
                f"'{sign_text}'"
            )
        
        # Add readability terms
        if "legible" not in enhanced.lower() and "readable" not in enhanced.lower():
            enhanced = f"{enhanced}, {readability_terms[1]}"
        
        return enhanced
    
    def get_negative_prompt(
        self,
        base_negative: str = "",
        is_text_heavy: bool = True,
    ) -> str:
        """
        Generate negative prompt for signage.
        
        Args:
            base_negative: User-provided negative prompt
            is_text_heavy: Whether the sign has significant text
            
        Returns:
            Complete negative prompt
        """
        negatives = []
        
        if base_negative:
            negatives.append(base_negative)
        
        # Standard quality negatives
        quality_negatives = [
            "blurry",
            "low quality",
            "pixelated",
            "amateur",
            "poorly lit",
        ]
        negatives.extend(quality_negatives)
        
        # Text-specific negatives
        if is_text_heavy:
            text_negatives = [
                "misspelled",
                "illegible",
                "distorted text",
                "wrong letters",
                "garbled text",
            ]
            negatives.extend(text_negatives)
        
        # Signage negatives
        signage_negatives = [
            "damaged sign",
            "broken",
            "old and worn",
            "dirty",
            "vandalized",
        ]
        negatives.extend(signage_negatives)
        
        return ", ".join(negatives)
    
    def suggest_improvements(
        self,
        prompt: str,
    ) -> list[dict]:
        """
        Suggest improvements to a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            List of suggestions with reasons
        """
        suggestions = []
        prompt_lower = prompt.lower()
        
        # Check for missing elements
        if not any(word in prompt_lower for word in ["sign", "signage", "letters"]):
            suggestions.append({
                "type": "missing_subject",
                "message": "Consider adding 'sign' or 'signage' for clarity",
                "example": f"{prompt}, commercial signage",
            })
        
        if not any(word in prompt_lower for word in ["photo", "photograph", "image"]):
            suggestions.append({
                "type": "missing_medium",
                "message": "Adding 'photograph' can improve realism",
                "example": f"Professional photograph of {prompt}",
            })
        
        if not any(word in prompt_lower for word in self.signage_keywords[4:]):
            if not any(word in prompt_lower for word in ["environment", "setting", "location"]):
                suggestions.append({
                    "type": "missing_context",
                    "message": "Consider adding environmental context",
                    "example": f"{prompt}, urban storefront setting",
                })
        
        # Check prompt length
        word_count = len(prompt.split())
        if word_count < 10:
            suggestions.append({
                "type": "too_short",
                "message": "Longer prompts often produce better results",
                "example": None,
            })
        elif word_count > 75:
            suggestions.append({
                "type": "too_long",
                "message": "Very long prompts may lose coherence",
                "example": None,
            })
        
        return suggestions
    
    def extract_sign_text(
        self,
        prompt: str,
    ) -> Optional[str]:
        """
        Extract the sign text from a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Extracted text or None
        """
        # Look for quoted text
        patterns = [
            r"'([^']+)'",  # Single quotes
            r'"([^"]+)"',  # Double quotes
            r"reading\s+['\"]?([^'\",.]+)",  # "reading X"
            r"displaying\s+['\"]?([^'\",.]+)",  # "displaying X"
            r"saying\s+['\"]?([^'\",.]+)",  # "saying X"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1).strip()
        
        return None
