"""
Prompt templates for SignForge.

Provides structured templates for consistent signage mockup prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    
    name: str
    template: str
    description: str
    variables: list[str] = field(default_factory=list)
    negative_prompt: str = ""
    
    # Suggested adapters
    suggested_adapters: list[str] = field(default_factory=list)
    
    def render(self, **kwargs: str) -> str:
        """
        Render the template with provided values.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Rendered prompt
        """
        result = self.template
        
        for var in self.variables:
            placeholder = f"{{{var}}}"
            value = kwargs.get(var, "")
            result = result.replace(placeholder, value)
        
        return result.strip()


# Default template library
TEMPLATE_LIBRARY: dict[str, PromptTemplate] = {
    "channel_letters": PromptTemplate(
        name="channel_letters",
        template=(
            "Professional photograph of channel letter signage reading '{text}', "
            "{material} letters with {lighting} illumination, "
            "mounted on {surface}, {perspective} view, "
            "{environment} setting, commercial photography, high quality"
        ),
        description="Channel letter signs with customizable text and style",
        variables=["text", "material", "lighting", "surface", "perspective", "environment"],
        negative_prompt=(
            "blurry, low quality, distorted text, misspelled, "
            "amateur, poorly lit, bad perspective"
        ),
        suggested_adapters=["sign_type/channel_letters"],
    ),
    
    "box_sign": PromptTemplate(
        name="box_sign",
        template=(
            "Commercial photograph of illuminated box sign displaying '{text}', "
            "{color} acrylic face with aluminum cabinet, "
            "{mounting} mounting, {perspective} angle, "
            "{environment}, professional signage photography"
        ),
        description="Cabinet/box signs with internal illumination",
        variables=["text", "color", "mounting", "perspective", "environment"],
        negative_prompt=(
            "blurry, pixelated, incorrect lighting, dim, "
            "damaged sign, dirty, old"
        ),
        suggested_adapters=["sign_type/box_sign"],
    ),
    
    "neon_sign": PromptTemplate(
        name="neon_sign",
        template=(
            "Atmospheric photograph of vintage neon sign reading '{text}', "
            "{color} neon tubes glowing brightly, "
            "{style} style, {environment}, "
            "night photography, bokeh lights, moody atmosphere"
        ),
        description="Classic neon tube signs",
        variables=["text", "color", "style", "environment"],
        negative_prompt=(
            "modern LED, digital display, daylight, "
            "broken tubes, dim, not glowing"
        ),
        suggested_adapters=["sign_type/neon", "environment/night"],
    ),
    
    "monument_sign": PromptTemplate(
        name="monument_sign",
        template=(
            "Professional photograph of monument sign for '{text}', "
            "{material} construction with {finish} finish, "
            "ground-mounted entrance sign, {landscaping}, "
            "{time_of_day} lighting, corporate architecture photography"
        ),
        description="Ground-mounted monument/entrance signs",
        variables=["text", "material", "finish", "landscaping", "time_of_day"],
        negative_prompt=(
            "floating sign, wall mounted, cheap materials, "
            "poor landscaping, dark"
        ),
        suggested_adapters=["sign_type/monument"],
    ),
    
    "storefront": PromptTemplate(
        name="storefront",
        template=(
            "Street-level photograph of {business_type} storefront with '{text}' signage, "
            "{sign_type} sign above entrance, "
            "{style} architecture, {environment}, "
            "urban photography, inviting atmosphere"
        ),
        description="Complete storefront scenes with signage",
        variables=["business_type", "text", "sign_type", "style", "environment"],
        negative_prompt=(
            "empty street, closed business, damaged, "
            "dirty windows, poor neighborhood"
        ),
        suggested_adapters=["environment/urban_storefront"],
    ),
    
    "minimal": PromptTemplate(
        name="minimal",
        template="{text} sign, {style}, high quality photograph",
        description="Minimal template for custom prompts",
        variables=["text", "style"],
        negative_prompt="blurry, low quality",
        suggested_adapters=[],
    ),
}


def get_template_library() -> dict[str, PromptTemplate]:
    """Get the template library."""
    return TEMPLATE_LIBRARY.copy()


def get_template(name: str) -> Optional[PromptTemplate]:
    """Get a template by name."""
    return TEMPLATE_LIBRARY.get(name)


def list_templates() -> list[dict]:
    """List all available templates."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "variables": t.variables,
            "suggested_adapters": t.suggested_adapters,
        }
        for t in TEMPLATE_LIBRARY.values()
    ]


# Common prompt elements
SIGN_MATERIALS = [
    "brushed aluminum",
    "polished stainless steel",
    "painted metal",
    "acrylic",
    "wood",
    "bronze",
    "copper",
    "chrome",
]

LIGHTING_STYLES = [
    "front-lit LED",
    "halo-lit",
    "backlit",
    "edge-lit",
    "neon tube",
    "warm white",
    "cool white",
    "RGB LED",
]

MOUNTING_STYLES = [
    "flush mounted",
    "raceway mounted",
    "standoff mounted",
    "suspended hanging",
    "pole mounted",
]

PERSPECTIVES = [
    "straight-on front",
    "45-degree angle",
    "street level looking up",
    "wide establishing shot",
    "close-up detail",
]

ENVIRONMENTS = [
    "busy urban street",
    "upscale shopping district",
    "indoor mall corridor",
    "office building lobby",
    "suburban shopping center",
    "historic downtown",
]


def get_prompt_vocabulary() -> dict[str, list[str]]:
    """Get vocabulary for prompt building."""
    return {
        "materials": SIGN_MATERIALS,
        "lighting": LIGHTING_STYLES,
        "mounting": MOUNTING_STYLES,
        "perspectives": PERSPECTIVES,
        "environments": ENVIRONMENTS,
    }
