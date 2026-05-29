"""Text encoder: Nomic-compatible interface + projection into d_model space."""

from encoder.nomic_encoder import NomicTextEncoder
from encoder.projection import EncoderProjection

__all__ = ["NomicTextEncoder", "EncoderProjection"]
