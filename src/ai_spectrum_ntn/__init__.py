"""Installable package marker for the AI Spectrum NTN project."""

__version__ = "0.1.0""""
AI Spectrum Management for Non-terrestrial Networks (NTN)
A comprehensive framework for 6G dynamic spectrum allocation using AI agents,
adversarial game theory, and autonomous optimization.

Integrates three core research frameworks:
- RAG-Practice: Multi-modal retrieval-augmented generation for LLM reasoning
- AutoResearch: Autonomous AI-driven experiment loop for continuous improvement
- NemoIR: Compiler framework for optimizing agent workflows

Pipeline:
1. Dataset Generation → 2. Data Processing → 3. Ground-Truth Optimization
→ 4. Model Training → 5. AI Agent Construction → 6. Tool Layer
→ 7. Workflow Compilation → 8. Evaluation → 9. Online Execution → 10. AutoResearch
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__description__ = "AI Agentic Spectrum Management for NTN"

# Import main entry points
from .pipeline import SpectrumManagementPipeline

__all__ = [
    'SpectrumManagementPipeline',
    '__version__',
    '__author__',
]
