"""Deterministic wireless tools callable by the spectrum-management agent."""

from .wireless_tools import ToolRegistry, default_tool_registry

__all__ = ["ToolRegistry", "default_tool_registry"]