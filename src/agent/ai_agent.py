"""Step 5 spectrum-management agent with perception, diagnosis, plan, and actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .action_generator import ActionGenerator, NetworkActions
from .environment_perception import EnvironmentPerception, EnvironmentSnapshot
from .memory_rag import default_knowledge
from .reasoning import Diagnosis, ReasoningEngine
from src.tools.wireless_tools import ToolRegistry, default_tool_registry


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]


class AIAgent:
    """Plan explainable spectrum actions from a processed network observation."""

    def __init__(self, agent_type: str = "llm", memory_size: int = 1000,
                 trained_models: dict[str, Any] | None = None,
                 agent_config: dict[str, Any] | None = None) -> None:
        if agent_type not in {"llm", "rl", "hybrid"}:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        self.agent_type = agent_type
        self.memory_size = memory_size
        self.trained_models = trained_models or {}
        self.config = agent_config or {}
        self.perception = EnvironmentPerception()
        self.reasoning = ReasoningEngine(default_knowledge())
        self.action_generator = ActionGenerator()
        self.tools: ToolRegistry = default_tool_registry()

    def plan(self, diagnosis: Diagnosis) -> list[ToolCall]:
        return [
            ToolCall("estimate_channel", {"method": "pilot_based"}),
            ToolCall("detect_jamming", {"sensitivity": "high"}),
            ToolCall("allocate_power", {"method": diagnosis.recommended_strategy}),
            ToolCall("allocate_spectrum", {"fairness": "proportional"}),
            ToolCall("reconfigure_beam", {"mode": "adaptive"}),
        ]

    def decide(self, snapshot: EnvironmentSnapshot) -> dict[str, Any]:
        diagnosis = self.reasoning.reason(snapshot.state_vector)
        plan = self.plan(diagnosis)
        tool_results: dict[str, Any] = {}
        for call in plan:
            tool_results[call.tool] = self.tools.execute(call.tool, snapshot.state_vector, **call.arguments)
        allocation = tool_results["allocate_power"]["allocation"]
        spectrum = tool_results["allocate_spectrum"]["user_per_subcarrier"]
        beam_mode = tool_results["reconfigure_beam"]["mode"]
        actions = NetworkActions(diagnosis.recommended_strategy, allocation, spectrum, beam_mode)
        return {
            "scenario_id": snapshot.scenario_id,
            "diagnosis": diagnosis.to_dict(),
            "plan": [asdict(call) for call in plan],
            "actions": actions.summary(),
            "policy_evaluation": self.tools.execute("evaluate_policy", snapshot.state_vector, action=allocation),
        }

    def run(self, scenario_index: int = 0, processed_path: str = "data/processed/learning_states.npz") -> dict[str, Any]:
        return self.decide(self.perception.observe(scenario_index, processed_path))


def build_agent(agent_type: str = "llm", memory_size: int = 1000,
                trained_models: dict[str, Any] | None = None,
                agent_config: dict[str, Any] | None = None) -> AIAgent:
    """Build an agent and persist its configuration without requiring an LLM key."""
    agent = AIAgent(agent_type, memory_size, trained_models, agent_config)
    destination = Path("data/models")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "agent_config.json").write_text(json.dumps({"agent_type": agent_type, "memory_size": memory_size, "offline_fallback": True}, indent=2))
    return agent