"""Mechanical actuator simulation for material sorting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..core.types import MaterialCategory


@dataclass
class ActuatorState:
    position: float
    active: bool
    cycle_time_ms: float


class ActuatorBank:
    """Tracks the state of actuators controlling material deflection."""

    def __init__(self) -> None:
        self._states: Dict[str, ActuatorState] = {}

    def register(self, actuator_id: str, cycle_time_ms: float) -> None:
        self._states[actuator_id] = ActuatorState(position=0.0, active=False, cycle_time_ms=cycle_time_ms)

    def trigger(self, actuator_id: str, direction: float) -> None:
        state = self._states[actuator_id]
        state.active = True
        state.position = max(-1.0, min(1.0, direction))

    def reset(self, actuator_id: str) -> None:
        state = self._states[actuator_id]
        state.active = False
        state.position = 0.0

    def status(self) -> Dict[str, ActuatorState]:
        return self._states


def default_actuator_bank() -> ActuatorBank:
    bank = ActuatorBank()
    for actuator in ("air_jet", "pusher", "electrostatic", "ballistic"):
        bank.register(actuator, cycle_time_ms=8.0)
    return bank
