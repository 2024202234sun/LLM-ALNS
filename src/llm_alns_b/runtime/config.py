"""Required external search configuration, with no research parameter defaults."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class SearchConfig:
    iterations: int
    rng_namespace: str
    destroy_slots: int
    repair_slots: int
    initial_destroy_weights: tuple
    initial_repair_weights: tuple
    selection_uniform_mixing: float
    weight_reaction: float
    reward_best: float
    reward_improving: float
    reward_accepted_changed: float
    reward_other: float
    initial_temperature_fraction: float
    temperature_decay: float
    minimum_temperature: float
    recent_window: int
    objective_improvement_tolerance: float
    tour_sum_absolute_tolerance: float
    best_sum_absolute_tolerance: float
    objective_relative_tolerance: float
    acceptance_exponent_minimum: float
    allowed_alphas: tuple
    removal_fraction: float

    def __post_init__(self):
        for name in ("iterations", "destroy_slots", "repair_slots", "recent_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.repair_slots > 4:
            raise ValueError("repair_slots exceeds the supported R1--R4 interfaces")
        if not isinstance(self.rng_namespace, str) or not self.rng_namespace:
            raise ValueError("rng_namespace must be a nonempty string")
        for name, count in (("initial_destroy_weights", self.destroy_slots),
                            ("initial_repair_weights", self.repair_slots)):
            values = self._numbers(name, getattr(self, name))
            if len(values) != count or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain one positive weight per slot")
            object.__setattr__(self, name, values)
        alphas = self._numbers("allowed_alphas", self.allowed_alphas)
        if not alphas or any(value <= 0 for value in alphas) or len(set(alphas)) != len(alphas):
            raise ValueError("allowed_alphas must contain unique positive numbers")
        object.__setattr__(self, "allowed_alphas", alphas)
        names = {field.name for field in fields(self)} - {
            "iterations", "rng_namespace", "destroy_slots", "repair_slots", "recent_window",
            "initial_destroy_weights", "initial_repair_weights", "allowed_alphas",
        }
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("selection_uniform_mixing", "weight_reaction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("reward_best", "reward_improving", "reward_accepted_changed", "reward_other",
                     "initial_temperature_fraction", "minimum_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("temperature_decay", "removal_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be greater than zero and at most one")
        for name in ("objective_improvement_tolerance", "tour_sum_absolute_tolerance",
                     "best_sum_absolute_tolerance", "objective_relative_tolerance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.acceptance_exponent_minimum >= 0:
            raise ValueError("acceptance_exponent_minimum must be negative")

    @staticmethod
    def _numbers(name, values):
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name} must be an array of finite numbers")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            raise ValueError(f"{name} must be an array of finite numbers")
        return tuple(float(value) for value in values)

    @classmethod
    def from_dict(cls, values):
        """Reject unknown and missing fields instead of filling implicit values."""
        if not isinstance(values, dict):
            raise ValueError("Search configuration must be an object")
        required = {field.name for field in fields(cls)}
        missing, unknown = required - values.keys(), values.keys() - required
        if missing or unknown:
            raise ValueError(f"Search configuration fields differ: missing={sorted(missing)}, unknown={sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        for name in ("initial_destroy_weights", "initial_repair_weights", "allowed_alphas"):
            values[name] = list(values[name])
        return values
