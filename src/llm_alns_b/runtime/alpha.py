"""Map a safely interpreted alpha policy to an externally configured removal count."""
import math
import time
from .config import SearchConfig
from .safe_program import ProgramError, SafeProgram


def alpha_count(n, alpha, config):
    if not isinstance(config, SearchConfig):
        raise TypeError("Explicit SearchConfig is required")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha not in config.allowed_alphas:
        raise ProgramError("Q must return one exact externally allowed alpha")
    return max(1, min(n - 1, math.floor(config.removal_fraction * n * alpha + .5)))


class AlphaProgram:
    def __init__(self, code, config):
        if not isinstance(config, SearchConfig):
            raise TypeError("Explicit SearchConfig is required")
        self.config = config
        self.program = SafeProgram(code, "Q")
        self.calls = 0
        self.seconds = 0.

    def __call__(self, state, rng):
        start = time.perf_counter()
        try:
            return alpha_count(state["n"], self.program(state, rng), self.config)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - start
