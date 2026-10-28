"""Search interfaces with explicitly supplied policies and numerical settings."""
from functools import partial
import math
import numpy as np
from .config import SearchConfig
from .problem import nearest_neighbor
from .primitives import PartialRoute
from .batch_repair import repair as batch_repair
from .safe_program import SafeProgram
from .alpha import AlphaProgram


def repair(kept, removed, operator, instance, rng):
    """Adapt the route-level interface to the fixed R1--R4 insertion rules."""
    class RNGAdapter:
        def choice(self, x):
            return rng.choice(x)

    route, trace = batch_repair(
        PartialRoute(tuple(kept), tuple(removed)),
        f"R{operator + 1}", instance, RNGAdapter(),
    )
    return tuple(route), trace


def selection(dw, rw, rng, *, config):
    def draw(weights):
        mix = config.selection_uniform_mixing
        probabilities = (1 - mix) * np.asarray(weights) / sum(weights) + mix / len(weights)
        return int(rng.choice(len(weights), p=probabilities))
    return draw(dw), draw(rw)


def weight_update(dw, rw, d, r, reward, *, config):
    reaction = config.weight_reaction
    dw[d] = (1 - reaction) * dw[d] + reaction * reward
    rw[r] = (1 - reaction) * rw[r] + reaction * reward


def acceptance(current, candidate, temperature, u, *, config):
    return candidate <= current or u < math.exp(max(config.acceptance_exponent_minimum,
                                                    (current - candidate) / temperature))


class Components:
    """Bind explicit policies and fixed interfaces to one SearchConfig.

    Source strings are interpreted safely. A Q string produces an alpha value;
    an already constructed Q callable must produce the actual removal count.
    """

    def __init__(self, D, Q, config, *, initial=nearest_neighbor, repair=repair):
        if not isinstance(config, SearchConfig):
            raise TypeError("Explicit SearchConfig is required")
        if isinstance(D, str):
            D = SafeProgram(D, "D")
        if isinstance(Q, str):
            Q = AlphaProgram(Q, config)
        if not callable(D) or not callable(Q):
            raise TypeError("Explicit callable D and Q policies are required")
        if isinstance(Q, AlphaProgram) and Q.config != config:
            raise ValueError("Q and search components must use the same SearchConfig")
        self.config = config
        self.initial = initial
        self.destroy = D
        self.degree = Q
        self.repair = repair
        self.selection = partial(selection, config=config)
        self.update = partial(weight_update, config=config)
        self.accept = partial(acceptance, config=config)
