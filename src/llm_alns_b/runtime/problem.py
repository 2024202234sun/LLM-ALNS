"""Single depot tour, normalized travel time; task service and transit differ."""
import hashlib
import json
import math
from dataclasses import dataclass
import numpy as np


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()

def integer_ids(route):
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in route):
        raise ValueError('Route IDs must be integers')
    return tuple(int(x) for x in route)

def _validate_route(route, context):
    route = integer_ids(route)
    if sorted(route) != list(context.node_ids[1:]):
        raise ValueError('Every task must be served exactly once; depot is implicit')
    return route

@dataclass
class Instance:
    name: str
    T: np.ndarray
    blocked: np.ndarray
    cells: list
    manifest: dict
    @property
    def n(self): return len(self.T) - 1
    @property
    def node_ids(self): return tuple(range(self.n + 1))
    def distance(self, a, b): return float(self.T[a, b])

def route_time(route, instance, full=True):
    route = _validate_route(route, instance) if full else integer_ids(route)
    return float(math.fsum(instance.distance(a, b) for a, b in zip((0, *route), (*route, 0))))

def nearest_neighbor(instance):
    # Resolve equal distances by node ID.
    pending = set(instance.node_ids[1:]); route = []; node = 0
    while pending:
        node = min(pending, key=lambda x: (instance.distance(node, x), x))
        route.append(node); pending.remove(node)
    return tuple(route)
