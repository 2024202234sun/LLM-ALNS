"""Eight-neighbor grid geometry with deterministic shortest paths."""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from typing import Iterable
import numpy as np
SQRT2=sqrt(2.0)
NEIGHBORS=((-1,0,1.),(-1,1,SQRT2),(0,1,1.),(1,1,SQRT2),(1,0,1.),(1,-1,SQRT2),(0,-1,1.),(-1,-1,SQRT2))
@dataclass(frozen=True)
class GridMap:
    blocked: np.ndarray
    base: tuple[int,int]
    @property
    def shape(self): return self.blocked.shape

def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)

def _neighbors(grid: GridMap, cell: tuple[int, int]) -> Iterable[tuple[tuple[int, int], float, int]]:
    r, c = cell
    h, w = grid.shape
    for order, (dr, dc, cost) in enumerate(NEIGHBORS):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or grid.blocked[nr, nc]:
            continue
        if dr and dc and (grid.blocked[r, nc] or grid.blocked[nr, c]):
            continue
        yield (nr, nc), cost, order

def astar(grid: GridMap, start: tuple[int, int], goal: tuple[int, int]) -> tuple[float, list[tuple[int, int]]]:
    if grid.blocked[start] or grid.blocked[goal]:
        return float("inf"), []
    if start == goal:
        return 0.0, [start]
    heap: list[tuple[float, float, int, int, int, tuple[int, int]]] = []
    heappush(heap, (octile(start, goal), 0.0, start[0], start[1], -1, start))
    g_score = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while heap:
        _, g, _, _, _, cell = heappop(heap)
        if g > g_score.get(cell, float("inf")) + 1e-12:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return g, path
        for nxt, cost, order in _neighbors(grid, cell):
            candidate = g + cost
            old = g_score.get(nxt, float("inf"))
            if candidate < old - 1e-12:
                g_score[nxt] = candidate
                parent[nxt] = cell
                heappush(heap, (candidate + octile(nxt, goal), candidate, nxt[0], nxt[1], order, nxt))
    return float("inf"), []

def dijkstra(grid: GridMap, start: tuple[int, int], goal: tuple[int, int]) -> tuple[float, list[tuple[int, int]]]:
    if grid.blocked[start] or grid.blocked[goal]:
        return float("inf"), []
    heap: list[tuple[float, int, int, int, tuple[int, int]]] = [(0.0, start[0], start[1], -1, start)]
    dist = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while heap:
        g, _, _, _, cell = heappop(heap)
        if g > dist.get(cell, float("inf")) + 1e-12:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return g, list(reversed(path))
        for nxt, cost, order in _neighbors(grid, cell):
            candidate = g + cost
            if candidate < dist.get(nxt, float("inf")) - 1e-12:
                dist[nxt] = candidate
                parent[nxt] = cell
                heappush(heap, (candidate, nxt[0], nxt[1], order, nxt))
    return float("inf"), []
