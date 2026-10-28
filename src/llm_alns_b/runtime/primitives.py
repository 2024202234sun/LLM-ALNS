"""Trusted route-level removals. Each call changes the partial tour immediately."""
from dataclasses import dataclass
import numpy as np
from .problem import integer_ids

@dataclass(frozen=True)
class PartialRoute:
    kept: tuple
    removed: tuple

def checked(route, q):
    route = list(integer_ids(route))
    if not route or 0 in route or len(set(route)) != len(route): raise ValueError('Invalid removal route')
    if isinstance(q, bool) or not isinstance(q, int) or not 1 <= q <= len(route):
        raise ValueError('Removal size outside remaining tasks')
    return route

def partition(route, removed):
    selected = set(removed)
    return [node for node in route if node not in selected], list(removed)

def savings(route, dist):
    # Savings in the depot-closed tour, using the distance-matrix interface.
    return [(float(dist[0 if i == 0 else route[i-1], node] +
                   dist[node, 0 if i+1 == len(route) else route[i+1]] -
                   dist[0 if i == 0 else route[i-1], 0 if i+1 == len(route) else route[i+1]]), node)
            for i, node in enumerate(route)]

def biased_pick(ordered, rng, power):
    if not isinstance(power,(int,float)) or not .25 <= power <= 8: raise ValueError('power in [.25,8]')
    return ordered[min(len(ordered)-1, int(rng.random() ** power * len(ordered)))]

def random_remove(route, q, dist, rng):
    route = checked(route, q)
    removed = [route[int(i)] for i in rng.choice(len(route), size=q, replace=False)]
    return partition(route, removed)

def worst_remove(route, q, dist, rng, power, recompute):
    original = checked(route,q); remaining = original[:]; removed = []
    order = [node for _,node in sorted(savings(remaining,dist), key=lambda x:(-x[0],x[1]))]
    for _ in range(q):
        if recompute:
            order = [node for _,node in sorted(savings(remaining,dist), key=lambda x:(-x[0],x[1]))]
        node = biased_pick(order,rng,power)
        removed.append(node); remaining.remove(node)
        if not recompute: order.remove(node)
    return partition(original, removed)

def seed_node(route, dist, rng, anchor):
    if anchor == 'random': return route[int(rng.integers(len(route)))]
    if anchor == 'worst': return min(savings(route,dist), key=lambda x:(-x[0],x[1]))[1]
    if anchor == 'long_edge':
        i = max(range(len(route)), key=lambda i:(dist[0 if i==0 else route[i-1],route[i]],-i))
        return route[i]
    raise ValueError('anchor must be random/worst/long_edge')

def related_remove(route, q, dist, rng, power, position_weight, anchor, pivot):
    original = checked(route,q); n=len(original); positions={node:i for i,node in enumerate(original)}
    if not 0 <= position_weight <= 3: raise ValueError('position_weight in [0,3]')
    removed=[seed_node(original,dist,rng,anchor)]
    scale=float(np.mean(dist)) / max(1,n)
    for _ in range(q-1):
        if pivot=='random': center=removed[int(rng.integers(len(removed)))]
        elif pivot=='last': center=removed[-1]
        elif pivot=='seed': center=removed[0]
        else: raise ValueError('pivot must be random/last/seed')
        remaining=[node for node in original if node not in removed]
        def key(node):
            delta=abs(positions[node]-positions[center]); delta=min(delta,n-delta)
            return (dist[center,node]+position_weight*scale*delta,node)
        removed.append(biased_pick(sorted(remaining,key=key),rng,power))
    return partition(original,removed)

def cluster_remove(route, q, dist, rng, power, anchor):
    original=checked(route,q); center=seed_node(original,dist,rng,anchor)
    remaining=sorted(original,key=lambda node:(dist[center,node],node)); removed=[]
    for _ in range(q):
        node=biased_pick(remaining,rng,power); remaining.remove(node); removed.append(node)
    return partition(original,removed)

def chain_remove(route, q, dist, rng, anchor, direction):
    original=checked(route,q); n=len(original); center=original.index(seed_node(original,dist,rng,anchor))
    if direction=='forward': offsets=list(range(q))
    elif direction=='backward': offsets=[-i for i in range(q)]
    elif direction=='both': offsets=[0]+[v for i in range(1,q+1) for v in (i,-i)]
    else: raise ValueError('direction must be forward/backward/both')
    removed=[]
    for offset in offsets:
        node=original[(center+offset)%n]
        if node not in removed: removed.append(node)
        if len(removed)==q: break
    return partition(original,removed)

PRIMITIVES = {f.__name__:f for f in (random_remove,worst_remove,related_remove,cluster_remove,chain_remove)}
