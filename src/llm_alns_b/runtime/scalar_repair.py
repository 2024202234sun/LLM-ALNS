"""Exact finite Top-k indices; every insertion delta is still evaluated.

Finite per-node tuple order is (delta, fixed_node, ascending_gap). Repeated
first argmin with selected entries set to +inf returns exactly sorted's first
k entries, including equal values and signed zero. Non-finite deltas retain
the full tuple ordering. R4 retains its original choice API and draw location.
"""
import numpy as np
from .problem import integer_ids, _validate_route
UNITS={"grid_length_units_per_time_unit":1.0}

def _finite_topk(deltas, node, k):
    work=deltas.copy()
    ordered=[]
    for unused in range(min(k,len(deltas))):
        gap=int(work.argmin())
        ordered.append((float(deltas[gap]),node,gap))
        work[gap]=np.inf
    return ordered

def repair(partial, name, context, rng, *, compact_trace=True):
    if name not in ('R1','R2','R3','R4'):
        raise ValueError('illegal repair')
    current=integer_ids(partial.kept);pending=list(integer_ids(partial.removed))
    _validate_route((*current,*pending),context)
    trace=[]
    while pending:
        left=np.asarray((0,*current));right=np.asarray((*current,0))
        edge_cost=context.T[left,right]
        per_node=[];node_arrays={};minima=[];all_finite=True
        is_regret=name in ('R2','R3')
        for node in pending:
            deltas=context.T[left,node]+context.T[node,right]-edge_cost
            finite=bool(np.isfinite(deltas).all())
            if is_regret:
                k=2 if name=='R2' else 3
                ordered=(_finite_topk(deltas,node,k) if finite else
                         sorted((float(delta),node,gap) for gap,delta in enumerate(deltas)))
                best=ordered[0]
                value=sum(x[0]-best[0] for x in ordered[1:k])
                per_node.append((-value,best[0],node,best[2],best))
            else:
                node_arrays[node]=deltas
                all_finite=all_finite and finite
                if finite:
                    gap=int(deltas.argmin())
                    minima.append((float(deltas[gap]),node,gap))
        eligible=None;regret=None
        if is_regret:
            selected=min(per_node)
            chosen=selected[-1]
            regret=-selected[0]
        elif all_finite:
            chosen=min(minima)
            if name=='R4':
                node=chosen[1]
                eligible=_finite_topk(node_arrays[node],node,3)
                chosen=eligible[int(rng.choice(np.arange(len(eligible))))]
        else:
            # Preserve Python's original non-total NaN comparison semantics.
            options=[];by_node={}
            for node in pending:
                local=[(float(delta),node,gap) for gap,delta in enumerate(node_arrays[node])]
                options.extend(local);by_node[node]=local
            chosen=min(options)
            if name=='R4':
                node=chosen[1];eligible=sorted(by_node[node])[:3]
                chosen=eligible[int(rng.choice(np.arange(len(eligible))))]
        delta,node,gap=chosen
        current=current[:gap]+(node,)+current[gap:]
        pending.remove(node)
        trace.append({'repair':name,'node':node,'gap':gap,'delta_distance':delta,
                      'delta_time':delta/UNITS['grid_length_units_per_time_unit'],
                      'tie_break':'-regret,delta,node,gap' if name in ('R2','R3') else 'delta,node,gap',
                      'regret':regret,'eligible_top3':eligible})
    _validate_route(current,context)
    return current,trace
