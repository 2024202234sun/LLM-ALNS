"""Exact per-call node/gap delta cache with split-edge updates.

Immutable finite float64 matrices use the cache branch. Other interfaces
delegate to the scalar implementation before any RNG consumption.
"""
import numpy as np
from .problem import integer_ids,_validate_route
from .scalar_repair import repair as scalar_fallback, _finite_topk
UNITS={"grid_length_units_per_time_unit":1.0}

def initial_cache(T,current,pending):
    left=np.asarray((0,*current));right=np.asarray((*current,0));nodes=np.asarray(pending)
    # Exact scalar operation order: add the two distances, then subtract edge cost.
    return T[left[None,:],nodes[:,None]]+T[nodes[:,None],right[None,:]]-T[left,right][None,:]

def split_cache(T,cache,selected_index,gap,inserted,left_node,right_node,pending):
    active=np.delete(cache,selected_index,axis=0)
    nodes=np.asarray(pending);left=np.asarray([left_node,inserted]);right=np.asarray([inserted,right_node])
    two=T[left[None,:],nodes[:,None]]+T[nodes[:,None],right[None,:]]-T[left,right][None,:]
    updated=np.empty((len(pending),cache.shape[1]+1),dtype=np.float64)
    updated[:,:gap]=active[:,:gap]
    updated[:,gap:gap+2]=two
    updated[:,gap+2:]=active[:,gap+1:]
    return updated

def repair(partial,name,context,rng,*,compact_trace=True):
    if name not in ('R1','R2','R3','R4'):raise ValueError('illegal repair')
    T=context.T
    if (type(T) is not np.ndarray or T.dtype!=np.float64 or T.flags.writeable
        or not np.isfinite(T).all() or np.max(np.abs(T))>np.finfo(np.float64).max/4):
        return scalar_fallback(partial,name,context,rng,compact_trace=compact_trace)
    current=integer_ids(partial.kept);pending=list(integer_ids(partial.removed))
    _validate_route((*current,*pending),context)
    trace=[]
    if not pending:return current,trace
    cache=initial_cache(T,current,pending)
    while pending:
        per_node=[];minima=[];is_regret=name in ('R2','R3')
        for ni,node in enumerate(pending):
            deltas=cache[ni]
            if is_regret:
                k=2 if name=='R2' else 3;ordered=_finite_topk(deltas,node,k);best=ordered[0]
                value=sum(x[0]-best[0] for x in ordered[1:k])
                per_node.append((-value,best[0],node,best[2],best))
            else:
                gap=int(deltas.argmin());minima.append((float(deltas[gap]),node,gap))
        eligible=None;regret=None
        if is_regret:
            selected=min(per_node);chosen=selected[-1];regret=-selected[0]
        else:
            chosen=min(minima)
            if name=='R4':
                node=chosen[1];eligible=_finite_topk(cache[pending.index(node)],node,3)
                chosen=eligible[int(rng.choice(np.arange(len(eligible))))]
        delta,node,gap=chosen
        selected_index=pending.index(node)
        left_node=0 if gap==0 else current[gap-1];right_node=0 if gap==len(current) else current[gap]
        current=current[:gap]+(node,)+current[gap:];pending.remove(node)
        trace.append({'repair':name,'node':node,'gap':gap,'delta_distance':delta,
                      'delta_time':delta/UNITS['grid_length_units_per_time_unit'],
                      'tie_break':'-regret,delta,node,gap' if name in ('R2','R3') else 'delta,node,gap',
                      'regret':regret,'eligible_top3':eligible})
        if pending:cache=split_cache(T,cache,selected_index,gap,node,left_node,right_node,pending)
    _validate_route(current,context)
    return current,trace
