"""Single-objective ALNS: exactly one complete candidate per iteration."""
import hashlib
import math
import time
from collections import Counter
import numpy as np
from .problem import route_time,_validate_route,integer_ids,digest
from .config import SearchConfig

def rng_for(instance,seed,iteration,role,config):
    # Component draws cannot shift repair, selection, or acceptance random streams.
    value=f'{config.rng_namespace}|{instance.name}|{seed}|{iteration}|{role}'
    derived=int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'big')
    return np.random.default_rng(derived)

def validate_destroy(parent,kept,removed,q):
    kept=integer_ids(kept); removed=integer_ids(removed)
    if len(removed)!=q or len(set(removed))!=q or 0 in removed:
        raise ValueError('Destroy must remove q different non-depot tasks')
    if set(kept)&set(removed) or sorted((*kept,*removed))!=sorted(parent):
        raise ValueError('Destroy partition mismatch')
    if kept!=tuple(x for x in parent if x not in set(removed)):
        raise ValueError('Destroy must preserve unremoved tour order')
    return kept,removed

def solve(instance,seed,config,components,trace_path=None,on_candidate=None):
    if not isinstance(config,SearchConfig):
        raise TypeError('Explicit SearchConfig is required')
    if components.config!=config:
        raise ValueError('Components and solver must use the same SearchConfig')
    budget=config.iterations
    wall=time.perf_counter(); cpu=time.process_time()
    route=_validate_route(components.initial(instance),instance)
    initial=route; initial_time=route_time(route,instance)
    current_time=best_time=initial_time; best=route
    dw=list(config.initial_destroy_weights); rw=list(config.initial_repair_weights); stagnation=0; recent=[]
    uses=Counter(); q_counts=Counter(); improved_count=accepted_count=changed_count=0
    records=[]
    trace=trace_path.open('x',encoding='utf-8') if trace_path else None
    try:
        for i in range(budget):
            if on_candidate: on_candidate()  # Persist reservation before any candidate work.
            state={'iteration':i,'max_iterations':budget,'n':instance.n,'progress':i/max(1,budget-1),
                   'stagnation':stagnation,'recent_improvement':sum(r[0] for r in recent)/max(1,len(recent)),
                   'recent_improvement_rate':sum(r[1] for r in recent)/max(1,len(recent)),
                   'recent_acceptance_rate':sum(r[2] for r in recent)/max(1,len(recent)),
                   'current_time':current_time,'best_time':best_time,'initial_time':initial_time}
            d,r=components.selection(dw,rw,rng_for(instance,seed,i,'selection',config))
            state['destroy_operator']=d; state['repair_operator']=r
            q=components.degree(dict(state),rng_for(instance,seed,i,'degree',config))
            if isinstance(q,bool) or not isinstance(q,int) or not 1<=q<instance.n:
                raise ValueError('Direct-Q must return an integer in [1,N-1]')
            parent=route; parent_time=current_time; prior_best=best_time
            kept,removed=components.destroy(list(parent),q,instance.T,dict(state),rng_for(instance,seed,i,'destroy',config))
            kept,removed=validate_destroy(parent,kept,removed,q)
            candidate,repair_trace=components.repair(kept,removed,r,instance,rng_for(instance,seed,i,'repair',config))
            candidate=_validate_route(candidate,instance); cost=route_time(candidate,instance)
            # Independently accumulate every edge in its tour order as a second objective check.
            check=sum(instance.distance(a,b) for a,b in zip((0,*candidate),(*candidate,0)))
            if not math.isclose(cost,check,rel_tol=config.objective_relative_tolerance,
                                abs_tol=config.tour_sum_absolute_tolerance): raise ValueError('Tour objective sum')
            improved=cost<best_time-config.objective_improvement_tolerance
            if cost<best_time or (cost==best_time and candidate<best): best,best_time=candidate,cost
            temperature=max(config.minimum_temperature, initial_time*config.initial_temperature_fraction *
                            (config.temperature_decay ** (i/max(1,budget-1))))
            accepted=components.accept(parent_time,cost,temperature,float(rng_for(instance,seed,i,'acceptance',config).random()))
            changed=candidate!=parent
            if accepted: route,current_time=candidate,cost
            reward=(config.reward_best if improved else config.reward_improving
                    if cost<parent_time-config.objective_improvement_tolerance
                    else config.reward_accepted_changed if accepted and changed else config.reward_other)
            components.update(dw,rw,d,r,reward)
            stagnation=0 if improved else stagnation+1
            recent.append((max(0.,prior_best-best_time)/initial_time,bool(improved),bool(accepted and changed)))
            recent=recent[-config.recent_window:]
            uses[f'D{d}']+=1; uses[f'R{r+1}']+=1; q_counts[q]+=1
            improved_count+=improved; accepted_count+=accepted and changed; changed_count+=changed
            row={'iteration':i+1,'state':state,'q':q,'destroy_slot':d,'repair':f'R{r+1}',
                 'parent_route':list(parent),'kept':list(kept),'removed':list(removed),
                 'candidate_route':list(candidate),'candidate_time':cost,'parent_time':parent_time,
                 'best_time':best_time,'best_route':list(best),'accepted':bool(accepted),
                 'improved_best':bool(improved),'changed_route':bool(changed),
                 'repair_insertions':len(repair_trace),'temperature':temperature}
            if len(repair_trace)!=q: raise ValueError('Repair insertion count')
            if trace:
                import json
                trace.write(json.dumps(row,separators=(',',':'))+'\n')
            else: records.append(row)
        _validate_route(best,instance)
        if not math.isclose(route_time(best,instance),best_time,rel_tol=config.objective_relative_tolerance,
                            abs_tol=config.best_sum_absolute_tolerance): raise ValueError('Best update error')
        return {'instance':instance.name,'seed':seed,'budget':budget,'evaluations':budget,
                'initial_route':list(initial),'initial_time':initial_time,'initial_hash':digest(list(initial)),
                'best_route':list(best),'best_time':best_time,'operator_calls':dict(uses),
                'q_histogram':dict(sorted(q_counts.items())),'best_improvements':int(improved_count),
                'accepted_changed':int(accepted_count),'changed_candidates':int(changed_count),
                'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu,
                'initialization_evaluations':1,'records':records}
    finally:
        if trace: trace.close()
