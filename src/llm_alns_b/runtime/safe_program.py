"""Small interpreter for a bounded Python subset. Never compile/eval/exec LLM code."""
import ast
import math
import operator
from .primitives import PRIMITIVES

class ProgramError(ValueError): pass
class Returned(Exception):
    def __init__(self, value): self.value=value

BIN = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod, ast.Pow:operator.pow}
CMP = {ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt, ast.LtE:operator.le,
       ast.Gt:operator.gt, ast.GtE:operator.ge}

def randint(rng, low, high):
    if not isinstance(low,int) or not isinstance(high,int) or not 0 <= low <= high <= 1000000:
        raise ProgramError('randint bounds')
    return int(rng.integers(low, high+1))

def uniform(rng, low, high):
    if not -1000000 <= low <= high <= 1000000: raise ProgramError('uniform bounds')
    return float(rng.uniform(low,high))

FUNCTIONS = {**PRIMITIVES, 'len':len, 'int':int, 'round':round, 'min':min, 'max':max, 'abs':abs,
             'sqrt':math.sqrt, 'log':math.log, 'cos':math.cos, 'sin':math.sin,
             'randint':randint, 'uniform':uniform}
ALLOWED = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.Return,ast.If,
           ast.Name,ast.Load,ast.Store,ast.Constant,ast.Tuple,ast.List,ast.Subscript,
           ast.Call,ast.keyword,ast.BinOp,ast.UnaryOp,ast.UAdd,ast.USub,ast.Not,
           ast.BoolOp,ast.And,ast.Or,ast.Compare,ast.IfExp,*BIN.keys(),*CMP.keys())

class SafeProgram:
    def __init__(self, source, kind):
        if kind not in ('D','Q') or not isinstance(source,str) or len(source)>10000:
            raise ProgramError('Invalid component/source size')
        self.source=source; self.kind=kind
        try: tree=ast.parse(source)
        except SyntaxError as e: raise ProgramError('Syntax error') from e
        name='destroy' if kind=='D' else 'degree'
        params=['route','q','dist','state','rng'] if kind=='D' else ['state','rng']
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
            raise ProgramError('Exactly one function, no top-level instructions')
        func=tree.body[0]; args=func.args
        if (func.name!=name or func.decorator_list or func.returns or args.defaults or args.kw_defaults or
            args.vararg or args.kwarg or args.posonlyargs or args.kwonlyargs or
            [a.arg for a in args.args]!=params or any(a.annotation for a in args.args)):
            raise ProgramError('Use the exact unannotated function signature')
        nodes=list(ast.walk(tree))
        if len(nodes)>600: raise ProgramError('AST too large')
        self.call_names=[]
        for node in nodes:
            if not isinstance(node,ALLOWED): raise ProgramError('Forbidden syntax: '+type(node).__name__)
            if isinstance(node,ast.Name) and ('__' in node.id or node.id.startswith('_')):
                raise ProgramError('Forbidden identifier')
            if isinstance(node,ast.Constant):
                if not isinstance(node.value,(int,float,str,bool,type(None))): raise ProgramError('Invalid literal')
                if isinstance(node.value,str) and len(node.value)>120: raise ProgramError('String too long')
                if isinstance(node.value,(int,float)) and (not math.isfinite(node.value) or abs(node.value)>1e9):
                    raise ProgramError('Numeric literal bound')
            if isinstance(node,ast.Call):
                if not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS:
                    raise ProgramError('Only trusted direct calls are permitted')
                if kind=='Q' and node.func.id in PRIMITIVES: raise ProgramError('Q cannot modify routes')
                if len(node.args)+len(node.keywords)>12 or any(k.arg is None for k in node.keywords):
                    raise ProgramError('No call expansion')
                self.call_names.append(node.func.id)
            if isinstance(node,ast.Assign):
                if len(node.targets)!=1: raise ProgramError('One assignment target')
                self._check_target(node.targets[0])
        self.params=params; self.body=func.body; self.tree=tree

    def _check_target(self,node):
        if isinstance(node,ast.Name):
            if node.id in FUNCTIONS or node.id=='PI': raise ProgramError('Cannot rebind trusted names')
        elif isinstance(node,(ast.Tuple,ast.List)):
            if len(node.elts)>8: raise ProgramError('Unpack size')
            for item in node.elts: self._check_target(item)
        else: raise ProgramError('No attribute/subscript assignment')

    def __call__(self,*args):
        if len(args)!=len(self.params): raise ProgramError('Argument count')
        env=dict(zip(self.params,args)); env['PI']=math.pi; steps=[0]
        try: self._block(self.body,env,steps)
        except Returned as r: return r.value
        raise ProgramError('Function did not return')

    def _tick(self,steps):
        steps[0]+=1
        if steps[0]>1000: raise ProgramError('Instruction cap')

    def _assign(self,target,value,env):
        if isinstance(target,ast.Name): env[target.id]=value
        else:
            if not isinstance(value,(tuple,list)) or len(value)!=len(target.elts): raise ProgramError('Unpack mismatch')
            for t,v in zip(target.elts,value): self._assign(t,v,env)

    def _block(self,nodes,env,steps):
        for node in nodes:
            self._tick(steps)
            if isinstance(node,ast.Assign): self._assign(node.targets[0],self._eval(node.value,env,steps),env)
            elif isinstance(node,ast.Return): raise Returned(self._eval(node.value,env,steps))
            elif isinstance(node,ast.If):
                self._block(node.body if self._eval(node.test,env,steps) else node.orelse,env,steps)
            else: raise ProgramError('Forbidden statement')

    def _eval(self,node,env,steps):
        self._tick(steps)
        get=lambda n:self._eval(n,env,steps)
        if isinstance(node,ast.Constant): return node.value
        if isinstance(node,ast.Name):
            if node.id not in env: raise ProgramError('Undefined name: '+node.id)
            return env[node.id]
        if isinstance(node,(ast.Tuple,ast.List)):
            if len(node.elts)>50: raise ProgramError('Container cap')
            values=[get(e) for e in node.elts]
            return tuple(values) if isinstance(node,ast.Tuple) else values
        if isinstance(node,ast.Subscript):
            obj=get(node.value); key=get(node.slice)
            return obj[key]
        if isinstance(node,ast.IfExp): return get(node.body) if get(node.test) else get(node.orelse)
        if isinstance(node,ast.BoolOp):
            value=get(node.values[0])
            for item in node.values[1:]:
                if isinstance(node.op,ast.And) and not value: return value
                if isinstance(node.op,ast.Or) and value: return value
                value=get(item)
            return value
        if isinstance(node,ast.UnaryOp):
            value=get(node.operand)
            if isinstance(node.op,ast.Not): return not value
            return bounded(value if isinstance(node.op,ast.UAdd) else -value)
        if isinstance(node,ast.Compare):
            left=get(node.left)
            for op,item in zip(node.ops,node.comparators):
                right=get(item)
                if not CMP[type(op)](left,right): return False
                left=right
            return True
        if isinstance(node,ast.BinOp):
            left=get(node.left); right=get(node.right)
            if isinstance(left,(list,tuple)):
                if not isinstance(node.op,ast.Add) or type(left)!=type(right) or len(left)+len(right)>100:
                    raise ProgramError('Only bounded list/tuple concatenation')
                return left+right
            if not isinstance(left,(int,float)) or not isinstance(right,(int,float)):
                raise ProgramError('Arithmetic requires scalar numbers')
            if isinstance(node.op,ast.Pow) and not -8 <= right <= 8: raise ProgramError('Exponent cap')
            return bounded(BIN[type(node.op)](left,right))
        if isinstance(node,ast.Call):
            value=FUNCTIONS[node.func.id](*[get(a) for a in node.args],**{k.arg:get(k.value) for k in node.keywords})
            return bounded(value)
        raise ProgramError('Forbidden expression')

def bounded(value):
    if isinstance(value,(float,int)) and (not math.isfinite(value) or abs(value)>1e9):
        raise ProgramError('Arithmetic result bound')
    return value

