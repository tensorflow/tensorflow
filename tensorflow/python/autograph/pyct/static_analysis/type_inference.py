# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Type inference.

This analysis annotates all symbols nodes of an AST with type information
extracted from static sources:
 * type annotations
 * global and local symbols visible to the function at analysis time
 * literals

Requires reaching function definitions analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Tuple

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos


class Resolver(object):
  """Resolver objects handle the process of looking up actual names and types.

  All resolve_* methods:
    * have a first namespace argument, mapping string to actual values
    * specify names as QN objects
    * specify types as a Set of inferred types

  All resolve_* methods must return either:
    * a set of `type` objects
    * None
  """

  def res_name(self, ns, name):
    """Resolves the type an external (e.g. closure, global) variable."""
    raise NotImplementedError('subclasses must implement')

  def res_value(self, ns, value):
    """Resolves the type a literal value."""
    raise NotImplementedError('subclasses must implement')

  # TODO(mdan): Allow caller to model side effects.
  def res_call(self, ns, name, target, args, keywords, starargs, kwargs):
    """Resolves the return type an external function or method call.

    Args:
      ns: namespace
      name: str, the function name
      target: if this is a method call, the types of the method target, None
          otherwise
      args: list or argument types
      keywords: dict of name to argument types
      starargs: list of types of the *args arguments (should be at most one)
      kwargs: list of types of the **kwargs arguments (in order of appearance)
    """
    raise NotImplementedError('subclasses must implement')

  def res_arg(self, ns, f_name, arg_name, type_anno):
    """Resolves the type of a (possibly annotated) function argument."""
    raise NotImplementedError('subclasses must implement')


class _SymbolTable(object):
  """Abstraction for the state of the CFG walk for type inference.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    value: Dict[qual_names.QN, Set[Type]], mapping symbols to the set of
      possible types.
  """

  def __init__(self, init_from=None):
    if init_from:
      assert isinstance(init_from, _SymbolTable)
      self.value = {
          s: set(other_types) for s, other_types in init_from.value.items()
      }
    else:
      self.value = {}

  def __eq__(self, other):
    if frozenset(self.value.keys()) != frozenset(other.value.keys()):
      return False
    ret = all(self.value[s] == other.value[s] for s in self.value)
    return ret

  def __ne__(self, other):
    return not self.__eq__(other)

  def __or__(self, other):
    assert isinstance(other, _SymbolTable)
    result = _SymbolTable(self)
    for s, other_types in other.value.items():
      if s not in result.value:
        self_types = set()
        result.value[s] = self_types
      else:
        self_types = result.value[s]
      self_types.update(other_types)
    return result

  def __repr__(self):
    return 'SymbolTable {}'.format(self.value)


_GETITEM = qual_names.QN('__getitem__')

_HANDLERS = {
    gast.Eq: qual_names.QN('__eq__'),
    gast.NotEq: qual_names.QN('__ne__'),
    gast.Lt: qual_names.QN('__lt__'),
    gast.LtE: qual_names.QN('__le__'),
    gast.Gt: qual_names.QN('__gt__'),
    gast.GtE: qual_names.QN('__ge__'),
    gast.In: qual_names.QN('__contains__'),
    # TODO(mdan): Is this actually correct?
    # NotIn(*) = Not(In(*))
    gast.NotIn: qual_names.QN('__not__'),

    gast.Add: qual_names.QN('__add__'),
    gast.Sub: qual_names.QN('__sub__'),
    gast.Mult: qual_names.QN('__mul__'),
    gast.Div: qual_names.QN('__div__'),
    gast.FloorDiv: qual_names.QN('__floordiv__'),
    gast.Mod: qual_names.QN('__mod__'),
    gast.Pow: qual_names.QN('__pow__'),
    gast.LShift: qual_names.QN('__lshift__'),
    gast.RShift: qual_names.QN('__rshift__'),
    gast.BitOr: qual_names.QN('__or__'),
    gast.BitXor: qual_names.QN('__xor__'),
    gast.BitAnd: qual_names.QN('__and__'),
    gast.MatMult: qual_names.QN('__matmul__'),
}

_FIXED_RETTYPES = {
    gast.Is: bool,
    gast.IsNot: bool,
}


class StmtInferrer(gast.NodeVisitor):
  """Runs type inference on a single AST statement.

  This visitor annotates most nodes with type information. It also sets types
  for the symbols modified by this statement in its types_out property.
  """

  def __init__(self, resolver, scope, namespace, closure_types, types_in):
    self.resolver = resolver
    self.scope = scope
    self.namespace = namespace
    self.closure_types = closure_types
    self.types_in = types_in
    self.new_symbols = {}
    self.rvalue = None

  def visit(self, node):
    types = super().visit(node)
    if types is not None:
      # TODO(mdan): Normalize by removing subtypes.
      anno.setanno(node, anno.Static.TYPES, tuple(types))
    return types

  def visit_FunctionDef(self, node):
    # Skip local function definitions. They are analyzed separately.
    return None

  def visit_Constant(self, node):
    return self.resolver.res_value(self.namespace, node.value)

  def visit_Tuple(self, node):
    if isinstance(node.ctx, gast.Load):
      for elt in node.elts:
        self.visit(elt)
      # TODO(mdan): Parameterize it.
      return {Tuple}

    assert isinstance(node.ctx, gast.Store)
    # TODO(mdan): Implement tuple unpacking.
    return None

  def visit_List(self, node):
    if isinstance(node.ctx, gast.Load):
      el_types = []
      for elt in node.elts:
        el_types.append(self.visit(elt))
      return {list}

    raise NotImplementedError('list unpacking')

  def visit_Set(self, node):
    raise NotImplementedError()

  def visit_Name(self, node):
    name = anno.getanno(node, anno.Basic.QN)
    if isinstance(node.ctx, gast.Load):
      types = self.types_in.value.get(name, None)
      if (types is None) and (name not in self.scope.bound):
        if name in self.closure_types:
          types = self.closure_types[name]
        else:
          types = self.resolver.res_name(self.namespace, name)
      return types

    elif isinstance(node.ctx, gast.Param):
      type_name = anno.getanno(node.annotation, anno.Basic.QN, None)
      types = self.resolver.res_arg(self.namespace, self.scope.function_name,
                                    name, type_name)
      if types is not None:
        self.new_symbols[name] = types
      return types

    elif isinstance(node.ctx, gast.Store):
      if self.rvalue is not None:
        self.new_symbols[name] = self.rvalue
      else:
        # No type information, assume Any.
        self.new_symbols[name] = {Any}
      return self.rvalue

    assert False, 'unknown ctx'

  def visit_Call(self, node):
    f_name = anno.getanno(node.func, anno.Basic.QN)

    kwargs = [self.visit(kw.value) for kw in node.keywords if kw.arg is None]
    keywords = {
        kw.arg: self.visit(kw.value)
        for kw in node.keywords
        if kw.arg is not None
    }
    is_starred = [isinstance(a, gast.Starred) for a in node.args]
    args = [
        self.visit(a)
        for a, starred in zip(node.args, is_starred)
        if not starred
    ]
    starargs = [
        self.visit(a.value)
        for a, starred in zip(node.args, is_starred)
        if starred
    ]

    if f_name in self.scope.bound:
      # Don't attempt external resolution of local functions.
      # TODO(mdan): Use type annotations of the local definition.
      return None

    return self.resolver.res_call(
        self.namespace, f_name, None, args, keywords, starargs, kwargs)

  def visit_Index(self, node):
    return self.visit(node.value)

  def visit_Assign(self, node):
    self.rvalue = self.visit(node.value)

    for t in node.targets:
      self.visit(t)

    self.rvalue = None

  def visit_Subscript(self, node):
    val_type = self.visit(node.value)
    slice_type = self.visit(node.slice)

    if val_type is None or slice_type is None:
      return None

    return self.resolver.res_call(self.namespace, _GETITEM, val_type,
                                  (slice_type,), {}, (), ())

  def visit_Compare(self, node):
    right_types = [self.visit(c) for c in node.comparators]
    op_types = [type(o) for o in node.ops]
    if len(op_types) > 1:
      raise NotImplementedError('chained comparisons')
    assert len(right_types) == 1

    left_type = self.visit(node.left)
    right_type, = right_types
    op_type, = op_types

    if left_type is None or right_type is None:
      return None

    f_name = _HANDLERS.get(op_type, None)
    if f_name is None:
      # Python doesn't allow overriding these operators. Their return types are
      # fixed.
      return {_FIXED_RETTYPES[op_type]}
    return self.resolver.res_call(self.namespace, _HANDLERS[op_type],
                                  left_type, (right_type,), {}, (), ())

  def visit_BinOp(self, node):
    left_type = self.visit(node.left)
    right_type = self.visit(node.right)

    if left_type is None or right_type is None:
      return None

    # TODO(mdan): This does not fully follow Python operator semantics.
    # For example, in `a + b` Python will try `a.__add__`, but also `b.__radd__`
    return self.resolver.res_call(self.namespace, _HANDLERS[type(node.op)],
                                  left_type, (right_type,), {}, (), ())


class Analyzer(cfg.GraphVisitor):
  """CFG visitor that propagates type information across statements."""

  def __init__(self, graph, resolver, namespace, scope, closure_types):
    """Creates a new analyzer.

    Args:
      graph: cfg.Graph
      resolver: Resolver
      namespace: Dict[str, Any]
      scope: activity.Scope
      closure_types: Dict[QN, Set]
    """
    super(Analyzer, self).__init__(graph)
    self.resolver = resolver
    self.namespace = namespace
    self.scope = scope
    self.closure_types = closure_types

  def init_state(self, _):
    return _SymbolTable()

  def _update_closure_types(self, ast_node, types):
    existing_types = anno.getanno(ast_node, anno.Static.CLOSURE_TYPES, None)

    if existing_types is None:
      existing_types = {}
      anno.setanno(ast_node, anno.Static.CLOSURE_TYPES, existing_types)

    for k, v in types.value.items():
      if k in existing_types:
        existing_types[k].update(v)
      else:
        existing_types[k] = set(v)

  def visit_node(self, node):
    prev_types_out = self.out[node]

    types_in = _SymbolTable()
    for n in node.prev:
      types_in |= self.out[n]

    types_out = _SymbolTable(types_in)
    ast_node = node.ast_node

    inferrer = StmtInferrer(
        self.resolver, self.scope, self.namespace, self.closure_types, types_in)
    inferrer.visit(ast_node)
    types_out.value.update(inferrer.new_symbols)

    reaching_fndefs = anno.getanno(ast_node, anno.Static.DEFINED_FNS_IN)
    node_scope = anno.getanno(ast_node, anno.Static.SCOPE, None)
    if node_scope is not None:
      # TODO(mdan): Check that it's actually safe to skip nodes without scope.
      reads = {str(qn) for qn in node_scope.read}
      for def_node in reaching_fndefs:
        if def_node.name in reads:
          self._update_closure_types(def_node, types_out)

    self.in_[node] = types_in
    self.out[node] = types_out

    return prev_types_out != types_out


class FunctionVisitor(transformer.Base):
  """AST visitor that applies type inference to each function separately."""

  def __init__(self, source_info, graphs, resolver):
    super(FunctionVisitor, self).__init__(source_info)
    self.graphs = graphs
    self.resolver = resolver

  def visit_FunctionDef(self, node):
    subgraph = self.graphs[node]
    scope = anno.getanno(node, annos.NodeAnno.ARGS_AND_BODY_SCOPE)
    closure_types = anno.getanno(node, anno.Static.CLOSURE_TYPES, {})

    analyzer = Analyzer(
        subgraph, self.resolver, self.ctx.info.namespace, scope, closure_types)
    analyzer.visit_forward()

    # Recursively process any remaining subfunctions.
    node.body = self.visit_block(node.body)

    return node


def resolve(node, source_info, graphs, resolver):
  """Performs type inference.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
    resolver: Resolver

  Returns:
    ast.AST
  """
  visitor = FunctionVisitor(source_info, graphs, resolver)
  node = visitor.visit(node)
  return node
