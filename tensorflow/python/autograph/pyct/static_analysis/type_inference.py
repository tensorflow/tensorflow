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

Important: This analysis is static, and does not detect dynamic type changes.
The analysis attempts to use the values of external symbols, if available. These
values are also considered static for the purpose of analysis.

Requires reaching function definitions analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos


class Resolver(object):
  """Resolver objects handle the process of looking up actual names and types.

  Unless noted otherwise, all resolve_* methods:
    * have a first namespace argument, mapping string to actual values
    * have a second types_namespace argument, mapping string to actual inferred
      types
    * specify names as QN objects
    * specify types as a Set of inferred types

  Unless noted otherwise, all resolve_* methods must return either:
    * a set of `type` objects
    * None
  """

  def res_name(self, ns, types_ns, name):
    """Resolves the type/value an external (e.g. closure, global) variable.

    Args:
      ns: namespace
      types_ns: types namespace
      name: symbol name
    Returns:
      Tuple (type, static_value). The first element is the type to use for
      inferrence. The second is the static value to use. Return None to treat it
      as unknown.
    """
    raise NotImplementedError('subclasses must implement')

  def res_value(self, ns, value):
    """Resolves the type a literal or static value."""
    raise NotImplementedError('subclasses must implement')

  def res_arg(self, ns, types_ns, f_name, name, type_anno):
    """Resolves the type of a (possibly annotated) function argument."""
    raise NotImplementedError('subclasses must implement')

  def res_call(self, ns, types_ns, node, args, keywords):
    """Resolves the return type an external function or method call.

    Args:
      ns: namespace
      types_ns: types namespace
      node: str, the function name
      args: types of each respective argument in node.args
      keywords: types of each respective argument in node.keywords

    Returns:
      Tuple (return_type, side_effect_types). The first element is just the
      return types of the function. The second element is a map from
      argument names to sets of types, and allow modelling side effects of
      functions (for example via global or nonlocal).
    """
    raise NotImplementedError('subclasses must implement')

  def res_subscript(self, ns, types_ns, node, value, slice_):
    """Resolves the return type of a unary operation."""
    raise NotImplementedError('subclasses must implement')

  def res_compare(self, ns, types_ns, node, left, right):
    """Resolves the return type of a unary operation."""
    raise NotImplementedError('subclasses must implement')

  def res_binop(self, ns, types_ns, node, left, right):
    """Resolves the return type of a unary operation."""
    raise NotImplementedError('subclasses must implement')


class _SymbolTable(object):
  """Abstraction for the state of the CFG walk for type inference.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    types: Dict[qual_names.QN, Set[Type]], mapping symbols to the set of
      possible types.
  """

  def __init__(self, init_from=None):
    if init_from:
      assert isinstance(init_from, _SymbolTable)
      self.types = {
          s: set(other_types) for s, other_types in init_from.types.items()
      }
    else:
      self.types = {}

  def __eq__(self, other):
    if frozenset(self.types.keys()) != frozenset(other.types.keys()):
      return False
    ret = all(self.types[s] == other.types[s] for s in self.types)
    return ret

  def __ne__(self, other):
    return not self.__eq__(other)

  def __or__(self, other):
    assert isinstance(other, _SymbolTable)
    result = _SymbolTable(self)
    for s, other_types in other.types.items():
      if s not in result.types:
        self_types = set()
        result.types[s] = self_types
      else:
        self_types = result.types[s]
      self_types.update(other_types)
    return result

  def __repr__(self):
    return 'SymbolTable {}'.format(self.types)


class StmtInferrer(gast.NodeVisitor):
  """Runs type inference on a single AST statement.

  This visitor annotates most nodes with type information. It also sets types
  for the symbols modified by this statement in its types_out property.

  Note: this inferrer is able to capture side effects of functions, however,
  these side effects will not be applied to the current expression. Doing so
  would create too much of a dependence on the runtime's internal rules about
  execution order.
  Example:

    def f():
      nonlocal a
      a = 1
      return a

    a = 0.0
    b = f() + a  # a = float; side effect of f() ignored
    print(a)  # a = int; side effect of f() accounted for
  """

  def __init__(self, resolver, scope, namespace, closure_types, types_in):
    self.resolver = resolver
    self.scope = scope
    self.namespace = namespace
    self.closure_types = closure_types
    self.types_in = types_in
    self.new_symbols = {}
    self.rtype = None

  def visit(self, node):
    types = super().visit(node)
    if types is not None:
      # TODO(mdan): Normalize by removing subtypes.
      anno.setanno(node, anno.Static.TYPES, tuple(types))
    return types

  def visit_FunctionDef(self, node):
    # Skip local function definitions. They are analyzed separately.
    # TODO(mdan): Don't skip. Analyze side effects instead.
    return None

  def _check_set(self, value):
    if value is not None and not isinstance(value, set):
      raise ValueError('{} method expected to return set, got {}'.format(
          self.resolver, value))

  def visit_Constant(self, node):
    types = self.resolver.res_value(self.namespace, node.value)
    if __debug__:
      self._check_set(types)
    return types

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
      types = self.types_in.types.get(name, None)
      if (types is None) and (name not in self.scope.bound):
        if name in self.closure_types:
          types = self.closure_types[name]
        else:
          types, value = self.resolver.res_name(
              self.namespace, self.types_in.types, name)
          if value is not None:
            anno.setanno(node, anno.Static.VALUE, value)

    elif isinstance(node.ctx, gast.Param):
      type_name = anno.getanno(node.annotation, anno.Basic.QN, None)
      types = self.resolver.res_arg(self.namespace, self.types_in.types,
                                    self.scope.function_name, name, type_name)
      if types is not None:
        self.new_symbols[name] = types

    elif isinstance(node.ctx, gast.Store):
      if self.rtype is not None:
        self.new_symbols[name] = self.rtype
      types = self.rtype

    else:
      assert False, 'unknown ctx'

    if __debug__:
      self._check_set(types)

    return types

  def visit_Attribute(self, node):
    parent_types = self.visit(node.value)

    # Attempt to use the static value if known.
    parent_value = anno.Static.VALUE.of(node.value, None)
    if parent_value is not None:
      static_value = getattr(parent_value, node.attr, None)

    else:
      # Fall back to the type if that is known.
      if parent_types is None:
        return None

      inferred_values = [getattr(t, node.attr, None) for t in parent_types]
      if not inferred_values:
        return None

      static_value = inferred_values[0]
      if static_value is None:
        return None

      if any(v is not static_value for v in inferred_values[1:]):
        # Static value not stable, assume it's dynamic.
        return None

    types = self.resolver.res_value(self.namespace, static_value)
    anno.setanno(node, anno.Static.VALUE, static_value)

    if __debug__:
      self._check_set(types)

    return types

  def visit_Call(self, node):
    self.visit(node.func)

    f_name = anno.getanno(node.func, anno.Basic.QN)
    if f_name in self.scope.bound:
      # Don't attempt external resolution of local functions.
      # TODO(mdan): Use type annotations of the local definition.
      return None

    arg_types = [self.visit(a) for a in node.args]
    keyword_types = [self.visit(kw.value) for kw in node.keywords]

    ret_type, side_effects = self.resolver.res_call(self.namespace,
                                                    self.types_in.types, node,
                                                    arg_types, keyword_types)
    if __debug__:
      self._check_set(ret_type)
      if side_effects:
        if not isinstance(side_effects, dict):
          raise ValueError(
              'side effects must be dict, got {}'.format(side_effects))
        for k, v in side_effects.items():
          if not isinstance(k, qual_names.QN):
            raise ValueError('side effect keys must be QNs, got {}'.format(k))
          self._check_set(v)

    if side_effects:
      self.new_symbols.update(side_effects)
    return ret_type

  def visit_Index(self, node):
    return self.visit(node.value)

  def visit_Assign(self, node):
    self.rtype = self.visit(node.value)

    for t in node.targets:
      self.visit(t)

    self.rtype = None

  def visit_Subscript(self, node):
    val_types = self.visit(node.value)
    slice_types = self.visit(node.slice)

    if val_types is None or slice_types is None:
      return None

    types = self.resolver.res_subscript(
        self.namespace, self.types_in.types, node, val_types, slice_types)

    if __debug__:
      self._check_set(types)

    return types

  def visit_Compare(self, node):
    left_types = self.visit(node.left)
    right_types = [self.visit(c) for c in node.comparators]

    if left_types is None or any(t is None for t in right_types):
      return None

    types = self.resolver.res_compare(
        self.namespace, self.types_in.types, node, left_types, right_types)

    if __debug__:
      self._check_set(types)

    return types

  def visit_BinOp(self, node):
    left_types = self.visit(node.left)
    right_types = self.visit(node.right)

    if left_types is None or right_types is None:
      return None

    types = self.resolver.res_binop(
        self.namespace, self.types_in.types, node, left_types, right_types)

    if __debug__:
      self._check_set(types)

    return types


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

    for k, v in types.types.items():
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

    inferrer = StmtInferrer(self.resolver, self.scope, self.namespace,
                            self.closure_types, types_in)
    inferrer.visit(ast_node)
    types_out.types.update(inferrer.new_symbols)

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

    analyzer = Analyzer(subgraph, self.resolver, self.ctx.info.namespace, scope,
                        closure_types)
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
