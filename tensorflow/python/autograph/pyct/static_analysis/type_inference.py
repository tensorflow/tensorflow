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

Requires activity analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos


class Resolver(object):
  """Resolver objects handle the process of looking up actual names and types.

  All resolve_* methods take:
    * a first namespace argument, mapping string to actual values
    * one or more name arguments, as QN objects

  All resolve_* methods must return either:
    * a set of `type` objects
    * None
  """

  def resolve_external_name(self, ns, name):
    """Resolves the type an external (e.g. closure, global) variable."""
    raise NotImplementedError('subclasses must implement')

  def resolve_external_call(self, ns, name):
    """Resolves the return type an external function call."""
    # TODO(mdan): This must accept argument value/types.
    raise NotImplementedError('subclasses must implement')

  def resolve_external_arg(self, ns, f_name, arg_name, type_anno):
    """Resolves the type of a (possibly annotated) function argument."""
    raise NotImplementedError('subclasses must implement')

  # TODO(mdan): More resolvers as needed.


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


class Analyzer(cfg.GraphVisitor):
  """CFG visitor that performs type inference at statement level."""

  def __init__(self, graph, resolver, namespace, scope):
    """Creates a new analyzer.

    Args:
      graph: cfg.Graph
      resolver: Resolver
      namespace: Dict[str, Any]
      scope: activity.Scope
    """
    super(Analyzer, self).__init__(graph)
    self.resolver = resolver
    self.namespace = namespace
    self.scope = scope

  def init_state(self, _):
    return _SymbolTable()

  def _infer_type(self, node, types_in):
    """Infers the return type of an expression."""
    if isinstance(node, gast.Name):
      # Normal variables: carry over their existing type.
      name = anno.getanno(node, anno.Basic.QN)
      types = types_in.value.get(name, None)
      if types is not None:
        return types
      # If type is unknown, resolve it.
      if name not in self.scope.bound:
        return self.resolver.resolve_external_name(self.namespace, name)
      return None

    if isinstance(node, gast.Call):
      # Function calls: resolve their return type.
      f_name = anno.getanno(node.func, anno.Basic.QN)
      return self.resolver.resolve_external_call(self.namespace, f_name)

    else:
      raise NotImplementedError(node)

  def _assignment_types(self, node, types_in):
    """Propagates types through an assignment operation."""
    targets = node.targets
    if len(targets) != 1:
      raise NotImplementedError('multiple assignment')

    target, = targets
    qn = anno.getanno(target, anno.Basic.QN)
    types = self._infer_type(node.value, types_in)
    if types is None:
      return ()

    return (qn, types),

  def _arg_type(self, node):
    """Looks up the type of an argument based on its annotation."""
    assert isinstance(node, gast.Name)
    name = anno.getanno(node, anno.Basic.QN)
    type_name = anno.getanno(node.annotation, anno.Basic.QN, None)

    type_ = self.resolver.resolve_external_arg(self.namespace,
                                               self.scope.function_name, name,
                                               type_name)
    return (name, type_),

  def _args_types(self, node):
    """Propagates types through argument annotations."""
    types = {}

    for n in node.posonlyargs:
      types.update(self._arg_type(n))
    for n in node.args:
      types.update(self._arg_type(n))
    for n in node.kwonlyargs:
      types.update(self._arg_type(n))

    if node.vararg:
      raise NotImplementedError('vararg')
    if node.kwarg:
      raise NotImplementedError('kwarg')

    # TODO(mdan): Use kw_defaults, defaults if available.

    return types

  def visit_node(self, node):
    prev_types_out = self.out[node]

    types_in = _SymbolTable()
    for n in node.prev:
      types_in |= self.out[n]

    types_out = _SymbolTable(types_in)
    ast_node = node.ast_node
    if isinstance(ast_node, gast.Assign):
      types_out.value.update(self._assignment_types(ast_node, types_in))
    elif isinstance(ast_node, gast.arguments):
      types_out.value.update(self._args_types(ast_node))

    self.in_[node] = types_in
    self.out[node] = types_out

    return prev_types_out != types_out


class TreeAnnotator(transformer.Base):
  """AST visitor that annotates each symbol with its possible types."""

  def __init__(self, source_info, graphs, resolver):
    super(TreeAnnotator, self).__init__(source_info)
    self.graphs = graphs
    self.resolver = resolver
    self.current_analyzer = None
    self.current_cfg_node = None

  def visit_FunctionDef(self, node):
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]
    scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

    analyzer = Analyzer(subgraph, self.resolver, self.ctx.info.namespace, scope)
    analyzer.visit_forward()

    # Recursively process any remaining subfunctions.
    self.current_analyzer = analyzer
    node.args = self.visit(node.args)
    node.body = self.visit_block(node.body)
    self.current_analyzer = parent_analyzer

    return node

  def visit_Name(self, node):
    if self.current_analyzer is None:
      # Names may appear outside function defs - for example in class
      # definitions.
      return node

    analyzer = self.current_analyzer
    cfg_node = self.current_cfg_node

    assert cfg_node is not None, ('name node, %s, outside of any statement?'
                                  % node.id)

    qn = anno.getanno(node, anno.Basic.QN)
    if isinstance(node.ctx, gast.Load):
      anno.setanno(node, anno.Static.TYPES,
                   tuple(analyzer.in_[cfg_node].value.get(qn, ())))
    else:
      anno.setanno(node, anno.Static.TYPES,
                   tuple(analyzer.out[cfg_node].value.get(qn, ())))

    return node

  def visit(self, node):
    parent = self.current_cfg_node

    if (self.current_analyzer is not None and
        node in self.current_analyzer.graph.index):
      self.current_cfg_node = self.current_analyzer.graph.index[node]
    node = super(TreeAnnotator, self).visit(node)

    self.current_cfg_node = parent
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
  visitor = TreeAnnotator(source_info, graphs, resolver)
  node = visitor.visit(node)
  return node
