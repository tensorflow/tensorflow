# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Reaching definition analysis.

This analysis attaches a set of a Definition objects to each symbol, one
for each distinct definition that may reach it. The Definition objects are
mutable and may be used by subsequent analyses to further annotate data like
static type and value information.
The analysis also attaches the set of the symbols defined at the entry of
control flow statements.

Requires activity analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer


class Definition(object):
  """Definition objects describe a unique definition of a variable.

  Subclasses of this may be used by passing an appropriate factory function to
  resolve.

  Attributes:
    param_of: Optional[ast.AST]
  """

  def __init__(self):
    self.param_of = None

  def __repr__(self):
    return '%s[%d]' % (self.__class__.__name__, id(self))


class _NodeState(object):
  """Abstraction for the state of the CFG walk for reaching definition analysis.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    value: Dict[qual_names.QN, Set[Definition, ...]], the defined symbols and
        their possible definitions
  """

  def __init__(self, init_from=None):
    if init_from:
      if isinstance(init_from, _NodeState):
        self.value = {
            s: set(other_infos) for s, other_infos in init_from.value.items()
        }
      elif isinstance(init_from, dict):
        self.value = {s: set((init_from[s],)) for s in init_from}
      else:
        assert False, init_from
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
    assert isinstance(other, _NodeState)
    result = _NodeState(self)
    for s, other_infos in other.value.items():
      if s in result.value:
        result.value[s].update(other_infos)
      else:
        result.value[s] = set(other_infos)
    return result

  def __sub__(self, other):
    assert isinstance(other, set)
    result = _NodeState(self)
    for s in other:
      result.value.pop(s, None)
    return result

  def __repr__(self):
    return 'NodeState[%s]=%s' % (id(self), repr(self.value))


class Analyzer(cfg.GraphVisitor):
  """CFG visitor that determines reaching definitions at statement level."""

  def __init__(self, graph, definition_factory):
    self._definition_factory = definition_factory
    super(Analyzer, self).__init__(graph)
    # This allows communicating that nodes have extra reaching definitions,
    # e.g. those that a function closes over.
    self.extra_in = {}

    self.gen_map = {}

  def init_state(self, _):
    return _NodeState()

  def visit_node(self, node):
    prev_defs_out = self.out[node]

    defs_in = _NodeState(self.extra_in.get(node.ast_node, None))
    for n in node.prev:
      defs_in |= self.out[n]

    if anno.hasanno(node.ast_node, anno.Static.SCOPE):
      node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)
      # The definition objects created by each node must be singletons because
      # their ids are used in equality checks.
      if node not in self.gen_map:
        node_symbols = {}
        # Every binding operation (assign, nonlocal, global, etc.) counts as a
        # definition, with the exception of del, which only deletes without
        # creating a new variable.
        newly_defined = ((node_scope.bound | node_scope.globals) -
                         node_scope.deleted)
        for s in newly_defined:
          def_ = self._definition_factory()
          node_symbols[s] = def_
        # Every param receives a definition. Params are not necessarily
        # considered as "modified".
        for s, p in node_scope.params.items():
          def_ = self._definition_factory()
          def_.param_of = weakref.ref(p)
          node_symbols[s] = def_
        self.gen_map[node] = _NodeState(node_symbols)

      gen = self.gen_map[node]
      kill = node_scope.modified | node_scope.deleted
      defs_out = gen | (defs_in - kill)

      gen = self.gen_map[node]
      defs_out = gen | (defs_in - kill)

    else:
      assert self.can_ignore(node), (node.ast_node, node)
      defs_out = defs_in

    self.in_[node] = defs_in
    self.out[node] = defs_out

    return prev_defs_out != defs_out


class TreeAnnotator(transformer.Base):
  """AST visitor that annotates each symbol name with its reaching definitions.

  Simultaneously, the visitor runs the dataflow analysis on each function node,
  accounting for the effect of closures. For example:

    def foo():
      bar = 1
      def baz():
        # bar = 1 reaches here
  """

  def __init__(self, source_info, graphs, definition_factory):
    super(TreeAnnotator, self).__init__(source_info)
    self.allow_skips = False
    self.definition_factory = definition_factory
    self.graphs = graphs
    self.current_analyzer = None
    self.current_cfg_node = None

  def visit_FunctionDef(self, node):
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]

    analyzer = Analyzer(subgraph, self.definition_factory)
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
      anno.setanno(node, anno.Static.DEFINITIONS,
                   tuple(analyzer.in_[cfg_node].value.get(qn, ())))
    else:
      anno.setanno(node, anno.Static.DEFINITIONS,
                   tuple(analyzer.out[cfg_node].value.get(qn, ())))

    return node

  def _aggregate_predecessors_defined_in(self, node):
    preds = self.current_analyzer.graph.stmt_prev[node]
    node_defined_in = set()
    for p in preds:
      node_defined_in |= set(self.current_analyzer.out[p].value.keys())
    anno.setanno(node, anno.Static.DEFINED_VARS_IN, frozenset(node_defined_in))

  def visit_If(self, node):
    self._aggregate_predecessors_defined_in(node)
    return self.generic_visit(node)

  def visit_For(self, node):
    self._aggregate_predecessors_defined_in(node)

    # Manually accounting for the shortcoming described in
    # cfg.AstToCfg.visit_For.
    parent = self.current_cfg_node
    self.current_cfg_node = self.current_analyzer.graph.index[node.iter]
    node.target = self.visit(node.target)
    self.current_cfg_node = parent

    node.iter = self.visit(node.iter)
    node.body = self.visit_block(node.body)
    node.orelse = self.visit_block(node.orelse)

    return node

  def visit_While(self, node):
    self._aggregate_predecessors_defined_in(node)
    return self.generic_visit(node)

  def visit_Try(self, node):
    self._aggregate_predecessors_defined_in(node)
    return self.generic_visit(node)

  def visit_ExceptHandler(self, node):
    self._aggregate_predecessors_defined_in(node)
    # TODO(mdan): Also track the exception type / name symbols.
    node.body = self.visit_block(node.body)
    return node

  def visit(self, node):
    parent = self.current_cfg_node

    if (self.current_analyzer is not None and
        node in self.current_analyzer.graph.index):
      self.current_cfg_node = self.current_analyzer.graph.index[node]
    node = super(TreeAnnotator, self).visit(node)

    self.current_cfg_node = parent
    return node


def resolve(node, source_info, graphs, definition_factory):
  """Resolves reaching definitions for each symbol.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
    definition_factory: Callable[[], Definition]
  Returns:
    ast.AST
  """
  visitor = TreeAnnotator(source_info, graphs, definition_factory)
  node = visitor.visit(node)
  return node
