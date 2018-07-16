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

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import cfg
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.static_analysis import annos


class Definition(object):
  """Definition objects describe a unique definition of a variable.

  Subclasses of this may be used by passing an appropriate factory fuction to
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
        for s in node_scope.modified:
          def_ = self._definition_factory()
          if s in node_scope.params:
            def_.param_of = node_scope.params[s]
          node_symbols[s] = def_
        self.gen_map[node] = _NodeState(node_symbols)

      gen = self.gen_map[node]
      kill = node_scope.modified
      defs_out = gen | (defs_in - kill)

    else:
      # Nodes that don't have a scope annotation are assumed not to touch any
      # symbols.
      # This Name node below is a literal name, e.g. False
      # This can also happen if activity.py forgot to annotate the node with a
      # scope object.
      assert isinstance(
          node.ast_node,
          (gast.Name, gast.Break, gast.Continue, gast.Raise)), (node.ast_node,
                                                                node)
      defs_out = defs_in

    self.in_[node] = defs_in
    self.out[node] = defs_out

    # TODO(mdan): Move this to the superclass?
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
    self.definition_factory = definition_factory
    self.graphs = graphs
    self.current_analyzer = None
    self.current_cfg_node = None

  def visit_FunctionDef(self, node):
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]

    # Preorder tree processing:
    #  1. if this is a child function, the parent was already analyzed and it
    #     has the proper state value for the subgraph's entry
    #  2. analyze the current function body
    #  2. recursively walk the subtree; child functions will be processed
    analyzer = Analyzer(subgraph, self.definition_factory)
    if parent_analyzer is not None:
      # Wire the state between the two subgraphs' analyzers.
      parent_out_state = parent_analyzer.out[parent_analyzer.graph.index[node]]
      # Exception: symbols modified in the child function are local to it
      body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
      parent_out_state -= body_scope.modified
      analyzer.extra_in[node.args] = parent_out_state

    # Complete the analysis for the local function and annotate its body.
    analyzer.visit_forward()

    # Recursively process any remaining subfunctions.
    self.current_analyzer = analyzer
    # Note: not visiting name, decorator_list and returns because they don't
    # apply to this anlysis.
    # TODO(mdan): Should we still process the function name?
    node.args = self.visit(node.args)
    node.body = self.visit_block(node.body)
    self.current_analyzer = parent_analyzer

    return node

  def visit_nonlocal(self, node):
    raise NotImplementedError()

  def visit_global(self, node):
    raise NotImplementedError()

  def visit_Name(self, node):
    if self.current_analyzer is None:
      # Names may appear outside function defs - for example in class
      # definitions.
      return node

    analyzer = self.current_analyzer
    cfg_node = self.current_cfg_node

    assert cfg_node is not None, 'name node outside of any statement?'

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
