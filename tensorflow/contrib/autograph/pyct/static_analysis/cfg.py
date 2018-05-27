# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Control flow graph analysis.

Given a Python AST we construct a control flow graph, with edges both to the
next and previous statements (so it can easily walk the graph both ways). Its
nodes contain the AST of the statements. It can then perform forward or backward
analysis on this CFG.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import operator

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct.static_analysis import activity


class CfgNode(object):
  """A node in the CFG."""
  __slots__ = ['next', 'value', 'prev']

  def __init__(self, value):
    self.next = set()
    self.prev = set()
    self.value = value


class Cfg(namedtuple('Cfg', ['entry', 'exit'])):
  """A Control Flow Graph.

  Each statement is represented as a node. For control flow statements such
  as conditionals and loops the conditional itself is a node which either
  branches or cycles, respectively.
  Attributes:
    entry: The entry node, which contains the `gast.arguments` node of the
        function definition.
    exit: The exit node. This node is special because it has no value (i.e. no
        corresponding AST node). This is because Python functions can have
        multiple return statements.
  """
  pass


class CfgBuilder(gast.NodeVisitor):
  """Construct a control flow graph.

  Construct a CFG starting from a FunctionDef node.
  Usage:
    cfg_obj = CfgBuilder().build_cfg(fndef_node)
  """

  def __init__(self):
    # The current leaves of the CFG
    self.current_leaves = []
    # TODO(alexbw): generalize to break, return, continue, yield, etc.
    # A stack of lists, tracking continue statements
    self.continue_ = []
    # A stack of lists tracking break nodes
    self.break_ = []

  def set_current_leaves(self, cfg_node):
    """Link this cfg_node to the current leaves.

    This is the central function for building the CFG. It links the current
    head cfg_nodes to the passed cfg_node. It then resets the head to the
    passed cfg_node.

    Args:
      cfg_node: A CfgNode instance.
    """
    for head in self.current_leaves:
      head.next.add(cfg_node)
      # While we're linking the CFG forward, add backlinks
      cfg_node.prev.add(head)
    self.current_leaves = [cfg_node]

  def build_cfg(self, node):
    """Build a CFG for a function.

    Implementation of building a CFG for dataflow analysis. See, e.g.:
    https://www.seas.harvard.edu/courses/cs252/2011sp/slides/Lec02-Dataflow.pdf

    Args:
      node: A function definition the body of which to analyze.
    Returns:
      A CFG object.
    Raises:
      TypeError: If the input is not a function definition.
    """
    if not isinstance(node, gast.FunctionDef):
      raise TypeError('input must be a function definition')
    entry_cfg_node = CfgNode(node.args)
    self.current_leaves = [entry_cfg_node]
    self.visit_statements(node.body)
    exit_cfg_node = CfgNode(None)
    self.set_current_leaves(exit_cfg_node)
    return Cfg(entry_cfg_node, exit_cfg_node)

  def visit_statements(self, nodes):
    for node in nodes:
      # Check for control flow
      if isinstance(node, (gast.For, gast.While, gast.If, gast.Try, gast.Break,
                           gast.Continue, gast.With)):
        self.visit(node)
      else:
        expr = CfgNode(node)
        self.set_current_leaves(expr)

  def generic_visit(self, node):
    raise ValueError('unknown control flow')

  def visit_If(self, node):
    # TODO(alexbw): change this to use immutable tuples instead of lists
    # The current head will hold the conditional
    test = CfgNode(node.test)
    self.set_current_leaves(test)
    # Handle the body
    self.visit_statements(node.body)
    body_exit = self.current_leaves
    self.current_leaves = [test]
    # Handle the orelse
    self.visit_statements(node.orelse)
    self.current_leaves.extend(body_exit)

  def visit_While(self, node):
    test = CfgNode(node.test)
    self.set_current_leaves(test)
    # Start a new level of nesting
    self.break_.append([])
    self.continue_.append([])
    # Handle the body
    self.visit_statements(node.body)
    body_exit = self.current_leaves
    self.current_leaves.extend(self.continue_.pop())
    self.set_current_leaves(test)
    # Handle the orelse
    self.visit_statements(node.orelse)
    # The break statements and the test go to the next node
    self.current_leaves.extend(self.break_.pop())
    # Body and orelse statements can reach out of the loop
    self.current_leaves.extend(body_exit)

  def visit_For(self, node):
    iter_ = CfgNode(node.iter)
    self.set_current_leaves(iter_)
    self.break_.append([])
    self.continue_.append([])
    self.visit_statements(node.body)
    body_exit = self.current_leaves
    self.current_leaves.extend(self.continue_.pop())
    self.set_current_leaves(iter_)
    # Handle the orelse
    self.visit_statements(node.orelse)
    # The break statements and the test go to the next node
    self.current_leaves.extend(self.break_.pop())
    # Body and orelse statements can reach out of the loop
    self.current_leaves.extend(body_exit)

  def visit_Break(self, node):
    self.break_[-1].extend(self.current_leaves)
    self.current_leaves[:] = []

  def visit_Continue(self, node):
    self.continue_[-1].extend(self.current_leaves)
    self.current_leaves[:] = []

  def visit_Try(self, node):
    self.visit_statements(node.body)
    body = self.current_leaves
    handlers = []
    for handler in node.handlers:
      self.current_leaves = body[:]
      self.visit_statements(handler.body)
      handlers.extend(self.current_leaves)
    self.current_leaves = body
    self.visit_statements(node.orelse)
    self.current_leaves = handlers + self.current_leaves
    self.visit_statements(node.finalbody)

  def visit_With(self, node):
    for item in node.items:
      self.set_current_leaves(CfgNode(item))
    self.visit_statements(node.body)


# TODO(alexbw): once CFG analysis occurs at a block level,
# this extra class will not be necessary
class PropagateAnalysis(gast.NodeVisitor):
  """Port analysis annotations from statements to their enclosing blocks."""

  def __init__(self, analysis):
    self.transfer_fn = analysis.transfer_fn
    self.in_label = analysis.in_label
    self.out_label = analysis.out_label
    super(PropagateAnalysis, self).__init__()

  def visit_If(self, node):
    # Depth-first.
    self.generic_visit(node)
    incoming = anno.getanno(node.body[0], self.in_label)
    incoming |= anno.getanno(node.test, self.in_label)
    outgoing = anno.getanno(node.body[-1], self.out_label)
    outgoing |= anno.getanno(node.test, self.out_label)
    if node.orelse:
      orelse_outgoing = anno.getanno(node.orelse[-1], self.out_label)
      outgoing = self.transfer_fn(outgoing, orelse_outgoing)
    anno.setanno(node, self.in_label, incoming)
    anno.setanno(node, self.out_label, outgoing)

  def visit_For(self, node):
    self.generic_visit(node)
    incoming = set(anno.getanno(node.body[0], self.in_label))
    incoming -= set((anno.getanno(node.target, anno.Basic.QN),))
    outgoing = anno.getanno(node.body[-1], self.out_label)
    if node.orelse:
      orelse_outgoing = anno.getanno(node.orelse[-1], self.out_label)
      outgoing = self.transfer_fn(outgoing, orelse_outgoing)
    anno.setanno(node, self.in_label, frozenset(incoming))
    anno.setanno(node, self.out_label, outgoing)

  def visit_While(self, node):
    self.generic_visit(node)
    incoming = anno.getanno(node.body[0], self.in_label)
    incoming |= anno.getanno(node.test, self.in_label)
    outgoing = anno.getanno(node.body[-1], self.out_label)
    if node.orelse:
      orelse_outgoing = anno.getanno(node.orelse[-1], self.out_label)
      outgoing = self.transfer_fn(outgoing, orelse_outgoing)
    anno.setanno(node, self.in_label, incoming)
    anno.setanno(node, self.out_label, outgoing)

  def visit_With(self, node):
    self.generic_visit(node)
    incoming = anno.getanno(node.body[0], self.in_label)
    for item in node.items:
      incoming |= anno.getanno(item, self.in_label)
    outgoing = anno.getanno(node.body[-1], self.out_label)
    anno.setanno(node, self.in_label, incoming)
    anno.setanno(node, self.out_label, outgoing)


# TODO(alexbw): Abstract the CFG walking machinery into a superclass
# which is parameterized on which fields it selects when walking.
# TODO(alexbw): Abstract the application of dataflow analysis
class Forward(object):
  """Forward analysis on CFG.

  Args:
    label: A name for this analysis e.g. 'active' for activity analysis. The AST
      nodes in the CFG will be given annotations 'name_in', 'name_out',
      'name_gen' and 'name_kill' which contain the incoming values, outgoing
      values, values generated by the statement, and values deleted by the
      statement respectively.
    transfer_fn: Either the AND or OR operator. If the AND operator is used it
      turns into forward must analysis (i.e. a value will only be carried
      forward if it appears on all incoming paths). The OR operator means that
      forward may analysis is done (i.e. the union of incoming values will be
      taken).
  """

  def __init__(self, label, context, transfer_fn=operator.or_):
    self.transfer_fn = transfer_fn
    self.context = context
    self.out_label = label + '_out'
    self.in_label = label + '_in'
    self.gen_label = label + '_gen'
    self.kill_label = label + '_kill'

  # TODO(alexbw): see if we can simplify by visiting breadth-first
  def visit(self, node):
    """Depth-first walking the CFG, applying dataflow information propagtion."""
    # node.value is None only for the exit CfgNode.
    if not node.value:
      return

    if anno.hasanno(node.value, self.out_label):
      before = hash(anno.getanno(node.value, self.out_label))
    else:
      before = None
    preds = [
        anno.getanno(pred.value, self.out_label)
        for pred in node.prev
        if anno.hasanno(pred.value, self.out_label)
    ]
    if preds:
      incoming = functools.reduce(self.transfer_fn, preds[1:], preds[0])
    else:
      incoming = frozenset()
    anno.setanno(node.value, self.in_label, incoming)
    gen, kill = self.get_gen_kill(node, incoming)
    anno.setanno(node.value, self.gen_label, gen)
    anno.setanno(node.value, self.kill_label, kill)
    anno.setanno(node.value, self.out_label, (incoming - kill) | gen)

    if hash(anno.getanno(node.value, self.out_label)) != before:
      for succ in node.next:
        self.visit(succ)

  def get_gen_kill(self, cfg_node, incoming):
    """Calculate Gen and Kill properties of a CFG node in dataflow analysis.

    A function which takes the CFG node as well as a set of incoming
    values. It must return a set of newly generated values by the statement as
    well as a set of deleted (killed) values.

    Args:
      cfg_node: A CfgNode instance.
      incoming:
    """
    raise NotImplementedError()


class Backward(Forward):
  """Backward analysis on CFG."""

  def visit(self, cfg_node):
    # cfg_node.value is None for the exit node, which will be visited only once
    if not cfg_node.value:
      for pred in cfg_node.prev:
        self.visit(pred)
      return

    if anno.hasanno(cfg_node.value, self.in_label):
      before = hash(anno.getanno(cfg_node.value, self.in_label))
    else:
      before = None
    succs = [
        anno.getanno(succ.value, self.in_label)
        for succ in cfg_node.next
        if anno.hasanno(succ.value, self.in_label)
    ]
    if succs:
      incoming = functools.reduce(self.transfer_fn, succs[1:], succs[0])
    else:
      incoming = frozenset()
    anno.setanno(cfg_node.value, self.out_label, incoming)
    gen, kill = self.get_gen_kill(cfg_node, incoming)
    anno.setanno(cfg_node.value, self.gen_label, gen)
    anno.setanno(cfg_node.value, self.kill_label, kill)
    anno.setanno(cfg_node.value, self.in_label, (incoming - kill) | gen)
    if hash(anno.getanno(cfg_node.value, self.in_label)) != before:
      for pred in cfg_node.prev:
        self.visit(pred)


def run_analyses(node, analyses):
  """Perform dataflow analysis on all functions within an AST.

  Args:
    node: An AST node on which to run dataflow analysis.
    analyses: Either an instance of the Forward or Backward dataflow analysis
      class, or a list or tuple of them.

  Returns:
    node: The node, but now with annotations on the AST nodes containing the
    results of the dataflow analyses.
  """
  if not isinstance(analyses, (tuple, list)):
    analyses = (analyses,)
  for analysis in analyses:
    if not isinstance(analysis, (Forward, Backward)):
      raise TypeError('not a valid forward analysis object')

  for child_node in gast.walk(node):
    if isinstance(child_node, gast.FunctionDef):
      cfg_obj = CfgBuilder().build_cfg(child_node)
      for analysis in analyses:
        if isinstance(analysis, Backward):
          analysis.visit(cfg_obj.exit)
        elif isinstance(analysis, Forward):
          analysis.visit(cfg_obj.entry)
  for analysis in analyses:
    PropagateAnalysis(analysis).visit(node)
  return node


class Liveness(Backward):
  """Perform a liveness analysis.

  Each statement is annotated with a set of variables that may be used
  later in the program.
  """

  def __init__(self, context):
    super(Liveness, self).__init__('live', context)

  def get_gen_kill(self, node, _):
    # A variable's parents are live if it is live
    # e.g. x is live if x.y is live. This means gen needs to return
    # all parents of a variable (if it's an Attribute or Subscript).
    # This doesn't apply to kill (e.g. del x.y doesn't affect liveness of x)
    gen = activity.get_read(node.value, self.context)
    gen = functools.reduce(lambda left, right: left | right.support_set, gen,
                           gen)
    kill = activity.get_updated(node.value, self.context)
    return gen, kill


class ReachingDefinitions(Forward):
  """Perform reaching definition analysis.

  Each statement is annotated with a set of (variable, definition) pairs.
  """

  def __init__(self, context):
    super(ReachingDefinitions, self).__init__('definitions', context)

  def get_gen_kill(self, node, incoming):
    definitions = activity.get_updated(node.value, self.context)
    gen = frozenset((id_, node.value) for id_ in definitions)
    kill = frozenset(def_ for def_ in incoming if def_[0] in definitions)
    return gen, kill


class Defined(Forward):
  """Perform defined variable analysis.

  Each statement is annotated with a set of variables which are guaranteed to
  be defined at that point.
  """

  def __init__(self, context):
    super(Defined, self).__init__('defined', context, transfer_fn=operator.and_)

  def get_gen_kill(self, node, _):
    gen = activity.get_updated(node.value, self.context)
    return gen, frozenset()
