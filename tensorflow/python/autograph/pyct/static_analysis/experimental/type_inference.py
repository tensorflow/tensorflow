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
"""Type inference.

This analysis annotates the AST with:
 * a dict mapping of read variables to a list of inferred types
 * a dict mapping of modified variables to a list of inferred types

This is experimental and does not infer types for variables
assigned to a python object.

Requires activity analysis.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg


class Analyzer(cfg.GraphVisitor):
  """CFG visitor that performs type inference at each node."""

  def __init__(self, graph):
    super(Analyzer, self).__init__(graph)

  def init_state(self, _):
    return

  def add_types_sets(self, node):
    if (anno.hasanno(node.ast_node, "type_anno_read")
        and anno.hasanno(node.ast_node, "type_anno_write")):
      return
    type_anno_read = {}
    type_anno_write = {}
    anno.setanno(node.ast_node, "type_anno_read", type_anno_read)
    anno.setanno(node.ast_node, "type_anno_write", type_anno_write)
    for n in node.next:
      self.add_types_sets(n)

  def set_type_annotations(self, node):
    types = anno.getanno(node.ast_node, "type_anno_read")
    for arg in node.ast_node.args:
      if arg.annotation is not None:
        type_annot = anno.getanno(arg.annotation, anno.Basic.QN)
        types[anno.getanno(arg, anno.Basic.QN)] = {type_annot}
        anno.setanno(arg, "type_anno", (type_annot))

  # Returns a set of all common types across read variables
  def overlap_read_types(self, var_types, reads):
    result = None
    for var in var_types:
      if var in reads:
        if result is None:
          result = var_types[var]
        else:
          result = result.intersection(var_types[var])
    return result

  def add_types(self, var, set_, annotations):
    if var in set_:
      for annot in annotations:
        set_[var].add(annot)
    else:
      set_[var] = annotations

  # Returns a map to pass down to children
  def combine_type_sets(self, node):
    type_reads = anno.getanno(node.ast_node, "type_anno_read")
    type_writes = anno.getanno(node.ast_node, "type_anno_write")
    combined = type_reads.copy()
    for var in type_writes:  # Overwrite read types if written to
      combined[var] = type_writes[var]
    return combined

  # Adds Tensor type to modified variable's list of inferred types
  def is_tensor(self, node, parent_types):
    node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)
    scope_reads = node_scope.read
    scope_writes = node_scope.modified
    type_writes = anno.getanno(node.ast_node, "type_anno_write")
    for var in scope_reads:
      for annot in parent_types.get(var, {}):
        if str(annot) == "ops.Tensor":  # TODO(rahulkamat): check if annotation equals Tensor class
          for w in scope_writes:
            self.add_types(w, type_writes, {annot})

  # Visits children and passes type information
  def pass_down_types(self, node):
    parent_types = self.combine_type_sets(node)
    for n in node.next:
      node_scope = anno.getanno(n.ast_node, anno.Static.SCOPE)
      scope_reads = node_scope.read
      scope_writes = node_scope.modified
      type_reads = anno.getanno(n.ast_node, "type_anno_read")
      type_writes = anno.getanno(n.ast_node, "type_anno_write")
      if isinstance(n.ast_node, gast.Assign) and len(scope_reads) == 0:
        continue
      for var in parent_types:
        # Add all type information from parent into type_anno_reads
        self.add_types(var, type_reads, parent_types[var])

        if var in scope_writes:
          self.is_tensor(n, parent_types)

          possible_types = self.overlap_read_types(parent_types, scope_reads)
          if possible_types is not None:
            for w in scope_writes:
              self.add_types(w, type_writes, possible_types)

  # Removes uneeded type information in node's scope
  def remove(self, node):
    node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)
    scope_reads = node_scope.read
    scope_writes = node_scope.modified
    type_reads = anno.getanno(node.ast_node, "type_anno_read")
    type_writes = anno.getanno(node.ast_node, "type_anno_write")

    for var in list(type_reads):
      if var not in scope_reads:
        type_reads.pop(var, None)

    for var in list(type_writes):
      if var not in scope_writes:
        type_writes.pop(var, None)

  def visit_node(self, node):
    self.add_types_sets(node)
    if hasattr(node.ast_node, "args"):
      self.set_type_annotations(node)
    self.pass_down_types(node)
    if not hasattr(node.ast_node, "args"):
      self.remove(node)
    return False  # Does not repeat


def resolve(node, graph):
  analyzer = Analyzer(graph)
  analyzer.visit_forward()
  return node
