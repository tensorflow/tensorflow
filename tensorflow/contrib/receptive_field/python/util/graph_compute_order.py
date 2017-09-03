# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Library to compute order of computations in a graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class GraphDefHelper(object):
  """Helper class to collect node names and definitions.

  Example:
    b = GraphDefHelper(graph_def)
    # Prints node that produces given output.
    print b.output_of['conv/foo/bar']
  """

  def __init__(self, gd):
    self.output_of = {}
    for each in gd.node:
      self.output_of[each.name] = each


# pylint: disable=invalid-name
_NodeEntry = collections.namedtuple('NodeEntry', field_names=['order', 'node'])


def _get_computed_nodes(g, output, seen):
  """Traverses the graph in topological order.

  Args:
    g: GraphDefHelper object.
    output: current node.
    seen: map of nodes we've already traversed.
  Returns:
    order in topological sort for 'output'.
  """
  if output in seen:
    return seen[output].order
  node_def = g.output_of.get(output, None)
  if node_def is None:
    seen[output] = _NodeEntry(0, None)
    return 0

  r = 0
  for each in node_def.input:
    # Parses name of input node.
    if each.startswith('^'):
      each = each[1:]
    each = each.split(':')[0]
    # Recursively computes ordering.
    new_v = _get_computed_nodes(g, each, seen)
    r = max(r, new_v + 1)

  seen[output] = _NodeEntry(r, node_def)

  return seen[output].order


def get_compute_order(graph_def):
  """Computes order of computation for a given graph.

  Args:
    graph_def: GraphDef object.
  Returns:
    map: name -> {order, node}
  """
  helper = GraphDefHelper(graph_def)
  seen = collections.defaultdict(_NodeEntry)
  for each in graph_def.node:
    _get_computed_nodes(helper, each.name, seen)
  return seen
