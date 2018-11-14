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
"""Logic to update a Tensorflow model graph with quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensorflow.contrib.quantize.python import common


class InputToOps(object):
  """Holds a mapping from tensor's name to ops that take it as input."""

  def __init__(self, graph):
    """Initializes mapping from tensor's name to ops that take it.

    Helps find edges between ops faster and avoids iterating over the whole
    graph.   The mapping is of type Dict[str, Set[tf.Operation]].

    Note: while inserting operations into the graph, we do not update the
    mapping, assuming that insertion points in the graph are never adjacent.
    With that restriction, an out of date mapping still works fine.

    Args:
      graph: Graph to process.
    """
    self.mapping = collections.defaultdict(set)
    for op in (op for op in graph.get_operations()):
      if op.name.startswith(common.SKIPPED_PREFIXES):
        continue
      for op_input in op.inputs:
        self.mapping[op_input].add(op)

  def ConsumerOperations(self, producer_op):
    """Looks through outputs of producer_op, finds ops that take them as input.

    Args:
      producer_op: Operation containing outputs to process.

    Returns:
      A Set[Operation] containing all operations taking input from producer_op
        outputs.
    """
    result = set()
    for inp in producer_op.outputs:
      result.update(self.mapping[inp])
    return result
