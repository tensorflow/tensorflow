# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Classes and functions used to construct graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops


__all__ = ['get_graph_from_inputs',
           'get_name_scope']


def get_graph_from_inputs(op_input_list, graph=None):
  """Returns the appropriate graph to use for the given inputs.

  1. If `graph` is provided, we validate that all inputs in `op_input_list` are
     from the same graph.
  2. Otherwise, we attempt to select a graph from the first Operation- or
     Tensor-valued input in `op_input_list`, and validate that all other
     such inputs are in the same graph.
  3. If the graph was not specified and it could not be inferred from
     `op_input_list`, we attempt to use the default graph.

  Args:
    op_input_list: A list of inputs to an operation, which may include `Tensor`,
      `Operation`, and other objects that may be converted to a graph element.
    graph: (Optional) The explicit graph to use.

  Raises:
    TypeError: If `op_input_list` is not a list or tuple, or if graph is not a
      Graph.
    ValueError: If a graph is explicitly passed and not all inputs are from it,
      or if the inputs are from multiple graphs, or we could not find a graph
      and there was no default graph.

  Returns:
    The appropriate graph to use for the given inputs.
  """
  # pylint: disable=protected-access
  return ops._get_graph_from_inputs(op_input_list, graph)


def get_name_scope():
  """Returns the current name scope of the default graph.

  For example:

    ```python
    with tf.name_scope('scope1'):
      with tf.name_scope('scope2'):
        print(tf.contrib.framework.get_name_scope())
    ```
    would print the string `scope1/scope2`.

  Returns:
    A string representing the current name scope.
  """
  return ops.get_default_graph().get_name_scope()
