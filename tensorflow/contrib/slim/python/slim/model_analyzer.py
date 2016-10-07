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
"""Tools for analyzing the operations and variables in a TensorFlow graph.

To analyze the operations in a graph:

  images, labels = LoadData(...)
  predictions = MyModel(images)

  slim.model_analyzer.analyze_ops(tf.get_default_graph(), print_info=True)

To analyze the model variables in a graph:

  variables = tf.model_variables()
  slim.model_analyzer.analyze_vars(variables, print_info=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def tensor_description(var):
  """Returns a compact and informative string about a tensor.

  Args:
    var: A tensor variable.

  Returns:
    a string with type and size, e.g.: (float32 1x8x8x1024).
  """
  description = '(' + str(var.dtype.name) + ' '
  sizes = var.get_shape()
  for i, size in enumerate(sizes):
    description += str(size)
    if i < len(sizes) - 1:
      description += 'x'
  description += ')'
  return description


def analyze_ops(graph, print_info=False):
  """Compute the estimated size of the ops.outputs in the graph.

  Args:
    graph: the graph containing the operations.
    print_info: Optional, if true print ops and their outputs.

  Returns:
    total size of the ops.outputs
  """
  if print_info:
    print('---------')
    print('Operations: name -> (type shapes) [size]')
    print('---------')
  total_size = 0
  for op in graph.get_operations():
    op_size = 0
    shapes = []
    for output in op.outputs:
      # if output.num_elements() is None or [] assume size 0.
      output_size = output.get_shape().num_elements() or 0
      if output.get_shape():
        shapes.append(tensor_description(output))
      op_size += output_size
    if print_info:
      print(op.name, '\t->', ', '.join(shapes), '[' + str(op_size) + ']')
    total_size += op_size
  return total_size


def analyze_vars(variables, print_info=False):
  """Prints the names and shapes of the variables.

  Args:
    variables: list of variables, for example tf.all_variables().
    print_info: Optional, if true print variables and their shape.

  Returns:
    (total size of the variables, total bytes of the variables)
  """
  if print_info:
    print('---------')
    print('Variables: name (type shape) [size]')
    print('---------')
  total_size = 0
  total_bytes = 0
  for var in variables:
    # if var.num_elements() is None or [] assume size 0.
    var_size = var.get_shape().num_elements() or 0
    var_bytes = var_size * var.dtype.size
    total_size += var_size
    total_bytes += var_bytes
    if print_info:
      print(var.name, tensor_description(var), '[%d, bytes: %d]' %
            (var_size, var_bytes))
  if print_info:
    print('Total size of variables: %d' % total_size)
    print('Total bytes of variables: %d' % total_bytes)
  return total_size, total_bytes
