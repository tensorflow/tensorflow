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
"""API to simulate quantization on a python graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantize.python import copy_graph
from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.contrib.quantize.python import quantize
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


def _create_graph(input_graph,
                  is_training,
                  elements=None,
                  device_name_or_function=None):
  """Returns a transformed training input_graph for simulated quantization.

  The forward pass has fake quantization ops inserted to simulate the error
  introduced by quantization.

  Args:
    input_graph: The tf.Graph to be transformed.
    is_training: Whether quantizing training or eval graph.
    elements: (Optional) List of Tensors and Operations in input_graph whose
        corresponding elements in the new graph will be returned.
    device_name_or_function: (Optional) The device name or function to use.

  Returns:
    g is new tf.Graph that is rewritten for simulated quantization.
    l is a list of Tensors/Operations in g corresponding to the provided input
        elements, if elements is not None.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  # TODO(suharshs): Describe the process in more detail in the doc string.
  g = copy_graph.CopyGraph(input_graph)
  with g.as_default():
    with ops.device(device_name_or_function):
      fold_batch_norms.FoldBatchNorms(g)
      quantize.Quantize(g, is_training=is_training)
  if elements is None:
    return g

  return_elements = []
  for element in elements:
    if isinstance(element, (ops.Tensor, variables.Variable)):
      return_elements.append(g.get_tensor_by_name(element.name))
    elif isinstance(element, ops.Operation):
      return_elements.append(g.get_operation_by_name(element.name))
    else:
      raise ValueError(
          'elements must consist of Tensor or Operation objects, got: ',
          str(element))
  return g, return_elements


def create_training_graph(input_graph,
                          elements=None,
                          device_name_or_function=None):
  """Returns a transformed training input_graph for simulated quantization.

  The forward pass has fake quantization ops inserted to simulate the error
  introduced by quantization.

  Args:
    input_graph: The tf.Graph to be transformed.
    elements: (Optional) List of Tensors and Operations in input_graph whose
        corresponding elements in the new graph will be returned.
    device_name_or_function: (Optional) The device name or function to use.

  Returns:
    g is new tf.Graph that is rewritten for simulated quantization.
    l is a list of Tensors/Operations in g corresponding to the provided input
        elements, if elements is not None.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  return _create_graph(
      input_graph=input_graph,
      is_training=True,
      elements=elements,
      device_name_or_function=device_name_or_function)


def create_eval_graph(input_graph, elements=None, device_name_or_function=None):
  """Returns a transformed eval input_graph for simulated quantization.

  The forward pass has fake quantization ops inserted to simulate the error
  introduced by quantization.

  Args:
    input_graph: The tf.Graph to be transformed.
    elements: (Optional) List of Tensors and Operations in input_graph whose
        corresponding elements in the new graph will be returned.
    device_name_or_function: (Optional) The device name or function to use.

  Returns:
    g is new tf.Graph that is rewritten for simulated quantization.
    l is a list of Tensors/Operations in g corresponding to the provided input
        elements, if elements is not None.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  return _create_graph(
      input_graph=input_graph,
      is_training=False,
      elements=elements,
      device_name_or_function=device_name_or_function)


def experimental_create_training_graph(input_graph,
                                       elements=None,
                                       device_name_or_function=None):
  """Returns a transformed training input_graph for simulated quantization.

  This function has additional experimental options not (yet) available to
  create_training_graph. The resulting behavior may be undefined.
  The forward pass has fake quantization ops inserted to simulate the error
  introduced by quantization.

  Args:
    input_graph: The tf.Graph to be transformed.
    elements: (Optional) List of Tensors and Operations in input_graph whose
        corresponding elements in the new graph will be returned.
    device_name_or_function: (Optional) The device name or function to use.

  Returns:
    g is new tf.Graph that is rewritten for simulated quantization.
    l is a list of Tensors/Operations in g corresponding to the provided input
        elements, if elements is not None.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  return _create_graph(
      input_graph=input_graph,
      is_training=True,
      elements=elements,
      device_name_or_function=device_name_or_function)


def experimental_create_eval_graph(input_graph,
                                   elements=None,
                                   device_name_or_function=None):
  """Returns a transformed eval input_graph for simulated quantization.

  This function has additional experimental options not (yet) available to
  create_eval_graph. The resulting behavior may be undefined.
  The forward pass has fake quantization ops inserted to simulate the error
  introduced by quantization.

  Args:
    input_graph: The tf.Graph to be transformed.
    elements: (Optional) List of Tensors and Operations in input_graph whose
        corresponding elements in the new graph will be returned.
    device_name_or_function: (Optional) The device name or function to use.

  Returns:
    g is new tf.Graph that is rewritten for simulated quantization.
    l is a list of Tensors/Operations in g corresponding to the provided input
        elements, if elements is not None.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  return _create_graph(
      input_graph=input_graph,
      is_training=False,
      elements=elements,
      device_name_or_function=device_name_or_function)
