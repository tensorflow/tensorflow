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
"""Common utilities used across this package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

# Skip all operations that are backprop related or export summaries.
SKIPPED_PREFIXES = (
    'gradients/', 'RMSProp/', 'Adagrad/', 'Const_', 'HistogramSummary',
    'ScalarSummary')

# Valid activation ops for quantization end points.
_ACTIVATION_OP_SUFFIXES = ['/Relu6', '/Relu', '/Identity']

# Regular expression for recognizing nodes that are part of batch norm group.
_BATCHNORM_RE = re.compile(r'^(.*)/BatchNorm/batchnorm')


def BatchNormGroups(graph):
  """Finds batch norm layers, returns their prefixes as a list of strings.

  Args:
    graph: Graph to inspect.

  Returns:
    List of strings, prefixes of batch norm group names found.
  """
  bns = []
  for op in graph.get_operations():
    match = _BATCHNORM_RE.search(op.name)
    if match:
      bn = match.group(1)
      if not bn.startswith(SKIPPED_PREFIXES):
        bns.append(bn)
  # Filter out duplicates.
  return list(collections.OrderedDict.fromkeys(bns))


def GetEndpointActivationOp(graph, prefix):
  """Returns an Operation with the given prefix and a valid end point suffix.

  Args:
    graph: Graph where to look for the operation.
    prefix: String, prefix of Operation to return.

  Returns:
    The Operation with the given prefix and a valid end point suffix or None if
    there are no matching operations in the graph for any valid suffix
  """
  for suffix in _ACTIVATION_OP_SUFFIXES:
    activation = _GetOperationByNameDontThrow(graph, prefix + suffix)
    if activation:
      return activation
  return None


def _GetOperationByNameDontThrow(graph, name):
  """Returns an Operation with the given name.

  Args:
    graph: Graph where to look for the operation.
    name: String, name of Operation to return.

  Returns:
    The Operation with the given name. None if the name does not correspond to
    any operation in the graph
  """
  try:
    return graph.get_operation_by_name(name)
  except KeyError:
    return None


def CreateOrGetQuantizationStep():
  """Returns a Tensor of the number of steps the quantized graph has run.

  Returns:
    Quantization step Tensor.
  """
  quantization_step_name = 'fake_quantization_step'
  quantization_step_tensor_name = quantization_step_name + '/Identity:0'
  g = ops.get_default_graph()
  try:
    return g.get_tensor_by_name(quantization_step_tensor_name)
  except KeyError:
    # Create in proper graph and base name_scope.
    with g.name_scope(None):
      quantization_step_tensor = variable_scope.get_variable(
          quantization_step_name,
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES])
      with g.name_scope(quantization_step_tensor.op.name + '/'):
        # We return the incremented variable tensor. Since this is used in conds
        # for quant_delay and freeze_bn_delay, it will run once per graph
        # execution. We return an identity to force resource variables and
        # normal variables to return a tensor of the same name.
        return array_ops.identity(
            state_ops.assign_add(quantization_step_tensor, 1))
