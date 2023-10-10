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
"""Helpers to traverse the Dataset dependency structure."""
import queue

from tensorflow.python.framework import dtypes


OP_TYPES_ALLOWLIST = ["DummyIterationCounter"]
# We allowlist all ops that produce variant tensors as output. This is a bit
# of overkill but the other dataset _inputs() traversal strategies can't
# cover the case of function inputs that capture dataset variants.
TENSOR_TYPES_ALLOWLIST = [dtypes.variant]


def _traverse(dataset, op_filter_fn):
  """Traverse a dataset graph, returning nodes matching `op_filter_fn`."""
  result = []
  bfs_q = queue.Queue()
  bfs_q.put(dataset._variant_tensor.op)  # pylint: disable=protected-access
  visited = []
  while not bfs_q.empty():
    op = bfs_q.get()
    visited.append(op)
    if op_filter_fn(op):
      result.append(op)
    for i in op.inputs:
      input_op = i.op
      if input_op not in visited:
        bfs_q.put(input_op)
  return result


def obtain_capture_by_value_ops(dataset):
  """Given an input dataset, finds all allowlisted ops used for construction.

  Allowlisted ops are stateful ops which are known to be safe to capture by
  value.

  Args:
    dataset: Dataset to find allowlisted stateful ops for.

  Returns:
    A list of variant_tensor producing dataset ops used to construct this
    dataset.
  """

  def capture_by_value(op):
    return (op.outputs[0].dtype in TENSOR_TYPES_ALLOWLIST or
            op.type in OP_TYPES_ALLOWLIST)

  return _traverse(dataset, capture_by_value)


def obtain_all_variant_tensor_ops(dataset):
  """Given an input dataset, finds all dataset ops used for construction.

  A series of transformations would have created this dataset with each
  transformation including zero or more Dataset ops, each producing a dataset
  variant tensor. This method outputs all of them.

  Args:
    dataset: Dataset to find variant tensors for.

  Returns:
    A list of variant_tensor producing dataset ops used to construct this
    dataset.
  """
  return _traverse(dataset, lambda op: op.outputs[0].dtype == dtypes.variant)
