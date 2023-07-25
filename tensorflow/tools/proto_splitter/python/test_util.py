# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for Python tests."""

from collections.abc import Sequence
import math
from typing import Optional

import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util


def make_graph_def_with_constant_nodes(
    node_sizes: Sequence[int],
    dtype: Optional[dtypes.DType] = None,
    **function_node_sizes,
) -> graph_pb2.GraphDef:
  """Creates a GraphDef with approximate node sizes.

  Args:
    node_sizes: list of ints, the approximate desired sizes of the nodes in the
      GraphDef.
    dtype: Dtype of encoded constant values (float32 or float64).
    **function_node_sizes: Map of function name to FunctionDef node sizes (see
      `node_sizes`).

  Returns:
    A GraphDef proto.
  """
  dtype = dtypes.float32
  graph_def = graph_pb2.GraphDef()
  n = 0

  def add_nodes(node_list, sizes):
    nonlocal n
    for s in sizes:
      node = node_list.add(name=f"Const_{n}", op="Const")

      # Add an empty value to compute the approximate size of the constant
      # value that will be added to the proto.
      node.attr["value"].tensor.MergeFrom(
          tensor_util.make_tensor_proto(np.ones([]), dtype=dtype)
      )
      remaining_size = s - node.ByteSize()
      if remaining_size < 0:
        raise ValueError(f"Unable to create node of size {s} bytes.")

      constant_size = [math.ceil(remaining_size / dtype.size)]
      node.attr["value"].tensor.Clear()
      node.attr["value"].tensor.MergeFrom(
          tensor_util.make_tensor_proto(
              np.random.random_sample(constant_size), dtype=dtype
          )
      )
      n += 1

  add_nodes(graph_def.node, node_sizes)
  for fn_name, fn_sizes in function_node_sizes.items():
    fn = graph_def.library.function.add()
    fn.signature.name = fn_name
    add_nodes(fn.node_def, fn_sizes)

  return graph_def
