# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Connects all float and double tensors to CheckNumericsOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


def verify_tensor_all_finite(t, msg, name=None):
  """Assert that the tensor does not contain any NaN's or Inf's.

  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).

  Returns:
    Same tensor as `t`.
  """
  with ops.op_scope([t], name, "VerifyFinite") as name:
    t = ops.convert_to_tensor(t, name="t")
    with ops.colocate_with(t):
      verify_input = array_ops.check_numerics(t, message=msg)
      out = control_flow_ops.with_dependencies([verify_input], t)
  return out


def add_check_numerics_ops():
  """Connect a `check_numerics` to every floating point tensor.

  `check_numerics` operations themselves are added for each `float` or `double`
  tensor in the graph. For all ops in the graph, the `check_numerics` op for
  all of its (`float` or `double`) inputs is guaranteed to run before the
  `check_numerics` op on any of its outputs.

  Returns:
    A `group` op depending on all `check_numerics` ops added.
  """
  check_op = []
  # This code relies on the ordering of ops in get_operations().
  # The consumer of a tensor always comes before that tensor's producer in
  # this list. This is true because get_operations() returns ops in the order
  # added, and ops can only be added once its inputs are added.
  for op in ops.get_default_graph().get_operations():
    for output in op.outputs:
      if output.dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
        message = op.name + ":" + str(output.value_index)
        with ops.control_dependencies(check_op):
          check_op = [array_ops.check_numerics(output, message=message)]
  return control_flow_ops.group(*check_op)
