# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Functions related to compiling TF code for the Graphcore IPU backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.contrib.compiler import xla

def compile(computation, inputs=None):
  """Builds an operator that compiles and runs `computation` with the Graphcore
     IPU XLA backend.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors.

      `computation` may return a list of operations and tensors.  Tensors must
      come before operations in the returned list.  The return value of
      `compile` is a list of tensors corresponding to the tensors from the
      output of `computation`.

      All `Operation`s returned from `computation` will be executed when
      evaluating any of the returned output tensors.
    inputs: A list of inputs or `None` (equivalent to an empty list). Each input
      can be a nested structure containing values that are convertible to
      tensors. Note that passing an N-dimension list of compatible values will
      result in a N-dimension list of scalar tensors rather than a single Rank-N
      tensors. If you need different behaviour, convert part of inputs to
      tensors with `tf.convert_to_tensor`.

  Returns:
    Same data structure as if computation(*inputs) is called directly with some
    exceptions for correctness. Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
  Raises:
    Exception: If the computation was not compiled for an IPU device.
  """
  old_op_list = ops.get_default_graph().get_operations()
  result = xla.compile(computation, inputs)

  new_op_list = ops.get_default_graph().get_operations()

  added_ops = set(old_op_list) ^ set(new_op_list)
  # Go over all the new added ops, check that they have been placed on an IPU
  # device.
  placed_on_ipu = False
  all_no_ops = True
  for o in added_ops:
    if o.device.startswith('/device:IPU'):
      placed_on_ipu = True
      break
    elif o.type != 'NoOp':
      all_no_ops = False

  if not placed_on_ipu and not all_no_ops:
    raise Exception("""\
A computation has been compiled, however it was not placed on an IPU device. \
This computation will not be executed on an IPU.
To execute it on an IPU use the `ipu_scope` from `tensorflow.contrib.ipu.ops`, \
for example:

  with ipu_scope('/device:IPU:0'):
    result = ipu_compiler.compile(comp, inputs)
""")
  return result
