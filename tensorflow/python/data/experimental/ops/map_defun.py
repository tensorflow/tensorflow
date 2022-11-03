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
"""Experimental API for optimizing `tf.data` pipelines."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops


def map_defun(fn,
              elems,
              output_dtypes,
              output_shapes,
              max_intra_op_parallelism=1):
  """Map a function on the list of tensors unpacked from `elems` on dimension 0.

  Args:
    fn: A function (`function.defun`) that takes a list of tensors and returns
      another list of tensors. The output list has the same types as
      output_dtypes. The elements of the output list have the same dimension 0
      as `elems`, and the remaining dimensions correspond to those of
      `fn_output_shapes`.
    elems: A list of tensors.
    output_dtypes: A list of dtypes corresponding to the output types of the
      function.
    output_shapes: A list of `TensorShape`s corresponding to the output shapes
      from each invocation of the function on slices of inputs.
    max_intra_op_parallelism: An integer. If positive, sets the max parallelism
      limit of each function call to this.

  Raises:
    ValueError: if any of the inputs are malformed.

  Returns:
    A list of `Tensor` objects with the same types as `output_dtypes`.
  """
  if not isinstance(elems, list):
    raise ValueError(f"`elems` must be a list of tensors, but was {elems}.")
  if not isinstance(output_dtypes, list):
    raise ValueError("`output_dtypes` must be a list of `tf.DType` objects, "
                     f"but was {output_dtypes}.")
  if not isinstance(output_shapes, list):
    raise ValueError("`output_shapes` must be a list of `tf.TensorShape` "
                     f"objects, but was {output_shapes}.")

  concrete_fn = fn.get_concrete_function()  # pylint: disable=protected-access
  # TODO(shivaniagrawal/rachelim): what about functions created without
  # input_signature.
  elems = [ops.convert_to_tensor(e) for e in elems]
  output_shapes = [tensor_shape.TensorShape(s) for s in output_shapes]
  return gen_dataset_ops.map_defun(elems, concrete_fn.captured_inputs,
                                   output_dtypes, output_shapes, concrete_fn,
                                   max_intra_op_parallelism)
