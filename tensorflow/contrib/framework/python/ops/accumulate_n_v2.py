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
"""Ops that will eventually be folded into tensorflow/python/ops/math_ops.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops



def accumulate_n_v2(inputs, shape=None, tensor_dtype=None, name=None):
  """Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  `tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
  wait for all of its inputs to be ready before beginning to sum. This can
  save memory if inputs are ready at different times, since minimum temporary
  storage is proportional to the output size rather than the inputs size.

  Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.

  For example:

  ```python
  a = tf.constant([[1, 2], [3, 4]])
  b = tf.constant([[5, 0], [0, 6]])
  tf.accumulate_n_v2([a, b, a])  # [[7, 4], [6, 14]]

  # Explicitly pass shape and type
  tf.accumulate_n_v2([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)
                                                                   # [[7,  4],
                                                                   #  [6, 14]]
  ```

  Args:
    inputs: A list of `Tensor` objects, each with same shape and type.
    shape: Shape of elements of `inputs`.
    tensor_dtype: The type of `inputs`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """
  _INPUTS_ERR_MSG = ValueError("inputs must be a list of at least one Tensor"
                               "with the same dtype and shape")
  if not inputs or not isinstance(inputs, (list, tuple)):
    raise _INPUTS_ERR_MSG
  inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(isinstance(x, ops.Tensor) for x in inputs):
    raise _INPUTS_ERR_MSG
  if not all(x.dtype == inputs[0].dtype for x in inputs):
    raise _INPUTS_ERR_MSG
  if shape is not None:
    shape = tensor_shape.as_shape(shape)
  else:
    shape = tensor_shape.unknown_shape()
  for input_tensor in inputs:
    if isinstance(input_tensor, ops.Tensor):
      shape = shape.merge_with(input_tensor.get_shape())

  # tensor_dtype is for safety only; operator's output type computed in C++
  if tensor_dtype is not None and tensor_dtype != inputs[0].dtype:
    raise TypeError("tensor_dtype is {}, but input is of type {}"
                    .format(tensor_dtype, inputs[0].dtype))

  if len(inputs) == 1 and name is None:
    return inputs[0]
  elif len(inputs) == 1 and name is not None:
    return array_ops.identity(inputs[0], name=name)
  elif context.in_eager_mode():
    # TemporaryVariable not currently supported in eager mode; fall back
    # onto AddN for now.
    # TODO(frreiss) remove this once the lifetime of eager variables gets
    # addressed
    return math_ops.add_n(inputs, name=name)
  else:
    return gen_math_ops._accumulate_nv2(inputs, name=name, shape=shape)

# The following code should eventually be merged into
# tensorflow/python/ops/math_grad.py
@ops.RegisterGradient("AccumulateNV2")
def _AddNGrad(op, grad):
  """Same as gradient for AddN. Copies the gradient to all inputs."""
  # Not broadcasting.
  return [grad] * len(op.inputs)

