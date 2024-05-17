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
"""Tensor shape utilities."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


def shape_tensor(shape):  # pylint: disable=invalid-name
  """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
  dtype = None
  if isinstance(shape, (tuple, list)):
    if not shape:
      dtype = dtypes.int32
    else:
      # If there are Dimension objects in the shape, unwrap them. This can be a
      # problem if v1 and v2 TensorShape objects get mixed up in partial
      # conversions, leading to shapes such as (1, 2, Dimension(5)), which are
      # not convertible to Tensors because of mixed content.
      shape = tuple(map(tensor_shape.dimension_value, shape))
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


# DO NOT USE: For testing only.
_ENABLE_MAYBE_SET_STATIC_SHAPE = True


def maybe_set_static_shape(tensor, shape):  # pylint: disable=invalid-name
  """Sets the shape of `tensor` to the `shape`'s constant value, if inferrable.

  This is a temporary workaround to fix shape inference across functional op
  boundaries. E.g.

  ```python
  shape = tf.constant([3])
  @tf.function
  def f():
    u = tf.random_uniform(shape)
    return u
  ```

  If we were to rely solely on C++ shape inference, the shape of `u` inside
  `f` would be unknown because C++ shape inference is not aware of the outer
  graph and all it sees is a Placeholder node when backtracing the captured
  tensor for `shape`. `maybe_set_static_shape` computes the static shape value
  of `shape` by traversing the `FuncGraph` boundaries and sets the correct
  shape.

  A longer term solution would be to fix C++ shape inference.

  Args:
    tensor: A tensor.
    shape: A shape tensor.
  """
  if (_ENABLE_MAYBE_SET_STATIC_SHAPE and not context.executing_eagerly() and
      ops.get_default_graph().building_function and
      not tensor.shape.is_fully_defined() and tensor_util.is_tensor(shape)):
    shape = shape_tensor(shape)
    const_shape = tensor_util.constant_value_as_shape(shape)
    tensor.set_shape(const_shape)
