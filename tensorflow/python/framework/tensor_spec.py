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
"""A TensorSpec class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util.tf_export import tf_export


@tf_export("TensorSpec")
class TensorSpec(object):
  """Describes a tf.Tensor.

  Metadata for describing the `tf.Tensor` objects accepted or returned
  by some TensorFlow APIs.
  """

  __slots__ = ["_shape", "_shape_tuple", "_dtype", "_name"]

  def __init__(self, shape, dtype=dtypes.float32, name=None):
    """Creates a TensorSpec.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      name: Optional name for the Tensor.

    Raises:
      TypeError: If shape is not convertible to a `tf.TensorShape`, or dtype is
        not convertible to a `tf.DType`.
    """
    self._shape = tensor_shape.TensorShape(shape)
    try:
      self._shape_tuple = tuple(self.shape.as_list())
    except ValueError:
      self._shape_tuple = None
    self._dtype = dtypes.as_dtype(dtype)
    self._name = name

  @classmethod
  def from_spec(cls, spec, name=None):
    return cls(spec.shape, spec.dtype, name or spec.name)

  @classmethod
  def from_tensor(cls, tensor, name=None):
    if isinstance(tensor, ops.EagerTensor):
      return TensorSpec(tensor.shape, tensor.dtype, name)
    elif isinstance(tensor, ops.Tensor):
      return TensorSpec(tensor.shape, tensor.dtype, name or tensor.op.name)
    else:
      raise ValueError("`tensor` should be a tf.Tensor")

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype

  @property
  def name(self):
    """Returns the (optionally provided) name of the described tensor."""
    return self._name

  def is_compatible_with(self, spec_or_tensor):
    """Returns True if spec_or_tensor is compatible with this TensorSpec.

    Two tensors are considered compatible if they have the same dtype
    and their shapes are compatible (see `tf.TensorShape.is_compatible_with`).

    Args:
      spec_or_tensor: A tf.TensorSpec or a tf.Tensor

    Returns:
      True if spec_or_tensor is compatible with self.
    """
    return (self._dtype.is_compatible_with(spec_or_tensor.dtype) and
            self._shape.is_compatible_with(spec_or_tensor.shape))

  def __repr__(self):
    return "TensorSpec(shape={}, dtype={}, name={})".format(
        self.shape, repr(self.dtype), repr(self.name))

  def __hash__(self):
    return hash((self._shape_tuple, self.dtype))

  def __eq__(self, other):
    return (self._shape_tuple == other._shape_tuple  # pylint: disable=protected-access
            and self.dtype == other.dtype
            and self._name == other._name)  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __reduce__(self):
    return TensorSpec, (self._shape, self._dtype, self._name)


class BoundedTensorSpec(TensorSpec):
  """A `TensorSpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
  tf_minimum = tf.convert_to_tensor(spec.minimum, dtype=spec.dtype)
  tf_maximum = tf.convert_to_tensor(spec.maximum, dtype=spec.dtype)
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by tensors
  with values in the set {0, 1, 2}:
  ```python
  spec = tensor_spec.BoundedTensorSpec((3, 5), tf.int32, 0, 2)
  ```
  """

  __slots__ = ("_minimum", "_maximum")

  def __init__(self, shape, dtype, minimum, maximum, name=None):
    """Initializes a new `BoundedTensorSpec`.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      minimum: Number or sequence specifying the minimum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not provided or not
        broadcastable to `shape`.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    super(BoundedTensorSpec, self).__init__(shape, dtype, name)

    if minimum is None or maximum is None:
      raise ValueError("minimum and maximum must be provided; but saw "
                       "'%s' and '%s'" % (minimum, maximum))

    try:
      minimum_shape = np.shape(minimum)
      common_shapes.broadcast_shape(
          tensor_shape.TensorShape(minimum_shape), self.shape)
    except ValueError as exception:
      raise ValueError("minimum is not compatible with shape. "
                       "Message: {!r}.".format(exception))

    try:
      maximum_shape = np.shape(maximum)
      common_shapes.broadcast_shape(
          tensor_shape.TensorShape(maximum_shape), self.shape)
    except ValueError as exception:
      raise ValueError("maximum is not compatible with shape. "
                       "Message: {!r}.".format(exception))

    self._minimum = np.array(minimum, dtype=self.dtype.as_numpy_dtype())
    self._minimum.setflags(write=False)

    self._maximum = np.array(maximum, dtype=self.dtype.as_numpy_dtype())
    self._maximum.setflags(write=False)

  @classmethod
  def from_spec(cls, spec):
    dtype = dtypes.as_dtype(spec.dtype)
    minimum = getattr(spec, "minimum", dtype.min)
    maximum = getattr(spec, "maximum", dtype.max)
    return BoundedTensorSpec(spec.shape, dtype, minimum, maximum, spec.name)

  @property
  def minimum(self):
    """Returns a NumPy array specifying the minimum bounds (inclusive)."""
    return self._minimum

  @property
  def maximum(self):
    """Returns a NumPy array specifying the maximum bounds (inclusive)."""
    return self._maximum

  def __repr__(self):
    s = "BoundedTensorSpec(shape={}, dtype={}, name={}, minimum={}, maximum={})"
    return s.format(self.shape, repr(self.dtype), repr(self.name),
                    repr(self.minimum), repr(self.maximum))

  def __eq__(self, other):
    tensor_spec_eq = super(BoundedTensorSpec, self).__eq__(other)
    return (tensor_spec_eq and np.allclose(self.minimum, other.minimum) and
            np.allclose(self.maximum, other.maximum))

  def __reduce__(self):
    return BoundedTensorSpec, (self._shape, self._dtype, self._minimum,
                               self._maximum, self._name)
