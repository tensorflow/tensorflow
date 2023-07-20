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
# =============================================================================
"""An extension type that represents WeakTensor."""


from typing import Optional

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry


_ALLOWED_WEAK_DTYPES = (
    dtypes.int32,
    dtypes.int64,
    dtypes.float32,
    dtypes.float64,
    dtypes.complex128,
)


class WeakTensorGradient(composite_tensor_gradient.CompositeTensorGradient):
  """CompositeTensorGradient for WeakTensor."""

  def get_gradient_components(self, weak_tensor):
    return weak_tensor.tensor

  def replace_gradient_components(self, weak_tensor, component_grads):
    return weak_tensor._type_spec._from_components([component_grads])  # pylint: disable=protected-access


# TODO(b/285024542): Modify the isinstance() checks to include WeakTensor.
# instance.
class WeakTensor(extension_type.ExtensionType):
  """A weakly typed Tensor.

  A simple wrapper class that contains a normal Tensor.

  A "weak" type means that its dtype is temporarily inferred by the system,
  and could defer to other dtypes.

  i.g. weak f64 + f16 => f16

  This information is used for auto dtype conversion.
  """

  # __name__ is required for serialization in SavedModel.
  __name__ = "tf.WeakTensor"
  tensor: tensor_lib.Tensor

  def __validate__(self):
    if self.tensor.dtype not in _ALLOWED_WEAK_DTYPES:
      raise TypeError(
          f"{self.tensor.dtype} not allowed "
          f"as a weak type. The allowed types are {_ALLOWED_WEAK_DTYPES}."
      )

  def __str__(self):
    return self._format_weak_tensor(is_repr=False)

  def __repr__(self):
    return self._format_weak_tensor(is_repr=True)

  def _format_weak_tensor(self, is_repr):
    tensor_str = self.tensor.__repr__() if is_repr else self.tensor.__str__()
    closing_char = tensor_str[len(tensor_str) - 1]
    last_index = tensor_str.rfind(closing_char)
    return tensor_str[:last_index] + ", weak=True" + closing_char

  def __getattr__(self, *args, **kwargs):
    # Fallback to `__getattr__` if `__getattribute__` fails, so that we can
    # directly expose Tensor's methods.
    return getattr(self.tensor, *args, **kwargs)

  def __array__(self, dtype=None):
    # We need to explicitly call np.array() because
    # self_tensor.__array__() for scalars raise:
    #     ValueError: object __array__ method not producing an array
    # resource_variable_ops also follows the same pattern.
    return np.array(self.tensor.__array__(dtype))

  def _disallow(self, task):
    raise errors.OperatorNotAllowedInGraphError(
        f"{task} is not allowed. You can attempt the following resolutions to"
        " the problem: If you are running in Graph mode, use Eager execution"
        " mode or decorate this function with @tf.function. If you are using"
        " AutoGraph, you can try decorating this function with @tf.function."
        " If that does not work, then you may be using an unsupported feature"
        " or your source code may not be visible to AutoGraph. See"
        " https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code"
        " for more information."
    )

  def _disallow_iteration(self):
    self._disallow("Iterating over a symbolic `tf.WeakTensor`")

  def _shape_as_list(self):
    if self.shape.ndims is not None:
      return [dim.value for dim in self.shape.dims]
    else:
      return None

  def __iter__(self):
    if not context.executing_eagerly():
      self._disallow_iteration()
    first_dim = self.tensor._get_first_dim()
    return _WeakTensorIterator(self, first_dim)

  def __hash__(self):
    return self.tensor.__hash__()

  def __copy__(self):
    # Weak Tensors are immutable so it's safe to return themselves as a copy.
    return self

  def __len__(self):
    return self.tensor.__len__()

  def __bool__(self):
    return self.tensor.__bool__()

  def __tf_tensor__(
      self, dtype: Optional[dtypes.DType] = None, name: Optional[str] = None
  ):
    return self.tensor.__tf_tensor__(dtype=dtype, name=name)

  def __format__(self, format_spec):
    return f"{self.tensor.__format__(format_spec)} weakly typed"

  def __complex__(self):
    return self.tensor.__complex__()

  def __int__(self):
    return self.tensor.__int__()

  def __float__(self):
    return self.tensor.__float__()

  def __index__(self):
    return self.tensor.__index__()

  def __deepcopy__(self, memo):
    # Eager Tensors are immutable so it's safe to return themselves as a copy.
    del memo
    return self

  def to_tensor(self):
    """Converts this 'WeakTensor' into a 'tf.Tensor'."""
    return self.tensor

  def numpy(self):
    """Copy of the contents of this WeakTensor into a NumPy array or scalar."""
    if not isinstance(self.tensor, ops.EagerTensor):
      raise ValueError("WeakTensor.numpy() is only supported in eager mode.")
    return self.tensor.numpy()

  def _as_graph_element(self):
    """Convert `self` to a graph element."""
    return self.tensor

  @classmethod
  def from_tensor(cls, tensor):
    """Converts a 'tf.Tensor' into a 'WeakTensor'."""
    return WeakTensor(tensor)

  # Redefine `shape` and `dtype` rather than relying on `getattr` because the
  # class derives from core.Tensor which returns None in the two methods.
  @property
  def dtype(self):
    return self.tensor.dtype

  @property
  def shape(self):
    return self.tensor.shape

  @property
  def is_tensor_like(self):
    return True

  __composite_gradient__ = WeakTensorGradient()


class _WeakTensorIterator(object):
  """Iterates over the leading dim of a WeakTensor. Performs no error checks."""

  __slots__ = ["_weak_tensor", "_index", "_limit"]

  def __init__(self, weak_tensor, dim0):
    self._weak_tensor = weak_tensor
    self._index = 0
    self._limit = dim0

  def __iter__(self):
    return self

  def __next__(self):
    if self._index == self._limit:
      raise StopIteration
    result = WeakTensor(self._weak_tensor.tensor[self._index])
    self._index += 1
    return result


def maybe_convert_to_weak_tensor(t, is_weak):
  return WeakTensor(t) if is_weak else t


# convert_to_tensor(WeakTensor) should return a WeakTensor because WeakTensor is
# a 'Tensor' with a special dtype.
def weak_tensor_conversion_function(t):
  if isinstance(t, WeakTensor):
    return t


tensor_conversion_registry.register_tensor_conversion_function(
    WeakTensor, weak_tensor_conversion_function
)
