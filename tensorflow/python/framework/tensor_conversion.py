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
"""Tensor conversion functions."""
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_export


def convert_to_tensor_v1(
    value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None
):
  """Converts the given `value` to a `Tensor` (with the TF1 API)."""
  preferred_dtype = deprecation.deprecated_argument_lookup(
      "dtype_hint", dtype_hint, "preferred_dtype", preferred_dtype
  )
  return convert_to_tensor_v2(value, dtype, preferred_dtype, name)


@tf_export.tf_export(v1=["convert_to_tensor"])
@dispatch.add_dispatch_support
def convert_to_tensor_v1_with_dispatch(
    value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None
):
  """Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars. For example:

  ```python
  import numpy as np

  def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

  # The following calls are equivalent.
  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  ```

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    preferred_dtype: Optional element type for the returned tensor, used when
      dtype is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so preferred_dtype can be used as a soft
      preference.  If the conversion to `preferred_dtype` is not possible, this
      argument has no effect.
    dtype_hint: same meaning as preferred_dtype, and overrides it.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  """
  return convert_to_tensor_v1(
      value,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      dtype_hint=dtype_hint,
  )


@tf_export.tf_export("convert_to_tensor", v1=[])
@dispatch.add_dispatch_support
def convert_to_tensor_v2_with_dispatch(
    value, dtype=None, dtype_hint=None, name=None
):
  """Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars.

  For example:

  >>> import numpy as np
  >>> def my_func(arg):
  ...   arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  ...   return arg

  >>> # The following calls are equivalent.
  ...
  >>> value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  >>> print(value_1)
  tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
  >>> value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  >>> print(value_2)
  tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
  >>> value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  >>> print(value_3)
  tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor, used when dtype
      is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so dtype_hint can be used as a soft preference. If
      the conversion to `dtype_hint` is not possible, this argument has no
      effect.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  """
  return convert_to_tensor_v2(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name
  )


def convert_to_tensor_v2(value, dtype=None, dtype_hint=None, name=None):
  """Converts the given `value` to a `Tensor`."""
  # preferred_dtype = preferred_dtype or dtype_hint
  return tensor_conversion_registry.convert(
      value, dtype, name, preferred_dtype=dtype_hint
  )
