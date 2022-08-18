# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for computing common window functions."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


def _check_params(window_length, dtype):
  """Check window_length and dtype params.

  Args:
    window_length: A scalar value or `Tensor`.
    dtype: The data type to produce. Must be a floating point type.

  Returns:
    window_length converted to a tensor of type int32.

  Raises:
    ValueError: If `dtype` is not a floating point type or window_length is not
      a scalar.
  """
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Found %s' % dtype)
  window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32)
  window_length.shape.assert_has_rank(0)
  return window_length


@tf_export('signal.kaiser_window')
@dispatch.add_dispatch_support
def kaiser_window(window_length, beta=12., dtype=dtypes.float32, name=None):
  """Generate a [Kaiser window][kaiser].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    beta: Beta parameter for Kaiser window, see reference below.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  [kaiser]:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.kaiser.html
  """
  with ops.name_scope(name, 'kaiser_window'):
    window_length = _check_params(window_length, dtype)
    window_length_const = tensor_util.constant_value(window_length)
    if window_length_const == 1:
      return array_ops.ones([1], dtype=dtype)
    # tf.range does not support float16 so we work with float32 initially.
    halflen_float = (
        math_ops.cast(window_length, dtype=dtypes.float32) - 1.0) / 2.0
    arg = math_ops.range(-halflen_float, halflen_float + 0.1,
                         dtype=dtypes.float32)
    # Convert everything into given dtype which can be float16.
    arg = math_ops.cast(arg, dtype=dtype)
    beta = math_ops.cast(beta, dtype=dtype)
    one = math_ops.cast(1.0, dtype=dtype)
    two = math_ops.cast(2.0, dtype=dtype)
    halflen_float = math_ops.cast(halflen_float, dtype=dtype)
    num = beta * math_ops.sqrt(
        one - math_ops.pow(arg, two) / math_ops.pow(halflen_float, two))
    window = math_ops.exp(num - beta) * (
        special_math_ops.bessel_i0e(num) / special_math_ops.bessel_i0e(beta))
  return window


@tf_export('signal.kaiser_bessel_derived_window')
@dispatch.add_dispatch_support
def kaiser_bessel_derived_window(window_length, beta=12.,
                                 dtype=dtypes.float32, name=None):
  """Generate a [Kaiser Bessel derived window][kbd].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    beta: Beta parameter for Kaiser window.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  [kbd]:
    https://en.wikipedia.org/wiki/Kaiser_window#Kaiser%E2%80%93Bessel-derived_(KBD)_window
  """
  with ops.name_scope(name, 'kaiser_bessel_derived_window'):
    window_length = _check_params(window_length, dtype)
    halflen = window_length // 2
    kaiserw = kaiser_window(halflen + 1, beta, dtype=dtype)
    kaiserw_csum = math_ops.cumsum(kaiserw)
    halfw = math_ops.sqrt(kaiserw_csum[:-1] / kaiserw_csum[-1])
    window = array_ops.concat((halfw, halfw[::-1]), axis=0)
  return window


@tf_export('signal.vorbis_window')
@dispatch.add_dispatch_support
def vorbis_window(window_length, dtype=dtypes.float32, name=None):
  """Generate a [Vorbis power complementary window][vorbis].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  [vorbis]:
    https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform#Window_functions
  """
  with ops.name_scope(name, 'vorbis_window'):
    window_length = _check_params(window_length, dtype)
    arg = math_ops.cast(math_ops.range(window_length), dtype=dtype)
    window = math_ops.sin(np.pi / 2.0 * math_ops.pow(math_ops.sin(
        np.pi / math_ops.cast(window_length, dtype=dtype) *
        (arg + 0.5)), 2.0))
  return window


@tf_export('signal.hann_window')
@dispatch.add_dispatch_support
def hann_window(window_length, periodic=True, dtype=dtypes.float32, name=None):
  """Generate a [Hann window][hann].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window. Periodic windows are typically used for spectral
      analysis while symmetric windows are typically used for digital
      filter design.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type.

  [hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
  """
  return _raised_cosine_window(name, 'hann_window', window_length, periodic,
                               dtype, 0.5, 0.5)


@tf_export('signal.hamming_window')
@dispatch.add_dispatch_support
def hamming_window(window_length, periodic=True, dtype=dtypes.float32,
                   name=None):
  """Generate a [Hamming][hamming] window.

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window. Periodic windows are typically used for spectral
      analysis while symmetric windows are typically used for digital
      filter design.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type.

  [hamming]:
    https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
  """
  return _raised_cosine_window(name, 'hamming_window', window_length, periodic,
                               dtype, 0.54, 0.46)


def _raised_cosine_window(name, default_name, window_length, periodic,
                          dtype, a, b):
  """Helper function for computing a raised cosine window.

  Args:
    name: Name to use for the scope.
    default_name: Default name to use for the scope.
    window_length: A scalar `Tensor` or integer indicating the window length.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window.
    dtype: A floating point `DType`.
    a: The alpha parameter to the raised cosine window.
    b: The beta parameter to the raised cosine window.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type or `window_length` is
      not scalar or `periodic` is not scalar.
  """
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Found %s' % dtype)

  with ops.name_scope(name, default_name, [window_length, periodic]):
    window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32,
                                          name='window_length')
    window_length.shape.assert_has_rank(0)
    window_length_const = tensor_util.constant_value(window_length)
    if window_length_const == 1:
      return array_ops.ones([1], dtype=dtype)
    periodic = math_ops.cast(
        ops.convert_to_tensor(periodic, dtype=dtypes.bool, name='periodic'),
        dtypes.int32)
    periodic.shape.assert_has_rank(0)
    even = 1 - math_ops.mod(window_length, 2)

    n = math_ops.cast(window_length + periodic * even - 1, dtype=dtype)
    count = math_ops.cast(math_ops.range(window_length), dtype)
    cos_arg = constant_op.constant(2 * np.pi, dtype=dtype) * count / n

    if window_length_const is not None:
      return math_ops.cast(a - b * math_ops.cos(cos_arg), dtype=dtype)
    return control_flow_ops.cond(
        math_ops.equal(window_length, 1),
        lambda: array_ops.ones([window_length], dtype=dtype),
        lambda: math_ops.cast(a - b * math_ops.cos(cos_arg), dtype=dtype))
