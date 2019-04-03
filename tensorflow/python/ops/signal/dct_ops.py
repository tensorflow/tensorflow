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
"""Discrete Cosine Transform ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as _math

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export


def _validate_dct_arguments(input_tensor, dct_type, n, axis, norm):
  """Checks that DCT/IDCT arguments are compatible and well formed."""
  if n is not None:
    raise NotImplementedError("The DCT length argument is not implemented.")
  if axis != -1:
    raise NotImplementedError("axis must be -1. Got: %s" % axis)
  if dct_type not in (1, 2, 3):
    raise ValueError("Only Types I, II and III (I)DCT are supported.")
  if dct_type == 1:
    if norm == "ortho":
      raise ValueError("Normalization is not supported for the Type-I DCT.")
    if input_tensor.shape[-1] is not None and input_tensor.shape[-1] < 2:
      raise ValueError(
          "Type-I DCT requires the dimension to be greater than one.")

  if norm not in (None, "ortho"):
    raise ValueError(
        "Unknown normalization. Expected None or 'ortho', got: %s" % norm)


# TODO(rjryan): Implement `n` and `axis` parameters.
@tf_export("signal.dct", v1=["signal.dct", "spectral.dct"])
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Discrete Cosine Transform (DCT)][dct] of `input`.

  Currently only Types I, II and III are supported.
  Type I is implemented using a length `2N` padded `tf.spectral.rfft`.
  Type II is implemented using a length `2N` padded `tf.spectral.rfft`, as
  described here:
  https://dsp.stackexchange.com/a/10606.
  Type III is a fairly straightforward inverse of Type II
  (i.e. using a length `2N` padded `tf.spectral.irfft`).

  @compatibility(scipy)
  Equivalent to scipy.fftpack.dct for Type-I, Type-II and Type-III DCT.
  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32` `Tensor` containing the signals to
      take the DCT of.
    type: The DCT type to perform. Must be 1, 2 or 3.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32` `Tensor` containing the DCT of `input`.

  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is
      not `-1`, or `norm` is not `None` or `'ortho'`.
    ValueError: If `type` is `1` and `norm` is `ortho`.

  [dct]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
  """
  _validate_dct_arguments(input, type, n, axis, norm)
  with _ops.name_scope(name, "dct", [input]):
    # We use the RFFT to compute the DCT and TensorFlow only supports float32
    # for FFTs at the moment.
    input = _ops.convert_to_tensor(input, dtype=_dtypes.float32)

    axis_dim = (tensor_shape.dimension_value(input.shape[-1])
                or _array_ops.shape(input)[-1])
    axis_dim_float = _math_ops.cast(axis_dim, _dtypes.float32)

    if type == 1:
      dct1_input = _array_ops.concat([input, input[..., -2:0:-1]], axis=-1)
      dct1 = _math_ops.real(fft_ops.rfft(dct1_input))
      return dct1

    if type == 2:
      scale = 2.0 * _math_ops.exp(
          _math_ops.complex(
              0.0, -_math_ops.range(axis_dim_float) * _math.pi * 0.5 /
              axis_dim_float))

      # TODO(rjryan): Benchmark performance and memory usage of the various
      # approaches to computing a DCT via the RFFT.
      dct2 = _math_ops.real(
          fft_ops.rfft(
              input, fft_length=[2 * axis_dim])[..., :axis_dim] * scale)

      if norm == "ortho":
        n1 = 0.5 * _math_ops.rsqrt(axis_dim_float)
        n2 = n1 * _math_ops.sqrt(2.0)
        # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
        weights = _array_ops.pad(
            _array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
            constant_values=n2)
        dct2 *= weights

      return dct2

    elif type == 3:
      if norm == "ortho":
        n1 = _math_ops.sqrt(axis_dim_float)
        n2 = n1 * _math_ops.sqrt(0.5)
        # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
        weights = _array_ops.pad(
            _array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
            constant_values=n2)
        input *= weights
      else:
        input *= axis_dim_float
      scale = 2.0 * _math_ops.exp(
          _math_ops.complex(
              0.0,
              _math_ops.range(axis_dim_float) * _math.pi * 0.5 /
              axis_dim_float))
      dct3 = _math_ops.real(
          fft_ops.irfft(
              scale * _math_ops.complex(input, 0.0),
              fft_length=[2 * axis_dim]))[..., :axis_dim]

      return dct3


# TODO(rjryan): Implement `n` and `axis` parameters.
@tf_export("signal.idct", v1=["signal.idct", "spectral.idct"])
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Inverse Discrete Cosine Transform (DCT)][idct] of `input`.

  Currently only Types I, II and III are supported. Type III is the inverse of
  Type II, and vice versa.

  Note that you must re-normalize by 1/(2n) to obtain an inverse if `norm` is
  not `'ortho'`. That is:
  `signal == idct(dct(signal)) * 0.5 / signal.shape[-1]`.
  When `norm='ortho'`, we have:
  `signal == idct(dct(signal, norm='ortho'), norm='ortho')`.

  @compatibility(scipy)
  Equivalent to scipy.fftpack.idct for Type-I, Type-II and Type-III DCT.
  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.idct.html
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32` `Tensor` containing the signals to take
      the DCT of.
    type: The IDCT type to perform. Must be 1, 2 or 3.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32` `Tensor` containing the IDCT of `input`.

  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is
      not `-1`, or `norm` is not `None` or `'ortho'`.

  [idct]:
  https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms
  """
  _validate_dct_arguments(input, type, n, axis, norm)
  inverse_type = {1: 1, 2: 3, 3: 2}[type]
  return dct(input, type=inverse_type, n=n, axis=axis, norm=norm, name=name)
