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
  if axis != -1:
    raise NotImplementedError("axis must be -1. Got: %s" % axis)
  if n is not None and n < 1:
    raise ValueError("n should be a positive integer or None")
  if dct_type not in (1, 2, 3, 4):
    raise ValueError("Types I, II, III and IV (I)DCT are supported.")
  if dct_type == 1:
    if norm == "ortho":
      raise ValueError("Normalization is not supported for the Type-I DCT.")
    if input_tensor.shape[-1] is not None and input_tensor.shape[-1] < 2:
      raise ValueError(
          "Type-I DCT requires the dimension to be greater than one.")

  if norm not in (None, "ortho"):
    raise ValueError(
        "Unknown normalization. Expected None or 'ortho', got: %s" % norm)


# TODO(rjryan): Implement `axis` parameter.
@tf_export("signal.dct", v1=["signal.dct", "spectral.dct"])
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Discrete Cosine Transform (DCT)][dct] of `input`.

  Types I, II, III and IV are supported.
  Type I is implemented using a length `2N` padded `tf.signal.rfft`.
  Type II is implemented using a length `2N` padded `tf.signal.rfft`, as
   described here: [Type 2 DCT using 2N FFT padded (Makhoul)]
   (https://dsp.stackexchange.com/a/10606).
  Type III is a fairly straightforward inverse of Type II
   (i.e. using a length `2N` padded `tf.signal.irfft`).
   Type IV is calculated through 2N length DCT2 of padded signal and
  picking the odd indices.

  @compatibility(scipy)
  Equivalent to [scipy.fftpack.dct]
   (https://docs.scipy.org/doc/scipy-1.4.0/reference/generated/scipy.fftpack.dct.html)
   for Type-I, Type-II, Type-III and Type-IV DCT.
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the
      signals to take the DCT of.
    type: The DCT type to perform. Must be 1, 2, 3 or 4.
    n: The length of the transform. If length is less than sequence length,
      only the first n elements of the sequence are considered for the DCT.
      If n is greater than the sequence length, zeros are padded and then
      the DCT is computed as usual.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32`/`float64` `Tensor` containing the DCT of
    `input`.

  Raises:
    ValueError: If `type` is not `1`, `2`, `3` or `4`, `axis` is
      not `-1`, `n` is not `None` or greater than 0,
      or `norm` is not `None` or `'ortho'`.
    ValueError: If `type` is `1` and `norm` is `ortho`.

  [dct]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
  """
  _validate_dct_arguments(input, type, n, axis, norm)
  with _ops.name_scope(name, "dct", [input]):
    input = _ops.convert_to_tensor(input)
    zero = _ops.convert_to_tensor(0.0, dtype=input.dtype)

    seq_len = (
        tensor_shape.dimension_value(input.shape[-1]) or
        _array_ops.shape(input)[-1])
    if n is not None:
      if n <= seq_len:
        input = input[..., 0:n]
      else:
        rank = len(input.shape)
        padding = [[0, 0] for _ in range(rank)]
        padding[rank - 1][1] = n - seq_len
        padding = _ops.convert_to_tensor(padding, dtype=_dtypes.int32)
        input = _array_ops.pad(input, paddings=padding)

    axis_dim = (tensor_shape.dimension_value(input.shape[-1])
                or _array_ops.shape(input)[-1])
    axis_dim_float = _math_ops.cast(axis_dim, input.dtype)

    if type == 1:
      dct1_input = _array_ops.concat([input, input[..., -2:0:-1]], axis=-1)
      dct1 = _math_ops.real(fft_ops.rfft(dct1_input))
      return dct1

    if type == 2:
      scale = 2.0 * _math_ops.exp(
          _math_ops.complex(
              zero, -_math_ops.range(axis_dim_float) * _math.pi * 0.5 /
              axis_dim_float))

      # TODO(rjryan): Benchmark performance and memory usage of the various
      # approaches to computing a DCT via the RFFT.
      dct2 = _math_ops.real(
          fft_ops.rfft(
              input, fft_length=[2 * axis_dim])[..., :axis_dim] * scale)

      if norm == "ortho":
        n1 = 0.5 * _math_ops.rsqrt(axis_dim_float)
        n2 = n1 * _math.sqrt(2.0)
        # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
        weights = _array_ops.pad(
            _array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
            constant_values=n2)
        dct2 *= weights

      return dct2

    elif type == 3:
      if norm == "ortho":
        n1 = _math_ops.sqrt(axis_dim_float)
        n2 = n1 * _math.sqrt(0.5)
        # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
        weights = _array_ops.pad(
            _array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
            constant_values=n2)
        input *= weights
      else:
        input *= axis_dim_float
      scale = 2.0 * _math_ops.exp(
          _math_ops.complex(
              zero,
              _math_ops.range(axis_dim_float) * _math.pi * 0.5 /
              axis_dim_float))
      dct3 = _math_ops.real(
          fft_ops.irfft(
              scale * _math_ops.complex(input, zero),
              fft_length=[2 * axis_dim]))[..., :axis_dim]

      return dct3

    elif type == 4:
      # DCT-2 of 2N length zero-padded signal, unnormalized.
      dct2 = dct(input, type=2, n=2*axis_dim, axis=axis, norm=None)
      # Get odd indices of DCT-2 of zero padded 2N signal to obtain
      # DCT-4 of the original N length signal.
      dct4 = dct2[..., 1::2]
      if norm == "ortho":
        dct4 *= _math.sqrt(0.5) * _math_ops.rsqrt(axis_dim_float)

      return dct4


# TODO(rjryan): Implement `n` and `axis` parameters.
@tf_export("signal.idct", v1=["signal.idct", "spectral.idct"])
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Inverse Discrete Cosine Transform (DCT)][idct] of `input`.

  Currently Types I, II, III, IV are supported. Type III is the inverse of
  Type II, and vice versa.

  Note that you must re-normalize by 1/(2n) to obtain an inverse if `norm` is
  not `'ortho'`. That is:
  `signal == idct(dct(signal)) * 0.5 / signal.shape[-1]`.
  When `norm='ortho'`, we have:
  `signal == idct(dct(signal, norm='ortho'), norm='ortho')`.

  @compatibility(scipy)
  Equivalent to [scipy.fftpack.idct]
   (https://docs.scipy.org/doc/scipy-1.4.0/reference/generated/scipy.fftpack.idct.html)
   for Type-I, Type-II, Type-III and Type-IV DCT.
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the
      signals to take the DCT of.
    type: The IDCT type to perform. Must be 1, 2, 3 or 4.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32`/`float64` `Tensor` containing the IDCT of
    `input`.

  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is
      not `-1`, or `norm` is not `None` or `'ortho'`.

  [idct]:
  https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms
  """
  _validate_dct_arguments(input, type, n, axis, norm)
  inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
  return dct(input, type=inverse_type, n=n, axis=axis, norm=norm, name=name)
