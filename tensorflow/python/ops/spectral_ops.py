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
"""Spectral operators (e.g. DCT, FFT, RFFT).

@@dct
@@fft
@@ifft
@@fft2d
@@ifft2d
@@fft3d
@@ifft3d
@@rfft
@@irfft
@@rfft2d
@@irfft2d
@@rfft3d
@@irfft3d
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as _math

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util.all_util import remove_undocumented


def _infer_fft_length_for_rfft(input_tensor, fft_rank):
  """Infers the `fft_length` argument for a `rank` RFFT from `input_tensor`."""
  # A TensorShape for the inner fft_rank dimensions.
  fft_shape = input_tensor.get_shape()[-fft_rank:]

  # If any dim is unknown, fall back to tensor-based math.
  if not fft_shape.is_fully_defined():
    return _array_ops.shape(input_tensor)[-fft_rank:]

  # Otherwise, return a constant.
  return _ops.convert_to_tensor(fft_shape.as_list(), _dtypes.int32)


def _infer_fft_length_for_irfft(input_tensor, fft_rank):
  """Infers the `fft_length` argument for a `rank` IRFFT from `input_tensor`."""
  # A TensorShape for the inner fft_rank dimensions.
  fft_shape = input_tensor.get_shape()[-fft_rank:]

  # If any dim is unknown, fall back to tensor-based math.
  if not fft_shape.is_fully_defined():
    fft_length = _array_ops.unstack(_array_ops.shape(input_tensor)[-fft_rank:])
    fft_length[-1] = _math_ops.maximum(0, 2 * (fft_length[-1] - 1))
    return _array_ops.stack(fft_length)

  # Otherwise, return a constant.
  fft_length = fft_shape.as_list()
  if fft_length:
    fft_length[-1] = max(0, 2 * (fft_length[-1] - 1))
  return _ops.convert_to_tensor(fft_length, _dtypes.int32)


def _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length, is_reverse=False):
  """Pads `input_tensor` to `fft_length` on its inner-most `fft_rank` dims."""
  fft_shape = _tensor_util.constant_value_as_shape(fft_length)

  # Edge case: skip padding empty tensors.
  if (input_tensor.shape.ndims is not None and
      any(dim.value == 0 for dim in input_tensor.shape)):
    return input_tensor

  # If we know the shapes ahead of time, we can either skip or pre-compute the
  # appropriate paddings. Otherwise, fall back to computing paddings in
  # TensorFlow.
  if fft_shape.is_fully_defined() and input_tensor.shape.ndims is not None:
    # Slice the last FFT-rank dimensions from input_tensor's shape.
    input_fft_shape = input_tensor.shape[-fft_shape.ndims:]

    if input_fft_shape.is_fully_defined():
      # In reverse, we only pad the inner-most dimension to fft_length / 2 + 1.
      if is_reverse:
        fft_shape = fft_shape[:-1].concatenate(fft_shape[-1].value // 2 + 1)

      paddings = [[0, max(fft_dim.value - input_dim.value, 0)]
                  for fft_dim, input_dim in zip(fft_shape, input_fft_shape)]
      if any(pad > 0 for _, pad in paddings):
        outer_paddings = [[0, 0]] * max((input_tensor.shape.ndims -
                                         fft_shape.ndims), 0)
        return _array_ops.pad(input_tensor, outer_paddings + paddings)
      return input_tensor

  # If we can't determine the paddings ahead of time, then we have to pad. If
  # the paddings end up as zero, tf.pad has a special-case that does no work.
  input_rank = _array_ops.rank(input_tensor)
  input_fft_shape = _array_ops.shape(input_tensor)[-fft_rank:]
  outer_dims = _math_ops.maximum(0, input_rank - fft_rank)
  outer_paddings = _array_ops.zeros([outer_dims], fft_length.dtype)
  # In reverse, we only pad the inner-most dimension to fft_length / 2 + 1.
  if is_reverse:
    fft_length = _array_ops.concat([fft_length[:-1],
                                    fft_length[-1:] // 2 + 1], 0)
  fft_paddings = _math_ops.maximum(0, fft_length - input_fft_shape)
  paddings = _array_ops.concat([outer_paddings, fft_paddings], 0)
  paddings = _array_ops.stack([_array_ops.zeros_like(paddings), paddings],
                              axis=1)
  return _array_ops.pad(input_tensor, paddings)


def _rfft_wrapper(fft_fn, fft_rank, default_name):
  """Wrapper around gen_spectral_ops.rfft* that infers fft_length argument."""

  def _rfft(input_tensor, fft_length=None, name=None):
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor, _dtypes.float32)
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_rfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length)
      return fft_fn(input_tensor, fft_length, name)
  _rfft.__doc__ = fft_fn.__doc__
  return _rfft


def _irfft_wrapper(ifft_fn, fft_rank, default_name):
  """Wrapper around gen_spectral_ops.irfft* that infers fft_length argument."""

  def _irfft(input_tensor, fft_length=None, name=None):
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor, _dtypes.complex64)
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_irfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length,
                                         is_reverse=True)
      return ifft_fn(input_tensor, fft_length, name)
  _irfft.__doc__ = ifft_fn.__doc__
  return _irfft


fft = gen_spectral_ops.fft
ifft = gen_spectral_ops.ifft
fft2d = gen_spectral_ops.fft2d
ifft2d = gen_spectral_ops.ifft2d
fft3d = gen_spectral_ops.fft3d
ifft3d = gen_spectral_ops.ifft3d
rfft = _rfft_wrapper(gen_spectral_ops.rfft, 1, "rfft")
irfft = _irfft_wrapper(gen_spectral_ops.irfft, 1, "irfft")
rfft2d = _rfft_wrapper(gen_spectral_ops.rfft2d, 2, "rfft2d")
irfft2d = _irfft_wrapper(gen_spectral_ops.irfft2d, 2, "irfft2d")
rfft3d = _rfft_wrapper(gen_spectral_ops.rfft3d, 3, "rfft3d")
irfft3d = _irfft_wrapper(gen_spectral_ops.irfft3d, 3, "irfft3d")


def _validate_dct_arguments(dct_type, n, axis, norm):
  if n is not None:
    raise NotImplementedError("The DCT length argument is not implemented.")
  if axis != -1:
    raise NotImplementedError("axis must be -1. Got: %s" % axis)
  if dct_type != 2:
    raise ValueError("Only the Type II DCT is supported.")
  if norm not in (None, "ortho"):
    raise ValueError(
        "Unknown normalization. Expected None or 'ortho', got: %s" % norm)


# TODO(rjryan): Implement `type`, `n` and `axis` parameters.
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Discrete Cosine Transform (DCT)][dct] of `input`.

  Currently only Type II is supported. Implemented using a length `2N` padded
  @{tf.spectral.rfft}, as described here: https://dsp.stackexchange.com/a/10606

  @compatibility(scipy)
  Equivalent to scipy.fftpack.dct for the Type-II DCT.
  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32` `Tensor` containing the signals to
      take the DCT of.
    type: The DCT type to perform. Must be 2.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32` `Tensor` containing the DCT of `input`.

  Raises:
    ValueError: If `type` is not `2`, `n` is not `None, `axis` is not `-1`, or
      `norm` is not `None` or `'ortho'`.

  [dct]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
  """
  _validate_dct_arguments(type, n, axis, norm)
  with _ops.name_scope(name, "dct", [input]):
    # We use the RFFT to compute the DCT and TensorFlow only supports float32
    # for FFTs at the moment.
    input = _ops.convert_to_tensor(input, dtype=_dtypes.float32)

    axis_dim = input.shape[-1].value or _array_ops.shape(input)[-1]
    axis_dim_float = _math_ops.to_float(axis_dim)
    scale = 2.0 * _math_ops.exp(_math_ops.complex(
        0.0, -_math.pi * _math_ops.range(axis_dim_float) /
        (2.0 * axis_dim_float)))

    # TODO(rjryan): Benchmark performance and memory usage of the various
    # approaches to computing a DCT via the RFFT.
    dct2 = _math_ops.real(
        rfft(input, fft_length=[2 * axis_dim])[..., :axis_dim] * scale)

    if norm == "ortho":
      n1 = 0.5 * _math_ops.rsqrt(axis_dim_float)
      n2 = n1 * _math_ops.sqrt(2.0)
      # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
      weights = _array_ops.pad(
          _array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
          constant_values=n2)
      dct2 *= weights

    return dct2

remove_undocumented(__name__)
