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
"""Fast-Fourier Transform ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


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
      any(dim.value == 0 for dim in input_tensor.shape.dims)):
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
        fft_shape = fft_shape[:-1].concatenate(
            fft_shape.dims[-1].value // 2 + 1)

      paddings = [[0, max(fft_dim.value - input_dim.value, 0)]
                  for fft_dim, input_dim in zip(
                      fft_shape.dims, input_fft_shape.dims)]
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
    """Wrapper around gen_spectral_ops.rfft* that infers fft_length argument."""
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor,
                                            preferred_dtype=_dtypes.float32)
      if input_tensor.dtype not in (_dtypes.float32, _dtypes.float64):
        raise ValueError(
            "RFFT requires tf.float32 or tf.float64 inputs, got: %s" %
            input_tensor)
      real_dtype = input_tensor.dtype
      if real_dtype == _dtypes.float32:
        complex_dtype = _dtypes.complex64
      else:
        assert real_dtype == _dtypes.float64
        complex_dtype = _dtypes.complex128
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_rfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length)

      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      return fft_fn(input_tensor, fft_length, Tcomplex=complex_dtype, name=name)
  _rfft.__doc__ = fft_fn.__doc__
  return _rfft


def _irfft_wrapper(ifft_fn, fft_rank, default_name):
  """Wrapper around gen_spectral_ops.irfft* that infers fft_length argument."""

  def _irfft(input_tensor, fft_length=None, name=None):
    """Wrapper irfft* that infers fft_length argument."""
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor,
                                            preferred_dtype=_dtypes.complex64)
      input_tensor.shape.with_rank_at_least(fft_rank)
      if input_tensor.dtype not in (_dtypes.complex64, _dtypes.complex128):
        raise ValueError(
            "IRFFT requires tf.complex64 or tf.complex128 inputs, got: %s" %
            input_tensor)
      complex_dtype = input_tensor.dtype
      real_dtype = complex_dtype.real_dtype
      if fft_length is None:
        fft_length = _infer_fft_length_for_irfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length,
                                         is_reverse=True)
      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      return ifft_fn(input_tensor, fft_length, Treal=real_dtype, name=name)
  _irfft.__doc__ = ifft_fn.__doc__
  return _irfft


# FFT/IFFT 1/2/3D are exported via
# third_party/tensorflow/core/api_def/python_api/
fft = gen_spectral_ops.fft
ifft = gen_spectral_ops.ifft
fft2d = gen_spectral_ops.fft2d
ifft2d = gen_spectral_ops.ifft2d
fft3d = gen_spectral_ops.fft3d
ifft3d = gen_spectral_ops.ifft3d
rfft = _rfft_wrapper(gen_spectral_ops.rfft, 1, "rfft")
tf_export("signal.rfft", v1=["signal.rfft", "spectral.rfft"])(
    dispatch.add_dispatch_support(rfft))
irfft = _irfft_wrapper(gen_spectral_ops.irfft, 1, "irfft")
tf_export("signal.irfft", v1=["signal.irfft", "spectral.irfft"])(
    dispatch.add_dispatch_support(irfft))
rfft2d = _rfft_wrapper(gen_spectral_ops.rfft2d, 2, "rfft2d")
tf_export("signal.rfft2d", v1=["signal.rfft2d", "spectral.rfft2d"])(
    dispatch.add_dispatch_support(rfft2d))
irfft2d = _irfft_wrapper(gen_spectral_ops.irfft2d, 2, "irfft2d")
tf_export("signal.irfft2d", v1=["signal.irfft2d", "spectral.irfft2d"])(
    dispatch.add_dispatch_support(irfft2d))
rfft3d = _rfft_wrapper(gen_spectral_ops.rfft3d, 3, "rfft3d")
tf_export("signal.rfft3d", v1=["signal.rfft3d", "spectral.rfft3d"])(
    dispatch.add_dispatch_support(rfft3d))
irfft3d = _irfft_wrapper(gen_spectral_ops.irfft3d, 3, "irfft3d")
tf_export("signal.irfft3d", v1=["signal.irfft3d", "spectral.irfft3d"])(
    dispatch.add_dispatch_support(irfft3d))


def _fft_size_for_grad(grad, rank):
  return _math_ops.reduce_prod(_array_ops.shape(grad)[-rank:])


@_ops.RegisterGradient("FFT")
def _fft_grad(_, grad):
  size = _math_ops.cast(_fft_size_for_grad(grad, 1), grad.dtype)
  return ifft(grad) * size


@_ops.RegisterGradient("IFFT")
def _ifft_grad(_, grad):
  rsize = _math_ops.cast(
      1. / _math_ops.cast(_fft_size_for_grad(grad, 1), grad.dtype.real_dtype),
      grad.dtype)
  return fft(grad) * rsize


@_ops.RegisterGradient("FFT2D")
def _fft2d_grad(_, grad):
  size = _math_ops.cast(_fft_size_for_grad(grad, 2), grad.dtype)
  return ifft2d(grad) * size


@_ops.RegisterGradient("IFFT2D")
def _ifft2d_grad(_, grad):
  rsize = _math_ops.cast(
      1. / _math_ops.cast(_fft_size_for_grad(grad, 2), grad.dtype.real_dtype),
      grad.dtype)
  return fft2d(grad) * rsize


@_ops.RegisterGradient("FFT3D")
def _fft3d_grad(_, grad):
  size = _math_ops.cast(_fft_size_for_grad(grad, 3), grad.dtype)
  return ifft3d(grad) * size


@_ops.RegisterGradient("IFFT3D")
def _ifft3d_grad(_, grad):
  rsize = _math_ops.cast(
      1. / _math_ops.cast(_fft_size_for_grad(grad, 3), grad.dtype.real_dtype),
      grad.dtype)
  return fft3d(grad) * rsize


def _rfft_grad_helper(rank, irfft_fn):
  """Returns a gradient function for an RFFT of the provided rank."""
  # Can't happen because we don't register a gradient for RFFT3D.
  assert rank in (1, 2), "Gradient for RFFT3D is not implemented."

  def _grad(op, grad):
    """A gradient function for RFFT with the provided `rank` and `irfft_fn`."""
    fft_length = op.inputs[1]
    complex_dtype = grad.dtype
    real_dtype = complex_dtype.real_dtype
    input_shape = _array_ops.shape(op.inputs[0])
    is_even = _math_ops.cast(1 - (fft_length[-1] % 2), complex_dtype)

    def _tile_for_broadcasting(matrix, t):
      expanded = _array_ops.reshape(
          matrix,
          _array_ops.concat([
              _array_ops.ones([_array_ops.rank(t) - 2], _dtypes.int32),
              _array_ops.shape(matrix)
          ], 0))
      return _array_ops.tile(
          expanded, _array_ops.concat([_array_ops.shape(t)[:-2], [1, 1]], 0))

    def _mask_matrix(length):
      """Computes t_n = exp(sqrt(-1) * pi * n^2 / line_len)."""
      # TODO(rjryan): Speed up computation of twiddle factors using the
      # following recurrence relation and cache them across invocations of RFFT.
      #
      # t_n = exp(sqrt(-1) * pi * n^2 / line_len)
      # for n = 0, 1,..., line_len-1.
      # For n > 2, use t_n = t_{n-1}^2 / t_{n-2} * t_1^2
      a = _array_ops.tile(
          _array_ops.expand_dims(_math_ops.range(length), 0), (length, 1))
      b = _array_ops.transpose(a, [1, 0])
      return _math_ops.exp(
          -2j * np.pi * _math_ops.cast(a * b, complex_dtype) /
          _math_ops.cast(length, complex_dtype))

    def _ymask(length):
      """A sequence of [1+0j, -1+0j, 1+0j, -1+0j, ...] with length `length`."""
      return _math_ops.cast(1 - 2 * (_math_ops.range(length) % 2),
                            complex_dtype)

    y0 = grad[..., 0:1]
    if rank == 1:
      ym = grad[..., -1:]
      extra_terms = y0 + is_even * ym * _ymask(input_shape[-1])
    elif rank == 2:
      # Create a mask matrix for y0 and ym.
      base_mask = _mask_matrix(input_shape[-2])

      # Tile base_mask to match y0 in shape so that we can batch-matmul the
      # inner 2 dimensions.
      tiled_mask = _tile_for_broadcasting(base_mask, y0)

      y0_term = _math_ops.matmul(tiled_mask, _math_ops.conj(y0))
      extra_terms = y0_term

      ym = grad[..., -1:]
      ym_term = _math_ops.matmul(tiled_mask, _math_ops.conj(ym))

      inner_dim = input_shape[-1]
      ym_term = _array_ops.tile(
          ym_term,
          _array_ops.concat([
              _array_ops.ones([_array_ops.rank(grad) - 1], _dtypes.int32),
              [inner_dim]
          ], 0)) * _ymask(inner_dim)

      extra_terms += is_even * ym_term

    # The gradient of RFFT is the IRFFT of the incoming gradient times a scaling
    # factor, plus some additional terms to make up for the components dropped
    # due to Hermitian symmetry.
    input_size = _math_ops.cast(
        _fft_size_for_grad(op.inputs[0], rank), real_dtype)
    the_irfft = irfft_fn(grad, fft_length)
    return 0.5 * (the_irfft * input_size + _math_ops.real(extra_terms)), None

  return _grad


def _irfft_grad_helper(rank, rfft_fn):
  """Returns a gradient function for an IRFFT of the provided rank."""
  # Can't happen because we don't register a gradient for IRFFT3D.
  assert rank in (1, 2), "Gradient for IRFFT3D is not implemented."

  def _grad(op, grad):
    """A gradient function for IRFFT with the provided `rank` and `rfft_fn`."""
    # Generate a simple mask like [1.0, 2.0, ..., 2.0, 1.0] for even-length FFTs
    # and [1.0, 2.0, ..., 2.0] for odd-length FFTs. To reduce extra ops in the
    # graph we special-case the situation where the FFT length and last
    # dimension of the input are known at graph construction time.
    fft_length = op.inputs[1]
    fft_length_static = _tensor_util.constant_value(fft_length)
    if fft_length_static is not None:
      fft_length = fft_length_static
    real_dtype = grad.dtype
    if real_dtype == _dtypes.float32:
      complex_dtype = _dtypes.complex64
    elif real_dtype == _dtypes.float64:
      complex_dtype = _dtypes.complex128
    is_odd = _math_ops.mod(fft_length[-1], 2)
    input_last_dimension = _array_ops.shape(op.inputs[0])[-1]
    mask = _array_ops.concat(
        [[1.0], 2.0 * _array_ops.ones(
            [input_last_dimension - 2 + is_odd], real_dtype),
         _array_ops.ones([1 - is_odd], real_dtype)], 0)

    rsize = _math_ops.reciprocal(_math_ops.cast(
        _fft_size_for_grad(grad, rank), real_dtype))

    # The gradient of IRFFT is the RFFT of the incoming gradient times a scaling
    # factor and a mask. The mask scales the gradient for the Hermitian
    # symmetric components of the RFFT by a factor of two, since these
    # components are de-duplicated in the RFFT.
    the_rfft = rfft_fn(grad, fft_length)
    return the_rfft * _math_ops.cast(rsize * mask, complex_dtype), None

  return _grad


@tf_export("signal.fftshift")
@dispatch.add_dispatch_support
def fftshift(x, axes=None, name=None):
  """Shift the zero-frequency component to the center of the spectrum.

  This function swaps half-spaces for all axes listed (defaults to all).
  Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

  @compatibility(numpy)
  Equivalent to numpy.fft.fftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html
  @end_compatibility

  For example:

  ```python
  x = tf.signal.fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
  x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
  ```

  Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple`, optional Axes over which to shift.  Default is
      None, which shifts all axes.
    name: An optional name for the operation.

  Returns:
    A `Tensor`, The shifted tensor.
  """
  with _ops.name_scope(name, "fftshift") as name:
    x = _ops.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = _array_ops.shape(x) // 2
    elif isinstance(axes, int):
      shift = _array_ops.shape(x)[axes] // 2
    else:
      rank = _array_ops.rank(x)
      # allows negative axis
      axes = _array_ops.where(_math_ops.less(axes, 0), axes + rank, axes)
      shift = _array_ops.gather(_array_ops.shape(x), axes) // 2

    return manip_ops.roll(x, shift, axes, name)


@tf_export("signal.ifftshift")
@dispatch.add_dispatch_support
def ifftshift(x, axes=None, name=None):
  """The inverse of fftshift.

  Although identical for even-length x,
  the functions differ by one sample for odd-length x.

  @compatibility(numpy)
  Equivalent to numpy.fft.ifftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifftshift.html
  @end_compatibility

  For example:

  ```python
  x = tf.signal.ifftshift([[ 0.,  1.,  2.],[ 3.,  4., -4.],[-3., -2., -1.]])
  x.numpy() # array([[ 4., -4.,  3.],[-2., -1., -3.],[ 1.,  2.,  0.]])
  ```

  Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple` Axes over which to calculate. Defaults to None,
      which shifts all axes.
    name: An optional name for the operation.

  Returns:
    A `Tensor`, The shifted tensor.
  """
  with _ops.name_scope(name, "ifftshift") as name:
    x = _ops.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = -(_array_ops.shape(x) // 2)
    elif isinstance(axes, int):
      shift = -(_array_ops.shape(x)[axes] // 2)
    else:
      rank = _array_ops.rank(x)
      # allows negative axis
      axes = _array_ops.where(_math_ops.less(axes, 0), axes + rank, axes)
      shift = -(_array_ops.gather(_array_ops.shape(x), axes) // 2)

    return manip_ops.roll(x, shift, axes, name)


_ops.RegisterGradient("RFFT")(_rfft_grad_helper(1, irfft))
_ops.RegisterGradient("IRFFT")(_irfft_grad_helper(1, rfft))
_ops.RegisterGradient("RFFT2D")(_rfft_grad_helper(2, irfft2d))
_ops.RegisterGradient("IRFFT2D")(_irfft_grad_helper(2, rfft2d))
