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
"""Gradients for operators defined in spectral_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops


def _FFTSizeForGrad(grad, rank):
  return math_ops.reduce_prod(array_ops.shape(grad)[-rank:])


@ops.RegisterGradient("FFT")
def _FFTGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 1), dtypes.float32)
  return spectral_ops.ifft(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT")
def _IFFTGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 1), dtypes.float32)
  return spectral_ops.fft(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("FFT2D")
def _FFT2DGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 2), dtypes.float32)
  return spectral_ops.ifft2d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT2D")
def _IFFT2DGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 2), dtypes.float32)
  return spectral_ops.fft2d(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("FFT3D")
def _FFT3DGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 3), dtypes.float32)
  return spectral_ops.ifft3d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT3D")
def _IFFT3DGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 3), dtypes.float32)
  return spectral_ops.fft3d(grad) * math_ops.complex(rsize, 0.)


def _RFFTGradHelper(rank, irfft_fn):
  """Returns a gradient function for an RFFT of the provided rank."""
  # Can't happen because we don't register a gradient for RFFT3D.
  assert rank in (1, 2), "Gradient for RFFT3D is not implemented."

  def _Grad(op, grad):
    """A gradient function for RFFT with the provided `rank` and `irfft_fn`."""
    fft_length = op.inputs[1]
    input_shape = array_ops.shape(op.inputs[0])
    is_even = math_ops.cast(1 - (fft_length[-1] % 2), dtypes.complex64)

    def _TileForBroadcasting(matrix, t):
      expanded = array_ops.reshape(
          matrix,
          array_ops.concat([
              array_ops.ones([array_ops.rank(t) - 2], dtypes.int32),
              array_ops.shape(matrix)
          ], 0))
      return array_ops.tile(
          expanded, array_ops.concat([array_ops.shape(t)[:-2], [1, 1]], 0))

    def _MaskMatrix(length):
      # TODO(rjryan): Speed up computation of twiddle factors using the
      # following recurrence relation and cache them across invocations of RFFT.
      #
      # t_n = exp(sqrt(-1) * pi * n^2 / line_len)
      # for n = 0, 1,..., line_len-1.
      # For n > 2, use t_n = t_{n-1}^2 / t_{n-2} * t_1^2
      a = array_ops.tile(
          array_ops.expand_dims(math_ops.range(length), 0), (length, 1))
      b = array_ops.transpose(a, [1, 0])
      return math_ops.exp(-2j * np.pi * math_ops.cast(a * b, dtypes.complex64) /
                          math_ops.cast(length, dtypes.complex64))

    def _YMMask(length):
      """A sequence of [1+0j, -1+0j, 1+0j, -1+0j, ...] with length `length`."""
      return math_ops.cast(1 - 2 * (math_ops.range(length) % 2),
                           dtypes.complex64)

    y0 = grad[..., 0:1]
    if rank == 1:
      ym = grad[..., -1:]
      extra_terms = y0 + is_even * ym * _YMMask(input_shape[-1])
    elif rank == 2:
      # Create a mask matrix for y0 and ym.
      base_mask = _MaskMatrix(input_shape[-2])

      # Tile base_mask to match y0 in shape so that we can batch-matmul the
      # inner 2 dimensions.
      tiled_mask = _TileForBroadcasting(base_mask, y0)

      y0_term = math_ops.matmul(tiled_mask, math_ops.conj(y0))
      extra_terms = y0_term

      ym = grad[..., -1:]
      ym_term = math_ops.matmul(tiled_mask, math_ops.conj(ym))

      inner_dim = input_shape[-1]
      ym_term = array_ops.tile(
          ym_term,
          array_ops.concat([
              array_ops.ones([array_ops.rank(grad) - 1], dtypes.int32),
              [inner_dim]
          ], 0)) * _YMMask(inner_dim)

      extra_terms += is_even * ym_term

    # The gradient of RFFT is the IRFFT of the incoming gradient times a scaling
    # factor, plus some additional terms to make up for the components dropped
    # due to Hermitian symmetry.
    input_size = math_ops.to_float(_FFTSizeForGrad(op.inputs[0], rank))
    irfft = irfft_fn(grad, fft_length)
    return 0.5 * (irfft * input_size + math_ops.real(extra_terms)), None

  return _Grad


def _IRFFTGradHelper(rank, rfft_fn):
  """Returns a gradient function for an IRFFT of the provided rank."""
  # Can't happen because we don't register a gradient for IRFFT3D.
  assert rank in (1, 2), "Gradient for IRFFT3D is not implemented."

  def _Grad(op, grad):
    """A gradient function for IRFFT with the provided `rank` and `rfft_fn`."""
    # Generate a simple mask like [1.0, 2.0, ..., 2.0, 1.0] for even-length FFTs
    # and [1.0, 2.0, ..., 2.0] for odd-length FFTs. To reduce extra ops in the
    # graph we special-case the situation where the FFT length and last
    # dimension of the input are known at graph construction time.
    fft_length = op.inputs[1]
    is_odd = math_ops.mod(fft_length[-1], 2)
    input_last_dimension = array_ops.shape(op.inputs[0])[-1]
    mask = array_ops.concat(
        [[1.0], 2.0 * array_ops.ones([input_last_dimension - 2 + is_odd]),
         array_ops.ones([1 - is_odd])], 0)

    rsize = math_ops.reciprocal(math_ops.to_float(_FFTSizeForGrad(grad, rank)))

    # The gradient of IRFFT is the RFFT of the incoming gradient times a scaling
    # factor and a mask. The mask scales the gradient for the Hermitian
    # symmetric components of the RFFT by a factor of two, since these
    # components are de-duplicated in the RFFT.
    rfft = rfft_fn(grad, fft_length)
    return rfft * math_ops.cast(rsize * mask, dtypes.complex64), None

  return _Grad


ops.RegisterGradient("RFFT")(_RFFTGradHelper(1, spectral_ops.irfft))
ops.RegisterGradient("IRFFT")(_IRFFTGradHelper(1, spectral_ops.rfft))
ops.RegisterGradient("RFFT2D")(_RFFTGradHelper(2, spectral_ops.irfft2d))
ops.RegisterGradient("IRFFT2D")(_IRFFTGradHelper(2, spectral_ops.rfft2d))
