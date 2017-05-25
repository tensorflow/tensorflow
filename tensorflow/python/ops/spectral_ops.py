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
"""Spectral operators (e.g. FFT, RFFT).

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

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
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


def _rfft_wrapper(fft_fn, fft_rank, default_name):
  """Wrapper around gen_spectral_ops.rfft* that infers fft_length argument."""

  def _rfft(input_tensor, fft_length=None, name=None):
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor, _dtypes.float32)
      if fft_length is None:
        fft_length = _infer_fft_length_for_rfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      return fft_fn(input_tensor, fft_length, name)
  _rfft.__doc__ = fft_fn.__doc__
  return _rfft


def _irfft_wrapper(ifft_fn, fft_rank, default_name):
  """Wrapper around gen_spectral_ops.irfft* that infers fft_length argument."""

  def _irfft(input_tensor, fft_length=None, name=None):
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor, _dtypes.complex64)
      if fft_length is None:
        fft_length = _infer_fft_length_for_irfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
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

remove_undocumented(__name__)
