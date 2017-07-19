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
"""Spectral operations (e.g. Short-time Fourier Transform)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.contrib.signal.python.ops import reconstruction_ops
from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.contrib.signal.python.ops import window_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops


def stft(signal, frame_length, frame_step, fft_length=None,
         window_fn=functools.partial(window_ops.hann_window, periodic=True),
         pad_end=False, name=None):
  """Computes the Short-time Fourier Transform of a batch of real signals.

  https://en.wikipedia.org/wiki/Short-time_Fourier_transform

  Implemented with GPU-compatible ops and supports gradients.

  Args:
    signal: A `[..., samples]` `float32` `Tensor` of real-valued signals.
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT to apply.
      If not provided, uses the smallest power of 2 enclosing `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    pad_end: Whether to pad the end of signal with zeros when the provided
      frame length and step produces a frame that lies partially past the end
      of `signal`.
    name: An optional name for the operation.

  Returns:
    A `[..., frames, fft_unique_bins]` `Tensor` of `complex64` STFT values where
    `fft_unique_bins` is `fft_length / 2 + 1` (the unique components of the
    FFT).

  Raises:
    ValueError: If `signal` is not at least rank 1, `frame_length` is
      not scalar, `frame_step` is not scalar, or `frame_length`
      is greater than `fft_length`.
  """
  with ops.name_scope(name, 'stft', [signal, frame_length,
                                     frame_step]):
    signal = ops.convert_to_tensor(signal, name='signal')
    signal.shape.with_rank_at_least(1)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)

    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')

    frame_length_static = tensor_util.constant_value(
        frame_length)
    fft_length_static = tensor_util.constant_value(fft_length)
    if (frame_length_static is not None and fft_length_static is not None and
        frame_length_static > fft_length_static):
      raise ValueError('frame_length (%d) may not be larger than '
                       'fft_length (%d)' % (frame_length_static,
                                            fft_length_static))

    framed_signal = shape_ops.frame(
        signal, frame_length, frame_step, pad_end=pad_end)

    # Optionally window the framed signal.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signal.dtype)
      framed_signal *= window

    # spectral_ops.rfft produces the (fft_length/2 + 1) unique components of the
    # FFT of the real windowed signals in framed_signal.
    return spectral_ops.rfft(framed_signal, [fft_length])


def inverse_stft(stfts,
                 frame_length,
                 frame_step,
                 fft_length,
                 window_fn=functools.partial(window_ops.hann_window,
                                             periodic=True),
                 name=None):
  """Computes the inverse Short-time Fourier Transform of a batch of STFTs.

  https://en.wikipedia.org/wiki/Short-time_Fourier_transform

  Implemented with GPU-compatible ops and supports gradients.

  Args:
    stfts: A `complex64` `[..., frames, fft_unique_bins]` `Tensor` of STFT bins
      representing a batch of `fft_length`-point STFTs.
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT that produced
      `stfts`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `Tensor` of `float32` signals representing the inverse
    STFT for each input STFT in `stfts`.

  Raises:
    ValueError: If `stfts` is not at least rank 2, `frame_length` is not scalar,
      `frame_step` is not scalar, or `fft_length` is not scalar.
  """
  with ops.name_scope(name, 'inverse_stft', [stfts]):
    stfts = ops.convert_to_tensor(stfts, name='stfts')
    stfts.shape.with_rank_at_least(2)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)
    fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
    fft_length.shape.assert_has_rank(0)
    real_frames = spectral_ops.irfft(stfts, [fft_length])[..., :frame_length]

    # Optionally window and overlap-add the inner 2 dimensions of real_frames
    # into a single [samples] dimension.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=stfts.dtype.real_dtype)
      real_frames *= window
    return reconstruction_ops.overlap_and_add(real_frames, frame_step)


def _enclosing_power_of_two(value):
  """Return 2**N for integer N such that 2**N >= value."""
  value_static = tensor_util.constant_value(value)
  if value_static is not None:
    return constant_op.constant(
        int(2**np.ceil(np.log(value_static) / np.log(2.0))), value.dtype)
  return math_ops.cast(
      math_ops.pow(2.0, math_ops.ceil(
          math_ops.log(math_ops.to_float(value)) / math_ops.log(2.0))),
      value.dtype)
