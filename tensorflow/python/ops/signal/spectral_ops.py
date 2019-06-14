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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.signal import reconstruction_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export('signal.stft')
def stft(signals, frame_length, frame_step, fft_length=None,
         window_fn=window_ops.hann_window,
         pad_end=False, name=None):
  """Computes the [Short-time Fourier Transform][stft] of `signals`.

  Implemented with GPU-compatible ops and supports gradients.

  Args:
    signals: A `[..., samples]` `float32` `Tensor` of real-valued signals.
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT to apply.
      If not provided, uses the smallest power of 2 enclosing `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    pad_end: Whether to pad the end of `signals` with zeros when the provided
      frame length and step produces a frame that lies partially past its end.
    name: An optional name for the operation.

  Returns:
    A `[..., frames, fft_unique_bins]` `Tensor` of `complex64` STFT values where
    `fft_unique_bins` is `fft_length // 2 + 1` (the unique components of the
    FFT).

  Raises:
    ValueError: If `signals` is not at least rank 1, `frame_length` is
      not scalar, or `frame_step` is not scalar.

  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  """
  with ops.name_scope(name, 'stft', [signals, frame_length,
                                     frame_step]):
    signals = ops.convert_to_tensor(signals, name='signals')
    signals.shape.with_rank_at_least(1)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)

    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')

    framed_signals = shape_ops.frame(
        signals, frame_length, frame_step, pad_end=pad_end)

    # Optionally window the framed signals.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signals.dtype)
      framed_signals *= window

    # fft_ops.rfft produces the (fft_length/2 + 1) unique components of the
    # FFT of the real windowed signals in framed_signals.
    return fft_ops.rfft(framed_signals, [fft_length])


@tf_export('signal.inverse_stft_window_fn')
def inverse_stft_window_fn(frame_step,
                           forward_window_fn=window_ops.hann_window,
                           name=None):
  """Generates a window function that can be used in `inverse_stft`.

  Constructs a window that is equal to the forward window with a further
  pointwise amplitude correction.  `inverse_stft_window_fn` is equivalent to
  `forward_window_fn` in the case where it would produce an exact inverse.

  See examples in `inverse_stft` documentation for usage.

  Args:
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    forward_window_fn: window_fn used in the forward transform, `stft`.
    name: An optional name for the operation.

  Returns:
    A callable that takes a window length and a `dtype` keyword argument and
      returns a `[window_length]` `Tensor` of samples in the provided datatype.
      The returned window is suitable for reconstructing original waveform in
      inverse_stft.
  """
  with ops.name_scope(name, 'inverse_stft_window_fn', [forward_window_fn]):
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)

  def inverse_stft_window_fn_inner(frame_length, dtype):
    """Computes a window that can be used in `inverse_stft`.

    Args:
      frame_length: An integer scalar `Tensor`. The window length in samples.
      dtype: Data type of waveform passed to `stft`.

    Returns:
      A window suitable for reconstructing original waveform in `inverse_stft`.

    Raises:
      ValueError: If `frame_length` is not scalar, `forward_window_fn` is not a
      callable that takes a window length and a `dtype` keyword argument and
      returns a `[window_length]` `Tensor` of samples in the provided datatype
      `frame_step` is not scalar, or `frame_step` is not scalar.
    """
    with ops.name_scope(name, 'inverse_stft_window_fn', [forward_window_fn]):
      frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
      frame_length.shape.assert_has_rank(0)

      # Use equation 7 from Griffin + Lim.
      forward_window = forward_window_fn(frame_length, dtype=dtype)
      denom = math_ops.square(forward_window)
      overlaps = -(-frame_length // frame_step)  # Ceiling division.
      denom = array_ops.pad(denom, [(0, overlaps * frame_step - frame_length)])
      denom = array_ops.reshape(denom, [overlaps, frame_step])
      denom = math_ops.reduce_sum(denom, 0, keepdims=True)
      denom = array_ops.tile(denom, [overlaps, 1])
      denom = array_ops.reshape(denom, [overlaps * frame_step])

      return forward_window / denom[:frame_length]
  return inverse_stft_window_fn_inner


@tf_export('signal.inverse_stft')
def inverse_stft(stfts,
                 frame_length,
                 frame_step,
                 fft_length=None,
                 window_fn=window_ops.hann_window,
                 name=None):
  """Computes the inverse [Short-time Fourier Transform][stft] of `stfts`.

  To reconstruct an original waveform, a complimentary window function should
  be used in inverse_stft. Such a window function can be constructed with
  tf.signal.inverse_stft_window_fn.

  Example:

  ```python
  frame_length = 400
  frame_step = 160
  waveform = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(waveform, frame_length, frame_step)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(frame_step))
  ```

  if a custom window_fn is used in stft, it must be passed to
  inverse_stft_window_fn:

  ```python
  frame_length = 400
  frame_step = 160
  window_fn = functools.partial(window_ops.hamming_window, periodic=True),
  waveform = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(
      waveform, frame_length, frame_step, window_fn=window_fn)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step, forward_window_fn=window_fn))
  ```

  Implemented with GPU-compatible ops and supports gradients.

  Args:
    stfts: A `complex64` `[..., frames, fft_unique_bins]` `Tensor` of STFT bins
      representing a batch of `fft_length`-point STFTs where `fft_unique_bins`
      is `fft_length // 2 + 1`
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT that produced
      `stfts`. If not provided, uses the smallest power of 2 enclosing
      `frame_length`.
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

  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  """
  with ops.name_scope(name, 'inverse_stft', [stfts]):
    stfts = ops.convert_to_tensor(stfts, name='stfts')
    stfts.shape.with_rank_at_least(2)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)
    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
      fft_length.shape.assert_has_rank(0)

    real_frames = fft_ops.irfft(stfts, [fft_length])

    # frame_length may be larger or smaller than fft_length, so we pad or
    # truncate real_frames to frame_length.
    frame_length_static = tensor_util.constant_value(frame_length)
    # If we don't know the shape of real_frames's inner dimension, pad and
    # truncate to frame_length.
    if (frame_length_static is None or real_frames.shape.ndims is None or
        real_frames.shape.as_list()[-1] is None):
      real_frames = real_frames[..., :frame_length]
      real_frames_rank = array_ops.rank(real_frames)
      real_frames_shape = array_ops.shape(real_frames)
      paddings = array_ops.concat(
          [array_ops.zeros([real_frames_rank - 1, 2],
                           dtype=frame_length.dtype),
           [[0, math_ops.maximum(0, frame_length - real_frames_shape[-1])]]], 0)
      real_frames = array_ops.pad(real_frames, paddings)
    # We know real_frames's last dimension and frame_length statically. If they
    # are different, then pad or truncate real_frames to frame_length.
    elif real_frames.shape.as_list()[-1] > frame_length_static:
      real_frames = real_frames[..., :frame_length_static]
    elif real_frames.shape.as_list()[-1] < frame_length_static:
      pad_amount = frame_length_static - real_frames.shape.as_list()[-1]
      real_frames = array_ops.pad(real_frames,
                                  [[0, 0]] * (real_frames.shape.ndims - 1) +
                                  [[0, pad_amount]])

    # The above code pads the inner dimension of real_frames to frame_length,
    # but it does so in a way that may not be shape-inference friendly.
    # Restore shape information if we are able to.
    if frame_length_static is not None and real_frames.shape.ndims is not None:
      real_frames.set_shape([None] * (real_frames.shape.ndims - 1) +
                            [frame_length_static])

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
      math_ops.pow(
          2.0,
          math_ops.ceil(
              math_ops.log(math_ops.cast(value, dtypes.float32)) /
              math_ops.log(2.0))), value.dtype)
