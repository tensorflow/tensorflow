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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.signal import reconstruction_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export('signal.stft')
@dispatch.add_dispatch_support
def stft(signals, frame_length, frame_step, fft_length=None,
         window_fn=window_ops.hann_window,
         pad_end=False, name=None):
  """Computes the [Short-time Fourier Transform][stft] of `signals`.

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    signals: A `[..., samples]` `float32`/`float64` `Tensor` of real-valued
      signals.
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
    A `[..., frames, fft_unique_bins]` `Tensor` of `complex64`/`complex128`
    STFT values where `fft_unique_bins` is `fft_length // 2 + 1` (the unique
    components of the FFT).

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
@dispatch.add_dispatch_support
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
      frame_step_ = ops.convert_to_tensor(frame_step, name='frame_step')
      frame_step_.shape.assert_has_rank(0)
      frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
      frame_length.shape.assert_has_rank(0)

      # Use equation 7 from Griffin + Lim.
      forward_window = forward_window_fn(frame_length, dtype=dtype)
      denom = math_ops.square(forward_window)
      overlaps = -(-frame_length // frame_step_)  # Ceiling division.  # pylint: disable=invalid-unary-operand-type
      denom = array_ops.pad(denom, [(0, overlaps * frame_step_ - frame_length)])
      denom = array_ops.reshape(denom, [overlaps, frame_step_])
      denom = math_ops.reduce_sum(denom, 0, keepdims=True)
      denom = array_ops.tile(denom, [overlaps, 1])
      denom = array_ops.reshape(denom, [overlaps * frame_step_])

      denom = denom[:frame_length]
      return array_ops.where(
          math_ops.equal(denom, 0.0),
          array_ops.zeros_like(forward_window),
          forward_window / denom,
      )
  return inverse_stft_window_fn_inner


@tf_export('signal.inverse_stft')
@dispatch.add_dispatch_support
def inverse_stft(stfts,
                 frame_length,
                 frame_step,
                 fft_length=None,
                 window_fn=window_ops.hann_window,
                 name=None):
  """Computes the inverse [Short-time Fourier Transform][stft] of `stfts`.

  To reconstruct an original waveform, a complementary window function should
  be used with `inverse_stft`. Such a window function can be constructed with
  `tf.signal.inverse_stft_window_fn`.
  Example:

  ```python
  frame_length = 400
  frame_step = 160
  waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(waveform, frame_length, frame_step)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(frame_step))
  ```

  If a custom `window_fn` is used with `tf.signal.stft`, it must be passed to
  `tf.signal.inverse_stft_window_fn`:

  ```python
  frame_length = 400
  frame_step = 160
  window_fn = tf.signal.hamming_window
  waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(
      waveform, frame_length, frame_step, window_fn=window_fn)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step, forward_window_fn=window_fn))
  ```

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    stfts: A `complex64`/`complex128` `[..., frames, fft_unique_bins]`
      `Tensor` of STFT bins representing a batch of `fft_length`-point STFTs
      where `fft_unique_bins` is `fft_length // 2 + 1`
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
    A `[..., samples]` `Tensor` of `float32`/`float64` signals representing
    the inverse STFT for each input STFT in `stfts`.

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


@tf_export('signal.mdct')
@dispatch.add_dispatch_support
def mdct(signals, frame_length, window_fn=window_ops.vorbis_window,
         pad_end=False, norm=None, name=None):
  """Computes the [Modified Discrete Cosine Transform][mdct] of `signals`.

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    signals: A `[..., samples]` `float32`/`float64` `Tensor` of real-valued
      signals.
    frame_length: An integer scalar `Tensor`. The window length in samples
      which must be divisible by 4.
    window_fn: A callable that takes a frame_length and a `dtype` keyword
      argument and returns a `[frame_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, a rectangular window with a scale of
      1/sqrt(2) is used. For perfect reconstruction of a signal from `mdct`
      followed by `inverse_mdct`, please use `tf.signal.vorbis_window`,
      `tf.signal.kaiser_bessel_derived_window` or `None`. If using another
      window function, make sure that w[n]^2 + w[n + frame_length // 2]^2 = 1
      and w[n] = w[frame_length - n - 1] for n = 0,...,frame_length // 2 - 1 to
      achieve perfect reconstruction.
    pad_end: Whether to pad the end of `signals` with zeros when the provided
      frame length and step produces a frame that lies partially past its end.
    norm: If it is None, unnormalized dct4 is used, if it is "ortho"
      orthonormal dct4 is used.
    name: An optional name for the operation.

  Returns:
    A `[..., frames, frame_length // 2]` `Tensor` of `float32`/`float64`
    MDCT values where `frames` is roughly `samples // (frame_length // 2)`
    when `pad_end=False`.

  Raises:
    ValueError: If `signals` is not at least rank 1, `frame_length` is
      not scalar, or `frame_length` is not a multiple of `4`.

  [mdct]: https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform
  """
  with ops.name_scope(name, 'mdct', [signals, frame_length]):
    signals = ops.convert_to_tensor(signals, name='signals')
    signals.shape.with_rank_at_least(1)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    # Assert that frame_length is divisible by 4.
    frame_length_static = tensor_util.constant_value(frame_length)
    if frame_length_static is not None:
      if frame_length_static % 4 != 0:
        raise ValueError('The frame length must be a multiple of 4.')
      frame_step = ops.convert_to_tensor(frame_length_static // 2,
                                         dtype=frame_length.dtype)
    else:
      frame_step = frame_length // 2

    framed_signals = shape_ops.frame(
        signals, frame_length, frame_step, pad_end=pad_end)

    # Optionally window the framed signals.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signals.dtype)
      framed_signals *= window
    else:
      framed_signals *= 1.0 / np.sqrt(2)

    split_frames = array_ops.split(framed_signals, 4, axis=-1)
    frame_firsthalf = -array_ops.reverse(split_frames[2],
                                         [-1]) - split_frames[3]
    frame_secondhalf = split_frames[0] - array_ops.reverse(split_frames[1],
                                                           [-1])
    frames_rearranged = array_ops.concat((frame_firsthalf, frame_secondhalf),
                                         axis=-1)
    # Below call produces the (frame_length // 2) unique components of the
    # type 4 orthonormal DCT of the real windowed signals in frames_rearranged.
    return dct_ops.dct(frames_rearranged, type=4, norm=norm)


@tf_export('signal.inverse_mdct')
@dispatch.add_dispatch_support
def inverse_mdct(mdcts,
                 window_fn=window_ops.vorbis_window,
                 norm=None,
                 name=None):
  """Computes the inverse modified DCT of `mdcts`.

  To reconstruct an original waveform, the same window function should
  be used with `mdct` and `inverse_mdct`.

  Example usage:

  >>> @tf.function
  ... def compare_round_trip():
  ...   samples = 1000
  ...   frame_length = 400
  ...   halflen = frame_length // 2
  ...   waveform = tf.random.normal(dtype=tf.float32, shape=[samples])
  ...   waveform_pad = tf.pad(waveform, [[halflen, 0],])
  ...   mdct = tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
  ...                         window_fn=tf.signal.vorbis_window)
  ...   inverse_mdct = tf.signal.inverse_mdct(mdct,
  ...                                         window_fn=tf.signal.vorbis_window)
  ...   inverse_mdct = inverse_mdct[halflen: halflen + samples]
  ...   return waveform, inverse_mdct
  >>> waveform, inverse_mdct = compare_round_trip()
  >>> np.allclose(waveform.numpy(), inverse_mdct.numpy(), rtol=1e-3, atol=1e-4)
  True

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    mdcts: A `float32`/`float64` `[..., frames, frame_length // 2]`
      `Tensor` of MDCT bins representing a batch of `frame_length // 2`-point
      MDCTs.
    window_fn: A callable that takes a frame_length and a `dtype` keyword
      argument and returns a `[frame_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, a rectangular window with a scale of
      1/sqrt(2) is used. For perfect reconstruction of a signal from `mdct`
      followed by `inverse_mdct`, please use `tf.signal.vorbis_window`,
      `tf.signal.kaiser_bessel_derived_window` or `None`. If using another
      window function, make sure that w[n]^2 + w[n + frame_length // 2]^2 = 1
      and w[n] = w[frame_length - n - 1] for n = 0,...,frame_length // 2 - 1 to
      achieve perfect reconstruction.
    norm: If "ortho", orthonormal inverse DCT4 is performed, if it is None,
      a regular dct4 followed by scaling of `1/frame_length` is performed.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `Tensor` of `float32`/`float64` signals representing
    the inverse MDCT for each input MDCT in `mdcts` where `samples` is
    `(frames - 1) * (frame_length // 2) + frame_length`.

  Raises:
    ValueError: If `mdcts` is not at least rank 2.

  [mdct]: https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform
  """
  with ops.name_scope(name, 'inverse_mdct', [mdcts]):
    mdcts = ops.convert_to_tensor(mdcts, name='mdcts')
    mdcts.shape.with_rank_at_least(2)
    half_len = math_ops.cast(mdcts.shape[-1], dtype=dtypes.int32)

    if norm is None:
      half_len_float = math_ops.cast(half_len, dtype=mdcts.dtype)
      result_idct4 = (0.5 / half_len_float) * dct_ops.dct(mdcts, type=4)
    elif norm == 'ortho':
      result_idct4 = dct_ops.dct(mdcts, type=4, norm='ortho')
    split_result = array_ops.split(result_idct4, 2, axis=-1)
    real_frames = array_ops.concat((split_result[1],
                                    -array_ops.reverse(split_result[1], [-1]),
                                    -array_ops.reverse(split_result[0], [-1]),
                                    -split_result[0]), axis=-1)

    # Optionally window and overlap-add the inner 2 dimensions of real_frames
    # into a single [samples] dimension.
    if window_fn is not None:
      window = window_fn(2 * half_len, dtype=mdcts.dtype)
      real_frames *= window
    else:
      real_frames *= 1.0 / np.sqrt(2)
    return reconstruction_ops.overlap_and_add(real_frames, half_len)
