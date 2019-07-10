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
"""General shape ops for frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import util_ops
from tensorflow.python.util.tf_export import tf_export


def _infer_frame_shape(signal, frame_length, frame_step, pad_end, axis):
  """Infers the shape of the return value of `frame`."""
  frame_length = tensor_util.constant_value(frame_length)
  frame_step = tensor_util.constant_value(frame_step)
  axis = tensor_util.constant_value(axis)
  if signal.shape.ndims is None:
    return None
  if axis is None:
    return [None] * (signal.shape.ndims + 1)

  signal_shape = signal.shape.as_list()
  num_frames = None
  frame_axis = signal_shape[axis]
  outer_dimensions = signal_shape[:axis]
  inner_dimensions = signal_shape[axis:][1:]
  if signal_shape and frame_axis is not None:
    if frame_step is not None and pad_end:
      # Double negative is so that we round up.
      num_frames = max(0, -(-frame_axis // frame_step))
    elif frame_step is not None and frame_length is not None:
      assert not pad_end
      num_frames = max(
          0, (frame_axis - frame_length + frame_step) // frame_step)
  return outer_dimensions + [num_frames, frame_length] + inner_dimensions


@tf_export("signal.frame")
def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1,
          name=None):
  """Expands `signal`'s `axis` dimension into frames of `frame_length`.

  Slides a window of size `frame_length` over `signal`'s `axis` dimension
  with a stride of `frame_step`, replacing the `axis` dimension with
  `[frames, frame_length]` frames.

  If `pad_end` is True, window positions that are past the end of the `axis`
  dimension are padded with `pad_value` until the window moves fully past the
  end of the dimension. Otherwise, only window positions that fully overlap the
  `axis` dimension are produced.

  For example:

  ```python
  pcm = tf.compat.v1.placeholder(tf.float32, [None, 9152])
  frames = tf.signal.frame(pcm, 512, 180)
  magspec = tf.abs(tf.signal.rfft(frames, [512]))
  image = tf.expand_dims(magspec, 3)
  ```

  Args:
    signal: A `[..., samples, ...]` `Tensor`. The rank and dimensions
      may be unknown. Rank must be at least 1.
    frame_length: The frame length in samples. An integer or scalar `Tensor`.
    frame_step: The frame hop size in samples. An integer or scalar `Tensor`.
    pad_end: Whether to pad the end of `signal` with `pad_value`.
    pad_value: An optional scalar `Tensor` to use where the input signal
      does not exist when `pad_end` is True.
    axis: A scalar integer `Tensor` indicating the axis to frame. Defaults to
      the last axis. Supports negative values for indexing from the end.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of frames with shape `[..., frames, frame_length, ...]`.

  Raises:
    ValueError: If `frame_length`, `frame_step`, `pad_value`, or `axis` are not
      scalar.
  """
  with ops.name_scope(name, "frame", [signal, frame_length, frame_step,
                                      pad_value]):
    signal = ops.convert_to_tensor(signal, name="signal")
    frame_length = ops.convert_to_tensor(frame_length, name="frame_length")
    frame_step = ops.convert_to_tensor(frame_step, name="frame_step")
    axis = ops.convert_to_tensor(axis, name="axis")

    signal.shape.with_rank_at_least(1)
    frame_length.shape.assert_has_rank(0)
    frame_step.shape.assert_has_rank(0)
    axis.shape.assert_has_rank(0)

    result_shape = _infer_frame_shape(signal, frame_length, frame_step, pad_end,
                                      axis)

    # Axis can be negative. Convert it to positive.
    signal_rank = array_ops.rank(signal)
    axis = math_ops.range(signal_rank)[axis]

    signal_shape = array_ops.shape(signal)
    outer_dimensions, length_samples, inner_dimensions = array_ops.split(
        signal_shape, [axis, 1, signal_rank - 1 - axis])
    length_samples = array_ops.reshape(length_samples, [])
    num_outer_dimensions = array_ops.size(outer_dimensions)
    num_inner_dimensions = array_ops.size(inner_dimensions)

    # If padding is requested, pad the input signal tensor with pad_value.
    if pad_end:
      pad_value = ops.convert_to_tensor(pad_value, signal.dtype)
      pad_value.shape.assert_has_rank(0)

      # Calculate number of frames, using double negatives to round up.
      num_frames = -(-length_samples // frame_step)

      # Pad the signal by up to frame_length samples based on how many samples
      # are remaining starting from last_frame_position.
      pad_samples = math_ops.maximum(
          0, frame_length + frame_step * (num_frames - 1) - length_samples)

      # Pad the inner dimension of signal by pad_samples.
      paddings = array_ops.concat(
          [array_ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
           [[0, pad_samples]],
           array_ops.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype)],
          0)
      signal = array_ops.pad(signal, paddings, constant_values=pad_value)

      signal_shape = array_ops.shape(signal)
      length_samples = signal_shape[axis]
    else:
      num_frames = math_ops.maximum(
          0, 1 + (length_samples - frame_length) // frame_step)

    subframe_length = util_ops.gcd(frame_length, frame_step)
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = length_samples // subframe_length

    slice_shape = array_ops.concat([outer_dimensions,
                                    [num_subframes * subframe_length],
                                    inner_dimensions], 0)
    subframe_shape = array_ops.concat([outer_dimensions,
                                       [num_subframes, subframe_length],
                                       inner_dimensions], 0)
    subframes = array_ops.reshape(array_ops.strided_slice(
        signal, array_ops.zeros_like(signal_shape),
        slice_shape), subframe_shape)

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = array_ops.reshape(
        math_ops.range(num_frames) * subframes_per_hop, [num_frames, 1])

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = array_ops.reshape(
        math_ops.range(subframes_per_frame), [1, subframes_per_frame])

    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most
    # subframes_per_frame dimension to stitch the subframes together into
    # frames. For example: [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector

    frames = array_ops.reshape(
        array_ops.gather(subframes, selector, axis=axis),
        array_ops.concat([outer_dimensions, [num_frames, frame_length],
                          inner_dimensions], 0))

    if result_shape:
      frames.set_shape(result_shape)
    return frames
