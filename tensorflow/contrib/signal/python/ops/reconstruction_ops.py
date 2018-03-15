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
"""Signal reconstruction via overlapped addition of frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.contrib.signal.python.ops import util_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _shuffle_to_front(input_tensor, k):
  """Shuffles the last `k` indices of `input_tensor` to the front.

  Transposes `input_tensor` to have the last `k` indices at the front. The input
  may have arbitrary rank and unknown shape.

  Args:
    input_tensor: A `Tensor` of arbitrary rank and unknown shape.
    k: A scalar `Tensor` specifying how many indices to shuffle.

  Returns:
    A transposed version of `input_tensor` with `k` indices shuffled to the
    front.

  Raises:
    ValueError: If `input_tensor` is not at least rank `k` or `k` is not scalar.
  """
  k = ops.convert_to_tensor(k, name="k")
  k.shape.with_rank(0)
  k_static = tensor_util.constant_value(k)
  if k_static is not None:
    input_tensor.shape.with_rank_at_least(k_static)

  rank = array_ops.rank(input_tensor)
  outer_indices, inner_indices = array_ops.split(math_ops.range(rank),
                                                 [rank - k, k])
  permutation = array_ops.concat([inner_indices, outer_indices], 0)

  return array_ops.transpose(input_tensor, perm=permutation)


def overlap_and_add(signal, frame_step, name=None):
  """Reconstructs a signal from a framed representation.

  Adds potentially overlapping frames of a signal with shape
  `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
  The resulting tensor has shape `[..., output_size]` where

      output_size = (frames - 1) * frame_step + frame_length

  Args:
    signal: A [..., frames, frame_length] `Tensor`. All dimensions may be
      unknown, and rank must be at least 2.
    frame_step: An integer or scalar `Tensor` denoting overlap offsets. Must be
      less than or equal to `frame_length`.
    name: An optional name for the operation.

  Returns:
    A `Tensor` with shape `[..., output_size]` containing the overlap-added
    frames of `signal`'s inner-most two dimensions.

  Raises:
    ValueError: If `signal`'s rank is less than 2, `frame_step` is not a scalar
      integer or `frame_step` is greater than `frame_length`.
  """
  with ops.name_scope(name, "overlap_and_add", [signal, frame_step]):
    signal = ops.convert_to_tensor(signal, name="signal")
    signal.shape.with_rank_at_least(2)
    frame_step = ops.convert_to_tensor(frame_step, name="frame_step")
    frame_step.shape.assert_has_rank(0)
    if not frame_step.dtype.is_integer:
      raise ValueError("frame_step must be an integer. Got %s" %
                       frame_step.dtype)

    # If frame_length and frame_step are known at graph construction time, check
    # frame_step is less than or equal to frame_length.
    frame_step_static = tensor_util.constant_value(frame_step)
    if (frame_step_static is not None and signal.shape.ndims is not None and
        signal.shape[-1].value is not None and
        frame_step_static > signal.shape[-1].value):
      raise ValueError(
          "frame_step (%d) must be less than or equal to frame_length (%d)" % (
              frame_step_static, signal.shape[-1].value))

    signal_shape = array_ops.shape(signal)

    # All dimensions that are not part of the overlap-and-add. Can be empty for
    # rank 2 inputs.
    outer_dimensions = signal_shape[:-2]

    signal_rank = array_ops.rank(signal)
    frames = signal_shape[-2]
    frame_length = signal_shape[-1]

    subframe_length = util_ops.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # To avoid overlap-adding sample-by-sample, we overlap-add at the "subframe"
    # level, where a subframe is gcd(frame_length, frame_step). Reshape signal
    # from [..., frames, frame_length] into [..., subframes, subframe_length].
    subframe_shape = array_ops.concat(
        [outer_dimensions, [-1, subframe_length]], 0)
    subframe_signal = array_ops.reshape(signal, subframe_shape)

    # Now we shuffle the last [subframes, subframe_length] dimensions to the
    # front.
    # TODO(rjryan): Add an axis argument to unsorted_segment_sum so we can
    # avoid this pair of transposes.
    subframe_signal = _shuffle_to_front(subframe_signal, 2)

    # Use unsorted_segment_sum to add overlapping subframes together.
    segment_ids = array_ops.reshape(shape_ops.frame(
        math_ops.range(output_subframes), subframes_per_frame, subframe_step,
        pad_end=False), [-1])
    result = math_ops.unsorted_segment_sum(subframe_signal, segment_ids,
                                           num_segments=output_subframes)

    # result is a [subframes, subframe_length, ...outer_dimensions] tensor. We
    # return a [...outer_dimensions, output_size] tensor with a transpose and
    # reshape.
    result_shape = array_ops.concat([outer_dimensions, [output_size]], 0)
    return array_ops.reshape(_shuffle_to_front(result, signal_rank - 2),
                             result_shape)
