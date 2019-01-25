# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""AudioMicrofrontend Op creates filterbanks from audio data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.lite.experimental.microfrontend.ops import gen_audio_microfrontend_op
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_audio_microfrontend_op = loader.load_op_library(
    resource_loader.get_path_to_datafile("_audio_microfrontend_op.so"))


def audio_microfrontend(audio,
                        sample_rate=16000,
                        window_size=25,
                        window_step=10,
                        num_channels=32,
                        upper_band_limit=7500.0,
                        lower_band_limit=125.0,
                        smoothing_bits=10,
                        even_smoothing=0.025,
                        odd_smoothing=0.06,
                        min_signal_remaining=0.05,
                        enable_pcan=True,
                        pcan_strength=0.95,
                        pcan_offset=80.0,
                        gain_bits=21,
                        enable_log=True,
                        scale_shift=6,
                        left_context=0,
                        right_context=0,
                        frame_stride=1,
                        zero_padding=False,
                        out_scale=1,
                        out_type=tf.uint16):
  """Audio Microfrontend Op.

  This Op converts a sequence of audio data into one or more
  feature vectors containing filterbanks of the input. The
  conversion process uses a lightweight library to perform:

  1. A slicing window function
  2. Short-time FFTs
  3. Filterbank calculations
  4. Noise reduction
  5. PCAN Auto Gain Control
  6. Logarithmic scaling

  Args:
    audio: 1D Tensor, int16 audio data in temporal ordering.
    sample_rate: Integer, the sample rate of the audio in Hz.
    window_size: Integer, length of desired time frames in ms.
    window_step: Integer, length of step size for the next frame in ms.
    num_channels: Integer, the number of filterbank channels to use.
    upper_band_limit: Float, the highest frequency included in the filterbanks.
    lower_band_limit: Float, the lowest frequency included in the filterbanks.
    smoothing_bits: Int, scale up signal by 2^(smoothing_bits) before reduction.
    even_smoothing: Float, smoothing coefficient for even-numbered channels.
    odd_smoothing: Float, smoothing coefficient for odd-numbered channels.
    min_signal_remaining: Float, fraction of signal to preserve in smoothing.
    enable_pcan: Bool, enable PCAN auto gain control.
    pcan_strength: Float, gain normalization exponent.
    pcan_offset: Float, positive value added in the normalization denominator.
    gain_bits: Int, number of fractional bits in the gain.
    enable_log: Bool, enable logarithmic scaling of filterbanks.
    scale_shift: Integer, scale filterbanks by 2^(scale_shift).
    left_context: Integer, number of preceding frames to attach to each frame.
    right_context: Integer, number of preceding frames to attach to each frame.
    frame_stride: Integer, M frames to skip over, where output[n] = frame[n*M].
    zero_padding: Bool, if left/right context is out-of-bounds, attach frame of
      zeroes. Otherwise, frame[0] or frame[size-1] will be copied.
    out_scale: Integer, divide all filterbanks by this number.
    out_type: DType, type of the output Tensor, defaults to UINT16.

  Returns:
    filterbanks: 2D Tensor, each row is a time frame, each column is a channel.

  Raises:
    ValueError: If the audio tensor is not explicitly a vector.
  """
  audio_shape = audio.get_shape()
  if audio_shape.ndims is None:
    raise ValueError("Input to `AudioMicrofrontend` should have known rank.")
  if len(audio_shape) > 1:
    audio = tf.reshape(audio, [-1])

  return gen_audio_microfrontend_op.audio_microfrontend(
      audio, sample_rate, window_size, window_step, num_channels,
      upper_band_limit, lower_band_limit, smoothing_bits, even_smoothing,
      odd_smoothing, min_signal_remaining, enable_pcan, pcan_strength,
      pcan_offset, gain_bits, enable_log, scale_shift, left_context,
      right_context, frame_stride, zero_padding, out_scale, out_type)


tf.NotDifferentiable("AudioMicrofrontend")
