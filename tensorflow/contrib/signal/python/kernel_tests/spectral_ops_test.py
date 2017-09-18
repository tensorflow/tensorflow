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
"""Tests for spectral_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.signal.python.ops import spectral_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.platform import test


class SpectralOpsTest(test.TestCase):

  @staticmethod
  def _np_hann_periodic_window(length):
    if length == 1:
      return np.ones(1)
    odd = length % 2
    if not odd:
      length += 1
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))
    if not odd:
      window = window[:-1]
    return window

  @staticmethod
  def _np_frame(data, window_length, hop_length):
    num_frames = 1 + int(np.floor((len(data) - window_length) // hop_length))
    shape = (num_frames, window_length)
    strides = (data.strides[0] * hop_length, data.strides[0])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

  @staticmethod
  def _np_stft(data, fft_length, hop_length, window_length):
    frames = SpectralOpsTest._np_frame(data, window_length, hop_length)
    window = SpectralOpsTest._np_hann_periodic_window(window_length)
    return np.fft.rfft(frames * window, fft_length)

  @staticmethod
  def _np_inverse_stft(stft, fft_length, hop_length, window_length):
    frames = np.fft.irfft(stft, fft_length)[..., :window_length]
    window = SpectralOpsTest._np_hann_periodic_window(window_length)
    return SpectralOpsTest._np_overlap_add(frames * window, hop_length)

  @staticmethod
  def _np_overlap_add(stft, hop_length):
    num_frames, window_length = np.shape(stft)
    # Output length will be one complete window, plus another hop_length's
    # worth of points for each additional window.
    output_length = window_length + (num_frames - 1) * hop_length
    output = np.zeros(output_length)
    for i in range(num_frames):
      output[i * hop_length:i * hop_length + window_length] += stft[i,]
    return output

  def _compare(self, signal, frame_length, frame_step, fft_length):
    with spectral_ops_test_util.fft_kernel_label_map(), (
        self.test_session(use_gpu=True)) as sess:
      actual_stft = spectral_ops.stft(
          signal, frame_length, frame_step, fft_length, pad_end=False)

      actual_inverse_stft = spectral_ops.inverse_stft(
          actual_stft, frame_length, frame_step, fft_length)

      actual_stft, actual_inverse_stft = sess.run(
          [actual_stft, actual_inverse_stft])

      expected_stft = SpectralOpsTest._np_stft(
          signal, fft_length, frame_step, frame_length)
      self.assertAllClose(expected_stft, actual_stft, 1e-4, 1e-4)

      expected_inverse_stft = SpectralOpsTest._np_inverse_stft(
          expected_stft, fft_length, frame_step, frame_length)
      self.assertAllClose(
          expected_inverse_stft, actual_inverse_stft, 1e-4, 1e-4)

  def _compare_round_trip(self, signal, frame_length, frame_step, fft_length):
    with spectral_ops_test_util.fft_kernel_label_map(), (
        self.test_session(use_gpu=True)) as sess:
      stft = spectral_ops.stft(signal, frame_length, frame_step, fft_length,
                               pad_end=False)
      inverse_stft = spectral_ops.inverse_stft(stft, frame_length, frame_step,
                                               fft_length)
      signal, inverse_stft = sess.run([signal, inverse_stft])

      # Since the shapes can differ due to padding, pad both signals to the max
      # of their lengths.
      max_length = max(signal.shape[0], inverse_stft.shape[0])
      signal = np.pad(signal, (0, max_length - signal.shape[0]), "constant")
      inverse_stft = np.pad(inverse_stft,
                            (0, max_length - inverse_stft.shape[0]), "constant")

      # Ignore the frame_length samples at either edge.
      start = frame_length
      end = signal.shape[0] - frame_length
      ratio = signal[start:end] / inverse_stft[start:end]

      # Check that the inverse and original signal are equal up to a constant
      # factor.
      self.assertLess(np.var(ratio), 2e-5)

  def test_shapes(self):
    with spectral_ops_test_util.fft_kernel_label_map(), (
        self.test_session(use_gpu=True)):
      signal = np.zeros((512,)).astype(np.float32)

      # If fft_length is not provided, the smallest enclosing power of 2 of
      # frame_length (8) is used.
      stft = spectral_ops.stft(signal, frame_length=7, frame_step=8,
                               pad_end=True)
      self.assertAllEqual([64, 5], stft.shape.as_list())
      self.assertAllEqual([64, 5], stft.eval().shape)

      stft = spectral_ops.stft(signal, frame_length=8, frame_step=8,
                               pad_end=True)
      self.assertAllEqual([64, 5], stft.shape.as_list())
      self.assertAllEqual([64, 5], stft.eval().shape)

      stft = spectral_ops.stft(signal, frame_length=8, frame_step=8,
                               fft_length=16, pad_end=True)
      self.assertAllEqual([64, 9], stft.shape.as_list())
      self.assertAllEqual([64, 9], stft.eval().shape)

      stft = np.zeros((32, 9)).astype(np.complex64)

      inverse_stft = spectral_ops.inverse_stft(stft, frame_length=8,
                                               fft_length=16, frame_step=8)
      expected_length = (stft.shape[0] - 1) * 8 + 8
      self.assertAllEqual([None], inverse_stft.shape.as_list())
      self.assertAllEqual([expected_length], inverse_stft.eval().shape)

  def test_stft_and_inverse_stft(self):
    """Test that spectral_ops.stft/inverse_stft match a NumPy implementation."""
    # Tuples of (signal_length, frame_length, frame_step, fft_length).
    test_configs = [
        (512, 64, 32, 64),
        (512, 64, 64, 64),
        (512, 64, 25, 64),
        (512, 25, 15, 36),
        (123, 23, 5, 42),
    ]

    for signal_length, frame_length, frame_step, fft_length in test_configs:
      signal = np.random.random(signal_length).astype(np.float32)
      self._compare(signal, frame_length, frame_step, fft_length)

  def test_stft_round_trip(self):
    # Tuples of (signal_length, frame_length, frame_step, fft_length).
    test_configs = [
        # 87.5% overlap.
        (4096, 256, 32, 256),
        # 75% overlap.
        (4096, 256, 64, 256),
        # Odd frame hop.
        (4096, 128, 25, 128),
        # Odd frame length.
        (4096, 127, 32, 128),
    ]

    for signal_length, frame_length, frame_step, fft_length in test_configs:
      # Generate a 440Hz signal at 8kHz sample rate.
      signal = math_ops.sin(2 * np.pi * 440 / 8000 *
                            math_ops.to_float(math_ops.range(signal_length)))
      self._compare_round_trip(signal, frame_length, frame_step, fft_length)

  @staticmethod
  def _compute_stft_gradient(signal, frame_length=32, frame_step=16,
                             fft_length=32):
    """Computes the gradient of the STFT with respect to `signal`."""
    stft = spectral_ops.stft(signal, frame_length, frame_step, fft_length)
    magnitude_stft = math_ops.abs(stft)
    loss = math_ops.reduce_sum(magnitude_stft)
    return gradients_impl.gradients([loss], [signal])[0]

  def test_gradients(self):
    """Test that spectral_ops.stft has a working gradient."""
    with spectral_ops_test_util.fft_kernel_label_map(), (
        self.test_session(use_gpu=True)) as sess:
      signal_length = 512

      # An all-zero signal has all zero gradients with respect to the sum of the
      # magnitude STFT.
      empty_signal = array_ops.zeros([signal_length], dtype=dtypes.float32)
      empty_signal_gradient = sess.run(
          self._compute_stft_gradient(empty_signal))
      self.assertTrue((empty_signal_gradient == 0.0).all())

      # A sinusoid will have non-zero components of its gradient with respect to
      # the sum of the magnitude STFT.
      sinusoid = math_ops.sin(
          2 * np.pi * math_ops.linspace(0.0, 1.0, signal_length))
      sinusoid_gradient = sess.run(self._compute_stft_gradient(sinusoid))
      self.assertFalse((sinusoid_gradient == 0.0).all())

  def test_gradients_numerical(self):
    with spectral_ops_test_util.fft_kernel_label_map(), (
        self.test_session(use_gpu=True)):
      # Tuples of (signal_length, frame_length, frame_step, fft_length,
      # stft_bound, inverse_stft_bound).
      # TODO(rjryan): Investigate why STFT gradient error is so high.
      test_configs = [
          (64, 16, 8, 16),
          (64, 16, 16, 16),
          (64, 16, 7, 16),
          (64, 7, 4, 9),
          (29, 5, 1, 10),
      ]

      for (signal_length, frame_length, frame_step, fft_length) in test_configs:
        signal_shape = [signal_length]
        signal = random_ops.random_uniform(signal_shape)
        stft_shape = [max(0, 1 + (signal_length - frame_length) // frame_step),
                      fft_length // 2 + 1]
        stft = spectral_ops.stft(signal, frame_length, frame_step, fft_length,
                                 pad_end=False)
        inverse_stft_shape = [(stft_shape[0] - 1) * frame_step + frame_length]
        inverse_stft = spectral_ops.inverse_stft(stft, frame_length, frame_step,
                                                 fft_length)
        stft_error = test.compute_gradient_error(signal, [signal_length],
                                                 stft, stft_shape)
        inverse_stft_error = test.compute_gradient_error(
            stft, stft_shape, inverse_stft, inverse_stft_shape)
        self.assertLess(stft_error, 2e-3)
        self.assertLess(inverse_stft_error, 5e-4)


if __name__ == "__main__":
  test.main()
