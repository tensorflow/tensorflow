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

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import spectral_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class SpectralOpsTest(test.TestCase, parameterized.TestCase):

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
    frames = np.fft.irfft(stft, fft_length)
    # Pad or truncate frames's inner dimension to window_length.
    frames = frames[..., :window_length]
    frames = np.pad(frames, [[0, 0]] * (frames.ndim - 1) +
                    [[0, max(0, window_length - frames.shape[-1])]], "constant")
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

  def _compare(self, signal, frame_length, frame_step, fft_length, tol):
    actual_stft = spectral_ops.stft(
        signal, frame_length, frame_step, fft_length, pad_end=False)
    signal_ph = array_ops.placeholder_with_default(signal, shape=signal.shape)
    actual_stft_from_ph = spectral_ops.stft(
        signal_ph, frame_length, frame_step, fft_length, pad_end=False)

    actual_inverse_stft = spectral_ops.inverse_stft(
        actual_stft, frame_length, frame_step, fft_length)

    actual_stft, actual_stft_from_ph, actual_inverse_stft = self.evaluate(
        [actual_stft, actual_stft_from_ph, actual_inverse_stft])

    actual_stft_ph = array_ops.placeholder_with_default(
        actual_stft, shape=actual_stft.shape)
    actual_inverse_stft_from_ph = self.evaluate(
        spectral_ops.inverse_stft(
            actual_stft_ph, frame_length, frame_step, fft_length))

    # Confirm that there is no difference in output when shape/rank is fully
    # unknown or known.
    self.assertAllClose(actual_stft, actual_stft_from_ph)
    self.assertAllClose(actual_inverse_stft, actual_inverse_stft_from_ph)

    expected_stft = SpectralOpsTest._np_stft(
        signal, fft_length, frame_step, frame_length)
    self.assertAllClose(expected_stft, actual_stft, rtol=tol, atol=tol)

    expected_inverse_stft = SpectralOpsTest._np_inverse_stft(
        expected_stft, fft_length, frame_step, frame_length)
    self.assertAllClose(
        expected_inverse_stft, actual_inverse_stft, rtol=tol, atol=tol)

  def test_shapes(self):
    signal = np.zeros((512,)).astype(np.float32)

    # If fft_length is not provided, the smallest enclosing power of 2 of
    # frame_length (8) is used.
    stft = spectral_ops.stft(signal, frame_length=7, frame_step=8,
                             pad_end=True)
    self.assertAllEqual([64, 5], stft.shape.as_list())
    self.assertAllEqual([64, 5], self.evaluate(stft).shape)

    stft = spectral_ops.stft(signal, frame_length=8, frame_step=8,
                             pad_end=True)
    self.assertAllEqual([64, 5], stft.shape.as_list())
    self.assertAllEqual([64, 5], self.evaluate(stft).shape)

    stft = spectral_ops.stft(signal, frame_length=8, frame_step=8,
                             fft_length=16, pad_end=True)
    self.assertAllEqual([64, 9], stft.shape.as_list())
    self.assertAllEqual([64, 9], self.evaluate(stft).shape)

    stft = spectral_ops.stft(signal, frame_length=16, frame_step=8,
                             fft_length=8, pad_end=True)
    self.assertAllEqual([64, 5], stft.shape.as_list())
    self.assertAllEqual([64, 5], self.evaluate(stft).shape)

    stft = np.zeros((32, 9)).astype(np.complex64)

    inverse_stft = spectral_ops.inverse_stft(stft, frame_length=8,
                                             fft_length=16, frame_step=8)
    expected_length = (stft.shape[0] - 1) * 8 + 8
    self.assertAllEqual([256], inverse_stft.shape.as_list())
    self.assertAllEqual([expected_length], self.evaluate(inverse_stft).shape)

  @parameterized.parameters(
      (512, 64, 32, 64, np.float32, 1e-4),
      (512, 64, 32, 64, np.float64, 1e-8),
      (512, 64, 64, 64, np.float32, 1e-4),
      (512, 64, 64, 64, np.float64, 1e-8),
      (512, 72, 64, 64, np.float32, 1e-4),
      (512, 72, 64, 64, np.float64, 1e-8),
      (512, 64, 25, 64, np.float32, 1e-4),
      (512, 64, 25, 64, np.float64, 1e-8),
      (512, 25, 15, 36, np.float32, 1e-4),
      (512, 25, 15, 36, np.float64, 1e-8),
      (123, 23, 5, 42, np.float32, 1e-4),
      (123, 23, 5, 42, np.float64, 1e-8))
  def test_stft_and_inverse_stft(self, signal_length, frame_length,
                                 frame_step, fft_length, np_rtype, tol):
    """Test that spectral_ops.stft/inverse_stft match a NumPy implementation."""
    signal = np.random.random(signal_length).astype(np_rtype)
    self._compare(signal, frame_length, frame_step, fft_length, tol)

  @parameterized.parameters(
      # 87.5% overlap.
      (4096, 256, 32, 256, np.float32, 1e-5, 1e-6),
      (4096, 256, 32, 256, np.float64, 1e-8, 1e-8),
      # 75% overlap.
      (4096, 256, 64, 256, np.float32, 1e-5, 1e-6),
      (4096, 256, 64, 256, np.float64, 1e-8, 1e-8),
      # Odd frame hop.
      (4096, 128, 25, 128, np.float32, 1e-3, 1e-6),
      (4096, 128, 25, 128, np.float64, 5e-4, 1e-8),
      # Odd frame length.
      (4096, 127, 32, 128, np.float32, 1e-3, 1e-6),
      (4096, 127, 32, 128, np.float64, 1e-3, 1e-8),
      # 50% overlap.
      (4096, 128, 64, 128, np.float32, 0.4, 1e-6),
      (4096, 128, 64, 128, np.float64, 0.4, 1e-8))
  def test_stft_round_trip(self, signal_length, frame_length, frame_step,
                           fft_length, np_rtype, threshold,
                           corrected_threshold):
    # Generate a random white Gaussian signal.
    signal = np.random.normal(size=signal_length).astype(np_rtype)

    stft = spectral_ops.stft(signal, frame_length, frame_step, fft_length,
                             pad_end=False)
    inverse_stft = spectral_ops.inverse_stft(stft, frame_length, frame_step,
                                             fft_length)
    inverse_stft_corrected = spectral_ops.inverse_stft(
        stft, frame_length, frame_step, fft_length,
        window_fn=spectral_ops.inverse_stft_window_fn(frame_step))
    inverse_stft, inverse_stft_corrected = self.evaluate(
        [inverse_stft, inverse_stft_corrected])

    # Truncate signal to the size of inverse stft.
    signal = signal[:inverse_stft.shape[0]]

    # Ignore the frame_length samples at either edge.
    signal = signal[frame_length:-frame_length]
    inverse_stft = inverse_stft[frame_length:-frame_length]
    inverse_stft_corrected = inverse_stft_corrected[
        frame_length:-frame_length]

    # Check that the inverse and original signal are close up to a scale
    # factor.
    inverse_stft_scaled = inverse_stft / np.mean(np.abs(inverse_stft))
    signal_scaled = signal / np.mean(np.abs(signal))
    self.assertLess(np.std(inverse_stft_scaled - signal_scaled), threshold)

    # Check that the inverse with correction and original signal are close.
    self.assertLess(np.std(inverse_stft_corrected - signal),
                    corrected_threshold)

  @parameterized.parameters(
      (256, 32),
      (256, 64),
      (128, 25),
      (127, 32),
      (128, 64))
  def test_inverse_stft_window_fn(self, frame_length, frame_step):
    """Test that inverse_stft_window_fn has unit gain at each window phase."""
    hann_window = window_ops.hann_window(frame_length, dtype=dtypes.float32)
    inverse_window_fn = spectral_ops.inverse_stft_window_fn(frame_step)
    inverse_window = inverse_window_fn(frame_length, dtype=dtypes.float32)
    hann_window, inverse_window = self.evaluate([hann_window, inverse_window])

    # Expect unit gain at each phase of the window.
    product_window = hann_window * inverse_window
    for i in range(frame_step):
      self.assertAllClose(1.0, np.sum(product_window[i::frame_step]))

  @parameterized.parameters((256, 64), (128, 32))
  def test_inverse_stft_window_fn_special_case(self, frame_length, frame_step):
    """Test inverse_stft_window_fn in special overlap = 3/4 case."""
    # Cases in which frame_length is an integer multiple of 4 * frame_step are
    # special because they allow exact reproduction of the waveform with a
    # squared Hann window (Hann window in both forward and reverse transforms).
    # In the case where frame_length = 4 * frame_step, that combination
    # produces a constant gain of 1.5, and so the corrected window will be the
    # Hann window / 1.5.
    hann_window = window_ops.hann_window(frame_length, dtype=dtypes.float32)
    inverse_window_fn = spectral_ops.inverse_stft_window_fn(frame_step)
    inverse_window = inverse_window_fn(frame_length, dtype=dtypes.float32)
    self.assertAllClose(hann_window, inverse_window * 1.5)

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
    # TODO(rjryan): Update gradient tests for Eager.
    if context.executing_eagerly():
      return
    with self.session() as sess:
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
      sinusoid_gradient = self.evaluate(self._compute_stft_gradient(sinusoid))
      self.assertFalse((sinusoid_gradient == 0.0).all())

  @parameterized.parameters(
      (64, 16, 8, 16, np.float32, 2e-3, 5e-4),
      (64, 16, 8, 16, np.float64, 1e-8, 1e-8),
      (64, 16, 16, 16, np.float32, 2e-3, 5e-4),
      (64, 16, 16, 16, np.float64, 1e-8, 1e-8),
      (64, 16, 7, 16, np.float32, 2e-3, 5e-4),
      (64, 16, 7, 16, np.float64, 1e-8, 1e-8),
      (64, 7, 4, 9, np.float32, 2e-3, 5e-4),
      (64, 7, 4, 9, np.float64, 1e-8, 1e-8),
      (29, 5, 1, 10, np.float32, 2e-3, 5e-4),
      (29, 5, 1, 10, np.float64, 1e-8, 1e-8))
  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message="On ROCm, this fails with mismatches at some locations "
      "(possibly due to peculiarities of rocFFT - investigate)")
  def test_gradients_numerical(self, signal_length, frame_length, frame_step,
                               fft_length, np_rtype, forward_tol, backward_tol):
    # TODO(rjryan): Investigate why STFT gradient error is so high.
    signal = np.random.rand(signal_length).astype(np_rtype) * 2 - 1

    def forward(signal):
      return spectral_ops.stft(
          signal, frame_length, frame_step, fft_length, pad_end=False)
    ((f_jacob_t,), (f_jacob_n,)) = gradient_checker_v2.compute_gradient(
        forward, [signal])
    self.assertAllClose(f_jacob_t, f_jacob_n,
                        rtol=forward_tol, atol=forward_tol)

    def backward(stft):
      return spectral_ops.inverse_stft(
          stft, frame_length, frame_step, fft_length)

    stft = forward(signal)
    ((b_jacob_t,), (b_jacob_n,)) = gradient_checker_v2.compute_gradient(
        backward, [stft])
    self.assertAllClose(b_jacob_t, b_jacob_n,
                        rtol=backward_tol, atol=backward_tol)

  @parameterized.parameters(
      itertools.product(
          (4000,),
          (256,),
          (np.float32, np.float64),
          ("ortho", None),
          ("vorbis", "kaiser_bessel_derived", None),
          (False, True)))
  def test_mdct_round_trip(self, signal_length, frame_length, np_rtype,
                           norm, window_type, pad_end):
    if np_rtype == np.float32:
      tol = 1e-5
    else:
      if window_type == "kaiser_bessel_derived":
        tol = 1e-6
      else:
        tol = 1e-8
    # Generate a random white Gaussian signal.
    signal = np.random.normal(size=signal_length).astype(np_rtype)
    if window_type == "vorbis":
      window_fn = window_ops.vorbis_window
    elif window_type == "kaiser_bessel_derived":
      window_fn = window_ops.kaiser_bessel_derived_window
    elif window_type is None:
      window_fn = None
    mdct = spectral_ops.mdct(signal, frame_length, norm=norm,
                             window_fn=window_fn, pad_end=pad_end)
    inverse_mdct = spectral_ops.inverse_mdct(mdct, norm=norm,
                                             window_fn=window_fn)
    inverse_mdct = self.evaluate(inverse_mdct)

    # Truncate signal and inverse_mdct to their minimum length.
    min_length = np.minimum(signal.shape[0], inverse_mdct.shape[0])
    # Ignore the half_len samples at either edge.
    half_len = frame_length // 2
    signal = signal[half_len:min_length-half_len]
    inverse_mdct = inverse_mdct[half_len:min_length-half_len]

    # Check that the inverse and original signal are close.
    self.assertAllClose(inverse_mdct, signal, atol=tol, rtol=tol)

if __name__ == "__main__":
  test.main()
