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
"""Tests for mel_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.kernel_tests.signal import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.signal import mel_ops
from tensorflow.python.platform import test

# mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
  """Convert frequencies to mel scale using HTK formula.

  Copied from
  https://github.com/tensorflow/models/blob/master/research/audioset/mel_features.py.

  Args:
    frequencies_hertz: Scalar or np.array of frequencies in hertz.

  Returns:
    Object of same size as frequencies_hertz containing corresponding values
    on the mel scale.
  """
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0,
                              unused_dtype=None):
  """Return a matrix that can post-multiply spectrogram rows to make mel.

  Copied from
  https://github.com/tensorflow/models/blob/master/research/audioset/mel_features.py.

  Returns a np.array matrix A that can be used to post-multiply a matrix S of
  spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
  "mel spectrogram" M of frames x num_mel_bins.  M = S A.

  The classic HTK algorithm exploits the complementarity of adjacent mel bands
  to multiply each FFT bin by only one mel weight, then add it, with positive
  and negative signs, to the two adjacent mel bands to which that bin
  contributes.  Here, by expressing this operation as a matrix multiply, we go
  from num_fft multiplies per frame (plus around 2*num_fft adds) to around
  num_fft^2 multiplies and adds.  However, because these are all presumably
  accomplished in a single call to np.dot(), it's not clear which approach is
  faster in Python.  The matrix multiplication has the attraction of being more
  general and flexible, and much easier to read.

  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.  This is
      the number of columns in the output matrix.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    audio_sample_rate: Samples per second of the audio at the input to the
      spectrogram. We need this to figure out the actual frequencies for
      each spectrogram bin, which dictates how they are mapped into mel.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum.  This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.

  Returns:
    An np.array with shape (num_spectrogram_bins, num_mel_bins).

  Raises:
    ValueError: if frequency edges are incorrectly ordered.
  """
  audio_sample_rate = tensor_util.constant_value(audio_sample_rate)
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
  # The i'th mel band (starting from i=1) has center frequency
  # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
  # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
  # the band_edges_mel arrays.
  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the *mel* domain, not hertz.
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  # HTK excludes the spectrogram DC bin; make sure it always gets a zero
  # coefficient.
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix


@tf_test_util.run_all_in_graph_and_eager_modes
class LinearToMelTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      # Defaults. Integer sample rate.
      (20, 129, 8000, False, 125.0, 3800.0, dtypes.float64),
      (20, 129, 8000, True, 125.0, 3800.0, dtypes.float64),
      # Defaults. Float sample rate.
      (20, 129, 8000.0, False, 125.0, 3800.0, dtypes.float64),
      (20, 129, 8000.0, True, 125.0, 3800.0, dtypes.float64),
      # Settings used by Tacotron (https://arxiv.org/abs/1703.10135).
      (80, 1025, 24000.0, False, 80.0, 12000.0, dtypes.float64))
  def test_matches_reference_implementation(
      self, num_mel_bins, num_spectrogram_bins, sample_rate,
      use_tensor_sample_rate, lower_edge_hertz, upper_edge_hertz, dtype):
    if use_tensor_sample_rate:
      sample_rate = constant_op.constant(sample_rate)
    mel_matrix_np = spectrogram_to_mel_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz, dtype)
    mel_matrix = mel_ops.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz, dtype)
    self.assertAllClose(mel_matrix_np, mel_matrix, atol=3e-6)

  @parameterized.parameters(dtypes.float32, dtypes.float64)
  def test_dtypes(self, dtype):
    # LinSpace is not supported for tf.float16.
    self.assertEqual(dtype,
                     mel_ops.linear_to_mel_weight_matrix(dtype=dtype).dtype)

  def test_error(self):
    # TODO(rjryan): Error types are different under eager.
    if context.executing_eagerly():
      return
    with self.assertRaises(ValueError):
      mel_ops.linear_to_mel_weight_matrix(num_mel_bins=0)
    with self.assertRaises(ValueError):
      mel_ops.linear_to_mel_weight_matrix(sample_rate=0.0)
    with self.assertRaises(ValueError):
      mel_ops.linear_to_mel_weight_matrix(lower_edge_hertz=-1)
    with self.assertRaises(ValueError):
      mel_ops.linear_to_mel_weight_matrix(lower_edge_hertz=100,
                                          upper_edge_hertz=10)
    with self.assertRaises(ValueError):
      mel_ops.linear_to_mel_weight_matrix(upper_edge_hertz=1000,
                                          sample_rate=800)
    with self.assertRaises(ValueError):
      mel_ops.linear_to_mel_weight_matrix(dtype=dtypes.int32)

  @parameterized.parameters(dtypes.float32, dtypes.float64)
  def test_constant_folding(self, dtype):
    """Mel functions should be constant foldable."""
    if context.executing_eagerly():
      return
    # TODO(rjryan): tf.bfloat16 cannot be constant folded by Grappler.
    g = ops.Graph()
    with g.as_default():
      mel_matrix = mel_ops.linear_to_mel_weight_matrix(
          sample_rate=constant_op.constant(8000.0, dtype=dtypes.float32),
          dtype=dtype)
      rewritten_graph = test_util.grappler_optimize(g, [mel_matrix])
      self.assertLen(rewritten_graph.node, 1)

  def test_num_spectrogram_bins_dynamic(self):
    num_spectrogram_bins = array_ops.placeholder_with_default(
        ops.convert_to_tensor(129, dtype=dtypes.int32), shape=())
    mel_matrix_np = spectrogram_to_mel_matrix(
        20, 129, 8000.0, 125.0, 3800.0)
    mel_matrix = mel_ops.linear_to_mel_weight_matrix(
        20, num_spectrogram_bins, 8000.0, 125.0, 3800.0)
    self.assertAllClose(mel_matrix_np, mel_matrix, atol=3e-6)


if __name__ == "__main__":
  test.main()
