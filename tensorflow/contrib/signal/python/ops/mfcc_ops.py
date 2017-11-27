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
"""Mel-Frequency Cepstral Coefficients (MFCCs) ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops


def mfccs_from_log_mel_spectrograms(log_mel_spectrograms, name=None):
  """Computes [MFCCs][mfcc] of `log_mel_spectrograms`.

  Implemented with GPU-compatible ops and supports gradients.

  [Mel-Frequency Cepstral Coefficient (MFCC)][mfcc] calculation consists of
  taking the DCT-II of a log-magnitude mel-scale spectrogram. [HTK][htk]'s MFCCs
  use a particular scaling of the DCT-II which is almost orthogonal
  normalization. We follow this convention.

  All `num_mel_bins` MFCCs are returned and it is up to the caller to select
  a subset of the MFCCs based on their application. For example, it is typical
  to only use the first few for speech recognition, as this results in
  an approximately pitch-invariant representation of the signal.

  For example:

  ```python
  sample_rate = 16000.0
  # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
  pcm = tf.placeholder(tf.float32, [None, None])

  # A 1024-point STFT with frames of 64 ms and 75% overlap.
  stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=256,
                                 fft_length=1024)
  spectrograms = tf.abs(stft)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1].value
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]
  ```

  Args:
    log_mel_spectrograms: A `[..., num_mel_bins]` `float32` `Tensor` of
      log-magnitude mel-scale spectrograms.
    name: An optional name for the operation.
  Returns:
    A `[..., num_mel_bins]` `float32` `Tensor` of the MFCCs of
    `log_mel_spectrograms`.

  Raises:
    ValueError: If `num_mel_bins` is not positive.

  [mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  [htk]: https://en.wikipedia.org/wiki/HTK_(software)
  """
  with ops.name_scope(name, 'mfccs_from_log_mel_spectrograms',
                      [log_mel_spectrograms]):
    # Compute the DCT-II of the resulting log-magnitude mel-scale spectrogram.
    # The DCT used in HTK scales every basis vector by sqrt(2/N), which is the
    # scaling required for an "orthogonal" DCT-II *except* in the 0th bin, where
    # the true orthogonal DCT (as implemented by scipy) scales by sqrt(1/N). For
    # this reason, we don't apply orthogonal normalization and scale the DCT by
    # `0.5 * sqrt(2/N)` manually.
    log_mel_spectrograms = ops.convert_to_tensor(log_mel_spectrograms,
                                                 dtype=dtypes.float32)
    if (log_mel_spectrograms.shape.ndims and
        log_mel_spectrograms.shape[-1].value is not None):
      num_mel_bins = log_mel_spectrograms.shape[-1].value
      if num_mel_bins == 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' %
                         log_mel_spectrograms)
    else:
      num_mel_bins = array_ops.shape(log_mel_spectrograms)[-1]

    dct2 = spectral_ops.dct(log_mel_spectrograms)
    return dct2 * math_ops.rsqrt(num_mel_bins * 2.0)
