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
"""Tests for mfcc_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.signal.python.ops import mfcc_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.platform import test


# TODO(rjryan): We have no open source tests for MFCCs at the moment. Internally
# at Google, this code is tested against a reference implementation that follows
# HTK conventions.
class MFCCTest(test.TestCase):

  def test_error(self):
    # num_mel_bins must be positive.
    with self.assertRaises(ValueError):
      signal = array_ops.zeros((2, 3, 0))
      mfcc_ops.mfccs_from_log_mel_spectrograms(signal)

    # signal must be float32
    with self.assertRaises(ValueError):
      signal = array_ops.zeros((2, 3, 5), dtype=dtypes.float64)
      mfcc_ops.mfccs_from_log_mel_spectrograms(signal)

  def test_basic(self):
    """A basic test that the op runs on random input."""
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session(use_gpu=True):
        signal = random_ops.random_normal((2, 3, 5))
        mfcc_ops.mfccs_from_log_mel_spectrograms(signal).eval()


if __name__ == "__main__":
  test.main()
