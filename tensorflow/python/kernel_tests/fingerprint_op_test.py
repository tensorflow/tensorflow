# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.fingerprint_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# Fingerprint op has C++ tests. This simple test case tests that fingerprint
# function is accessible via Python API.
class FingerprintTest(test.TestCase):

  def test_default_values(self):
    data = np.arange(10)
    data = np.expand_dims(data, axis=0)
    fingerprint0 = self.evaluate(array_ops.fingerprint(data))
    fingerprint1 = self.evaluate(array_ops.fingerprint(data[:, 1:]))
    self.assertEqual(fingerprint0.ndim, 2)
    self.assertTupleEqual(fingerprint0.shape, fingerprint1.shape)
    self.assertTrue(np.any(fingerprint0 != fingerprint1))

  def test_empty(self):
    f0 = self.evaluate(array_ops.fingerprint([]))
    self.assertEqual(f0.ndim, 2)
    self.assertEqual(f0.shape, (0, 8))


if __name__ == "__main__":
  test.main()
