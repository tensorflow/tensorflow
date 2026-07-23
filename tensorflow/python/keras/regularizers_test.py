# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras regularizers validation."""

from tensorflow.python.framework import constant_op
from tensorflow.python.keras import regularizers
from tensorflow.python.platform import test as test_lib


class CheckPenaltyNumberTest(test_lib.TestCase):

  def testNegativeScalarRaises(self):
    with self.assertRaisesRegex(ValueError, 'expected a non-negative'):
      regularizers.l1(-0.1)

  def testPositiveScalarPasses(self):
    # Should not raise
    regularizers.l1(0.0)
    regularizers.l1(0.5)
    regularizers.l1(1.0)

  def testTensorBypassesValidation(self):
    # Even out-of-range Tensors bypass static validation
    regularizers.l1(constant_op.constant(-1.0))
    regularizers.l1(constant_op.constant(1.5))


if __name__ == '__main__':
  test_lib.main()
