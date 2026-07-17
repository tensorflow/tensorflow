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
"""Tests for Keras callbacks validation."""

from tensorflow.python.keras import callbacks
from tensorflow.python.platform import test as test_lib


class EarlyStoppingValidationTest(test_lib.TestCase):

  def testPatienceNegative(self):
    with self.assertRaisesRegex(ValueError, r'patience.*must be >= 0'):
      callbacks.EarlyStopping(patience=-1)

  def testPatienceZero(self):
    callbacks.EarlyStopping(patience=0)

  def testPatiencePositive(self):
    callbacks.EarlyStopping(patience=3)


if __name__ == '__main__':
  test_lib.main()
