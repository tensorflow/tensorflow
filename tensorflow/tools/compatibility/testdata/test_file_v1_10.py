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
"""Tests for tf upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 2.0."""

  def testRenames(self):
    with self.cached_session():
      self.assertAllClose(1.04719755, tf.acos(0.5).eval())
      self.assertAllClose(0.5, tf.rsqrt(4.0).eval())

if __name__ == "__main__":
  test_lib.main()
