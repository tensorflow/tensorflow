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
"""Tests for forward and backwards compatibility utilties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compat import compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class DisableV2BehaviorTest(test.TestCase):

  def test_basic(self):
    t = constant_op.constant([1, 2, 3])  # creates a hidden context
    self.assertTrue(isinstance(t, ops.EagerTensor))
    compat.disable_v2_behavior()
    t = constant_op.constant([1, 2, 3])
    self.assertFalse(isinstance(t, ops.EagerTensor))


if __name__ == '__main__':
  compat.enable_v2_behavior()
  test.main()
