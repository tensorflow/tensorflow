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
"""Tests for tf.contrib.stateless API.

The real tests are in python/kernel_tests/random/stateless_random_ops_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import stateless
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


class StatelessOpsTest(test.TestCase):

  def testAPI(self):
    self.assertIs(stateless.stateless_random_uniform,
                  stateless_random_ops.stateless_random_uniform)
    self.assertIs(stateless.stateless_random_normal,
                  stateless_random_ops.stateless_random_normal)
    self.assertIs(stateless.stateless_truncated_normal,
                  stateless_random_ops.stateless_truncated_normal)
    self.assertIs(stateless.stateless_multinomial,
                  stateless_random_ops.stateless_multinomial)


if __name__ == '__main__':
  test.main()
