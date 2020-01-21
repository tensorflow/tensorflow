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
"""Tests for tensorflow.compiler.tests.unstack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class UnstackOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _test(self, size):
    with self.session() as sess:
      x_tf = array_ops.placeholder(np.float32, shape=[size, 512])
      with self.test_scope():
        ret = array_ops.unstack(x_tf)
      ret_vals = sess.run([ret], feed_dict={x_tf: np.zeros([size, 512])})
      self.assertLen(ret_vals[0], size)
      for ret_val in ret_vals[0]:
        self.assertTrue(np.all(ret_val == 0.))

  def testLarge2000(self):
    self._test(2000)


if __name__ == "__main__":
  test.main()
