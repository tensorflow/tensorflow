# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for where op."""

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import bincount_ops
from tensorflow.python.platform import test
# pylint: enable=g-direct-tensorflow-import


class WhereOpTest(xla_test.XLATestCase):

  def testBincount(self):
    self.skipTest("TODO: this a dummy kernel")
    """Test first form of where (return indices)."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.int32)
        values = bincount_ops.bincount(x)

      # Output of the computation is dynamic.
      feed = [1, 1, 2, 3, 2, 4, 4, 5]
      self.assertAllEqual([0, 2, 2, 1, 2, 1],
                          sess.run(values, {x: feed}))

  
if __name__ == "__main__":
  test.main()
