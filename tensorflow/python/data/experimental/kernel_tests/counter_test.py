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
"""Tests for `tf.data.experimental.Counter`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class CounterTest(test_base.DatasetTestBase):

  @test_util.run_deprecated_v1
  def testCounter(self):
    """Test dataset construction using `count`."""
    iterator = dataset_ops.make_one_shot_iterator(
        counter.Counter(start=3, step=4))
    get_next = iterator.get_next()
    self.assertEqual([], get_next.shape.as_list())
    self.assertEqual(dtypes.int64, get_next.dtype)

    negative_iterator = dataset_ops.make_one_shot_iterator(
        counter.Counter(start=0, step=-1))
    negative_get_next = negative_iterator.get_next()

    with self.cached_session() as sess:
      self.assertEqual(3, self.evaluate(get_next))
      self.assertEqual(3 + 4, self.evaluate(get_next))
      self.assertEqual(3 + 2 * 4, self.evaluate(get_next))

      self.assertEqual(0, self.evaluate(negative_get_next))
      self.assertEqual(-1, self.evaluate(negative_get_next))
      self.assertEqual(-2, self.evaluate(negative_get_next))


if __name__ == "__main__":
  test.main()
