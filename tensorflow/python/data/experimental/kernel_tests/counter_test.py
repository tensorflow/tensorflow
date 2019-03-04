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


@test_util.run_all_in_graph_and_eager_modes
class CounterTest(test_base.DatasetTestBase):

  def testCounter(self):
    """Test dataset construction using `count`."""
    dataset = counter.Counter(start=3, step=4)
    self.assertEqual(
        [], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    self.assertEqual(dtypes.int64, dataset_ops.get_legacy_output_types(dataset))
    get_next = self.getNext(dataset)

    negative_dataset = counter.Counter(start=0, step=-1)
    negative_get_next = self.getNext(negative_dataset)

    self.assertEqual(3, self.evaluate(get_next()))
    self.assertEqual(3 + 4, self.evaluate(get_next()))
    self.assertEqual(3 + 2 * 4, self.evaluate(get_next()))

    self.assertEqual(0, self.evaluate(negative_get_next()))
    self.assertEqual(-1, self.evaluate(negative_get_next()))
    self.assertEqual(-2, self.evaluate(negative_get_next()))


if __name__ == "__main__":
  test.main()
