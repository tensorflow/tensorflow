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
"""Test RangeDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import counter
from tensorflow.contrib.data.python.ops import enumerate_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class RangeDatasetTest(test.TestCase):

  def testEnumerateDataset(self):
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)

    iterator = (dataset_ops.Dataset.from_tensor_slices(components).apply(
        enumerate_ops.enumerate_dataset(start)).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int64, get_next[0].dtype)
    self.assertEqual((), get_next[0].shape)
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [t.shape for t in get_next[1]])

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual((20, (b"a", 1, 37.0)), sess.run(get_next))
      self.assertEqual((21, (b"b", 2, 38.0)), sess.run(get_next))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCounter(self):
    """Test dataset construction using `count`."""
    iterator = (counter.Counter(start=3, step=4)
                .make_one_shot_iterator())
    get_next = iterator.get_next()
    self.assertEqual([], get_next.shape.as_list())
    self.assertEqual(dtypes.int64, get_next.dtype)

    negative_iterator = (counter.Counter(start=0, step=-1)
                         .make_one_shot_iterator())
    negative_get_next = negative_iterator.get_next()

    with self.test_session() as sess:
      self.assertEqual(3, sess.run(get_next))
      self.assertEqual(3 + 4, sess.run(get_next))
      self.assertEqual(3 + 2 * 4, sess.run(get_next))

      self.assertEqual(0, sess.run(negative_get_next))
      self.assertEqual(-1, sess.run(negative_get_next))
      self.assertEqual(-2, sess.run(negative_get_next))


if __name__ == "__main__":
  test.main()
