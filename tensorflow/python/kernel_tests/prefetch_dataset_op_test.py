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
"""Test PrefetchDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class PrefetchDatasetTest(test.TestCase):
  def testBufferSize(self):
    buffer_size = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(10).prefetch(
      buffer_size=buffer_size).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={buffer_size: 5})
      for m in range(10):
        self.assertEqual(m, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testInvalidBufferSize(self):
    buffer_size = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(10).prefetch(
      buffer_size=buffer_size).make_initializable_iterator()
    init_op = iterator.initializer

    with self.assertRaisesRegexp(errors.InvalidArgumentError, "buffer_size"):
      with self.test_session() as sess:
        sess.run(init_op, feed_dict={buffer_size: 0})

    with self.assertRaisesRegexp(errors.InvalidArgumentError, "buffer_size"):
      with self.test_session() as sess:
        sess.run(init_op, feed_dict={buffer_size: -5})


if __name__ == "__main__":
  test.main()
