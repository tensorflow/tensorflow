# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for learn.dataframe.transforms.reader_source."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# pylint: disable=g-import-not-at-top
from tensorflow.contrib.learn.python.learn.dataframe.transforms import reader_source as rs
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class ReaderSourceTestCase(test.TestCase):
  """Test class for ReaderSource."""

  def setUp(self):
    super(ReaderSourceTestCase, self).setUp()
    self.work_units = [str(x) for x in range(1000)]

  def testNoShuffle(self):
    id_source = rs.ReaderSource(
        reader_cls=io_ops.IdentityReader,
        work_units=self.work_units,
        batch_size=1,
        shuffle=False,
        num_threads=1)
    index_column, value_column = id_source()
    index_tensor = index_column.build()
    value_tensor = value_column.build()
    self.assertEqual([1], index_tensor.get_shape().as_list())
    self.assertEqual([1], value_tensor.get_shape().as_list())
    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
      for i in range(50):
        index, value = sess.run([index_tensor, value_tensor])
        self.assertEqual(i, int(index[0]))
        self.assertEqual(i, int(value[0]))
      coord.request_stop()
      coord.join(threads)

  def testYesShuffle(self):
    id_source = rs.ReaderSource(
        reader_cls=io_ops.IdentityReader,
        work_units=self.work_units,
        batch_size=1,
        shuffle=True,
        num_threads=10,
        seed=1234)
    index_column, value_column = id_source()
    cache = {}
    index_tensor = index_column.build(cache)
    value_tensor = value_column.build(cache)
    self.assertEqual([1], index_tensor.get_shape().as_list())
    self.assertEqual([1], value_tensor.get_shape().as_list())
    seen = set([])
    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
      for _ in range(500):
        index, value = sess.run([index_tensor, value_tensor])
        self.assertEqual(index, value)
        self.assertNotIn(int(value[0]), seen)
        seen.add(int(value[0]))
      coord.request_stop()
      coord.join(threads)


if __name__ == "__main__":
  test.main()
