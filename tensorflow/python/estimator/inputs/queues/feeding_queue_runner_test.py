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
"""Tests `FeedingQueueRunner` using arrays and `DataFrames`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.estimator.inputs.pandas_import import HAS_PANDAS
from tensorflow.python.estimator.inputs.queues import feeding_functions as ff
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

if HAS_PANDAS:
  # pylint: disable=g-import-not-at-top
  import pandas as pd


def get_rows(array, row_indices):
  rows = [array[i] for i in row_indices]
  return np.vstack(rows)


class FeedingQueueRunnerTestCase(test.TestCase):
  """Tests for `FeedingQueueRunner`."""

  def testArrayFeeding(self):
    with ops.Graph().as_default():
      array = np.arange(32).reshape([16, 2])
      q = ff._enqueue_data(array, capacity=100)
      batch_size = 3
      dq_op = q.dequeue_many(batch_size)
      with session.Session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        for i in range(100):
          indices = [
              j % array.shape[0]
              for j in range(batch_size * i, batch_size * (i + 1))
          ]
          expected_dq = get_rows(array, indices)
          dq = sess.run(dq_op)
          np.testing.assert_array_equal(indices, dq[0])
          np.testing.assert_array_equal(expected_dq, dq[1])
        coord.request_stop()
        coord.join(threads)

  def testArrayFeedingMultiThread(self):
    with ops.Graph().as_default():
      array = np.arange(256).reshape([128, 2])
      q = ff._enqueue_data(array, capacity=128, num_threads=8, shuffle=True)
      batch_size = 3
      dq_op = q.dequeue_many(batch_size)
      with session.Session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        for _ in range(100):
          dq = sess.run(dq_op)
          indices = dq[0]
          expected_dq = get_rows(array, indices)
          np.testing.assert_array_equal(expected_dq, dq[1])
        coord.request_stop()
        coord.join(threads)

  def testPandasFeeding(self):
    if not HAS_PANDAS:
      return
    with ops.Graph().as_default():
      array1 = np.arange(32)
      array2 = np.arange(32, 64)
      df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(64, 96))
      q = ff._enqueue_data(df, capacity=100)
      batch_size = 5
      dq_op = q.dequeue_many(5)
      with session.Session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        for i in range(100):
          indices = [
              j % array1.shape[0]
              for j in range(batch_size * i, batch_size * (i + 1))
          ]
          expected_df_indices = df.index[indices]
          expected_rows = df.iloc[indices]
          dq = sess.run(dq_op)
          np.testing.assert_array_equal(expected_df_indices, dq[0])
          for col_num, col in enumerate(df.columns):
            np.testing.assert_array_equal(expected_rows[col].values,
                                          dq[col_num + 1])
        coord.request_stop()
        coord.join(threads)

  def testPandasFeedingMultiThread(self):
    if not HAS_PANDAS:
      return
    with ops.Graph().as_default():
      array1 = np.arange(128, 256)
      array2 = 2 * array1
      df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(128))
      q = ff._enqueue_data(df, capacity=128, num_threads=8, shuffle=True)
      batch_size = 5
      dq_op = q.dequeue_many(batch_size)
      with session.Session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        for _ in range(100):
          dq = sess.run(dq_op)
          indices = dq[0]
          expected_rows = df.iloc[indices]
          for col_num, col in enumerate(df.columns):
            np.testing.assert_array_equal(expected_rows[col].values,
                                          dq[col_num + 1])
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
  test.main()
