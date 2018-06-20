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
"""Tests for the experimental input pipeline statistics gathering ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np

from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.contrib.data.python.ops import unique
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class OverrideThreadpoolDatasetTest(test.TestCase):

  def testNumThreads(self):

    def get_thread_id(_):
      # Python creates a dummy thread object to represent the current
      # thread when called from an "alien" thread (such as a
      # `PrivateThreadPool` thread in this case). It does not include
      # the TensorFlow-given display name, but it has a unique
      # identifier that maps one-to-one with the underlying OS thread.
      return np.array(threading.current_thread().ident).astype(np.int64)

    for num_threads in [1, 2, 4, 8, 16]:

      dataset = (
          dataset_ops.Dataset.range(1000).map(
              lambda x: script_ops.py_func(get_thread_id, [x], dtypes.int64),
              num_parallel_calls=32).apply(unique.unique()))

      dataset = threadpool.override_threadpool(
          dataset,
          threadpool.PrivateThreadPool(
              num_threads, display_name="private_thread_pool_%d" % num_threads))

      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()

      with self.test_session() as sess:
        sess.run(iterator.initializer)
        thread_ids = []
        try:
          while True:
            thread_ids.append(sess.run(next_element))
        except errors.OutOfRangeError:
          pass
        self.assertEqual(len(thread_ids), len(set(thread_ids)))
        self.assertGreater(len(thread_ids), 0)
        # NOTE(mrry): We don't control the thread pool scheduling, and
        # so cannot guarantee that all of the threads in the pool will
        # perform work.
        self.assertLessEqual(len(thread_ids), num_threads)


if __name__ == "__main__":
  test.main()
