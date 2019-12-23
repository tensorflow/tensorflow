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
"""Tests for the private `override_threadpool()` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.experimental.ops import threading_options
from tensorflow.python.data.experimental.ops import threadpool
from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class OverrideThreadpoolTest(test_base.DatasetTestBase,
                             parameterized.TestCase):

  def _testNumThreadsHelper(self, num_threads, override_threadpool_fn):

    def get_thread_id(_):
      # Python creates a dummy thread object to represent the current
      # thread when called from an "alien" thread (such as a
      # `PrivateThreadPool` thread in this case). It does not include
      # the TensorFlow-given display name, but it has a unique
      # identifier that maps one-to-one with the underlying OS thread.
      return np.array(threading.current_thread().ident).astype(np.int64)

    dataset = (
        dataset_ops.Dataset.range(1000).map(
            lambda x: script_ops.py_func(get_thread_id, [x], dtypes.int64),
            num_parallel_calls=32).apply(unique.unique()))
    dataset = override_threadpool_fn(dataset)
    next_element = self.getNext(dataset, requires_initialization=True)

    thread_ids = []
    try:
      while True:
        thread_ids.append(self.evaluate(next_element()))
    except errors.OutOfRangeError:
      pass
    self.assertLen(thread_ids, len(set(thread_ids)))
    self.assertNotEmpty(thread_ids)
    if num_threads:
      # NOTE(mrry): We don't control the thread pool scheduling, and
      # so cannot guarantee that all of the threads in the pool will
      # perform work.
      self.assertLessEqual(len(thread_ids), num_threads)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_threads=[1, 2, 4, 8, 16], max_intra_op_parallelism=[None]) +
          combinations.combine(
              num_threads=[4], max_intra_op_parallelism=[-1, 0, 4])))
  def testNumThreadsDeprecated(self, num_threads, max_intra_op_parallelism):

    def override_threadpool_fn(dataset):
      return threadpool.override_threadpool(
          dataset,
          threadpool.PrivateThreadPool(
              num_threads,
              max_intra_op_parallelism=max_intra_op_parallelism,
              display_name="private_thread_pool_%d" % num_threads))

    self._testNumThreadsHelper(num_threads, override_threadpool_fn)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_threads=[1, 2, 4, 8, 16], max_intra_op_parallelism=[None]) +
          combinations.combine(
              num_threads=[None], max_intra_op_parallelism=[0, 1, 4]) +
          combinations.combine(
              num_threads=[4], max_intra_op_parallelism=[0, 1, 4]) +
          combinations.combine(
              num_threads=[None], max_intra_op_parallelism=[None])))
  def testNumThreads(self, num_threads, max_intra_op_parallelism):

    def override_threadpool_fn(dataset):
      t_options = threading_options.ThreadingOptions()
      if max_intra_op_parallelism is not None:
        t_options.max_intra_op_parallelism = max_intra_op_parallelism
      if num_threads is not None:
        t_options.private_threadpool_size = num_threads
      options = dataset_ops.Options()
      options.experimental_threading = t_options
      return dataset.with_options(options)

    self._testNumThreadsHelper(num_threads, override_threadpool_fn)

  @combinations.generate(test_base.default_test_combinations())
  def testMaxIntraOpParallelismAsGraphDefInternal(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset_ops._MaxIntraOpParallelismDataset(dataset, 1)
    graph = graph_pb2.GraphDef().FromString(
        self.evaluate(dataset._as_serialized_graph()))
    self.assertTrue(
        any(node.op != "MaxIntraOpParallelismDataset" for node in graph.node))


if __name__ == "__main__":
  test.main()
