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
"""Tests for `tf.data.Iterator` using distributed sessions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class IteratorClusterTest(test.TestCase):

  def testRemoteIteratorWithoutRemoteCallFail(self):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2
    worker, _ = test_util.create_local_cluster(
        1, 1, worker_config=worker_config)

    with ops.device("/job:worker/replica:0/task:0/cpu:1"):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
      iterator_3_handle = iterator_3.string_handle()

    with ops.device("/job:worker/replica:0/task:0/cpu:0"):
      remote_it = iterator_ops.Iterator.from_string_handle(
          iterator_3_handle, dataset_3.output_types, dataset_3.output_shapes)
      get_next_op = remote_it.get_next()

    with session.Session(worker[0].target) as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next_op)

  def _testRemoteIteratorHelper(self, device0, device1, target):
    with ops.device(device1):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
      iterator_3_handle = iterator_3.string_handle()

    @function.Defun(dtypes.string)
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, dataset_3.output_types, dataset_3.output_shapes)
      return remote_iterator.get_next()

    with ops.device(device0):
      target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      remote_op = functional_ops.remote_call(
          args=[iterator_3_handle],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target=target_placeholder)

    with session.Session(target) as sess:
      elem = sess.run(remote_op, feed_dict={target_placeholder: device1})
      self.assertEqual(elem, [1])
      # Fails when target is cpu:0 where the resource is not located.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(remote_op, feed_dict={target_placeholder: device0})
      elem = sess.run(iterator_3.get_next())
      self.assertEqual(elem, [2])
      elem = sess.run(remote_op, feed_dict={target_placeholder: device1})
      self.assertEqual(elem, [3])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(remote_op, feed_dict={target_placeholder: device1})

  def testRemoteIteratorUsingRemoteCallOp(self):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2
    worker, _ = test_util.create_local_cluster(
        1, 1, worker_config=worker_config)

    self._testRemoteIteratorHelper("/job:worker/replica:0/task:0/cpu:0",
                                   "/job:worker/replica:0/task:0/cpu:1",
                                   worker[0].target)

  def testRemoteIteratorUsingRemoteCallOpCrossProcess(self):
    workers, _ = test_util.create_local_cluster(2, 1)

    self._testRemoteIteratorHelper("/job:worker/replica:0/task:0/cpu:0",
                                   "/job:worker/replica:0/task:1/cpu:0",
                                   workers[0].target)

  def testCaptureHashTableInSharedIterator(self):
    worker, _ = test_util.create_local_cluster(1, 1)

    # NOTE(mrry): We must use the V2 variants of `HashTable`
    # etc. because these produce a `tf.resource`-typed output that is
    # compatible with the in-graph function implementation.
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.HashTable(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        shared_name="shared_table")

    input_sentences = dataset_ops.Dataset.from_tensor_slices(
        ["brain brain tank salad surgery", "surgery brain"])

    iterator = (
        input_sentences.map(lambda x: string_ops.string_split([x]).values).map(
            table.lookup)
        .make_initializable_iterator(shared_name="shared_iterator"))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with session.Session(worker[0].target) as sess:
      sess.run(table.initializer)
      sess.run(init_op)
      self.assertAllEqual([0, 0, -1, 1, 2], sess.run(get_next))

    with session.Session(worker[0].target) as sess:
      self.assertAllEqual([2, 0], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testImplicitDisposeParallelMapDataset(self):
    # Tests whether a parallel map dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(None) -> PrefetchDataset(100).
    worker, _ = test_util.create_local_cluster(1, 1)

    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(None).prefetch(10000))

    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with session.Session(worker[0].target) as sess:
      sess.run(init_op)
      for _ in range(3):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
