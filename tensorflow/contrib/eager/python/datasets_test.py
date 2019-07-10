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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import threading
import time

import numpy as np

from tensorflow.contrib import lookup
from tensorflow.contrib.eager.python import datasets
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental.ops import threadpool
from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as trackable_utils


class IteratorTest(test.TestCase):

  def testBasic(self):
    got = []
    for t in datasets.Iterator(Dataset.range(4)):
      got.append(t.numpy())
    self.assertAllEqual([0, 1, 2, 3], got)

  def testBasicOneShotIterator(self):
    got = []
    for t in Dataset.range(4).make_one_shot_iterator():
      got.append(t.numpy())
    self.assertAllEqual([0, 1, 2, 3], got)

  def testBasicImplicitIterator(self):
    got = []
    for t in Dataset.range(4):
      got.append(t.numpy())
    self.assertAllEqual([0, 1, 2, 3], got)

  def testGetNext(self):
    iterator = datasets.Iterator(Dataset.range(4))
    self.assertEqual(0, iterator.get_next().numpy())
    self.assertEqual(1, iterator.get_next().numpy())
    self.assertEqual(2, iterator.get_next().numpy())
    self.assertEqual(3, iterator.get_next().numpy())
    with self.assertRaises(errors.OutOfRangeError):
      iterator.get_next()

  def testGetNextOneShotIterator(self):
    iterator = Dataset.range(4).make_one_shot_iterator()
    self.assertEqual(0, iterator.get_next().numpy())
    self.assertEqual(1, iterator.get_next().numpy())
    self.assertEqual(2, iterator.get_next().numpy())
    self.assertEqual(3, iterator.get_next().numpy())
    with self.assertRaises(errors.OutOfRangeError):
      iterator.get_next()

  def testMultipleIteratorsOnTheSameDataset(self):
    ds = Dataset.range(4)
    it1 = datasets.Iterator(ds)
    it2 = datasets.Iterator(ds)
    got = [x.numpy() for x in it1]
    self.assertAllEqual([0, 1, 2, 3], got)

    got = [x.numpy() for x in it2]
    self.assertAllEqual([0, 1, 2, 3], got)

  def testNestedOutputs(self):
    ds = Dataset.zip((Dataset.range(4), Dataset.zip((Dataset.range(4),
                                                     Dataset.range(4)))))
    total = 0
    # The Iterator will return a nested structure of Tensor objects.
    # Some funkiness to compare against simple integers.
    for (i, x) in enumerate(datasets.Iterator(ds)):
      want = (i, (i, i))
      got = (x[0].numpy(), (x[1][0].numpy(), x[1][1].numpy()))
      self.assertEqual(got, want)
      total += 1
    self.assertEqual(4, total)

  def testMapAndFilter(self):
    def even(x):
      return math_ops.equal(math_ops.mod(x, 2), 0)

    it = datasets.Iterator(Dataset.range(8).map(math_ops.square).filter(even))
    got = [x.numpy() for x in it]
    self.assertAllEqual([0, 4, 16, 36], got)

  def testMapCaptureLookupTable(self):
    default_val = -1
    keys = constant_op.constant(['brain', 'salad', 'surgery'])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup.HashTable(
        lookup.KeyValueTensorInitializer(keys, values), default_val)
    dataset = Dataset.from_tensor_slices(['brain', 'salad', 'surgery'])
    dataset = dataset.map(table.lookup)
    it = datasets.Iterator(dataset)
    got = [x.numpy() for x in it]
    self.assertAllEqual([0, 1, 2], got)

  def testMultipleIteratorsOnADatasetThatUsesFunctions(self):
    ds = Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6]).map(math_ops.square)

    got1 = [x.numpy() for x in datasets.Iterator(ds)]
    self.assertAllEqual([1, 4, 9, 16, 25, 36], got1)
    got2 = [x.numpy() for x in datasets.Iterator(ds)]
    self.assertAllEqual(got1, got2)

  def assertSparseValuesEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def testSparseTensorElements(self):
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 0], [1, 0], [2, 0]]),
        values=np.array([0, 0, 0]),
        dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    expected = [
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[0]]),
             values=np.array([1]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[1]]),
             values=np.array([2]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[2]]),
             values=np.array([3]),
             dense_shape=np.array([3]))),
    ]

    for i, result in enumerate(
        datasets.Iterator(Dataset.from_tensor_slices(components))):
      self.assertSparseValuesEqual(expected[i][0], result[0])
      self.assertSparseValuesEqual(expected[i][1], result[1])

  def testPyFunc(self):

    def my_map(inp):
      return [[x + 1 for x in inp]]

    ds = Dataset.range(4).map(
        lambda x: script_ops.py_func(my_map, [[x]], dtypes.int64))
    got = [x.numpy() for x in datasets.Iterator(ds)]
    self.assertAllEqual([[1], [2], [3], [4]], got)

  def testTensorsPlacedOnDevice(self):
    ds = Dataset.from_tensors([0., 1.])
    with ops.device(test.gpu_device_name()):
      x = datasets.Iterator(ds).next()
      x = math_ops.add(x, x)
    self.assertAllEqual([0., 2.], x.numpy())

  def testGpuTensor(self):
    ds = Dataset.from_tensors([0., 1.])
    with ops.device(test.gpu_device_name()):
      for x in ds:
        y = math_ops.add(x, x)
    self.assertAllEqual([0., 2.], y.numpy())

  def testOverrideThreadPool(self):

    def get_thread_id(_):
      # Python creates a dummy thread object to represent the current
      # thread when called from an "alien" thread (such as a
      # `PrivateThreadPool` thread in this case). It does not include
      # the TensorFlow-given display name, but it has a unique
      # identifier that maps one-to-one with the underlying OS thread.
      return np.array(threading.current_thread().ident).astype(np.int64)

    for num_threads in [1, 2, 4, 8, 16]:

      dataset = (
          Dataset.range(1000).map(
              lambda x: script_ops.py_func(get_thread_id, [x], dtypes.int64),
              num_parallel_calls=32).apply(unique.unique()))

      dataset = threadpool.override_threadpool(
          dataset,
          threadpool.PrivateThreadPool(
              num_threads, display_name='private_thread_pool_%d' % num_threads))

      thread_ids = []
      for next_element in datasets.Iterator(dataset):
        thread_ids.append(next_element)
      self.assertEqual(len(thread_ids), len(set(thread_ids)))
      self.assertGreater(len(thread_ids), 0)
      # NOTE(mrry): We don't control the thread pool scheduling, and
      # so cannot guarantee that all of the threads in the pool will
      # perform work.
      self.assertLessEqual(len(thread_ids), num_threads)

  def testSaveRestore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    dataset = Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset = dataset.map(math_ops.square).batch(2)
    iterator = datasets.Iterator(dataset)
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual([1, 4], iterator.get_next().numpy())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([9, 16], iterator.get_next().numpy())
    self.assertAllEqual([25, 36], iterator.get_next().numpy())
    checkpoint.restore(save_path)
    self.assertAllEqual([9, 16], iterator.get_next().numpy())
    self.assertAllEqual([25, 36], iterator.get_next().numpy())

  def testSaveRestoreMultipleIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    dataset = Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset = dataset.map(math_ops.square).batch(2)
    iterator_1 = datasets.Iterator(dataset)
    iterator_2 = datasets.Iterator(dataset)
    dataset_2 = Dataset.range(10)
    iterator_3 = datasets.Iterator(dataset_2)

    checkpoint = trackable_utils.Checkpoint(
        iterator_1=iterator_1, iterator_2=iterator_2, iterator_3=iterator_3)
    self.assertAllEqual([1, 4], iterator_1.get_next().numpy())
    self.assertEqual(0, iterator_3.get_next().numpy())
    self.assertEqual(1, iterator_3.get_next().numpy())
    self.assertEqual(2, iterator_3.get_next().numpy())

    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([1, 4], iterator_2.get_next().numpy())
    self.assertAllEqual([9, 16], iterator_2.get_next().numpy())
    self.assertEqual(3, iterator_3.get_next().numpy())
    checkpoint.restore(save_path)
    self.assertAllEqual([9, 16], iterator_1.get_next().numpy())
    self.assertAllEqual([1, 4], iterator_2.get_next().numpy())
    self.assertEqual(3, iterator_3.get_next().numpy())

  def testRestoreExhaustedIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    dataset = Dataset.range(3)
    iterator = datasets.Iterator(dataset)

    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertEqual(0, iterator.get_next().numpy())
    self.assertEqual(1, iterator.get_next().numpy())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertEqual(2, iterator.get_next().numpy())
    checkpoint.restore(save_path)
    self.assertEqual(2, iterator.get_next().numpy())

  def testRestoreInReconstructedIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    dataset = Dataset.range(10)
    for i in range(5):
      iterator = datasets.Iterator(dataset)
      checkpoint = trackable_utils.Checkpoint(iterator=iterator)
      checkpoint.restore(checkpoint_management.latest_checkpoint(
          checkpoint_directory))
      for j in range(2):
        self.assertEqual(i * 2 + j, iterator.get_next().numpy())
      checkpoint.save(file_prefix=checkpoint_prefix)


class DatasetConstructorBenchmark(test.Benchmark):

  def benchmarkSliceRepeatBatchEager(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        Dataset.from_tensor_slices(input_data).repeat(num_epochs)
        .batch(batch_size))
    iterator = datasets.Iterator(dataset)

    ends = [time.time()]
    for _ in iterator:
      ends.append(time.time())

    deltas = np.ediff1d(ends)
    median_wall_time = np.median(deltas)
    print(
        'Slice/repeat/batch eager input size: %d batch size: %d Median wall '
        'time per element: %f'
        % (input_size, batch_size, median_wall_time))
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name='benchmark_slice_repeat_batch_eager_input_%d_batch_%d' %
        (input_size, batch_size))

  def benchmarkSliceBatchCacheRepeatCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        Dataset.from_tensor_slices(input_data).batch(batch_size).cache()
        .repeat(num_epochs))
    iterator = datasets.Iterator(dataset)

    ends = [time.time()]
    for _ in iterator:
      ends.append(time.time())

    deltas = np.ediff1d(ends)
    median_wall_time = np.median(deltas)
    print(
        'Slice/batch/cache/repeat eager input size: %d batch size: %d Median '
        'wall time per element: %f'
        % (input_size, batch_size, median_wall_time))
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name='benchmark_slice_batch_cache_repeat_eager_input_%d_batch_%d' %
        (input_size, batch_size))


if __name__ == '__main__':
  test.main()
