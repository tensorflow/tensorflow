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
"""Tests for `tf.data.experimental.prefetch_to_device()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class PrefetchToDeviceTest(test_base.DatasetTestBase):

  def testPrefetchToDevice(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    # NOTE(mrry): This device block creates the "host" dataset and iterator on
    # /cpu:0, and ensures that the prefetching is across devices. In typical use
    # this would not be necessary, because the GPU device would not support any
    # of the dataset-related ops.
    with ops.device("/cpu:0"):
      iterator = device_dataset.make_one_shot_iterator()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    next_element = iterator.get_next()
    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testPrefetchToSameDevice(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device(
            "/job:localhost/replica:0/task:0/device:CPU:0"))

    # NOTE(mrry): This device block creates the "host" dataset and iterator on
    # /cpu:0, and ensures that the prefetching is across devices. In typical use
    # this would not be necessary, because the GPU device would not support any
    # of the dataset-related ops.
    with ops.device("/cpu:0"):
      iterator = device_dataset.make_one_shot_iterator()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    next_element = iterator.get_next()
    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testPrefetchDictToDevice(self):
    host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {"a": x})
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    # NOTE(mrry): This device block creates the "host" dataset and iterator on
    # /cpu:0, and ensures that the prefetching is across devices. In typical use
    # this would not be necessary, because the GPU device would not support any
    # of the dataset-related ops.
    with ops.device("/cpu:0"):
      iterator = device_dataset.make_one_shot_iterator()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    next_element = iterator.get_next()
    self.assertEqual(dtypes.int64, next_element["a"].dtype)
    self.assertEqual([], next_element["a"].shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual({"a": i}, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testPrefetchSparseTensorsToDevice(self):
    def make_tensor(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i*[1]), dense_shape=[2, 2])
    host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)

    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    # NOTE(mrry): This device block creates the "host" dataset and iterator on
    # /cpu:0, and ensures that the prefetching is across devices. In typical use
    # this would not be necessary, because the GPU device would not support any
    # of the dataset-related ops.
    with ops.device("/cpu:0"):
      iterator = device_dataset.make_one_shot_iterator()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    next_element = iterator.get_next()
    self.assertEqual(dtypes.int64, next_element.dtype)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        actual = self.evaluate(next_element)
        self.assertAllEqual([i], actual.values)
        self.assertAllEqual([[0, 0]], actual.indices)
        self.assertAllEqual([2, 2], actual.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testPrefetchToDeviceGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

    iterator = device_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testPrefetchToDeviceWithReInit(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/cpu:1"))

    # NOTE(mrry): This device block creates the "host" dataset and iterator on
    # /cpu:0, and ensures that the prefetching is across devices. In typical use
    # this would not be necessary, because the GPU device would not support any
    # of the dataset-related ops.
    with ops.device("/cpu:0"):
      iterator = device_dataset.make_initializable_iterator()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    next_element = iterator.get_next()
    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      self.evaluate(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, self.evaluate(next_element))
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testPrefetchToDeviceGpuWithReInit(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

    iterator = device_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, self.evaluate(next_element))
      self.evaluate(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)


if __name__ == "__main__":
  test.main()
