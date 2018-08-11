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
"""Tests for prefetching_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compat import compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class PrefetchingKernelsOpsTest(test.TestCase):

  def setUp(self):
    self._event = threading.Event()

  def _create_ds_and_iterator(self, device0, initializable=False):

    def gen():
      for i in range(1, 10):
        yield [float(i)]
        if i == 6:
          self._event.set()

    with ops.device(device0):
      ds = dataset_ops.Dataset.from_generator(gen, (dtypes.float32))
      if initializable:
        ds_iterator = ds.make_initializable_iterator()
      else:
        ds_iterator = ds.make_one_shot_iterator()
      return (ds, ds_iterator)

  def _create_ops(self, ds, ds_iterator, buffer_name, device0, device1):
    ds_iterator_handle = ds_iterator.string_handle()

    @function.Defun(dtypes.string)
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, ds.output_types, ds.output_shapes)
      return remote_iterator.get_next()

    target = constant_op.constant(device0)
    with ops.device(device1):
      buffer_resource_handle = prefetching_ops.function_buffering_resource(
          f=_remote_fn,
          output_types=[dtypes.float32],
          target_device=target,
          string_arg=ds_iterator_handle,
          buffer_size=3,
          shared_name=buffer_name)

    with ops.device(device1):
      prefetch_op = prefetching_ops.function_buffering_resource_get_next(
          function_buffer_resource=buffer_resource_handle,
          output_types=[dtypes.float32])
      reset_op = prefetching_ops.function_buffering_resource_reset(
          function_buffer_resource=buffer_resource_handle)
      destroy_op = resource_variable_ops.destroy_resource_op(
          buffer_resource_handle, ignore_lookup_error=True)

    return (prefetch_op, reset_op, destroy_op)

  def _prefetch_fn_helper_one_shot(self, buffer_name, device0, device1):
    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})

    ds, ds_iterator = self._create_ds_and_iterator(device0, initializable=False)
    prefetch_op, _, destroy_op = self._create_ops(ds, ds_iterator, buffer_name,
                                                  device0, device1)

    with self.test_session(config=worker_config) as sess:
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [1.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [2.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [3.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [4.0])
      self._event.wait()
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [5.0])
      sess.run(destroy_op)

  def testSameDeviceCPU(self):
    self._prefetch_fn_helper_one_shot("same_device_cpu",
                                      "/job:localhost/replica:0/task:0/cpu:0",
                                      "/job:localhost/replica:0/task:0/cpu:0")

  def testDifferentDeviceCPU(self):
    self._prefetch_fn_helper_one_shot("diff_device_cpu",
                                      "/job:localhost/replica:0/task:0/cpu:0",
                                      "/job:localhost/replica:0/task:0/cpu:1")

  def testDifferentDeviceCPUGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    self._prefetch_fn_helper_one_shot("cpu_gpu",
                                      "/job:localhost/replica:0/task:0/cpu:0",
                                      "/job:localhost/replica:0/task:0/gpu:0")

  def testReinitialization(self):
    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})

    device0 = "/job:localhost/replica:0/task:0/cpu:0"
    device1 = "/job:localhost/replica:0/task:0/cpu:1"
    ds, ds_iterator = self._create_ds_and_iterator(device0, initializable=True)
    prefetch_op, reset_op, destroy_op = self._create_ops(
        ds, ds_iterator, "reinit", device0, device1)

    with self.test_session(config=worker_config) as sess:
      sess.run(ds_iterator.initializer)
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [1.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [2.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [3.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [4.0])
      self._event.wait()
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [5.0])
      # Lets reset the function buffering resource and reinitialize the
      # iterator. Should be able to go through this again.
      self._event.clear()
      sess.run(reset_op)
      sess.run(ds_iterator.initializer)
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [1.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [2.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [3.0])
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [4.0])
      self._event.wait()
      elem = sess.run(prefetch_op)
      self.assertEqual(elem, [5.0])
      sess.run(destroy_op)

  def testReinitializationOutOfRange(self):
    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})

    device0 = "/job:localhost/replica:0/task:0/cpu:0"
    device1 = "/job:localhost/replica:0/task:0/cpu:1"
    ds, ds_iterator = self._create_ds_and_iterator(device0, initializable=True)
    prefetch_op, reset_op, destroy_op = self._create_ops(
        ds, ds_iterator, "reinit", device0, device1)

    with self.test_session(config=worker_config) as sess:
      sess.run(ds_iterator.initializer)
      for i in range(1, 10):
        elem = sess.run(prefetch_op)
        self.assertEqual(elem, [float(i)])
      # Try fetching after its over twice to test out end of sequence.
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(prefetch_op)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(prefetch_op)

      # Now reset everything and try it out again.
      self._event.clear()
      sess.run(reset_op)
      sess.run(ds_iterator.initializer)
      for i in range(1, 10):
        elem = sess.run(prefetch_op)
        self.assertEqual(elem, [float(i)])
      # Try fetching after its over twice to test out end of sequence.
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(prefetch_op)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(prefetch_op)

      sess.run(destroy_op)

  def testStringsGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    device0 = "/job:localhost/replica:0/task:0/cpu:0"
    device1 = "/job:localhost/replica:0/task:0/gpu:0"

    ds = dataset_ops.Dataset.from_tensor_slices(["a", "b", "c"])
    ds_iterator = ds.make_one_shot_iterator()
    ds_iterator_handle = ds_iterator.string_handle()

    @function.Defun(dtypes.string)
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, ds.output_types, ds.output_shapes)
      return remote_iterator.get_next()

    target = constant_op.constant(device0)
    with ops.device(device1):
      buffer_resource_handle = prefetching_ops.function_buffering_resource(
          f=_remote_fn,
          output_types=[dtypes.string],
          target_device=target,
          string_arg=ds_iterator_handle,
          buffer_size=3,
          shared_name="strings")

    with ops.device(device1):
      prefetch_op = prefetching_ops.function_buffering_resource_get_next(
          function_buffer_resource=buffer_resource_handle,
          output_types=[dtypes.string])
      destroy_op = resource_variable_ops.destroy_resource_op(
          buffer_resource_handle, ignore_lookup_error=True)

    with self.test_session() as sess:
      self.assertEqual([b"a"], sess.run(prefetch_op))
      self.assertEqual([b"b"], sess.run(prefetch_op))
      self.assertEqual([b"c"], sess.run(prefetch_op))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(prefetch_op)

      sess.run(destroy_op)


class PrefetchToDeviceTest(test.TestCase):

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
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

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

    with self.test_session() as sess:
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

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
        self.assertEqual({"a": i}, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

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
        actual = sess.run(next_element)
        self.assertAllEqual([i], actual.values)
        self.assertAllEqual([[0, 0]], actual.indices)
        self.assertAllEqual([2, 2], actual.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testPrefetchToDeviceGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

    iterator = device_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

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
      sess.run(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, sess.run(next_element))
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testPrefetchToDeviceGpuWithReInit(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.prefetch_to_device("/gpu:0"))

    iterator = device_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, sess.run(next_element))
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)


class CopyToDeviceTest(test.TestCase):

  def testCopyToDevice(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceInt32(self):
    host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int32, next_element.dtype)
    self.assertEqual((4,), next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      self.assertAllEqual([0, 1, 2, 3], sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToSameDevice(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:0"))

    with ops.device("/cpu:0"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceWithPrefetch(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyDictToDevice(self):
    host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {"a": x})
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element["a"].dtype)
    self.assertEqual([], next_element["a"].shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual({"a": i}, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyDictToDeviceWithPrefetch(self):
    host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {"a": x})
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element["a"].dtype)
    self.assertEqual([], next_element["a"].shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        self.assertEqual({"a": i}, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopySparseTensorsToDevice(self):

    def make_tensor(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[2, 2])

    host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)

    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        actual = sess.run(next_element)
        self.assertAllEqual([i], actual.values)
        self.assertAllEqual([[0, 0]], actual.indices)
        self.assertAllEqual([2, 2], actual.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopySparseTensorsToDeviceWithPrefetch(self):

    def make_tensor(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[2, 2])

    host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)

    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      for i in range(10):
        actual = sess.run(next_element)
        self.assertAllEqual([i], actual.values)
        self.assertAllEqual([[0, 0]], actual.indices)
        self.assertAllEqual([2, 2], actual.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuWithPrefetch(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0")).prefetch(1)

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuInt32(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      self.assertAllEqual([0, 1, 2, 3], sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuInt32AndPrefetch(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0")).prefetch(1)

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      self.assertAllEqual([0, 1, 2, 3], sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuStrings(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors(["a", "b", "c"])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      self.assertAllEqual([b"a", b"b", b"c"], sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuStringsAndPrefetch(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.from_tensors(["a", "b", "c"])
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      self.assertAllEqual([b"a", b"b", b"c"], sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDevicePingPongCPUGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with compat.forward_compatibility_horizon(2018, 8, 4):
      host_dataset = dataset_ops.Dataset.range(10)
      device_dataset = host_dataset.apply(
          prefetching_ops.copy_to_device("/gpu:0", source_device="/cpu:0"))
      back_to_cpu_dataset = device_dataset.apply(
          prefetching_ops.copy_to_device("/cpu:0", source_device="/gpu:0"))

      with ops.device("/cpu:0"):
        iterator = back_to_cpu_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

      with self.test_session() as sess:
        sess.run(iterator.initializer)
        for i in range(10):
          self.assertEqual(i, sess.run(next_element))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(next_element)

  def testCopyToDeviceWithReInit(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1"))

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      sess.run(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, sess.run(next_element))
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceWithReInitAndPrefetch(self):
    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/cpu:1")).prefetch(1)

    with ops.device("/cpu:1"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    self.assertEqual(host_dataset.output_types, device_dataset.output_types)
    self.assertEqual(host_dataset.output_types, iterator.output_types)
    self.assertEqual(host_dataset.output_shapes, device_dataset.output_shapes)
    self.assertEqual(host_dataset.output_shapes, iterator.output_shapes)
    self.assertEqual(host_dataset.output_classes, device_dataset.output_classes)
    self.assertEqual(host_dataset.output_classes, iterator.output_classes)

    self.assertEqual(dtypes.int64, next_element.dtype)
    self.assertEqual([], next_element.shape)

    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=worker_config) as sess:
      sess.run(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, sess.run(next_element))
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuWithReInit(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, sess.run(next_element))
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testCopyToDeviceGpuWithReInitAndPrefetch(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0")).prefetch(1)

    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(5):
        self.assertEqual(i, sess.run(next_element))
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testIteratorGetNextAsOptionalOnGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(3)
    device_dataset = host_dataset.apply(
        prefetching_ops.copy_to_device("/gpu:0"))
    with ops.device("/gpu:0"):
      iterator = device_dataset.make_initializable_iterator()
      next_elem = iterator_ops.get_next_as_optional(iterator)
      elem_has_value_t = next_elem.has_value()
      elem_value_t = next_elem.get_value()

    with self.test_session() as sess:
      # Before initializing the iterator, evaluating the optional fails with
      # a FailedPreconditionError.
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(elem_has_value_t)
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(elem_value_t)

      # For each element of the dataset, assert that the optional evaluates to
      # the expected value.
      sess.run(iterator.initializer)
      for i in range(3):
        elem_has_value, elem_value = sess.run([elem_has_value_t, elem_value_t])
        self.assertTrue(elem_has_value)
        self.assertEqual(i, elem_value)

      # After exhausting the iterator, `next_elem.has_value()` will evaluate to
      # false, and attempting to get the value will fail.
      for _ in range(2):
        self.assertFalse(sess.run(elem_has_value_t))
        with self.assertRaises(errors.InvalidArgumentError):
          sess.run(elem_value_t)


class MultiDeviceIteratorTest(test.TestCase):

  def testBasic(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      sess.run(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        self.assertEqual(i, sess.run(elem_on_1))
        self.assertEqual(i + 1, sess.run(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elem_on_1)
        sess.run(elem_on_2)

  def testOneOnSameDevice(self):
    with ops.device("/cpu:0"):
      dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        dataset, ["/cpu:0", "/cpu:1"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()

    config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=config) as sess:
      sess.run(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        self.assertEqual(i, sess.run(elem_on_1))
        self.assertEqual(i + 1, sess.run(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elem_on_1)
        sess.run(elem_on_2)

  def testRepeatDevices(self):
    with ops.device("/cpu:0"):
      dataset = dataset_ops.Dataset.range(20)
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2", "/cpu:1", "/cpu:2"])
    elements = multi_device_iterator.get_next()
    elem_on_1, elem_on_2, elem_on_3, elem_on_4 = elements

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      sess.run(multi_device_iterator.initializer)
      for i in range(0, 20, 4):
        self.assertEqual(i, sess.run(elem_on_1))
        self.assertEqual(i + 1, sess.run(elem_on_2))
        self.assertEqual(i + 2, sess.run(elem_on_3))
        self.assertEqual(i + 3, sess.run(elem_on_4))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elem_on_1)
        sess.run(elem_on_2)
        sess.run(elem_on_3)
        sess.run(elem_on_4)

  def testNotFullyDivisible(self):
    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      sess.run(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        self.assertEqual(i, sess.run(elem_on_1))
        self.assertEqual(i + 1, sess.run(elem_on_2))
      self.assertEqual(8, sess.run(elem_on_1))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elem_on_1)
        sess.run(elem_on_2)

  def testUneven(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"], max_buffer_size=4)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      sess.run(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        self.assertEqual(i, sess.run(elem_on_1))
      for i in range(0, 10, 2):
        self.assertEqual(i + 1, sess.run(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elem_on_1)
        sess.run(elem_on_2)

  def testMultipleInitializations(self):
    with ops.device("/cpu:0"):
      epoch = array_ops.placeholder(dtypes.int64, shape=[])
      dataset1 = dataset_ops.Dataset.from_tensors(epoch).repeat(1000)
      dataset2 = dataset_ops.Dataset.range(1000)
      dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"], prefetch_buffer_size=4)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()
    init_op = multi_device_iterator.initializer

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      for i in range(1000):
        sess.run(init_op, feed_dict={epoch: i})
        self.assertEqual([(i, 0), (i, 1)], sess.run([elem_on_1, elem_on_2]))

  def testBasicGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with compat.forward_compatibility_horizon(2018, 8, 4):
      dataset = dataset_ops.Dataset.range(10)
      multi_device_iterator = prefetching_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/gpu:0"])
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()

      config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
      with self.test_session(config=config) as sess:
        sess.run(multi_device_iterator.initializer)
        for i in range(0, 10, 2):
          self.assertEqual(i, sess.run(elem_on_1))
          self.assertEqual(i + 1, sess.run(elem_on_2))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(elem_on_1)
          sess.run(elem_on_2)

  def testUnevenGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with compat.forward_compatibility_horizon(2018, 8, 4):
      dataset = dataset_ops.Dataset.range(10)
      multi_device_iterator = prefetching_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/gpu:0"], max_buffer_size=4)
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()

      config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
      with self.test_session(config=config) as sess:
        sess.run(multi_device_iterator.initializer)
        for i in range(0, 10, 2):
          self.assertEqual(i, sess.run(elem_on_1))
        for i in range(0, 10, 2):
          self.assertEqual(i + 1, sess.run(elem_on_2))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(elem_on_1)
          sess.run(elem_on_2)


if __name__ == "__main__":
  test.main()
