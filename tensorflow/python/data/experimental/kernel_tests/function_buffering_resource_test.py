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
"""Tests for the private `FunctionBufferingResource` used in prefetching."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class FunctionBufferingResourceTest(test_base.DatasetTestBase):

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

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, ds.output_types, ds.output_shapes)
      return remote_iterator.get_next()

    target = constant_op.constant(device0)
    with ops.device(device1):
      buffer_resource_handle = prefetching_ops.function_buffering_resource(
          f=_remote_fn.get_concrete_function(),
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
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [1.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [2.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [3.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [4.0])
      self._event.wait()
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [5.0])
      self.evaluate(destroy_op)

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
      self.evaluate(ds_iterator.initializer)
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [1.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [2.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [3.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [4.0])
      self._event.wait()
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [5.0])
      # Lets reset the function buffering resource and reinitialize the
      # iterator. Should be able to go through this again.
      self._event.clear()
      self.evaluate(reset_op)
      self.evaluate(ds_iterator.initializer)
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [1.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [2.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [3.0])
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [4.0])
      self._event.wait()
      elem = self.evaluate(prefetch_op)
      self.assertEqual(elem, [5.0])
      self.evaluate(destroy_op)

  def testReinitializationOutOfRange(self):
    worker_config = config_pb2.ConfigProto(device_count={"CPU": 2})

    device0 = "/job:localhost/replica:0/task:0/cpu:0"
    device1 = "/job:localhost/replica:0/task:0/cpu:1"
    ds, ds_iterator = self._create_ds_and_iterator(device0, initializable=True)
    prefetch_op, reset_op, destroy_op = self._create_ops(
        ds, ds_iterator, "reinit", device0, device1)

    with self.test_session(config=worker_config) as sess:
      self.evaluate(ds_iterator.initializer)
      for i in range(1, 10):
        elem = self.evaluate(prefetch_op)
        self.assertEqual(elem, [float(i)])
      # Try fetching after its over twice to test out end of sequence.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(prefetch_op)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(prefetch_op)

      # Now reset everything and try it out again.
      self._event.clear()
      self.evaluate(reset_op)
      self.evaluate(ds_iterator.initializer)
      for i in range(1, 10):
        elem = self.evaluate(prefetch_op)
        self.assertEqual(elem, [float(i)])
      # Try fetching after its over twice to test out end of sequence.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(prefetch_op)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(prefetch_op)

      self.evaluate(destroy_op)

  def testStringsGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    device0 = "/job:localhost/replica:0/task:0/cpu:0"
    device1 = "/job:localhost/replica:0/task:0/gpu:0"

    ds = dataset_ops.Dataset.from_tensor_slices(["a", "b", "c"])
    ds_iterator = ds.make_one_shot_iterator()
    ds_iterator_handle = ds_iterator.string_handle()

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, ds.output_types, ds.output_shapes)
      return remote_iterator.get_next()

    target = constant_op.constant(device0)
    with ops.device(device1):
      buffer_resource_handle = prefetching_ops.function_buffering_resource(
          f=_remote_fn.get_concrete_function(),
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

    with self.cached_session() as sess:
      self.assertEqual([b"a"], self.evaluate(prefetch_op))
      self.assertEqual([b"b"], self.evaluate(prefetch_op))
      self.assertEqual([b"c"], self.evaluate(prefetch_op))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(prefetch_op)

      self.evaluate(destroy_op)


if __name__ == "__main__":
  test.main()
