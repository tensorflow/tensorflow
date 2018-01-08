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

import itertools
import threading

from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class StagingAreaOpsTest(test.TestCase):

  def setUp(self):
    self._event = threading.Event()

  def _prefetch_fn_helper(self, buffer_name, device0, device1):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2

    def gen():
      for i in itertools.count(start=1, step=1):
        yield [i + 0.0]
        if i == 6:
          self._event.set()

    with ops.device(device0):
      dataset_3 = dataset_ops.Dataset.from_generator(gen, (dtypes.float32))
      iterator_3 = dataset_3.make_one_shot_iterator()
      iterator_3_handle = iterator_3.string_handle()

    @function.Defun(dtypes.string)
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, dataset_3.output_types, dataset_3.output_shapes)
      return remote_iterator.get_next()

    target = constant_op.constant(device0)
    with ops.device(device1):
      buffer_resource_handle = prefetching_ops.function_buffering_resource(
          f=_remote_fn,
          target_device=target,
          string_arg=iterator_3_handle,
          buffer_size=3,
          thread_pool_size=2,
          shared_name=buffer_name)

    with ops.device(device1):
      prefetch_op = prefetching_ops.function_buffering_resource_get_next(
          function_buffer_resource=buffer_resource_handle,
          output_types=[dtypes.float32])

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
      sess.run(
          resource_variable_ops.destroy_resource_op(
              buffer_resource_handle, ignore_lookup_error=True))

  def testSameDeviceCPU(self):
    self._prefetch_fn_helper("same_device_cpu",
                             "/job:localhost/replica:0/task:0/cpu:0",
                             "/job:localhost/replica:0/task:0/cpu:0")

  def testDifferentDeviceCPU(self):
    self._prefetch_fn_helper("diff_device_cpu",
                             "/job:localhost/replica:0/task:0/cpu:0",
                             "/job:localhost/replica:0/task:0/cpu:1")

  def testDifferentDeviceCPUGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    self._prefetch_fn_helper("cpu_gpu", "/job:localhost/replica:0/task:0/cpu:0",
                             "/job:localhost/replica:0/task:0/gpu:0")


if __name__ == "__main__":
  test.main()
