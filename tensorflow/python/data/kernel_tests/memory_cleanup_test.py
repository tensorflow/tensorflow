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
"""Verify that memory usage is minimal in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


@test_util.run_all_in_graph_and_eager_modes
class MemoryCleanupTest(test_base.DatasetTestBase):

  def assertNotIncreasingMemory(self,
                                f,
                                num_iters=100000,
                                increase_threshold_absolute_mb=10):
    """Assert memory usage doesn't increase beyond given threshold for f."""
    with context.eager_mode():
      # Warm up.
      f()
      # Wait for background threads to start up and take over memory.
      # FIXME: The nature of this test leaves few other options. Maybe there
      # is a better way to do this.
      time.sleep(4)
      initial = memory_profiler.memory_usage(-1)[0]
      for _ in six.moves.range(num_iters):
        f()
      increase = memory_profiler.memory_usage(-1)[0] - initial
      logging.info("Memory increase observed: %f MB" % increase)
      assert increase < increase_threshold_absolute_mb, (
          "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
          "Maximum allowed increase: %f") % (initial, increase,
                                             increase_threshold_absolute_mb)

  @test_util.run_v1_only("b/121264236")
  def testEagerMemoryUsageWithReset(self):
    if not context.executing_eagerly():
      self.skipTest("Only eager mode test")
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    def f():
      self.evaluate(multi_device_iterator.get_next())
      multi_device_iterator._eager_reset()

    self.assertNotIncreasingMemory(
        f, num_iters=100, increase_threshold_absolute_mb=350)

  @test_util.run_v1_only("b/121264236")
  def testEagerMemoryUsageWithRecreation(self):
    if not context.executing_eagerly():
      self.skipTest("Only eager mode test")
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    dataset = dataset_ops.Dataset.range(10)

    def f():
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/cpu:2"])
      self.evaluate(multi_device_iterator.get_next())
      del multi_device_iterator

    # TODO(b/123316347): Reduce threshold once bug is fixed.
    self.assertNotIncreasingMemory(
        f, num_iters=100, increase_threshold_absolute_mb=500)


if __name__ == "__main__":
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={"CPU": 3, "GPU": 1}))
  test.main()
