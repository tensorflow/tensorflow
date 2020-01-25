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

import gc
import time

from absl.testing import parameterized
import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_like
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


class MemoryCleanupTest(test_base.DatasetTestBase, parameterized.TestCase):

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

  # TODO(b/121264236): Support v2 behavior
  @combinations.generate(combinations.combine(tf_api_version=1, mode="eager"))
  def testEagerMemoryUsageWithReset(self):
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

  # TODO(b/121264236): Support v2 behavior
  @combinations.generate(
      combinations.combine(tf_api_version=1, mode="eager"))
  def testEagerMemoryUsageWithRecreation(self):
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

  def _testIteratorMemoryLeak(self, get_dataset):

    def run():
      get_next = self.getNext(get_dataset())
      for _ in range(100):
        self.evaluate(get_next())

    for _ in range(10):
      run()

    gc.collect()
    tensors = [
        o for o in gc.get_objects() if isinstance(o, tensor_like._TensorLike)
    ]
    self.assertEmpty(tensors, "%d Tensors are still alive." % len(tensors))

  @combinations.generate(test_base.eager_only_combinations())
  def testFilter(self):

    def get_dataset():

      def fn(_):
        return True

      return dataset_ops.Dataset.range(0, 100).filter(fn)

    self._testIteratorMemoryLeak(get_dataset)

  @combinations.generate(combinations.combine(tf_api_version=1, mode="eager"))
  def testFilterLegacy(self):

    def get_dataset():

      def fn(_):
        return True

      return dataset_ops.Dataset.range(0, 100).filter_with_legacy_function(fn)

    self._testIteratorMemoryLeak(get_dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testFlatMap(self):

    def get_dataset():

      def fn(x):
        return dataset_ops.Dataset.from_tensors(x * x)

      return dataset_ops.Dataset.range(0, 100).flat_map(fn)

    self._testIteratorMemoryLeak(get_dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testFromGenerator(self):

    def get_dataset():

      def fn():
        return six.moves.range(100)

      return dataset_ops.Dataset.from_generator(fn, output_types=dtypes.float32)

    self._testIteratorMemoryLeak(get_dataset)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_parallel_calls=[None, 10])))
  def testMap(self, num_parallel_calls):

    def get_dataset():

      def fn(x):
        return x * x

      return dataset_ops.Dataset.range(0, 100).map(
          fn, num_parallel_calls=num_parallel_calls)

    self._testIteratorMemoryLeak(get_dataset)

  @combinations.generate(
      combinations.combine(
          tf_api_version=1, mode="eager", num_parallel_calls=[None, 10]))
  def testMapLegacy(self, num_parallel_calls):

    def get_dataset():

      def fn(x):
        return x * x

      return dataset_ops.Dataset.range(0, 100).map_with_legacy_function(
          fn, num_parallel_calls=num_parallel_calls)

    self._testIteratorMemoryLeak(get_dataset)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(num_parallel_calls=[None, 10])))
  def testInterleave(self, num_parallel_calls):

    def get_dataset():

      def fn(x):
        return dataset_ops.Dataset.from_tensors(x * x)

      return dataset_ops.Dataset.range(0, 100).interleave(
          fn, num_parallel_calls=num_parallel_calls, cycle_length=10)

    self._testIteratorMemoryLeak(get_dataset)


if __name__ == "__main__":
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={"CPU": 3, "GPU": 1}))
  test.main()
