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
"""Tests for memory leaks in eager execution.

It is possible that this test suite will eventually become flaky due to taking
too long to run (since the tests iterate many times), but for now they are
helpful for finding memory leaks since not all PyObject leaks are found by
introspection (test_util decorators). Please be careful adding new tests here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six

from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.variables import Variable

# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


class SingleLayerNet(keras.Model):
  """Simple keras model used to ensure that there are no leaks."""

  def __init__(self):
    super(SingleLayerNet, self).__init__()
    self.fc1 = keras.layers.Dense(5)

  def call(self, x):
    return self.fc1(x)


class MemoryTest(test.TestCase):

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

      assert increase < increase_threshold_absolute_mb, (
          "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
          "Maximum allowed increase: %f") % (initial, increase,
                                             increase_threshold_absolute_mb)

  def testMemoryLeakAnonymousVariable(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    def f():
      inputs = Variable(array_ops.zeros([32, 100], dtypes.float32))
      del inputs

    self.assertNotIncreasingMemory(f, num_iters=10000)

  def testMemoryLeakInSimpleModelForwardOnly(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    inputs = array_ops.zeros([32, 100], dtypes.float32)
    net = SingleLayerNet()

    def f():
      with backprop.GradientTape():
        net(inputs)

    self.assertNotIncreasingMemory(f)

  def testMemoryLeakInSimpleModelForwardAndBackward(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    inputs = array_ops.zeros([32, 100], dtypes.float32)
    net = SingleLayerNet()

    def f():
      with backprop.GradientTape() as tape:
        result = net(inputs)

      tape.gradient(result, net.variables)

      del tape

    self.assertNotIncreasingMemory(f)


if __name__ == "__main__":
  test.main()
