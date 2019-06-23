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

import os
import time
import six

from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.variables import Variable
from tensorflow.python.training import server_lib

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


def assert_no_leak(f, num_iters=100000, increase_threshold_absolute_mb=10):
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


class MemoryTest(test.TestCase):

  def testMemoryLeakAnonymousVariable(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    def f():
      inputs = Variable(array_ops.zeros([32, 100], dtypes.float32))
      del inputs

    assert_no_leak(f, num_iters=10000)

  def testMemoryLeakInSimpleModelForwardOnly(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    inputs = array_ops.zeros([32, 100], dtypes.float32)
    net = SingleLayerNet()

    def f():
      with backprop.GradientTape():
        net(inputs)

    assert_no_leak(f)

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

    assert_no_leak(f)

  def testMemoryLeakInFunction(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    def f():

      @def_function.function
      def graph(x):
        return x * x + x

      graph(constant_op.constant(42))

    assert_no_leak(f, num_iters=1000, increase_threshold_absolute_mb=30)


class RemoteWorkerMemoryTest(test.TestCase):

  def __init__(self, method):
    super(RemoteWorkerMemoryTest, self).__init__(method)

    # used for remote worker tests
    os.environ["TF_EAGER_REMOTE_USE_SEND_TENSOR_RPC"] = "1"
    self._cached_server = server_lib.Server.create_local_server()
    self._cached_server_target = self._cached_server.target[len("grpc://"):]

  def testMemoryLeakInLocalCopy(self):
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    remote.connect_to_remote_host(self._cached_server_target)

    # Run a function locally with the input on a remote worker and ensure we
    # do not leak a reference to the remote tensor.

    @def_function.function
    def local_func(i):
      return i

    def func():
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        x = array_ops.zeros([1000, 1000], dtypes.int32)

      local_func(x)

    assert_no_leak(func, num_iters=100, increase_threshold_absolute_mb=50)


if __name__ == "__main__":
  test.main()
