# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for memory leaks in remote eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.eager.memory_tests import memory_test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import server_lib


class RemoteWorkerMemoryTest(test.TestCase):

  def __init__(self, method):
    super(RemoteWorkerMemoryTest, self).__init__(method)

    # used for remote worker tests
    self._cached_server = server_lib.Server.create_local_server()
    self._cached_server_target = self._cached_server.target[len("grpc://"):]

  def testMemoryLeakInLocalCopy(self):
    if not memory_test_util.memory_profiler_is_available():
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

    memory_test_util.assert_no_leak(
        func, num_iters=100, increase_threshold_absolute_mb=50)


if __name__ == "__main__":
  test.main()
