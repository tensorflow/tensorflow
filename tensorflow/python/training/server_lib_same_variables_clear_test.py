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
"""Tests for tf.GrpcServer."""

from tensorflow.python.client import session
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class SameVariablesClearTest(test.TestCase):

  # Verifies behavior of tf.Session.reset().
  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_deprecated_v1
  def testSameVariablesClear(self):
    server = server_lib.Server.create_local_server()

    # Creates a graph with 2 variables.
    v0 = variables.Variable([[2, 1]], name="v0")
    v1 = variables.Variable([[1], [2]], name="v1")
    v2 = math_ops.matmul(v0, v1)

    # Verifies that both sessions connecting to the same target return
    # the same results.
    sess_1 = session.Session(server.target)
    sess_2 = session.Session(server.target)
    sess_1.run(variables.global_variables_initializer())
    self.assertAllEqual([[4]], sess_1.run(v2))
    self.assertAllEqual([[4]], sess_2.run(v2))

    # Resets target. sessions abort. Use sess_2 to verify.
    session.Session.reset(server.target)
    with self.assertRaises(errors_impl.AbortedError):
      self.assertAllEqual([[4]], sess_2.run(v2))

    # Connects to the same target. Device memory for the variables would have
    # been released, so they will be uninitialized.
    sess_2 = session.Session(server.target)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      sess_2.run(v2)
    # Reinitializes the variables.
    sess_2.run(variables.global_variables_initializer())
    self.assertAllEqual([[4]], sess_2.run(v2))
    sess_2.close()


if __name__ == "__main__":
  test.main()
