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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class SameVariablesClearContainerTest(test.TestCase):

  # Verifies behavior of tf.Session.reset() with multiple containers using
  # default container names as defined by the target name.
  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_deprecated_v1
  def testSameVariablesClearContainer(self):
    # Starts two servers with different names so they map to different
    # resource "containers".
    server0 = server_lib.Server(
        {
            "local0": ["localhost:0"]
        }, protocol="grpc", start=True)
    server1 = server_lib.Server(
        {
            "local1": ["localhost:0"]
        }, protocol="grpc", start=True)

    # Creates a graph with 2 variables.
    v0 = variables.Variable(1.0, name="v0")
    v1 = variables.Variable(2.0, name="v0")

    # Initializes the variables. Verifies that the values are correct.
    sess_0 = session.Session(server0.target)
    sess_1 = session.Session(server1.target)
    sess_0.run(v0.initializer)
    sess_1.run(v1.initializer)
    self.assertAllEqual(1.0, sess_0.run(v0))
    self.assertAllEqual(2.0, sess_1.run(v1))

    # Resets container "local0". Verifies that v0 is no longer initialized.
    session.Session.reset(server0.target, ["local0"])
    sess = session.Session(server0.target)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      self.evaluate(v0)
    # Reinitializes v0 for the following test.
    self.evaluate(v0.initializer)

    # Verifies that v1 is still valid.
    self.assertAllEqual(2.0, sess_1.run(v1))

    # Resets container "local1". Verifies that v1 is no longer initialized.
    session.Session.reset(server1.target, ["local1"])
    sess = session.Session(server1.target)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      self.evaluate(v1)
    # Verifies that v0 is still valid.
    sess = session.Session(server0.target)
    self.assertAllEqual(1.0, self.evaluate(v0))


if __name__ == "__main__":
  test.main()
