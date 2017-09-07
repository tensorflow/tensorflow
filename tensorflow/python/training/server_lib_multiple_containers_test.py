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
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class MultipleContainersTest(test.TestCase):

  # Verifies behavior of tf.Session.reset() with multiple containers using
  # tf.container.
  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  def testMultipleContainers(self):
    with ops.container("test0"):
      v0 = variables.Variable(1.0, name="v0")
    with ops.container("test1"):
      v1 = variables.Variable(2.0, name="v0")
    server = server_lib.Server.create_local_server()
    sess = session.Session(server.target)
    sess.run(variables.global_variables_initializer())
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))

    # Resets container. Session aborts.
    session.Session.reset(server.target, ["test0"])
    with self.assertRaises(errors_impl.AbortedError):
      sess.run(v1)

    # Connects to the same target. Device memory for the v0 would have
    # been released, so it will be uninitialized. But v1 should still
    # be valid.
    sess = session.Session(server.target)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      sess.run(v0)
    self.assertAllEqual(2.0, sess.run(v1))


if __name__ == "__main__":
  test.main()
