# Copyright 2016 Google Inc. All Rights Reserved.
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

import tensorflow as tf


class GrpcServerTest(tf.test.TestCase):

  def _localServer(self):
    server_def = tf.ServerDef(protocol="grpc")
    job_def = server_def.cluster.job.add()
    job_def.name = "local"
    job_def.tasks[0] = "localhost:0"
    server_def.job_name = job_def.name
    server_def.task_index = 0
    return server_def

  def testRunStep(self):
    server = tf.GrpcServer(self._localServer())
    server.start()

    with tf.Session(server.target) as sess:
      c = tf.constant([[2, 1]])
      d = tf.constant([[1], [2]])
      e = tf.matmul(c, d)
      print(sess.run(e))
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  def testMultipleSessions(self):
    server = tf.GrpcServer(self._localServer())
    server.start()

    c = tf.constant([[2, 1]])
    d = tf.constant([[1], [2]])
    e = tf.matmul(c, d)

    sess_1 = tf.Session(server.target)
    sess_2 = tf.Session(server.target)

    sess_1.run(e)
    sess_2.run(e)

    sess_1.close()
    sess_2.close()
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.


if __name__ == "__main__":
  tf.test.main()
