# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import time

import numpy as np
import tensorflow as tf


class GrpcServerTest(tf.test.TestCase):

  def testRunStep(self):
    server = tf.train.Server.create_local_server()

    with tf.Session(server.target) as sess:
      c = tf.constant([[2, 1]])
      d = tf.constant([[1], [2]])
      e = tf.matmul(c, d)
      self.assertAllEqual([[4]], sess.run(e))
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  def testMultipleSessions(self):
    server = tf.train.Server.create_local_server()

    c = tf.constant([[2, 1]])
    d = tf.constant([[1], [2]])
    e = tf.matmul(c, d)

    sess_1 = tf.Session(server.target)
    sess_2 = tf.Session(server.target)

    self.assertAllEqual([[4]], sess_1.run(e))
    self.assertAllEqual([[4]], sess_2.run(e))

    sess_1.close()
    sess_2.close()
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  # Verifies behavior of multiple variables with multiple sessions connecting to
  # the same server.
  def testSameVariablesNoClear(self):
    server = tf.train.Server.create_local_server()

    with tf.Session(server.target) as sess_1:
      v0 = tf.Variable([[2, 1]], name="v0")
      v1 = tf.Variable([[1], [2]], name="v1")
      v2 = tf.matmul(v0, v1)
      sess_1.run([v0.initializer, v1.initializer])
      self.assertAllEqual([[4]], sess_1.run(v2))

    with tf.Session(server.target) as sess_2:
      new_v0 = tf.get_default_graph().get_tensor_by_name("v0:0")
      new_v1 = tf.get_default_graph().get_tensor_by_name("v1:0")
      new_v2 = tf.matmul(new_v0, new_v1)
      self.assertAllEqual([[4]], sess_2.run(new_v2))

  # Verifies behavior of tf.Session.reset().
  def testSameVariablesClear(self):
    server = tf.train.Server.create_local_server()

    # Creates a graph with 2 variables.
    v0 = tf.Variable([[2, 1]], name="v0")
    v1 = tf.Variable([[1], [2]], name="v1")
    v2 = tf.matmul(v0, v1)

    # Verifies that both sessions connecting to the same target return
    # the same results.
    sess_1 = tf.Session(server.target)
    sess_2 = tf.Session(server.target)
    sess_1.run(tf.initialize_all_variables())
    self.assertAllEqual([[4]], sess_1.run(v2))
    self.assertAllEqual([[4]], sess_2.run(v2))

    # Resets target. sessions abort. Use sess_2 to verify.
    tf.Session.reset(server.target)
    with self.assertRaises(tf.errors.AbortedError):
      self.assertAllEqual([[4]], sess_2.run(v2))

    # Connects to the same target. Device memory for the variables would have
    # been released, so they will be unitialized.
    sess_2 = tf.Session(server.target)
    with self.assertRaises(tf.errors.FailedPreconditionError):
      sess_2.run(v2)
    # Reinitialzes the variables.
    sess_2.run(tf.initialize_all_variables())
    self.assertAllEqual([[4]], sess_2.run(v2))
    sess_2.close()

  # Verifies behavior of tf.Session.reset() with multiple containers using
  # default container names as defined by the target name.
  def testSameVariablesClearContainer(self):
    # Starts two servers with different names so they map to different
    # resource "containers".
    server0 = tf.train.Server({"local0": ["localhost:0"]}, protocol="grpc",
                              start=True)
    server1 = tf.train.Server({"local1": ["localhost:0"]}, protocol="grpc",
                              start=True)

    # Creates a graph with 2 variables.
    v0 = tf.Variable(1.0, name="v0")
    v1 = tf.Variable(2.0, name="v0")

    # Initializes the variables. Verifies that the values are correct.
    sess_0 = tf.Session(server0.target)
    sess_1 = tf.Session(server1.target)
    sess_0.run(v0.initializer)
    sess_1.run(v1.initializer)
    self.assertAllEqual(1.0, sess_0.run(v0))
    self.assertAllEqual(2.0, sess_1.run(v1))

    # Resets container "local0". Verifies that v0 is no longer initialized.
    tf.Session.reset(server0.target, ["local0"])
    sess = tf.Session(server0.target)
    with self.assertRaises(tf.errors.FailedPreconditionError):
      sess.run(v0)
    # Reinitializes v0 for the following test.
    sess.run(v0.initializer)

    # Verifies that v1 is still valid.
    self.assertAllEqual(2.0, sess_1.run(v1))

    # Resets container "local1". Verifies that v1 is no longer initialized.
    tf.Session.reset(server1.target, ["local1"])
    sess = tf.Session(server1.target)
    with self.assertRaises(tf.errors.FailedPreconditionError):
      sess.run(v1)
    # Verifies that v0 is still valid.
    sess = tf.Session(server0.target)
    self.assertAllEqual(1.0, sess.run(v0))

  # Verifies behavior of tf.Session.reset() with multiple containers using
  # tf.container.
  def testMultipleContainers(self):
    with tf.container("test0"):
      v0 = tf.Variable(1.0, name="v0")
    with tf.container("test1"):
      v1 = tf.Variable(2.0, name="v0")
    server = tf.train.Server.create_local_server()
    sess = tf.Session(server.target)
    sess.run(tf.initialize_all_variables())
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))

    # Resets container. Session aborts.
    tf.Session.reset(server.target, ["test0"])
    with self.assertRaises(tf.errors.AbortedError):
      sess.run(v1)

    # Connects to the same target. Device memory for the v0 would have
    # been released, so it will be unitialized. But v1 should still
    # be valid.
    sess = tf.Session(server.target)
    with self.assertRaises(tf.errors.FailedPreconditionError):
      sess.run(v0)
    self.assertAllEqual(2.0, sess.run(v1))

  # Verifies various reset failures.
  def testResetFails(self):
    # Creates variable with container name.
    with tf.container("test0"):
      v0 = tf.Variable(1.0, name="v0")
    # Creates variable with default container.
    v1 = tf.Variable(2.0, name="v1")
    # Verifies resetting the non-existent target returns error.
    with self.assertRaises(tf.errors.NotFoundError):
      tf.Session.reset("nonexistent", ["test0"])

    # Verifies resetting with config.
    # Verifies that resetting target with no server times out.
    with self.assertRaises(tf.errors.DeadlineExceededError):
      tf.Session.reset("grpc://localhost:0", ["test0"],
                       config=tf.ConfigProto(operation_timeout_in_ms=5))

    # Verifies no containers are reset with non-existent container.
    server = tf.train.Server.create_local_server()
    sess = tf.Session(server.target)
    sess.run(tf.initialize_all_variables())
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))
    # No container is reset, but the server is reset.
    tf.Session.reset(server.target, ["test1"])
    # Verifies that both variables are still valid.
    sess = tf.Session(server.target)
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))

  def testLargeConstant(self):
    server = tf.train.Server.create_local_server()
    with tf.Session(server.target) as sess:
      const_val = np.empty([10000, 3000], dtype=np.float32)
      const_val.fill(0.5)
      c = tf.constant(const_val)
      shape_t = tf.shape(c)
      self.assertAllEqual([10000, 3000], sess.run(shape_t))

  def testLargeFetch(self):
    server = tf.train.Server.create_local_server()
    with tf.Session(server.target) as sess:
      c = tf.fill([10000, 3000], 0.5)
      expected_val = np.empty([10000, 3000], dtype=np.float32)
      expected_val.fill(0.5)
      self.assertAllEqual(expected_val, sess.run(c))

  def testLargeFeed(self):
    server = tf.train.Server.create_local_server()
    with tf.Session(server.target) as sess:
      feed_val = np.empty([10000, 3000], dtype=np.float32)
      feed_val.fill(0.5)
      p = tf.placeholder(tf.float32, shape=[10000, 3000])
      min_t = tf.reduce_min(p)
      max_t = tf.reduce_max(p)
      min_val, max_val = sess.run([min_t, max_t], feed_dict={p: feed_val})
      self.assertEqual(0.5, min_val)
      self.assertEqual(0.5, max_val)

  def testCloseCancelsBlockingOperation(self):
    server = tf.train.Server.create_local_server()
    sess = tf.Session(server.target)

    q = tf.FIFOQueue(10, [tf.float32])
    enqueue_op = q.enqueue(37.0)
    dequeue_t = q.dequeue()

    sess.run(enqueue_op)
    sess.run(dequeue_t)

    def blocking_dequeue():
      with self.assertRaises(tf.errors.CancelledError):
        sess.run(dequeue_t)

    blocking_thread = self.checkedThread(blocking_dequeue)
    blocking_thread.start()
    time.sleep(0.5)
    sess.close()
    blocking_thread.join()

  def testSetConfiguration(self):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))

    # Configure a server using the default local server options.
    server = tf.train.Server.create_local_server(config=config, start=False)
    self.assertEqual(
        0.1,
        server.server_def.default_session_config
        .gpu_options.per_process_gpu_memory_fraction)

    # Configure a server using an explicit ServerDefd with an
    # overridden config.
    cluster_def = tf.train.ClusterSpec(
        {"localhost": ["localhost:0"]}).as_cluster_def()
    server_def = tf.train.ServerDef(
        cluster=cluster_def, job_name="localhost", task_index=0,
        protocol="grpc")
    server = tf.train.Server(server_def, config=config, start=False)
    self.assertEqual(
        0.1,
        server.server_def.default_session_config
        .gpu_options.per_process_gpu_memory_fraction)

  def testInvalidHostname(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "port"):
      _ = tf.train.Server({"local": ["localhost"]},
                          job_name="local",
                          task_index=0)

  def testInteractiveSession(self):
    server = tf.train.Server.create_local_server()
    # TODO(b/29900832): Remove this assertion when the bug is fixed.
    a = tf.constant(1.0)
    with self.assertRaisesRegexp(tf.errors.UnimplementedError, "pruned"):
      sess = tf.InteractiveSession(target=server.target)
      sess.run(a)

    # TODO(b/29900832): The following code fails (without the unimplemented
    # check in `tensorflow::MasterSession`):
    # a = tf.constant(1.0)
    # b = tf.constant(2.0)
    # self.assertEqual(1.0, sess.run(a))
    # self.assertEqual(2.0, sess.run(b))


class ServerDefTest(tf.test.TestCase):

  def testLocalServer(self):
    cluster_def = tf.train.ClusterSpec(
        {"local": ["localhost:2222"]}).as_cluster_def()
    server_def = tf.train.ServerDef(
        cluster=cluster_def, job_name="local", task_index=0, protocol="grpc")

    self.assertProtoEquals("""
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }
    }
    job_name: 'local' task_index: 0 protocol: 'grpc'
    """, server_def)

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = tf.train.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testTwoProcesses(self):
    cluster_def = tf.train.ClusterSpec(
        {"local": ["localhost:2222", "localhost:2223"]}).as_cluster_def()
    server_def = tf.train.ServerDef(
        cluster=cluster_def, job_name="local", task_index=1, protocol="grpc")

    self.assertProtoEquals("""
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
                          tasks { key: 1 value: 'localhost:2223' } }
    }
    job_name: 'local' task_index: 1 protocol: 'grpc'
    """, server_def)

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = tf.train.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testTwoJobs(self):
    cluster_def = tf.train.ClusterSpec(
        {"ps": ["ps0:2222", "ps1:2222"],
         "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
    ).as_cluster_def()
    server_def = tf.train.ServerDef(
        cluster=cluster_def, job_name="worker", task_index=2, protocol="grpc")

    self.assertProtoEquals("""
    cluster {
      job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                       tasks { key: 1 value: 'ps1:2222' } }
      job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                           tasks { key: 1 value: 'worker1:2222' }
                           tasks { key: 2 value: 'worker2:2222' } }
    }
    job_name: 'worker' task_index: 2 protocol: 'grpc'
    """, server_def)

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = tf.train.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testClusterSpec(self):
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ["ps0:2222", "ps1:2222"],
         "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]})

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """

    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, tf.train.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        tf.train.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        tf.train.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())


if __name__ == "__main__":
  tf.test.main()
