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

import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import server_lib


class GrpcServerTest(test.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(GrpcServerTest, self).__init__(methodName)
    self._cached_server = server_lib.Server.create_local_server()

  def testRunStep(self):
    server = self._cached_server

    with session.Session(server.target) as sess:
      c = constant_op.constant([[2, 1]])
      d = constant_op.constant([[1], [2]])
      e = math_ops.matmul(c, d)
      self.assertAllEqual([[4]], sess.run(e))
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  def testMultipleSessions(self):
    server = self._cached_server

    c = constant_op.constant([[2, 1]])
    d = constant_op.constant([[1], [2]])
    e = math_ops.matmul(c, d)

    sess_1 = session.Session(server.target)
    sess_2 = session.Session(server.target)

    self.assertAllEqual([[4]], sess_1.run(e))
    self.assertAllEqual([[4]], sess_2.run(e))

    sess_1.close()
    sess_2.close()
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  # Verifies various reset failures.
  def testResetFails(self):
    # Creates variable with container name.
    with ops.container("test0"):
      v0 = variables.Variable(1.0, name="v0")
    # Creates variable with default container.
    v1 = variables.Variable(2.0, name="v1")
    # Verifies resetting the non-existent target returns error.
    with self.assertRaises(errors_impl.NotFoundError):
      session.Session.reset("nonexistent", ["test0"])

    # Verifies resetting with config.
    # Verifies that resetting target with no server times out.
    with self.assertRaises(errors_impl.DeadlineExceededError):
      session.Session.reset(
          "grpc://localhost:0", ["test0"],
          config=config_pb2.ConfigProto(operation_timeout_in_ms=5))

    # Verifies no containers are reset with non-existent container.
    server = self._cached_server
    sess = session.Session(server.target)
    sess.run(variables.global_variables_initializer())
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))
    # No container is reset, but the server is reset.
    session.Session.reset(server.target, ["test1"])
    # Verifies that both variables are still valid.
    sess = session.Session(server.target)
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))

  def _useRPCConfig(self):
    """Return a `tf.ConfigProto` that ensures we use the RPC stack for tests.

    This configuration ensures that we continue to exercise the gRPC
    stack when testing, rather than using the in-process optimization,
    which avoids using gRPC as the transport between a client and
    master in the same process.

    Returns:
      A `tf.ConfigProto`.
    """
    return config_pb2.ConfigProto(rpc_options=config_pb2.RPCOptions(
        use_rpc_for_inprocess_master=True))

  def testLargeConstant(self):
    server = self._cached_server
    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      const_val = np.empty([10000, 3000], dtype=np.float32)
      const_val.fill(0.5)
      c = constant_op.constant(const_val)
      shape_t = array_ops.shape(c)
      self.assertAllEqual([10000, 3000], sess.run(shape_t))

  def testLargeFetch(self):
    server = self._cached_server
    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      c = array_ops.fill([10000, 3000], 0.5)
      expected_val = np.empty([10000, 3000], dtype=np.float32)
      expected_val.fill(0.5)
      self.assertAllEqual(expected_val, sess.run(c))

  def testLargeFeed(self):
    server = self._cached_server
    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      feed_val = np.empty([10000, 3000], dtype=np.float32)
      feed_val.fill(0.5)
      p = array_ops.placeholder(dtypes.float32, shape=[10000, 3000])
      min_t = math_ops.reduce_min(p)
      max_t = math_ops.reduce_max(p)
      min_val, max_val = sess.run([min_t, max_t], feed_dict={p: feed_val})
      self.assertEqual(0.5, min_val)
      self.assertEqual(0.5, max_val)

  def testCloseCancelsBlockingOperation(self):
    server = self._cached_server
    sess = session.Session(server.target, config=self._useRPCConfig())

    q = data_flow_ops.FIFOQueue(10, [dtypes.float32])
    enqueue_op = q.enqueue(37.0)
    dequeue_t = q.dequeue()

    sess.run(enqueue_op)
    sess.run(dequeue_t)

    def blocking_dequeue():
      with self.assertRaisesRegexp(errors_impl.CancelledError,
                                   "Session::Close"):
        sess.run(dequeue_t)

    blocking_thread = self.checkedThread(blocking_dequeue)
    blocking_thread.start()
    time.sleep(0.5)
    sess.close()
    blocking_thread.join()

  def testInteractiveSession(self):
    server = self._cached_server
    # Session creation will warn (in C++) that the place_pruned_graph option
    # is not supported, but it should successfully ignore it.
    sess = session.InteractiveSession(server.target)
    c = constant_op.constant(42.0)
    self.assertEqual(42.0, c.eval())
    sess.close()

  def testSetConfiguration(self):
    config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.1))

    # Configure a server using the default local server options.
    server = server_lib.Server.create_local_server(config=config, start=False)
    self.assertEqual(0.1, server.server_def.default_session_config.gpu_options.
                     per_process_gpu_memory_fraction)

    # Configure a server using an explicit ServerDefd with an
    # overridden config.
    cluster_def = server_lib.ClusterSpec({
        "localhost": ["localhost:0"]
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def,
        job_name="localhost",
        task_index=0,
        protocol="grpc")
    server = server_lib.Server(server_def, config=config, start=False)
    self.assertEqual(0.1, server.server_def.default_session_config.gpu_options.
                     per_process_gpu_memory_fraction)

  def testInvalidHostname(self):
    with self.assertRaisesRegexp(errors_impl.InvalidArgumentError, "port"):
      _ = server_lib.Server(
          {
              "local": ["localhost"]
          }, job_name="local", task_index=0)

  def testTimeoutRaisesException(self):
    server = self._cached_server
    q = data_flow_ops.FIFOQueue(1, [dtypes.float32])
    blocking_t = q.dequeue()

    with session.Session(server.target) as sess:
      with self.assertRaises(errors_impl.DeadlineExceededError):
        sess.run(blocking_t, options=config_pb2.RunOptions(timeout_in_ms=1000))

    with session.Session(server.target, config=self._useRPCConfig()) as sess:
      with self.assertRaises(errors_impl.DeadlineExceededError):
        sess.run(blocking_t, options=config_pb2.RunOptions(timeout_in_ms=1000))

  def testTwoServersSamePort(self):
    # Starting a server with the same target as the cached server should fail.
    server = self._cached_server
    with self.assertRaises(errors_impl.UnknownError):
      _ = server_lib.Server(
          {"local_2": [server.target[len("grpc://"):]]})

  def testExtendAfterQueueRunners(self):
    server = self._cached_server
    with session.Session(server.target) as sess:
      input_queue = input_ops.input_producer(constant_op.constant(
          [0.], dtype=dtypes.float32))
      self.assertIsNotNone(input_queue)

      var = variables.Variable(1., dtype=dtypes.float32, trainable=False,
                               name="var")

      sess.run(variables.global_variables_initializer())
      queue_runner_impl.start_queue_runners(sess)
      sess.run(var.assign(3.0))

  def testIsolateSessionState(self):
    server = self._cached_server

    init_value = array_ops.placeholder(dtypes.int32)
    v = variables.Variable(init_value, validate_shape=False, name="v")

    sharing_config = config_pb2.ConfigProto(isolate_session_state=False)
    sharing_sess_0 = session.Session(server.target, config=sharing_config)
    sharing_sess_1 = session.Session(server.target, config=sharing_config)

    isolate_config = config_pb2.ConfigProto(isolate_session_state=True)
    isolate_sess_0 = session.Session(server.target, config=isolate_config)
    isolate_sess_1 = session.Session(server.target, config=isolate_config)

    # Initially all variables are initialized.
    for sess in [sharing_sess_0, sharing_sess_1,
                 isolate_sess_0, isolate_sess_1]:
      with self.assertRaises(errors_impl.FailedPreconditionError):
        sess.run(v)

    # Shared sessions will see each other's updates, but isolated sessions
    # will not.
    sharing_sess_0.run(v.initializer, feed_dict={init_value: 86})
    self.assertAllEqual(86, sharing_sess_0.run(v))
    self.assertAllEqual(86, sharing_sess_1.run(v))
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_0.run(v)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_1.run(v)

    # Changing the shape works because `validate_shape` is False.
    sharing_sess_1.run(v.initializer, feed_dict={init_value: [86, 99]})
    self.assertAllEqual([86, 99], sharing_sess_0.run(v))
    self.assertAllEqual([86, 99], sharing_sess_1.run(v))
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_0.run(v)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_1.run(v)

    # Initializing in an isolated session will only affect the state in that
    # session.
    isolate_sess_0.run(v.initializer, feed_dict={init_value: 37})
    self.assertAllEqual([86, 99], sharing_sess_0.run(v))
    self.assertAllEqual([86, 99], sharing_sess_1.run(v))
    self.assertAllEqual(37, isolate_sess_0.run(v))
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_1.run(v)

    # Isolated sessions can have different shapes for the same variable.
    isolate_sess_1.run(v.initializer, feed_dict={init_value: [19, 86]})
    self.assertAllEqual([86, 99], sharing_sess_0.run(v))
    self.assertAllEqual([86, 99], sharing_sess_1.run(v))
    self.assertAllEqual(37, isolate_sess_0.run(v))
    self.assertAllEqual([19, 86], isolate_sess_1.run(v))

  def testShapeChangingIsolateState(self):
    server = self._cached_server
    sharing_config = config_pb2.ConfigProto(isolate_session_state=False)
    isolate_config = config_pb2.ConfigProto(isolate_session_state=True)

    with ops.Graph().as_default():
      w_vector = variables.Variable([1, 2, 3], name="w")
      with session.Session(server.target, config=sharing_config) as sess:
        with self.assertRaises(errors_impl.FailedPreconditionError):
          sess.run(w_vector)
        sess.run(w_vector.initializer)
        self.assertAllEqual([1, 2, 3], sess.run(w_vector))

    with ops.Graph().as_default():
      w_vector = variables.Variable([4, 5, 6], name="w")
      with session.Session(server.target, config=sharing_config) as sess:
        self.assertAllEqual([1, 2, 3], sess.run(w_vector))
        sess.run(w_vector.initializer)
        self.assertAllEqual([4, 5, 6], sess.run(w_vector))

    with ops.Graph().as_default():
      w_scalar = variables.Variable(86, name="w")
      with session.Session(server.target, config=sharing_config) as sess:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          sess.run(w_scalar.initializer)

    with ops.Graph().as_default():
      w_scalar = variables.Variable(37, name="w")
      with session.Session(server.target, config=isolate_config) as sess:
        with self.assertRaises(errors_impl.FailedPreconditionError):
          sess.run(w_scalar)
        sess.run(w_scalar.initializer)
        self.assertAllEqual(37, sess.run(w_scalar))


class ServerDefTest(test.TestCase):

  def testLocalServer(self):
    cluster_def = server_lib.ClusterSpec({
        "local": ["localhost:2222"]
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="local", task_index=0, protocol="grpc")

    self.assertProtoEquals("""
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }
    }
    job_name: 'local' task_index: 0 protocol: 'grpc'
    """, server_def)

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testTwoProcesses(self):
    cluster_def = server_lib.ClusterSpec({
        "local": ["localhost:2222", "localhost:2223"]
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="local", task_index=1, protocol="grpc")

    self.assertProtoEquals("""
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
                          tasks { key: 1 value: 'localhost:2223' } }
    }
    job_name: 'local' task_index: 1 protocol: 'grpc'
    """, server_def)

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testTwoJobs(self):
    cluster_def = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
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
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

  def testDenseAndSparseJobs(self):
    cluster_def = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": {
            0: "worker0:2222",
            2: "worker2:2222"
        }
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name="worker", task_index=2, protocol="grpc")

    self.assertProtoEquals("""
    cluster {
      job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                       tasks { key: 1 value: 'ps1:2222' } }
      job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                           tasks { key: 2 value: 'worker2:2222' } }
    }
    job_name: 'worker' task_index: 2 protocol: 'grpc'
    """, server_def)

    # Verifies round trip from Proto->Spec->Proto is correct.
    cluster_spec = server_lib.ClusterSpec(cluster_def)
    self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())


class ClusterSpecTest(test.TestCase):

  def testProtoDictDefEquivalences(self):
    cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })

    expected_proto = """
    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                     tasks { key: 1 value: 'ps1:2222' } }
    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                         tasks { key: 1 value: 'worker1:2222' }
                         tasks { key: 2 value: 'worker2:2222' } }
    """

    self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
    self.assertProtoEquals(
        expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
    self.assertProtoEquals(
        expected_proto,
        server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

  def testClusterSpecAccessors(self):
    original_dict = {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"],
        "sparse": {
            0: "sparse0:2222",
            3: "sparse3:2222"
        }
    }
    cluster_spec = server_lib.ClusterSpec(original_dict)

    self.assertEqual(original_dict, cluster_spec.as_dict())

    self.assertEqual(2, cluster_spec.num_tasks("ps"))
    self.assertEqual(3, cluster_spec.num_tasks("worker"))
    self.assertEqual(2, cluster_spec.num_tasks("sparse"))
    with self.assertRaises(ValueError):
      cluster_spec.num_tasks("unknown")

    self.assertEqual("ps0:2222", cluster_spec.task_address("ps", 0))
    self.assertEqual("sparse0:2222", cluster_spec.task_address("sparse", 0))
    with self.assertRaises(ValueError):
      cluster_spec.task_address("unknown", 0)
    with self.assertRaises(ValueError):
      cluster_spec.task_address("sparse", 2)

    self.assertEqual([0, 1], cluster_spec.task_indices("ps"))
    self.assertEqual([0, 1, 2], cluster_spec.task_indices("worker"))
    self.assertEqual([0, 3], cluster_spec.task_indices("sparse"))
    with self.assertRaises(ValueError):
      cluster_spec.task_indices("unknown")

    # NOTE(mrry): `ClusterSpec.job_tasks()` is not recommended for use
    # with sparse jobs.
    self.assertEqual(["ps0:2222", "ps1:2222"], cluster_spec.job_tasks("ps"))
    self.assertEqual(["worker0:2222", "worker1:2222", "worker2:2222"],
                     cluster_spec.job_tasks("worker"))
    self.assertEqual(["sparse0:2222", None, None, "sparse3:2222"],
                     cluster_spec.job_tasks("sparse"))
    with self.assertRaises(ValueError):
      cluster_spec.job_tasks("unknown")

  def testEmptyClusterSpecIsFalse(self):
    self.assertFalse(server_lib.ClusterSpec({}))

  def testNonEmptyClusterSpecIsTrue(self):
    self.assertTrue(server_lib.ClusterSpec({"job": ["host:port"]}))

  def testEq(self):
    self.assertEquals(server_lib.ClusterSpec({}), server_lib.ClusterSpec({}))
    self.assertEquals(
        server_lib.ClusterSpec({
            "job": ["host:2222"]
        }),
        server_lib.ClusterSpec({
            "job": ["host:2222"]
        }),)
    self.assertEquals(
        server_lib.ClusterSpec({
            "job": {
                0: "host:2222"
            }
        }), server_lib.ClusterSpec({
            "job": ["host:2222"]
        }))

  def testNe(self):
    self.assertNotEquals(
        server_lib.ClusterSpec({}),
        server_lib.ClusterSpec({
            "job": ["host:2223"]
        }),)
    self.assertNotEquals(
        server_lib.ClusterSpec({
            "job1": ["host:2222"]
        }),
        server_lib.ClusterSpec({
            "job2": ["host:2222"]
        }),)
    self.assertNotEquals(
        server_lib.ClusterSpec({
            "job": ["host:2222"]
        }),
        server_lib.ClusterSpec({
            "job": ["host:2223"]
        }),)
    self.assertNotEquals(
        server_lib.ClusterSpec({
            "job": ["host:2222", "host:2223"]
        }),
        server_lib.ClusterSpec({
            "job": ["host:2223", "host:2222"]
        }),)


if __name__ == "__main__":
  test.main()
