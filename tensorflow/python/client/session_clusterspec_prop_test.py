# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.client.session.Session's ClusterSpec Propagation.

These tests exercise the ClusterSpec Propagation capabilities of distributed
Sessions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


# NOTE(mrry): Dummy shape registration for ops used in the tests, since they
# don't have C++ op registrations on which to attach C++ shape fns.
ops.RegisterShape('ConstructionFails')(common_shapes.unknown_shape)


class SessionClusterSpecPropagationTest(test_util.TensorFlowTestCase):

  def testClusterSpecPropagationSimple(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config)
    output = sess.run(const)
    self.assertEqual(17, output)

  def testClusterSpecPropagationWorker2Placement(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device('/job:worker/task:1'):
      with ops.device('/cpu:0'):
        const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(const, options=run_options, run_metadata=run_metadata)
    self.assertEqual(17, output)
    self.assertEqual(1,
                     len([
                         node_stats
                         for dev_stats in run_metadata.step_stats.dev_stats
                         for node_stats in dev_stats.node_stats
                         if '/job:worker/replica:0/task:1/device:CPU:0' ==
                         dev_stats.device and 'Const' == node_stats.node_name
                     ]))

  def testClusterSpecPropagationWorker1Placement(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device('/job:worker/task:0'):
      const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    output = sess.run(const)
    self.assertEqual(17, output)

  def testCanonicalDeviceNames(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device(
        '/job:worker/task:1/device:CPU:0'):
      const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(const, options=run_options, run_metadata=run_metadata)
    self.assertEqual(17, output)
    self.assertEqual(1,
                     len([
                         node_stats
                         for dev_stats in run_metadata.step_stats.dev_stats
                         for node_stats in dev_stats.node_stats
                         if '/job:worker/replica:0/task:1/device:CPU:0' ==
                         dev_stats.device and 'Const' == node_stats.node_name
                     ]))

  def testFullDeviceNames(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'renamed_worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device(
        '/job:renamed_worker/replica:0/task:1/device:CPU:0'):
      const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(const, options=run_options, run_metadata=run_metadata)
    self.assertEqual(17, output)
    self.assertEqual(1,
                     len([
                         node_stats
                         for dev_stats in run_metadata.step_stats.dev_stats
                         for node_stats in dev_stats.node_stats
                         if '/job:renamed_worker/replica:0/task:1/device:CPU:0'
                         == dev_stats.device and 'Const' == node_stats.node_name
                     ]))

  def testMultipleLocalDevices(self):
    # Note: CPU->CPU transfers have a fast-path in
    # BaseRemoteRendezvous::SameWorkerRecvDone that means the test doesn't
    # actually capture the motivating bug unless run on a GPU machine.
    #
    # Example error message (before bugfix -- line breaks added because  lint):
    #
    # W0718 17:14:41.521534  190121 device_mgr.cc:107] Unknown device:
    #     /job:worker/replica:0/task:0/device:CPU:0 all devices:
    #     /job:local/replica:0/task:0/device:GPU:0,
    #     /job:local/replica:0/task:0/device:GPU:0,
    #     /job:local/replica:0/task:0/cpu:1, CPU:0, GPU:0,
    #     /job:local/replica:0/task:0/device:CPU:1,
    #     /job:local/replica:0/task:0/device:CPU:0, CPU:1,
    #     /job:local/replica:0/task:0/cpu:0
    server_config = config_pb2.ConfigProto(device_count={'CPU': 2})
    server1 = server_lib.Server.create_local_server(config=server_config)
    server2 = server_lib.Server.create_local_server(config=server_config)
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g:
      with ops.device('/job:worker/task:1/cpu:1'):
        input1 = constant_op.constant(17, dtypes.float32)
      with ops.device('/job:worker/task:0/cpu:1'):
        input2 = constant_op.constant(3, dtypes.float32)
      with ops.device('/job:worker/task:1/cpu:0'):
        sum1 = input1 + input2

      if test.is_gpu_available():
        device_str = '/job:worker/task:0/device:GPU:0'
      else:
        device_str = '/job:worker/task:0/cpu:1'
      with ops.device(device_str):
        sum2 = input2 + input1

      with ops.device('/job:worker/task:0/cpu:0'):
        sum3 = sum1 + sum2
    sess = session.Session(server1.target, config=config, graph=g)
    output = sess.run(sum3)
    self.assertEqual(40, output)

  def testLegacyDeviceNames(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.Graph().as_default() as g, ops.device('/job:worker/task:1/cpu:0'):
      const = constant_op.constant(17)
    sess = session.Session(server1.target, config=config, graph=g)
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(const, options=run_options, run_metadata=run_metadata)
    self.assertEqual(17, output)
    self.assertEqual(1,
                     len([
                         node_stats
                         for dev_stats in run_metadata.step_stats.dev_stats
                         for node_stats in dev_stats.node_stats
                         if '/job:worker/replica:0/task:1/device:CPU:0' ==
                         dev_stats.device and 'Const' == node_stats.node_name
                     ]))

  def testClusterSpecPropagationThreeServers2Graphs(self):
    """Boots 3 servers, creates 2 sessions, ensures appropriate operations.

    We create 2 clusterspecs:
     1. server2 as the master, server1 as a worker
     2. server2 as the master, server3 as a worker

    We ensure that variables on the workers are independent.
    """
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    server3 = server_lib.Server.create_local_server()
    cluster_def1 = cluster_pb2.ClusterDef()
    job1 = cluster_def1.job.add()
    job1.name = 'worker1'
    job1.tasks[0] = server2.target[len('grpc://'):]
    job1.tasks[1] = server1.target[len('grpc://'):]

    cluster_def2 = cluster_pb2.ClusterDef()
    job2 = cluster_def2.job.add()
    job2.name = 'worker2'
    job2.tasks[0] = server2.target[len('grpc://'):]
    job2.tasks[1] = server3.target[len('grpc://'):]

    config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
    config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)

    with ops.Graph().as_default() as g1:
      with ops.device('/job:worker1/task:1'):
        var1 = variables.Variable(array_ops.zeros([2]), name='var1')
        update_op1 = state_ops.assign_add(
            var1, array_ops.ones([2]), name='var1_assign_add')
        init1 = variables.global_variables_initializer()

    with ops.Graph().as_default() as g2:
      with ops.device('/job:worker2/task:1'):
        var2 = variables.Variable(array_ops.zeros([2]), name='var2')
        update_op2 = state_ops.assign_add(
            var2, array_ops.ones([2]), name='var2_assign_add')
        init2 = variables.global_variables_initializer()

    sess1 = session.Session(server2.target, graph=g1, config=config1)
    sess2 = session.Session(server2.target, graph=g2, config=config2)

    init1.run(session=sess1)
    init2.run(session=sess2)

    expected_zeros = np.zeros([2])
    expected_ones = np.ones([2])

    self.assertAllEqual(expected_zeros, sess1.run(var1))
    self.assertAllEqual(expected_zeros, sess2.run(var2))

    self.assertAllEqual(expected_ones, sess1.run(update_op1))
    self.assertAllEqual(expected_ones, sess1.run(var1))
    self.assertAllEqual(expected_zeros, sess2.run(var2))
    self.assertAllEqual(expected_ones, sess2.run(update_op2))
    self.assertAllEqual(expected_ones + expected_ones, sess1.run(update_op1))
    self.assertAllEqual(expected_ones, sess2.run(var2))
    self.assertAllEqual(expected_ones + expected_ones, sess1.run(var1))

  def testClusterSpecPropagationThreeServers(self):
    """Boots 3 servers, creates 2 sessions, ensures appropriate operations.

    We create 2 clusterspecs:
     1. server2 as the master, server1 as a worker
     2. server2 as the master, server3 as a worker

    We ensure that variables on the workers are independent.
    """
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    server3 = server_lib.Server.create_local_server()
    cluster_def1 = cluster_pb2.ClusterDef()
    job1 = cluster_def1.job.add()
    job1.name = 'worker'
    job1.tasks[0] = server2.target[len('grpc://'):]
    job1.tasks[1] = server1.target[len('grpc://'):]

    cluster_def2 = cluster_pb2.ClusterDef()
    job2 = cluster_def2.job.add()
    job2.name = 'worker'
    job2.tasks[0] = server2.target[len('grpc://'):]
    job2.tasks[1] = server3.target[len('grpc://'):]

    config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
    config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)

    with ops.device('/job:worker/task:1'):
      var = variables.Variable(array_ops.zeros([2]), name='var')
      feed = array_ops.placeholder(dtypes.float32, shape=(2))
      update_op = var.assign_add(feed)

    sess1 = session.Session(server2.target, config=config1)
    sess2 = session.Session(server2.target, config=config2)

    variables.global_variables_initializer().run(session=sess1)
    variables.global_variables_initializer().run(session=sess2)

    expected_zeros = np.zeros([2])
    expected_ones = np.ones([2])

    self.assertAllEqual(expected_zeros, sess1.run(var))
    self.assertAllEqual(expected_zeros, sess2.run(var))
    self.assertAllEqual(expected_ones,
                        sess1.run(update_op, feed_dict={feed: expected_ones}))
    self.assertAllEqual(expected_ones, sess1.run(var))
    self.assertAllEqual(expected_zeros, sess2.run(var))
    self.assertAllEqual(expected_ones,
                        sess2.run(update_op, feed_dict={feed: expected_ones}))
    self.assertAllEqual(expected_ones + expected_ones,
                        sess1.run(update_op, feed_dict={feed: expected_ones}))
    self.assertAllEqual(expected_ones, sess2.run(var))
    self.assertAllEqual(expected_ones + expected_ones, sess1.run(var))

  def testClusterSpecPropagationThreeServersOneCluster(self):
    """Boots 3 servers, ensures appropriate communication across workers.

    Additionally, in this cluster, we ensure the master is not the 0-th worker.

    Note: this test only uses one session.
    """
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()
    server3 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server3.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    job.tasks[2] = server1.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    # Add ops to the devices in non-linear order.

    with ops.device('/job:worker/task:1'):
      feed1 = array_ops.placeholder(dtypes.float32, shape=(2))
      const1 = constant_op.constant(2.0)
      mul1 = const1 * feed1

    with ops.device('/job:worker/task:2'):
      feed2 = array_ops.placeholder(dtypes.float32, shape=(2))
      const2 = constant_op.constant(2.0)
      mul2 = const2 * feed2

    with ops.device('/job:worker/task:0'):
      feed0 = array_ops.placeholder(dtypes.float32, shape=(2))
      const0 = constant_op.constant(2.0)
      mul0 = const0 * feed0

    sum_op = mul0 + mul1 + mul2

    ones = np.ones([2])
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    # Run!
    with session.Session(server1.target, config=config) as sess:
      output = sess.run(
          sum_op,
          options=run_options,
          run_metadata=run_metadata,
          feed_dict={feed1: ones,
                     feed2: ones,
                     feed0: ones})
      self.assertAllEqual(6 * ones, output)

      self.assertEqual(
          3,
          len([
              dev_stats.device
              for dev_stats in run_metadata.step_stats.dev_stats
              for node_stats in dev_stats.node_stats
              if '/job:worker/replica:0/task:' in dev_stats.device and
              node_stats.node_name.startswith('Const')
          ]), run_metadata)

  def testClusterSpecPropagationIsolation(self):
    """Test that two sessions using ClusterSpec propagation are isolated."""
    server = server_lib.Server.create_local_server()
    init_value = array_ops.placeholder(dtypes.int32, shape=[])
    v = variables.Variable(init_value)

    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    sess1 = session.Session(server.target, config=config)
    sess2 = session.Session(server.target, config=config)

    # Initially, the variable is uninitialized in both sessions.
    with self.assertRaises(errors.FailedPreconditionError):
      sess1.run(v)
    with self.assertRaises(errors.FailedPreconditionError):
      sess2.run(v)

    # An update in sess1 should be visible in sess1 only.
    sess1.run(v.initializer, feed_dict={init_value: 37})
    self.assertEqual(37, sess1.run(v))
    with self.assertRaises(errors.FailedPreconditionError):
      sess2.run(v)

    # An update in sess2 should be visible in sess2 only.
    sess2.run(v.initializer, feed_dict={init_value: 86})
    self.assertEqual(37, sess1.run(v))
    self.assertEqual(86, sess2.run(v))

    # Closing sess2 has no effect on the state of sess1.
    sess2.close()
    self.assertEqual(37, sess1.run(v))

    # Subsequent sessions will not see the state of existing sessions.
    sess3 = session.Session(server.target, config=config)
    self.assertEqual(37, sess1.run(v))
    with self.assertRaises(errors.FailedPreconditionError):
      sess3.run(v)

  def testClusterSpecPropagationPartialRun(self):
    """Test successful partial run with ClusterSpec propagation."""
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    with ops.device('/job:worker/task:0'):
      a = array_ops.placeholder(dtypes.float32, shape=[])
    with ops.device('/job:worker/task:1'):
      b = array_ops.placeholder(dtypes.float32, shape=[])
      c = array_ops.placeholder(dtypes.float32, shape=[])
      r1 = math_ops.add(a, b)
    with ops.device('/job:worker/task:0'):
      r2 = math_ops.multiply(r1, c)

    with session.Session(server1.target, config=config) as sess:
      h = sess.partial_run_setup([r1, r2], [a, b, c])
      res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
      self.assertEqual(3, res)
      res = sess.partial_run(h, r2, feed_dict={c: 3})
      self.assertEqual(9, res)


if __name__ == '__main__':
  googletest.main()
