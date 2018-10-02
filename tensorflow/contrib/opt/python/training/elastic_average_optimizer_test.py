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
"""Tests for ElasticAverageOptimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import portpicker
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import training
from tensorflow.python.training import training_util

from tensorflow.contrib.opt.python.training.elastic_average_optimizer import \
  ElasticAverageOptimizer, ElasticAverageCustomGetter, GLOBAL_VARIABLE_NAME


def create_local_cluster(num_workers, num_ps, protocol="grpc"):
  """Create local GRPC servers and return them."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs, job_name="worker", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs, job_name="ps", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_ps)
  ]

  return cluster_dict, workers, ps_servers


# Creates the workers and return their sessions, graphs, train_ops.
# Chief worker will update at last
def _get_workers(num_workers, period, workers, moving_rate, num_ps=1):
  sessions = []
  graphs = []
  train_ops = []
  savers = []
  for worker_id in range(num_workers):
    graph = ops.Graph()
    is_chief = (worker_id == 0)
    with graph.as_default():
      worker_device = "/job:worker/task:%d/cpu:0" % (worker_id)
      ea_custom = ElasticAverageCustomGetter(worker_device=worker_device)
      with variable_scope.variable_scope(
          "", custom_getter=ea_custom), ops.device(
              device_setter.replica_device_setter(
                  worker_device=worker_device,
                  ps_device="/job:ps/task:0/cpu:0",
                  ps_tasks=1)):
        global_step = training_util.get_or_create_global_step()
        var_0 = variable_scope.get_variable(initializer=0.0, name="v0")
        var_1 = variable_scope.get_variable(initializer=1.0, name="v1")
      if num_ps > 1:
        with variable_scope.variable_scope(
            "",
            partitioner=partitioned_variables.fixed_size_partitioner(
                num_ps, axis=0),
            custom_getter=ea_custom), ops.device(
                device_setter.replica_device_setter(
                    worker_device=worker_device,
                    ps_device="/job:ps/task:0/cpu:0",
                    ps_tasks=num_ps)):

          partition_var = variable_scope.get_variable(
              'partition_var',
              shape=[2, 4],
              initializer=init_ops.ones_initializer)
          part_0 = list(partition_var)[0]
          part_1 = list(partition_var)[1]

      with ops.device("/job:worker/task:" + str(worker_id)):
        grads_0 = constant_op.constant(-1.0)
        grads_1 = constant_op.constant(-1.0)
        grads_part_0 = constant_op.constant([[-1., -1., -1., -1.]])
        grads_part_1 = constant_op.constant([[-1., -1., -1., -1.]])

        sgd_opt = gradient_descent.GradientDescentOptimizer(1.0)
        opt = ElasticAverageOptimizer(
            opt=sgd_opt,
            num_worker=num_workers,
            moving_rate=moving_rate,
            communication_period=period,
            ea_custom_getter=ea_custom)
        if num_ps == 1:
          train_op = [
              opt.apply_gradients(([grads_0, var_0], [grads_1, var_1]),
                                  global_step)
          ]
        else:
          train_op = [
              opt.apply_gradients(([grads_0, var_0],
                                   [grads_1, var_1],
                                   [grads_part_0, part_0],
                                   [grads_part_1, part_1]),
                                  global_step)
          ]
        easgd_hook = opt.make_session_run_hook(is_chief, worker_id)
        saver = opt.swapping_saver()
      # Creates MonitoredSession
      sess = training.MonitoredTrainingSession(
          workers[worker_id].target, hooks=[easgd_hook])

    sessions.append(sess)
    graphs.append(graph)
    train_ops.append(train_op)
    savers.append(saver)

  return sessions, graphs, train_ops, savers


class ElasticAverageOptimizerTest(test.TestCase):

  def _run(self, train_op, sess):
    sess.run(train_op)

  def test1Workers2Period(self):
    num_workers = 1
    communication_period = 2
    num_ps = 1
    cluster, workers, _ = create_local_cluster(
        num_workers=num_workers, num_ps=num_ps)

    sessions, graphs, train_ops, savers = _get_workers(
        num_workers, communication_period, workers, 1.0)

    var_0 = graphs[0].get_tensor_by_name("v0:0")
    var_1 = graphs[0].get_tensor_by_name("v1:0")
    global_step = training_util.get_global_step(graphs[0])
    var_0_g = graphs[0].get_tensor_by_name(GLOBAL_VARIABLE_NAME + "/v0:0")
    var_1_g = graphs[0].get_tensor_by_name(GLOBAL_VARIABLE_NAME + "/v1:0")
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(var_0_g))
    self.assertAllEqual(1.0, sessions[0].run(var_1_g))
    self.assertAllEqual(0, sessions[0].run(global_step))

    sessions[0].run(train_ops[0])

    self.assertAllEqual(1.0, sessions[0].run(var_0))
    self.assertAllEqual(2.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(var_0_g))
    self.assertAllEqual(1.0, sessions[0].run(var_1_g))
    self.assertAllEqual(0, sessions[0].run(global_step))

    # iteration 2, global variable update
    sessions[0].run(train_ops[0])

    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(2.0, sessions[0].run(var_0_g))
    self.assertAllEqual(3.0, sessions[0].run(var_1_g))
    self.assertAllEqual(1, sessions[0].run(global_step))

    # iteration 3
    sessions[0].run(train_ops[0])

    self.assertAllEqual(1.0, sessions[0].run(var_0))
    self.assertAllEqual(2.0, sessions[0].run(var_1))
    self.assertAllEqual(2.0, sessions[0].run(var_0_g))
    self.assertAllEqual(3.0, sessions[0].run(var_1_g))
    self.assertAllEqual(1, sessions[0].run(global_step))
    sessions[0].run(train_ops[0])

    # save, data will be global value
    outfile = os.path.join(test.get_temp_dir(), "model")
    savers[0].save(sessions[0]._sess._sess._sess._sess,
                   save_path=outfile)
    ops.reset_default_graph()   # restore on a new graph
    with session.Session() as sess:
      v0 = variable_scope.get_variable(initializer=0.0, name="v0")
      v1 = variable_scope.get_variable(initializer=1.0, name="v1")
      sess.run(variables.local_variables_initializer())
      saver_opt = saver.Saver(var_list=[v1, v0])
      saver_opt.restore(sess, outfile)
      self.assertAllEqual(2.0, sess.run(v0))
      self.assertAllEqual(3.0, sess.run(v1))

  def test2Worker1Period(self):
    num_workers = 2
    communication_period = 1
    num_ps = 2
    cluster, workers, _ = create_local_cluster(
        num_workers=num_workers, num_ps=num_ps)

    sessions, graphs, train_ops, savers = _get_workers(
        num_workers, communication_period, workers, 0.5, num_ps=2)

    var_0 = graphs[0].get_tensor_by_name("v0:0")
    var_1 = graphs[0].get_tensor_by_name("v1:0")

    var_0_1 = graphs[1].get_tensor_by_name("v0:0")
    var_1_1 = graphs[1].get_tensor_by_name("v1:0")

    var_0_g = graphs[0].get_tensor_by_name(GLOBAL_VARIABLE_NAME + "/v0:0")
    var_1_g = graphs[0].get_tensor_by_name(GLOBAL_VARIABLE_NAME + "/v1:0")
    part_0_g = graphs[0].get_tensor_by_name(
        GLOBAL_VARIABLE_NAME + "/partition_var/part_0:0")

    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[1].run(var_0_1))
    self.assertAllEqual(1.0, sessions[1].run(var_1_1))
    self.assertAllEqual(0.0, sessions[0].run(var_0_g))
    self.assertAllEqual(1.0, sessions[0].run(var_1_g))

    sessions[0].run(train_ops[0])
    sessions[1].run(train_ops[1])

    self.assertAllEqual(0.5, sessions[0].run(var_0))
    self.assertAllEqual(1.5, sessions[0].run(var_1))
    self.assertAllEqual(0.75, sessions[0].run(var_0_g))
    self.assertAllEqual(1.75, sessions[0].run(var_1_g))
    self.assertAllEqual(0.75, sessions[1].run(var_0_1))
    self.assertAllEqual(1.75, sessions[1].run(var_1_1))
    # part_0 of global_center copy
    part_0_g = sessions[0].run(part_0_g)

    outfile = os.path.join(test.get_temp_dir(), "model")
    savers[0].save(sessions[0]._sess._sess._sess._sess,
                   save_path=outfile)

    # verify restore of partitioned_variables
    ops.reset_default_graph()   # restore on a new graph
    g = ops.get_default_graph()
    with session.Session() as sess, g.as_default():
      with variable_scope.variable_scope(
          "",
          partitioner=partitioned_variables.fixed_size_partitioner(
              num_ps, axis=0)):
        partition_var = variable_scope.get_variable(
            'partition_var',
            shape=[2, 4],
            initializer=init_ops.ones_initializer)
      s = saver.Saver(var_list=[partition_var])
      s.restore(sess, outfile)
      part_0 = g.get_tensor_by_name('partition_var/part_0:0')
      self.assertAllEqual(part_0_g, sess.run(part_0))

  def testPS2TasksWithClusterSpecClass(self):
    cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    ea_custom = ElasticAverageCustomGetter(worker_device="/job:worker/task:0")
    from tensorflow.python.training import device_setter
    with ops.device(
        device_setter.replica_device_setter(cluster=cluster_spec,
                                            worker_device="/job:worker/task:0",
                                            ps_device="/job:ps")), \
        variable_scope.variable_scope("", custom_getter=ea_custom):
      v = variable_scope.get_variable(initializer=[1, 2], name="v")
      w = variable_scope.get_variable(initializer=[2, 1], name="w")
      v_g, w_g = ea_custom._global_map[v], ea_custom._global_map[w]
      self.assertDeviceEqual("/job:worker/task:0", v.device)
      self.assertDeviceEqual("job:ps/task:0", v_g.device)
      self.assertDeviceEqual("/job:worker/task:0", w.device)
      self.assertDeviceEqual("job:ps/task:1", w_g.device)


if __name__ == "__main__":
  test.main()
