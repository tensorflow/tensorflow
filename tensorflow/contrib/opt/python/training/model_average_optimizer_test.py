# 2017 Contrib.
# ==============================================================================
"""Tests for model_average_optimizer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import portpicker

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.training import training
from tensorflow.contrib.opt.python.training import model_average_optimizer


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

  return workers, ps_servers


# Creates the workers and return their sessions, graphs, train_ops.
def get_workers(num_workers, replicas_to_aggregate, workers):
  sessions = []
  graphs = []
  train_ops = []
  for worker_id in range(num_workers):
    graph = ops.Graph()
    is_chief = (worker_id == 0)
    with graph.as_default():
      with ops.device(model_average_optimizer.model_average_device_setter(
          worker_device="/job:worker/task:%d/gpu:%d" % (worker_id, worker_id))):
        k = variables.Variable(0.0, name="k")
        if is_chief:
          xs = np.linspace(-0.5, 0.49, 100)
        else:
          xs = np.linspace(-1.0, 0.99, 100)
        ys = 42.0 * xs
        x = ops.convert_to_tensor(xs, np.float32)
        y = ops.convert_to_tensor(ys, np.float32)

        y_hat = math_ops.multiply(k, x, name="y_hat")
        sse = math_ops.reduce_sum((y - y_hat) * (y - y_hat), name="sse")
        opt = training.GradientDescentOptimizer(learning_rate=0.02)
        train_op = opt.minimize(sse)
        ma = model_average_optimizer.ModelAverageOptimizer(
            replicas_to_aggregate, 1)
        ma_hook = ma.make_ma_run_hook()
        ma_replicas_hook = ma.make_session_run_hook(is_chief,
                                                    num_tokens=num_workers)

      # Creates MonitoredSession
      session = training.MonitoredTrainingSession(
          master=workers[worker_id].target,
          is_chief=is_chief,
          hooks=[ma_replicas_hook, ma_hook])

    sessions.append(session)
    graphs.append(graph)
    train_ops.append(train_op)

  return sessions, graphs, train_ops


class ModelAverageOptimizerTest(test.TestCase):
  def _run(self, train_op, sess):
    sess.run(train_op)

  def test2Workers(self):
    num_workers = 2
    replicas_to_aggregate = 2
    num_ps = 1
    workers, _ = create_local_cluster(num_workers=num_workers, num_ps=num_ps)

    # Creates and returns all the workers.
    sessions, graphs, train_ops = get_workers(num_workers,
                                              replicas_to_aggregate, workers)
    var_0 = graphs[0].get_tensor_by_name("k:0")
    var_1 = graphs[1].get_tensor_by_name("k:0")
    var_g = graphs[0].get_tensor_by_name("modelAverage_g0:0")

    self.assertAllEqual(0.0, sessions[0]._tf_sess().run(var_0))

    # We have initial tokens in the queue so we can call this one by one. After
    # the first step, this will no longer work as there will be no more extra
    # tokens in the queue.
    sessions[0].run(train_ops[0])
    sessions[1].run(train_ops[1])

    sessions[0].run(train_ops[0])
    sessions[1].run(train_ops[1])
    # Will just use session 1 to verify all the variables later.
    a = 14.00279999
    b = 56.56562805
    ma = (a+b)/2.0
    self.assertAllClose(ma, sessions[0]._tf_sess().run(var_0))
    self.assertAllClose(ma, sessions[1]._tf_sess().run(var_1))
    self.assertAllClose(ma, sessions[0]._tf_sess().run(var_g))

  _cluster_spec = server_lib.ClusterSpec({
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
  })

  def testPS2TasksWithClusterSpecClass(self):
    with ops.device(
        model_average_optimizer.model_average_device_setter(
            cluster=self._cluster_spec)):
      v = variables.Variable([1, 2], name="modelAverage_v")
      v_1 = variables.Variable([1, 2], name="modelAverage_v_1")
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", v_1.device)
      self.assertDeviceEqual("/job:ps/task:1", v_1.initializer.device)
      self.assertDeviceEqual("/job:worker", w.device)
      self.assertDeviceEqual("/job:worker", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

if __name__ == "__main__":
  test.main()
