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
"""Tests for DropStaleGradientOptimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import portpicker

from tensorflow.contrib.opt.python.training import drop_stale_gradient_optimizer
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import server_lib
from tensorflow.python.training import training_util


# Creates the workers and return their sessions, graphs, train_ops.
def _get_workers(num_workers, staleness):
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  cluster_dict = {
      'worker': ['localhost:%s' % port for port in worker_ports],
      'ps': ['localhost:%s' % portpicker.pick_unused_port()]
  }
  cs = server_lib.ClusterSpec(cluster_dict)
  workers = [
      server_lib.Server(
          cs, job_name='worker', task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  server_lib.Server(cs, job_name='ps', task_index=0, start=True)

  sessions = []
  graphs = []
  train_ops = []

  # To simulate stale cases, maintaining two queues for computing and
  # applying gradients respectively. In the phase of computing gradients,
  # all workers except chief worker compute gradients together and chief worker
  # computes after all other worers' computing finished. In the phase of
  # applying gradients, chief worker will first apply gradients, then all other
  # workers will apply gradients one by one. Therefore, the chief worker will
  # always have 0 staleness, each of all other workers will have a unique
  # staleness value from [1, num_workers).
  for worker_id in range(num_workers):
    graph = ops.Graph()
    with graph.as_default():
      global_step = training_util.create_global_step()
      var_0 = variables.Variable(0.0, name='v0')
      var_1 = variables.Variable(1.0, name='v1')
      compute_gradients_queue = data_flow_ops.FIFOQueue(
          -1, global_step.dtype.base_dtype, shapes=(),
          name='compute_gradients_queue', shared_name='compute_gradients_queue')
      apply_gradients_queue = data_flow_ops.FIFOQueue(
          -1, global_step.dtype.base_dtype, shapes=(),
          name='apply_gradients_queue', shared_name='apply_gradients_queue')

      # Gradients for loss on var_0 and var_1 will be 1.0.
      loss = 0 - var_0 - var_1
      sgd_opt = gradient_descent.GradientDescentOptimizer(1.0)
      stale_check_opt = (
          drop_stale_gradient_optimizer.DropStaleGradientOptimizer(
              sgd_opt, staleness))

      # Compute gradients.
      if worker_id == 0:
        with ops.control_dependencies(
            [compute_gradients_queue.dequeue_many(num_workers - 1)]):
          grad_and_vars = stale_check_opt.compute_gradients(loss)
      else:
        grad_and_vars = stale_check_opt.compute_gradients(loss)
        with ops.control_dependencies([t[0] for t in grad_and_vars]):
          worker_enqueue_op = compute_gradients_queue.enqueue(global_step)

      # Apply gradients.
      if worker_id == 0:
        with ops.control_dependencies(
            [stale_check_opt.apply_gradients(grad_and_vars, global_step)]):
          train_op = apply_gradients_queue.enqueue(global_step)
      else:
        with ops.control_dependencies([worker_enqueue_op]):
          with ops.control_dependencies([apply_gradients_queue.dequeue()]):
            with ops.control_dependencies(
                [stale_check_opt.apply_gradients(
                    grad_and_vars, global_step)]):
              train_op = apply_gradients_queue.enqueue(global_step)

      sess = session.Session(workers[worker_id].target)

    sessions.append(sess)
    graphs.append(graph)
    train_ops.append(train_op)

  return sessions, graphs, train_ops


class DropStaleGradientOptimizerTest(test.TestCase):

  def _run(self, train_op, sess):
    sess.run(train_op)

  def test1Worker(self):
    num_workers = 1
    sessions, graphs, train_ops = _get_workers(num_workers, 0)
    with graphs[0].as_default():
      sessions[0].run(variables.global_variables_initializer())
    global_step = training_util.get_global_step(graphs[0])
    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    stale_counter = graphs[0].get_tensor_by_name('stale_counter:0')
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0, sessions[0].run(global_step))

    sessions[0].run(train_ops[0])

    # Verify the updated value after 1 step.
    self.assertAllEqual(1, sessions[0].run(global_step))
    self.assertAllEqual(0.0 + 1.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0 + 1.0, sessions[0].run(var_1))
    self.assertAllEqual(1, sessions[0].run(global_step))

  def test1WorkerNegativeStaleness(self):
    num_workers = 1
    sessions, graphs, train_ops = _get_workers(num_workers, -1)
    with graphs[0].as_default():
      sessions[0].run(variables.global_variables_initializer())
    global_step = training_util.get_global_step(graphs[0])
    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    stale_counter = graphs[0].get_tensor_by_name('stale_counter:0')
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0, sessions[0].run(global_step))

    sessions[0].run(train_ops[0])

    # Verify no updates because max staleness is negative.
    self.assertAllEqual(0, sessions[0].run(global_step))
    self.assertAllEqual(1.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))

  def test2WorkersStaleness0(self):
    num_workers = 2
    sessions, graphs, train_ops = _get_workers(num_workers, 0)
    with graphs[0].as_default():
      sessions[0].run(variables.global_variables_initializer())
    global_step = training_util.get_global_step(graphs[0])
    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    stale_counter = graphs[0].get_tensor_by_name('stale_counter:0')
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0, sessions[0].run(global_step))

    thread_0 = self.checkedThread(
        target=self._run, args=(train_ops[0], sessions[0]))
    thread_1 = self.checkedThread(
        target=self._run, args=(train_ops[1], sessions[1]))
    thread_0.start()
    thread_1.start()
    thread_0.join()
    thread_1.join()

    # With 2 workers and max staleness set to 0, only chief worker will update
    # var_0 and var_1.
    self.assertAllEqual(1, sessions[0].run(global_step))
    self.assertAllEqual(1.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0.0 + 1.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0 + 1.0, sessions[0].run(var_1))

  def test2WorkersStaleness1(self):
    num_workers = 2
    sessions, graphs, train_ops = _get_workers(num_workers, 1)
    with graphs[0].as_default():
      sessions[0].run(variables.global_variables_initializer())
    global_step = training_util.get_global_step(graphs[0])
    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    stale_counter = graphs[0].get_tensor_by_name('stale_counter:0')
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0, sessions[0].run(global_step))

    thread_0 = self.checkedThread(
        target=self._run, args=(train_ops[0], sessions[0]))
    thread_1 = self.checkedThread(
        target=self._run, args=(train_ops[1], sessions[1]))
    thread_0.start()
    thread_1.start()
    thread_0.join()
    thread_1.join()

    # With 2 workers and max staleness set to 1, both workers will update
    # var_0 and var_1.
    self.assertAllEqual(2, sessions[0].run(global_step))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0.0 + 2.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0 + 2.0, sessions[0].run(var_1))

  def test3WorkersStaleness0(self):
    num_workers = 3
    sessions, graphs, train_ops = _get_workers(num_workers, 0)
    with graphs[0].as_default():
      sessions[0].run(variables.global_variables_initializer())
    global_step = training_util.get_global_step(graphs[0])
    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    stale_counter = graphs[0].get_tensor_by_name('stale_counter:0')
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0, sessions[0].run(global_step))

    thread_0 = self.checkedThread(
        target=self._run, args=(train_ops[0], sessions[0]))
    thread_1 = self.checkedThread(
        target=self._run, args=(train_ops[1], sessions[1]))
    thread_2 = self.checkedThread(
        target=self._run, args=(train_ops[2], sessions[2]))
    thread_0.start()
    thread_1.start()
    thread_2.start()
    thread_0.join()
    thread_1.join()
    thread_2.join()

    # With 3 workers and max staleness set to 0, only chief worker will update
    # var_0 and var_1.
    self.assertAllEqual(1, sessions[0].run(global_step))
    self.assertAllEqual(2.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0.0 + 1.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0 + 1.0, sessions[0].run(var_1))

  def test3WorkersStaleness1(self):
    num_workers = 3
    sessions, graphs, train_ops = _get_workers(num_workers, 1)
    with graphs[0].as_default():
      sessions[0].run(variables.global_variables_initializer())
    global_step = training_util.get_global_step(graphs[0])
    var_0 = graphs[0].get_tensor_by_name('v0:0')
    var_1 = graphs[0].get_tensor_by_name('v1:0')
    stale_counter = graphs[0].get_tensor_by_name('stale_counter:0')
    # Verify the initialized value.
    self.assertAllEqual(0.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0, sessions[0].run(var_1))
    self.assertAllEqual(0.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0, sessions[0].run(global_step))

    thread_0 = self.checkedThread(
        target=self._run, args=(train_ops[0], sessions[0]))
    thread_1 = self.checkedThread(
        target=self._run, args=(train_ops[1], sessions[1]))
    thread_2 = self.checkedThread(
        target=self._run, args=(train_ops[2], sessions[2]))
    thread_0.start()
    thread_1.start()
    thread_2.start()
    thread_0.join()
    thread_1.join()
    thread_2.join()

    # With 3 workers and max staleness set to 1, chief worker and only one of
    # the two other workers will update var_0 and var_1.
    self.assertAllEqual(2, sessions[0].run(global_step))
    self.assertAllEqual(1.0, sessions[0].run(stale_counter))
    self.assertAllEqual(0.0 + 2.0, sessions[0].run(var_0))
    self.assertAllEqual(1.0 + 2.0, sessions[0].run(var_1))


if __name__ == '__main__':
  test.main()
