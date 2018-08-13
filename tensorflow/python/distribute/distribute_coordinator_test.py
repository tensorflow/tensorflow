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
"""Tests for distribute coordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import os
import sys
import threading
import six

# pylint: disable=invalid-name
_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:
  _portpicker_import_error = _error
  portpicker = None
# pylint: enable=invalid-name

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

CHIEF = distribute_coordinator._TaskType.CHIEF
WORKER = distribute_coordinator._TaskType.WORKER
PS = distribute_coordinator._TaskType.PS
EVALUATOR = distribute_coordinator._TaskType.EVALUATOR

SPLIT_CLIENT = distribute_coordinator.CoordinatorMode.SPLIT_CLIENT
INDEPENDENT_WORKER = distribute_coordinator.CoordinatorMode.INDEPENDENT_WORKER

RUN_STD_SERVER_METHOD = "tensorflow.python.distribute.distribute_coordinator._run_std_server"

NUM_WORKERS = 3
NUM_PS = 2


def _bytes_to_str(maybe_bytes):
  if isinstance(maybe_bytes, six.string_types):
    return maybe_bytes
  else:
    return str(maybe_bytes, "utf-8")


def _strip_protocol(target):
  # cluster_spec expects "host:port" strings.
  if "//" in target:
    return target.split("//")[1]
  else:
    return target


class MockServer(object):

  def __init__(self):
    self._joined = False

  def join(self):
    assert not self._joined
    self._joined = True

  @property
  def joined(self):
    return self._joined


class DistributeCoordinatorTestBase(test.TestCase):

  @classmethod
  def setUpClass(cls):
    # We have to create a global in-process cluster because once an in-process
    # tensorflow server is created, there is no way to terminate it. Please see
    # multi_worker_test_base.py for more details.
    cls._workers, cls._ps = test_util.create_local_cluster(
        NUM_WORKERS, num_ps=NUM_PS)
    cls._cluster_spec = {
        WORKER: [
            _strip_protocol(_bytes_to_str(w.target)) for w in cls._workers
        ],
        PS: [_strip_protocol(_bytes_to_str(ps.target)) for ps in cls._ps]
    }

  def setUp(self):
    self._result_correct = 0
    self._lock = threading.Lock()
    self._worker_context = {}
    self._std_servers = {}
    self._barrier = distribute_coordinator._Barrier(NUM_WORKERS)

  @contextlib.contextmanager
  def _test_session(self, target):
    config = config_pb2.ConfigProto(allow_soft_placement=True)
    config.graph_options.optimizer_options.opt_level = -1
    with session.Session(graph=None, config=config, target=target) as sess:
      yield sess

  def _create_cluster_spec(self,
                           has_chief=False,
                           num_workers=1,
                           num_ps=0,
                           has_eval=False):
    if _portpicker_import_error:
      raise _portpicker_import_error  # pylint: disable=raising-bad-type

    cluster_spec = {}
    if has_chief:
      cluster_spec[CHIEF] = ["localhost:%s" % portpicker.pick_unused_port()]
    if num_workers:
      cluster_spec[WORKER] = [
          "localhost:%s" % portpicker.pick_unused_port()
          for _ in range(num_workers)
      ]
    if num_ps:
      cluster_spec[PS] = [
          "localhost:%s" % portpicker.pick_unused_port() for _ in range(num_ps)
      ]
    if has_eval:
      cluster_spec[EVALUATOR] = ["localhost:%s" % portpicker.pick_unused_port()]
    return cluster_spec

  def _in_graph_worker_fn(self):
    context = distribute_coordinator.get_current_worker_context()
    self.assertTrue(context is not None)
    with self._test_session(target=context.master_target) as sess:
      xs = []
      expected = 0.0
      for i in range(context.num_workers):
        with ops.device("/job:worker/task:%d" % i):
          x = variable_scope.get_variable("x_%d" % i, initializer=10.0)
          x_add = x.assign_add(float(i))
          xs.append(x_add)
          expected += i + 10.0

      with ops.device("/job:worker/task:0"):
        result = math_ops.add_n(xs)

      variables.global_variables_initializer().run()
      result_value = sess.run(result)
    self.assertEqual(result_value, expected)
    if result_value == expected:
      self._result_correct += 1

  def _run_coordinator_in_thread(self, worker_fn, **kwargs):
    t = threading.Thread(
        target=distribute_coordinator.run_distribute_coordinator,
        args=(worker_fn,),
        kwargs=kwargs)
    t.start()
    return t

  def _run_multiple_coordinator_in_threads(self, worker_fn, cluster_spec,
                                           **kwargs):
    threads = {}
    for task_type in cluster_spec.keys():
      threads[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        t = self._run_coordinator_in_thread(
            worker_fn,
            cluster_spec=cluster_spec,
            task_type=task_type,
            task_id=task_id,
            **kwargs)
        threads[task_type].append(t)
    return threads

  def _between_graph_worker_fn(self):
    context = distribute_coordinator.get_current_worker_context()
    self.assertTrue(context is not None)
    with self._test_session(target=context.master_target) as sess:
      with ops.device("/job:ps/task:0"):
        # TODO(yuefengz): investigate why not using resource variable will make
        # the test flaky.
        x = variable_scope.get_variable(
            "x", initializer=10.0, use_resource=True)
      with ops.device("/job:ps/task:1"):
        y = variable_scope.get_variable(
            "y", initializer=20.0, use_resource=True)

      x_add = x.assign_add(2.0)
      y_sub = y.assign_sub(2.0)
      train_op = control_flow_ops.group([x_add, y_sub])

      if context.is_chief:
        variables.global_variables_initializer().run()

      # Synchronize workers after initializaton.
      if context.has_barrier:
        context.wait_for_other_workers()
      else:
        while True:
          uninit_vars = sess.run(variables.report_uninitialized_variables())
          # pylint: disable=g-explicit-length-test
          if len(uninit_vars) == 0:
            break

      sess.run(train_op)

      # Synchronize workers after one step to make sure they all have finished
      # training.
      if context.has_barrier:
        context.wait_for_other_workers()
      else:
        self._barrier.wait()

      x_val, y_val = sess.run([x, y])

      self.assertEqual(x_val, 16.0)
      self.assertEqual(y_val, 14.0)
      if x_val == 16.0 and y_val == 14.0:
        with self._lock:
          self._result_correct += 1

  def _dump_worker_context(self):
    """Dumps the propoerties of each worker context.

    It dumps the context properties to a dict mapping from task_type to a list
    of tuples of master_target, num_workers, is_chief and distribute_mode, where
    the list is indexed by the task_id.
    """
    context = distribute_coordinator.get_current_worker_context()
    self.assertTrue(context is not None)
    task_type = str(context.task_type)
    task_id = context.task_id or 0
    with self._lock:
      if task_type not in self._worker_context:
        self._worker_context[task_type] = []
      while len(self._worker_context[task_type]) <= task_id:
        self._worker_context[task_type].append(None)
      self._worker_context[task_type][task_id] = (context.master_target,
                                                  context.num_workers,
                                                  context.is_chief,
                                                  context.distributed_mode)

  def _run_mock_std_server(self,
                           session_config=None,
                           cluster_spec=None,
                           task_type=None,
                           task_id=None,
                           rpc_layer=None):
    task_type = str(task_type)
    task_id = task_id or 0
    with self._lock:
      if task_type not in self._std_servers:
        self._std_servers[task_type] = []
      while len(self._std_servers[task_type]) <= task_id:
        self._std_servers[task_type].append(None)

      server = MockServer()
      self._std_servers[task_type][task_id] = server
    return server


class DistributeCoordinatorTestSplitMode(DistributeCoordinatorTestBase):

  def testInGraphSplitMode(self):
    """Test it runs in-graph replication in split client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._in_graph_worker_fn,
        cluster_spec=self._cluster_spec,
        between_graph=False)
    self.assertEqual(self._result_correct, 1)

  def testBetweenGraph(self):
    """Test it runs between-graph replication in split client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._between_graph_worker_fn,
        cluster_spec=self._cluster_spec,
        between_graph=True)

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  def testBetweenGraphContext(self):
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        cluster_spec=self._cluster_spec,
        between_graph=True)

    # There is only one type of task and there three such tasks.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue(WORKER in self._worker_context)
    self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context[WORKER][0],
        (_bytes_to_str(self._workers[0].target), NUM_WORKERS, True, True))
    self.assertEqual(
        self._worker_context[WORKER][1],
        (_bytes_to_str(self._workers[1].target), NUM_WORKERS, False, True))
    self.assertEqual(
        self._worker_context[WORKER][2],
        (_bytes_to_str(self._workers[2].target), NUM_WORKERS, False, True))

  def testInGraphContext(self):
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        cluster_spec=self._cluster_spec,
        between_graph=False)

    # There is only a "None" task in the dumped task context.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context["None"][0],
        (_bytes_to_str(self._workers[0].target), NUM_WORKERS, True, True))

  def testLocalContext(self):
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context, cluster_spec=None, between_graph=True)

    # There is only a "None" task.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0], ("local", 0, True, False))

  def testBetweenGraphContextWithChief(self):
    # Adds a chief node, so there are NUM_WORKERS + 1 workers in total.
    cluster_spec = copy.deepcopy(self._cluster_spec)
    cluster_spec[CHIEF] = ["fake_chief"]

    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        cluster_spec=cluster_spec,
        between_graph=True,
        rpc_layer="grpc")

    # There are one CHIEF and three workers.
    self.assertEqual(len(self._worker_context), 2)
    self.assertTrue(CHIEF in self._worker_context)
    self.assertTrue(WORKER in self._worker_context)
    self.assertEqual(len(self._worker_context[CHIEF]), 1)
    self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context[CHIEF][0],
                     ("grpc://fake_chief", 4, True, True))
    self.assertEqual(
        self._worker_context[WORKER][0],
        (_bytes_to_str(self._workers[0].target), NUM_WORKERS + 1, False, True))
    self.assertEqual(
        self._worker_context[WORKER][1],
        (_bytes_to_str(self._workers[1].target), NUM_WORKERS + 1, False, True))
    self.assertEqual(
        self._worker_context[WORKER][2],
        (_bytes_to_str(self._workers[2].target), NUM_WORKERS + 1, False, True))

  def testInGraphContextWithEval(self):
    # Adds a EVALUATOR job.
    cluster_spec = copy.deepcopy(self._cluster_spec)
    cluster_spec[EVALUATOR] = ["fake_evaluator"]

    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        cluster_spec=cluster_spec,
        between_graph=False,
        rpc_layer=None)

    # There are one "None" task and one EVALUATOR task.
    self.assertEqual(len(self._worker_context), 2)
    self.assertTrue("None" in self._worker_context)
    self.assertTrue(EVALUATOR in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)
    self.assertEqual(len(self._worker_context[EVALUATOR]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0], (_strip_protocol(
        _bytes_to_str(self._workers[0].target)), 3, True, True))
    self.assertEqual(self._worker_context[EVALUATOR][0],
                     ("fake_evaluator", 3, True, False))


class DistributeCoordinatorTestInpendentWorkerMode(
    DistributeCoordinatorTestBase):

  def testInGraph(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    threads = self._run_multiple_coordinator_in_threads(
        self._in_graph_worker_fn,
        cluster_spec,
        between_graph=False,
        mode=INDEPENDENT_WORKER)
    threads[WORKER][0].join()
    self.assertEqual(self._result_correct, 1)

  def testBetweenGraph(self):
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, num_ps=NUM_PS)
    threads = self._run_multiple_coordinator_in_threads(
        self._between_graph_worker_fn,
        cluster_spec,
        between_graph=True,
        mode=INDEPENDENT_WORKER)
    for task_id in range(NUM_WORKERS):
      threads[WORKER][task_id].join()

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  def testBetweenGraphContext(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          between_graph=True,
          rpc_layer=None)
      for task_id in range(NUM_WORKERS):
        threads[WORKER][task_id].join()

    # There is only one type of task and three such tasks.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue(WORKER in self._worker_context)
    self.assertEqual(len(self._worker_context[WORKER]), NUM_WORKERS)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context[WORKER][0],
        (_bytes_to_str(cluster_spec[WORKER][0]), NUM_WORKERS, True, True))
    self.assertEqual(
        self._worker_context[WORKER][1],
        (_bytes_to_str(cluster_spec[WORKER][1]), NUM_WORKERS, False, True))
    self.assertEqual(
        self._worker_context[WORKER][2],
        (_bytes_to_str(cluster_spec[WORKER][2]), NUM_WORKERS, False, True))

    # Make sure each worker runs a std server.
    self.assertEqual(len(self._std_servers), 1)
    self.assertTrue(WORKER in self._std_servers)
    self.assertEqual(len(self._std_servers[WORKER]), 3)
    self.assertFalse(self._std_servers[WORKER][0].joined)
    self.assertFalse(self._std_servers[WORKER][1].joined)
    self.assertFalse(self._std_servers[WORKER][2].joined)

  def testInGraphContext(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          between_graph=False,
          rpc_layer=None)
      for task_id in range(NUM_WORKERS):
        threads[WORKER][task_id].join()

    # There is only a "None" task in the dumped task context.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(
        self._worker_context["None"][0],
        (_bytes_to_str(cluster_spec[WORKER][0]), NUM_WORKERS, True, True))

    # Make sure each worker runs a std server.
    self.assertEqual(len(self._std_servers), 1)
    self.assertTrue(WORKER in self._std_servers)
    self.assertEqual(len(self._std_servers[WORKER]), 3)
    self.assertFalse(self._std_servers[WORKER][0].joined)
    self.assertTrue(self._std_servers[WORKER][1].joined)
    self.assertTrue(self._std_servers[WORKER][2].joined)

  def testInGraphContextWithEval(self):
    # Adds a EVALUATOR job.
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, has_eval=True)

    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          between_graph=False,
          rpc_layer=None)
      for task_id in range(NUM_WORKERS):
        threads[WORKER][task_id].join()
      threads[EVALUATOR][0].join()

    # There are one "None" task and one EVALUATOR task.
    self.assertEqual(len(self._worker_context), 2)
    self.assertTrue("None" in self._worker_context)
    self.assertTrue(EVALUATOR in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)
    self.assertEqual(len(self._worker_context[EVALUATOR]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0],
                     (_bytes_to_str(cluster_spec[WORKER][0]), 3, True, True))
    self.assertEqual(self._worker_context[EVALUATOR][0],
                     (cluster_spec[EVALUATOR][0], 3, True, False))

    # Make sure each worker runs a std server.
    self.assertEqual(len(self._std_servers), 2)
    self.assertTrue(WORKER in self._std_servers)
    self.assertTrue(EVALUATOR in self._std_servers)
    self.assertEqual(len(self._std_servers[WORKER]), 3)
    self.assertEqual(len(self._std_servers[EVALUATOR]), 1)
    self.assertFalse(self._std_servers[WORKER][0].joined)
    self.assertTrue(self._std_servers[WORKER][1].joined)
    self.assertTrue(self._std_servers[WORKER][2].joined)
    self.assertFalse(self._std_servers[EVALUATOR][0].joined)


if __name__ == "__main__":
  # TODO(yuefengz): find a smart way to terminite std server threads.
  with test.mock.patch.object(sys, "exit", os._exit):
    test.main()
