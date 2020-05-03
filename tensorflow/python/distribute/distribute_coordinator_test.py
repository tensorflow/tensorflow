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
"""Tests for Distribute Coordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import json
import os
import sys
import threading
import time

import six

_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:  # pylint: disable=invalid-name
  _portpicker_import_error = _error
  portpicker = None

# pylint: disable=g-import-not-at-top
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_manager


CHIEF = distribute_coordinator._TaskType.CHIEF
WORKER = distribute_coordinator._TaskType.WORKER
PS = distribute_coordinator._TaskType.PS
EVALUATOR = distribute_coordinator._TaskType.EVALUATOR

STANDALONE_CLIENT = distribute_coordinator.CoordinatorMode.STANDALONE_CLIENT
INDEPENDENT_WORKER = distribute_coordinator.CoordinatorMode.INDEPENDENT_WORKER

NUM_WORKERS = 3
NUM_PS = 2

original_sys_exit = sys.exit


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


class MockExtended(object):

  def __init__(self,
               between_graph=False,
               should_init=None,
               should_checkpoint=None,
               should_save_summary=None):
    self.experimental_between_graph = between_graph
    self.experimental_should_init = should_init
    self.should_checkpoint = should_checkpoint
    self.should_save_summary = should_save_summary


class MockStrategy(object):

  def __init__(self,
               between_graph=False,
               should_init=None,
               should_checkpoint=None,
               should_save_summary=None):
    self.extended = MockExtended(between_graph, should_init, should_checkpoint,
                                 should_save_summary)

  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    if self.extended.experimental_should_init is None:
      if task_id == 0:
        self.extended.experimental_should_init = True
      else:
        self.extended.experimental_should_init = False
    if self.extended.should_checkpoint is None:
      if task_id == 0:
        self.extended.should_checkpoint = True
      else:
        self.extended.should_checkpoint = False
    if self.extended.should_save_summary is None:
      if task_id == 0:
        self.extended.should_save_summary = True
      else:
        self.extended.should_save_summary = False

    if session_config:
      if (cluster_spec and task_type and task_id is not None and
          self.extended.experimental_between_graph):
        session_config.intra_op_parallelism_threads += 1
        if task_type in ["chief", "worker"]:
          session_config.device_filters.extend(
              ["/job:%s/task:%d" % (task_type, task_id), "/job:ps"])
      else:
        session_config.inter_op_parallelism_threads += 1
        session_config.device_filters.append("/job:somejob")


class MockServer(object):

  def __init__(self):
    self._joined = False
    self._started = False

  def start(self):
    self._started = True

  def join(self):
    assert not self._joined
    self._joined = True

  @property
  def joined(self):
    return self._joined

  @property
  def started(self):
    return self._started


class DistributeCoordinatorTestBase(test.TestCase):

  @classmethod
  def setUpClass(cls):
    # We have to create a global in-process cluster because once an in-process
    # tensorflow server is created, there is no way to terminate it. Please see
    # multi_worker_test_base.py for more details.
    # TODO(yuefengz): use the utitliy from multi_worker_test_base.
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
    self._strategy_property = {}
    self._std_servers = {}
    self._barrier = distribute_coordinator._Barrier(NUM_WORKERS)
    self._coord = coordinator.Coordinator()

  @contextlib.contextmanager
  def _test_session(self, target):
    config = config_pb2.ConfigProto(allow_soft_placement=True)
    config.graph_options.optimizer_options.opt_level = -1
    with session.Session(graph=None, config=config, target=target) as sess:
      yield sess

  # TODO(yuefengz): use the utitliy from multi_worker_test_base.
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

  def _in_graph_worker_fn(self, strategy):
    context = distribute_coordinator_context.get_current_worker_context()
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

      self.evaluate(variables.global_variables_initializer())
      result_value = sess.run(result)
    self.assertEqual(result_value, expected)
    if result_value == expected:
      self._result_correct += 1

  def _wrapped_worker_fn(self, worker_fn):
    def wrapped(*args, **kwargs):
      with self._coord.stop_on_exception():
        return worker_fn(*args, **kwargs)
    return wrapped

  def _run_coordinator_in_thread(self, worker_fn, strategy, **kwargs):
    t = threading.Thread(
        target=distribute_coordinator.run_distribute_coordinator,
        args=(self._wrapped_worker_fn(worker_fn), strategy),
        kwargs=kwargs)
    t.start()
    return t

  def _run_multiple_coordinator_in_threads(self, worker_fn, strategy,
                                           cluster_spec, **kwargs):
    threads = {}
    for task_type in cluster_spec.keys():
      threads[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        t = self._run_coordinator_in_thread(
            worker_fn,
            strategy,
            cluster_spec=cluster_spec,
            task_type=task_type,
            task_id=task_id,
            **kwargs)
        threads[task_type].append(t)
    return threads

  def _join_threads(self, threads):
    try:
      self._coord.join(threads)
    except errors.UnknownError as e:
      if "Could not start gRPC server" in e.message:
        self.skipTest("Cannot start std servers.")
      else:
        raise

  def _between_graph_worker_fn(self, strategy):
    context = distribute_coordinator_context.get_current_worker_context()
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
        self.evaluate(variables.global_variables_initializer())

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

  def _between_graph_with_monitored_session(self, strategy):
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)
    with ops.device("/job:ps/task:0"):
      # TODO(yuefengz): investigate why not using resource variable will make
      # the test flaky.
      x = variable_scope.get_variable("xx", initializer=10.0, use_resource=True)
    with ops.device("/job:ps/task:1"):
      y = variable_scope.get_variable("yy", initializer=20.0, use_resource=True)

    x_add = x.assign_add(2.0)
    y_sub = y.assign_sub(2.0)
    train_op = control_flow_ops.group([x_add, y_sub])

    # The monitored session will run init or ready ops.
    with monitored_session.MonitoredSession() as sess:
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

  def _dump_worker_context(self, strategy):
    """Dumps the propoerties of each worker context.

    It dumps the context properties to a dict mapping from task_type to a list
    of tuples of master_target, num_workers, is_chief and distribute_mode, where
    the list is indexed by the task_id.

    Args:
      strategy: a `DistributionStrategy` object.
    """
    context = distribute_coordinator_context.get_current_worker_context()
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

  def _dump_strategy_property(self, strategy):
    context = distribute_coordinator_context.get_current_worker_context()
    self.assertTrue(context is not None)

    self.assertEqual(context._strategy.extended.experimental_should_init,
                     strategy.extended.experimental_should_init)
    self.assertEqual(context.should_checkpoint,
                     strategy.extended.should_checkpoint)
    self.assertEqual(context.should_save_summary,
                     strategy.extended.should_save_summary)

    task_type = str(context.task_type)
    task_id = context.task_id or 0
    with self._lock:
      if task_type not in self._strategy_property:
        self._strategy_property[task_type] = []
      while len(self._strategy_property[task_type]) <= task_id:
        self._strategy_property[task_type].append(None)
      self._strategy_property[task_type][task_id] = (
          context._strategy.extended.experimental_should_init,
          context.should_checkpoint,
          context.should_save_summary)

  def _run_mock_std_server(self,
                           session_config=None,
                           cluster_spec=None,
                           task_type=None,
                           task_id=None,
                           rpc_layer=None,
                           environment=None):
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


class DistributeCoordinatorTestStandaloneMode(DistributeCoordinatorTestBase):

  def testInGraphStandaloneMode(self):
    """Test it runs in-graph replication in standalone client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._in_graph_worker_fn,
        MockStrategy(between_graph=False),
        cluster_spec=self._cluster_spec)
    self.assertEqual(self._result_correct, 1)

  def testBetweenGraph(self):
    """Test it runs between-graph replication in standalone client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._between_graph_worker_fn,
        MockStrategy(between_graph=True),
        cluster_spec=self._cluster_spec)

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  @test_util.run_v1_only("MonitoredSession removed from v2")
  def testBetweenGraphWithMonitoredSession(self):
    """Test monitored session in standalone client mode."""
    distribute_coordinator.run_distribute_coordinator(
        self._between_graph_with_monitored_session,
        MockStrategy(between_graph=True),
        cluster_spec=self._cluster_spec)

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  def testBetweenGraphContext(self):
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=True),
        cluster_spec=self._cluster_spec)

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

  def testBetweenGraphStrategyProperties(self):
    # Dumps properties of the strategy objects.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_strategy_property,
        MockStrategy(between_graph=True, should_init=True),
        cluster_spec=self._cluster_spec)

    # There is only one type of task and there three such tasks.
    self.assertEqual(len(self._strategy_property), 1)
    self.assertTrue(WORKER in self._strategy_property)
    self.assertEqual(len(self._strategy_property[WORKER]), NUM_WORKERS)

    # Check whether each task has the right properties of should_init,
    # should_checkpoint and should_save_summary.
    self.assertEqual(self._strategy_property[WORKER][0], (True, True, True))
    self.assertEqual(self._strategy_property[WORKER][1], (True, False, False))
    self.assertEqual(self._strategy_property[WORKER][2], (True, False, False))

  def testInGraphContext(self):
    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=False),
        cluster_spec=self._cluster_spec)

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
        self._dump_worker_context,
        MockStrategy(between_graph=False),
        cluster_spec=None)

    # There is only a "None" task.
    self.assertEqual(len(self._worker_context), 1)
    self.assertTrue("None" in self._worker_context)
    self.assertEqual(len(self._worker_context["None"]), 1)

    # Check whether each task has the right master_target, num_workers, is_chief
    # and distributed_mode.
    self.assertEqual(self._worker_context["None"][0], ("", 0, True, False))

  def testBetweenGraphContextWithChief(self):
    # Adds a chief node, so there are NUM_WORKERS + 1 workers in total.
    cluster_spec = copy.deepcopy(self._cluster_spec)
    cluster_spec[CHIEF] = ["fake_chief"]

    # Dumps the task contexts to the self._worker_context dict.
    distribute_coordinator.run_distribute_coordinator(
        self._dump_worker_context,
        MockStrategy(between_graph=True),
        cluster_spec=cluster_spec,
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
        MockStrategy(between_graph=False),
        cluster_spec=cluster_spec,
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


class DistributeCoordinatorTestIndependentWorkerMode(
    DistributeCoordinatorTestBase):

  def testInGraph(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    threads = self._run_multiple_coordinator_in_threads(
        self._in_graph_worker_fn,
        MockStrategy(between_graph=False),
        cluster_spec,
        mode=INDEPENDENT_WORKER)
    self._join_threads([threads[WORKER][0]])
    self.assertEqual(self._result_correct, 1)

  def testBetweenGraph(self):
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, num_ps=NUM_PS)
    threads = self._run_multiple_coordinator_in_threads(
        self._between_graph_worker_fn,
        MockStrategy(between_graph=True),
        cluster_spec,
        mode=INDEPENDENT_WORKER)
    self._join_threads(threads[WORKER])

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  @test_util.run_v1_only("MonitoredSession removed from v2")
  def testBetweenGraphWithMonitoredSession(self):
    cluster_spec = self._create_cluster_spec(
        num_workers=NUM_WORKERS, num_ps=NUM_PS)
    threads = self._run_multiple_coordinator_in_threads(
        self._between_graph_with_monitored_session,
        MockStrategy(between_graph=True),
        cluster_spec,
        mode=INDEPENDENT_WORKER)
    self._join_threads(threads[WORKER])

    # Each finished worker will increment self._result_correct.
    self.assertEqual(self._result_correct, NUM_WORKERS)

  def testBetweenGraphContext(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          MockStrategy(between_graph=True),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])

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

  def testBetweenGraphStrategyProperties(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps properties of the strategy objects.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_strategy_property,
          MockStrategy(between_graph=True, should_init=True),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])

    # There is only one type of task and there three such tasks.
    self.assertEqual(len(self._strategy_property), 1)
    self.assertTrue(WORKER in self._strategy_property)
    self.assertEqual(len(self._strategy_property[WORKER]), NUM_WORKERS)

    # Check whether each task has the right properties of should_init,
    # should_checkpoint and should_save_summary.
    self.assertEqual(self._strategy_property[WORKER][0], (True, True, True))
    self.assertEqual(self._strategy_property[WORKER][1], (True, False, False))
    self.assertEqual(self._strategy_property[WORKER][2], (True, False, False))

  def testInGraphContext(self):
    cluster_spec = self._create_cluster_spec(num_workers=NUM_WORKERS)
    # Dumps the task contexts and std server arguments.
    with test.mock.patch.object(distribute_coordinator, "_run_std_server",
                                self._run_mock_std_server):
      threads = self._run_multiple_coordinator_in_threads(
          self._dump_worker_context,
          MockStrategy(between_graph=False),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])

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
          MockStrategy(between_graph=False),
          cluster_spec,
          mode=INDEPENDENT_WORKER,
          rpc_layer=None)
      self._join_threads(threads[WORKER])
      self._join_threads([threads[EVALUATOR][0]])

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

  def testRunStdServerInGoogleEnvironment(self):
    cluster_spec = {"worker": ["fake_worker"], "ps": ["localhost:0"]}
    tf_config = {"cluster": cluster_spec, "environment": "google"}

    joined = [False]

    def _fake_sleep(_):
      joined[0] = True
      original_sys_exit(0)

    def _thread_fn(cluster_spec):
      distribute_coordinator.run_distribute_coordinator(
          None,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="ps",
          task_id=0)

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            time, "sleep", _fake_sleep):
      t = threading.Thread(target=_thread_fn, args=(cluster_spec,))
      t.start()
      t.join()
    self.assertTrue(joined[0])

  def testRpcLayerEnvironmentVariable(self):
    cluster_spec = {"worker": ["fake_worker"], "ps": ["fake_ps"]}
    tf_config = {"cluster": cluster_spec, "rpc_layer": "cake"}

    rpc_layer_from_coordinator = [None]

    def _run_mock_server(cluster_spec=None,
                         task_type=None,
                         task_id=None,
                         session_config=None,
                         rpc_layer=None,
                         environment=None):
      del cluster_spec, task_type, task_id, session_config, environment
      rpc_layer_from_coordinator[0] = rpc_layer
      return MockServer()

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            distribute_coordinator, "_run_std_server", _run_mock_server):
      distribute_coordinator.run_distribute_coordinator(
          None,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="ps",
          task_id=0)
    self.assertEqual(rpc_layer_from_coordinator[0], "cake")


class StrategyConfigureTest(test.TestCase):

  def setUp(self):
    self._device_filters = []
    self._intra_op_parallelism_threads = None
    self._inter_op_parallelism_threads = None
    super(StrategyConfigureTest, self).setUp()

  def _dump_device_filters(self, *args, **kwargs):
    session_config = kwargs.get("session_config", None)
    self._device_filters.extend(session_config.device_filters)
    self._intra_op_parallelism_threads = (
        session_config.intra_op_parallelism_threads)
    self._inter_op_parallelism_threads = (
        session_config.inter_op_parallelism_threads)
    return MockServer()

  def _worker_fn(self, strategy):
    worker_context = distribute_coordinator_context.get_current_worker_context()
    session_config = worker_context._session_config
    self._device_filters.extend(session_config.device_filters)
    self._intra_op_parallelism_threads = (
        session_config.intra_op_parallelism_threads)
    self._inter_op_parallelism_threads = (
        session_config.inter_op_parallelism_threads)
    return MockServer()

  def test_session_config_in_std_server(self):
    cluster_spec = {"worker": ["fake_worker"], "ps": ["fake_ps"]}
    tf_config = {"cluster": cluster_spec}

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            distribute_coordinator, "_run_std_server",
            self._dump_device_filters):
      distribute_coordinator.run_distribute_coordinator(
          lambda _: None,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="worker",
          task_id=0)
    self.assertEqual(self._intra_op_parallelism_threads, 1)
    self.assertEqual(self._inter_op_parallelism_threads, 0)

  def test_session_config_in_session_creator(self):
    cluster_spec = {"worker": ["localhost:0"]}
    tf_config = {"cluster": cluster_spec}

    # Reset the saved Server state.
    distribute_coordinator._thread_local = threading.local()  # pylint: disable=protected-access

    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(tf_config)}):
      distribute_coordinator.run_distribute_coordinator(
          self._worker_fn,
          MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="worker",
          task_id=0)
    self.assertEqual(self._device_filters, ["/job:worker/task:0", "/job:ps"])
    self.assertEqual(self._intra_op_parallelism_threads, 2)
    self.assertEqual(self._inter_op_parallelism_threads, 0)

  def test_eval_strategy_configure(self):
    cluster_spec = {"evaluator": ["localhost:0"]}
    tf_config = {"cluster": cluster_spec}

    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(tf_config)}):
      distribute_coordinator.run_distribute_coordinator(
          lambda _: None,
          MockStrategy(between_graph=False),
          eval_fn=self._worker_fn,
          eval_strategy=MockStrategy(between_graph=True),
          mode=INDEPENDENT_WORKER,
          cluster_spec=cluster_spec,
          task_type="evaluator",
          task_id=0)
    self.assertEqual(self._device_filters, ["/job:somejob"])
    self.assertEqual(self._intra_op_parallelism_threads, 0)
    self.assertEqual(self._inter_op_parallelism_threads, 2)


class RunStandardTensorflowServerTest(test.TestCase):

  def test_std_server_arguments(self):
    cs = {"worker": ["fake_worker"], "ps": ["fake_ps"]}
    tf_config = {"cluster": cs, "task": {"type": "ps", "id": 0}}

    def _mock_run_std_server(cluster_spec=None,
                             task_type=None,
                             task_id=None,
                             session_config=None,
                             rpc_layer=None):
      self.assertEqual(cluster_spec.as_dict(), cs)
      self.assertEqual(task_type, "ps")
      self.assertEqual(task_id, 0)
      self.assertEqual(session_config.experimental.collective_group_leader,
                       "/job:worker/replica:0/task:0")
      self.assertEqual(session_config.intra_op_parallelism_threads, 1)
      self.assertEqual(rpc_layer, "grpc")

      return MockServer()

    with test.mock.patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), test.mock.patch.object(
            distribute_coordinator, "_run_std_server", _mock_run_std_server):
      session_config = config_pb2.ConfigProto()
      session_config.intra_op_parallelism_threads = 1
      mock_server = distribute_coordinator.run_standard_tensorflow_server(
          session_config)
      self.assertTrue(mock_server.started)


if __name__ == "__main__":
  # TODO(yuefengz): find a smart way to terminate std server threads.
  with test.mock.patch.object(sys, "exit", os._exit):
    # Reduce `recovery_wait_secs` from 30 seconds so the test completes quickly.
    orig_init = session_manager.SessionManager.__init__

    def new_init(*args, **kwargs):
      kwargs.pop("recovery_wait_secs", None)
      kwargs["recovery_wait_secs"] = 0.5
      orig_init(*args, **kwargs)

    session_manager.SessionManager.__init__ = new_init

    test.main()
