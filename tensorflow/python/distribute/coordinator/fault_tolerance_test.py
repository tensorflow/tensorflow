# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fault tolerance test for parameter server training in TF2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import time

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib

_RPC_ERROR_FROM_WORKER = "GRPC error information from remote target /job:worker"
_RPC_ERROR_FROM_PS = "GRPC error information from remote target /job:ps"


class Model(object):

  def __init__(self, coordinator):
    self.cluster_coord = coordinator
    self.strategy = self.cluster_coord.strategy
    with self.cluster_coord.strategy.scope():
      self.build()

  def build(self):
    self.w = variables.Variable(
        initial_value=random_ops.random_uniform((1000, 1000)),
        dtype=dtypes.float32)
    self.iterations = variables.Variable(initial_value=0, dtype=dtypes.int32)

    def dataset_fn():
      data = random_ops.random_uniform((1000, 1000))
      dataset = dataset_ops.DatasetV2.from_tensors([data]).repeat()
      return dataset

    self.iterator = iter(
        self.cluster_coord.create_per_worker_dataset(dataset_fn))

  @def_function.function
  def train_fn(self, iterator):
    for _ in math_ops.range(5):
      x = math_ops.matmul(array_ops.squeeze(next(iterator)), self.w)
      x = math_ops.matmul(random_ops.random_uniform((1000, 1000)), x)
      self.w.assign_add(x)
    self.iterations.assign_add(1)

  def schedule_training_functions(self, num_steps):
    with self.strategy.scope():
      for _ in range(num_steps):
        self.cluster_coord.schedule(self.train_fn, args=(self.iterator,))

  def join_training_functions(self):
    self.cluster_coord.join()


class FaultToleranceTest(test.TestCase):  # pylint: disable=missing-docstring

  NUM_WORKERS = 2
  NUM_PS = 2

  def setUp(self):
    super(FaultToleranceTest, self).setUp()

    # Set the environment variable to prevent hanging upon job failure and
    # restart. Note that it defaults to 'use_caller' at Google, but defaults
    # to False in OSS.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    self._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=FaultToleranceTest.NUM_WORKERS,
        num_ps=FaultToleranceTest.NUM_PS,
        rpc_layer="grpc")
    self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
    self._cluster_def["chief"] = [
        "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    cluster_resolver = SimpleClusterResolver(
        server_lib.ClusterSpec(self._cluster_def), rpc_layer="grpc")

    # The strategy's constructor would connect to the cluster.
    self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cluster_resolver)
    self.cluster_coord = cluster_coordinator.ClusterCoordinator(self.strategy)

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[])

  def tearDown(self):
    super(FaultToleranceTest, self).tearDown()
    self._cluster.stop()

  def _restart(self, downtime_secs, job):
    """Kills `job` (index: 0) and restarts it after `downtime_secs`.

    Args:
      downtime_secs: secs before restarting the job.
      job: a string specifying the job to restart.
    """
    self._cluster.kill_task(job, 0)
    time.sleep(downtime_secs)
    self.assertFalse(context.check_alive("/job:%s/replica:0/task:0" % job))
    self._cluster.start_task(job, 0)
    while not context.check_alive("/job:%s/replica:0/task:0" % job):
      time.sleep(1)

  def _restart_in_thread(self, downtime_secs, restart_job):

    def _restart_fn():
      with self.thread_coord.stop_on_exception():
        self._restart(downtime_secs, restart_job)

    restart_thread = threading.Thread(target=_restart_fn)
    restart_thread.start()
    return restart_thread

  def testOneWorkerPreemption(self):
    # A blackbox test to make sure the model can still train when there is
    # worker preemption.
    model = Model(self.cluster_coord)
    model.schedule_training_functions(10)

    time.sleep(1)  # Let it run a couple steps.
    self.assertFalse(
        self.cluster_coord.done(), "cluster finishes work before restart, this"
        " is most likely due to the test runs in more powerful machine"
        " compared to the one it previously runs. This setup is brittle but"
        " there are no easy better alternatives. To fix the failure, consider"
        " adding more work to the cluster, e.g, scheduling more functions.")
    self._restart(5, "worker")

    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)

  def testOneWorkerPreemptionWithCancellation(self):

    @def_function.function
    def normal_function():
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(x, y))

    @def_function.function
    def error_function():
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      check_ops.assert_non_positive_v2(
          math_ops.reduce_sum(math_ops.matmul(x, y)))
      return x

    @def_function.function
    def long_function():
      x = random_ops.random_uniform((1000, 1000))
      for _ in math_ops.range(10000):
        a = random_ops.random_uniform((1000, 1000))
        b = random_ops.random_uniform((1000, 1000))
        x += math_ops.matmul(a, b)
      return x

    for _ in range(3):
      self.cluster_coord.schedule(normal_function)
    long_function_result = self.cluster_coord.schedule(long_function)
    self.cluster_coord.schedule(error_function)

    time.sleep(1)  # Let it run a couple steps.
    self._restart(1, "worker")

    with self.assertRaises(errors.InvalidArgumentError):
      self.cluster_coord.join()

    with self.assertRaises(errors.CancelledError):
      long_function_result.fetch()

    for _ in range(3):
      self.cluster_coord.schedule(normal_function)
    self.cluster_coord.join()

  def testHandleDatasetCreationFailure(self):
    model = Model(self.cluster_coord)

    restart_thread = self._restart_in_thread(5, "worker")

    model.schedule_training_functions(3)
    model.join_training_functions()

    self.thread_coord.join([restart_thread])
    self.assertGreaterEqual(model.iterations.numpy(), 3)

  def testWorkerPreemptionErrorType(self):

    @def_function.function
    def worker_train_fn():
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(x, y))

    def run_fn():
      with self.thread_coord.stop_on_exception():
        with ops.device("/job:worker/replica:0/task:0"):
          for _ in range(3):
            for _ in range(3):
              worker_train_fn()
            time.sleep(5)

    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)  # Let it run a couple steps.
    self._restart(2, "worker")

    try:
      self.thread_coord.join([run_thread])
    except errors.UnavailableError as e:
      logging.info("Got exception %r, error message is %s", e, e)

      self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))  # pylint: disable=g-assert-in-except
      self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))

      self.assertTrue("failed to connect to all addresses" in str(e) or
                      "Unable to find a context_id" in str(e) or
                      "Socket closed" in str(e) or
                      "Connection reset by peer" in str(e) or
                      "Transport closed" in str(e))

  def testWorkerPreemptionErrorTypeWithPythonFunction(self):

    def worker_train_fn():
      x = random_ops.random_uniform((2, 10))
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(x, y))

    def run_fn():
      with self.thread_coord.stop_on_exception():
        with ops.device("/job:worker/replica:0/task:0"):
          for _ in range(3):
            for _ in range(3):
              worker_train_fn()
            time.sleep(5)

    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)  # Let it run a couple steps.
    self._restart(2, "worker")

    try:
      self.thread_coord.join([run_thread])
    except errors.UnavailableError as e:
      logging.info("Got exception %r, error message is %s", e, e)

      self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))  # pylint: disable=g-assert-in-except
      self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))

      self.assertTrue("failed to connect to all addresses" in str(e) or
                      "Unable to find a context_id" in str(e) or
                      "Socket closed" in str(e) or
                      "Connection reset by peer" in str(e) or
                      "Transport closed" in str(e))

  def testPSPreemptionErrorType(self):

    with ops.device("/job:ps/replica:0/task:0"):
      v = variables.Variable(
          initial_value=random_ops.random_uniform((2, 10)),
          dtype=dtypes.float32)

    @def_function.function
    def worker_train_fn():
      y = random_ops.random_uniform((10, 2))
      return math_ops.reduce_mean(math_ops.matmul(v, y))

    def run_fn():
      with self.thread_coord.stop_on_exception():
        with ops.device("/job:worker/replica:0/task:0"):
          for _ in range(3):
            for _ in range(3):
              worker_train_fn()
            time.sleep(5)

    run_thread = threading.Thread(target=run_fn)
    run_thread.start()
    time.sleep(1)  # Let it run a couple steps.

    # Use a short restart delay to cover the case that RPC channel is reused
    self._restart(1, "ps")

    try:
      self.thread_coord.join([run_thread])
    except (errors.UnavailableError, errors.AbortedError) as e:
      logging.info("Got exception %r, error message is %s", e, e)
      self.assertIn(_RPC_ERROR_FROM_PS, str(e))  # pylint: disable=g-assert-in-except

      if isinstance(e, errors.UnavailableError):
        self.assertTrue("failed to connect to all addresses" in str(e) or
                        "Unable to find a context_id" in str(e) or
                        "Socket closed" in str(e) or
                        "Connection reset by peer" in str(e) or
                        "Transport closed" in str(e))

      if isinstance(e, errors.AbortedError):
        self.assertIn("RecvTensor expects a different device incarnation",
                      str(e))

  def testTwoWorkersPreempted(self):
    model = Model(self.cluster_coord)
    model.schedule_training_functions(10)

    time.sleep(1)
    self.assertFalse(self.cluster_coord.done())
    self._cluster.kill_task("worker", 0)
    self._cluster.kill_task("worker", 1)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:1"))
    self._cluster.start_task("worker", 0)
    self._cluster.start_task("worker", 1)
    time.sleep(2)
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:1"))

    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)

  def testWorkerContinuousFailure(self):
    model = Model(self.cluster_coord)
    model.schedule_training_functions(10)

    time.sleep(1)
    self.assertFalse(self.cluster_coord.done())
    self._cluster.kill_task("worker", 0)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self._cluster.start_task("worker", 0)
    time.sleep(2)
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))
    self._cluster.kill_task("worker", 0)
    time.sleep(2)
    self.assertFalse(context.check_alive("/job:worker/replica:0/task:0"))
    self._cluster.start_task("worker", 0)
    time.sleep(2)
    self.assertTrue(context.check_alive("/job:worker/replica:0/task:0"))

    model.join_training_functions()
    self.assertGreaterEqual(model.iterations.numpy(), 10)

  def testClusterStateNotDisrupted(self):
    # This test has side effects and can disrupt other tests, even if the
    # resource created by it will not be used in following tests.
    # TODO(b/155209534): enable this test.
    # self.testPSPreemptionErrorType()

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[])
    self.testOneWorkerPreemption()

    self.thread_coord = thread_coordinator.Coordinator(
        clean_stop_exception_types=[])
    self.testWorkerPreemptionErrorType()

    # In previous tests, workers may fail after training is done. But the
    # following tests start with creating resources where failure is not
    # handled.
    # TODO(b/153888707): enable the following two tests.
    # self.testTwoWorkersPreempted()
    # self.testWorkerContinuousFailure()


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
