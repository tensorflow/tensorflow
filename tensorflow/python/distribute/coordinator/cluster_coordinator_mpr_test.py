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
"""Multi-process runner tests for `ClusterCoordinator` with PSv2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


class ClusterCoordinatorMprTest(test.TestCase):

  # TODO(b/168772720): Merge or remove the following task failure tests once
  # MultiProcessCluster is made available in OSS.
  def testStrategyRun_withWorkerFailures(self):
    self._testStrategyRun("worker")

  def testStrategyRun_withPsFailures(self):
    self._testStrategyRun("ps")

  def testStrategyRun_withoutFailures(self):
    self._testStrategyRun(None)

  def _testStrategyRun(self, failure_task_type):

    def fn(functions_scheduled_event):
      # TODO(b/170664373): This is needed for TF2 parameter server training in
      # OSS. Remove this when resolved.
      os.environ["GRPC_FAIL_FAST"] = "use_caller"

      cluster_resolver = TFConfigClusterResolver()
      if cluster_resolver.task_type != "chief":
        utils.start_server(cluster_resolver, "grpc")
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          cluster_resolver)
      ps_client = coordinator_lib.ClusterCoordinator(strategy)

      with strategy.scope():
        v = variables.Variable(initial_value=1)

        @def_function.function
        def worker_fn(input_tensor):

          def replica_fn(input_tensor):
            return input_tensor + v

          run_result = strategy.run(replica_fn, args=(input_tensor,))
          check_ops.assert_equal_v2(run_result, 4)
          return run_result

      for i in range(5000):
        if i % 500 == 0:
          logging.info("Scheduling function-{}...".format(i))
        result = ps_client.schedule(worker_fn, args=(constant_op.constant(3),))
      functions_scheduled_event.set()
      logging.info("Joining...")
      ps_client.join()
      logging.info("Finished joining.")
      if result.fetch() != 4:
        raise AssertionError("Unexpected RemoteValue result: {}".format(
            result.fetch()))
      logging.info("testStrategyRun succeeded")

    manager = multi_process_runner.manager()
    functions_scheduled_event = manager.Event()
    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1, num_ps=1, has_eval=False),
        args=(functions_scheduled_event,),
        rpc_layer="grpc",
        return_output=True)
    mpr.start()

    if failure_task_type is not None:
      functions_scheduled_event.wait()
      logging.info("Before interrupting {}-0.".format(failure_task_type))
      mpr.terminate(failure_task_type, 0)

      if failure_task_type == "ps":
        with self.assertRaises(errors.UnavailableError):
          mpr.join()
        return

      time.sleep(10)
      logging.info("Before restarting {}-0.".format(failure_task_type))
      mpr.start_single_process(task_type="worker", task_id=0)

    self.assertTrue(
        any(["testStrategyRun succeeded" in msg for msg in mpr.join().stdout]))

  def testScheduleTranslatePSFailureError(self):
    self._test_translate_ps_failure_error(test_schedule=True)

  def testJoinTranslatePSFailureError(self):
    self._test_translate_ps_failure_error(test_join=True)

  def _test_translate_ps_failure_error(self,
                                       test_schedule=False,
                                       test_join=False):

    def fn(functions_scheduled_event, test_finished_event):
      # TODO(b/170664373): This is needed for TF2 parameter server training in
      # OSS. Remove this when resolved.
      os.environ["GRPC_FAIL_FAST"] = "use_caller"

      cluster_resolver = TFConfigClusterResolver()
      if cluster_resolver.task_type != "chief":
        utils.start_server(cluster_resolver, "grpc")
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          cluster_resolver)
      ps_coordinator = coordinator_lib.ClusterCoordinator(strategy)

      with strategy.scope():
        v = variables.Variable(initial_value=0, dtype=dtypes.int32)

      @def_function.function
      def worker_fn():
        # An ever-running function.
        for _ in math_ops.range(100000):
          v.assign_add(1)

      # Keep the two workers occupied.
      ps_coordinator.schedule(worker_fn)
      ps_coordinator.schedule(worker_fn)
      # Now the main process can terminate.
      functions_scheduled_event.set()

      # Verified that join and schedule indeed raise UnavailableError.
      try:
        if test_join:
          ps_coordinator.join()
        if test_schedule:
          while ps_coordinator.cluster._closure_queue._error is None:
            time.sleep(1)
          ps_coordinator.schedule(worker_fn)
      except errors.UnavailableError:
        # The following verifies that after PS fails, continue executing
        # functions on workers should fail and indicate it's PS failure.
        for worker_id in range(3):
          with ops.device("/job:worker/replica:0/task:{}".format(worker_id)):
            try:
              # Executing a function after PS fails should result in a PS
              # failure.
              worker_fn()
            except Exception as e:  # pylint: disable=broad-except
              if coordinator_lib._is_ps_failure(e):
                if worker_id < 2:
                  continue
                logging.info("_test_translate_ps_failure_error ends properly.")
                # Now we can safely exit the test.
                test_finished_event.set()
                return
            raise RuntimeError("Executing a function after PS fails, should "
                               "result in a PS failure.")

      raise RuntimeError("UnavailableError supposed to be raised.")

    manager = multi_process_runner.manager()
    functions_scheduled_event = manager.Event()
    test_finished_event = manager.Event()
    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=3, num_ps=1, has_eval=False),
        args=(functions_scheduled_event, test_finished_event),
        rpc_layer="grpc",
        return_output=True,
        use_dill_for_args=False)

    mpr.start()
    functions_scheduled_event.wait()
    mpr.terminate("ps", 0)
    while mpr.process_exists("ps", 0):
      time.sleep(0.01)
    test_finished_event.wait()
    self.assertTrue(
        any("_test_translate_ps_failure_error ends properly" in msg
            for msg in mpr.join().stdout))

  def test_numpy_fetched_after_worker_failure(self):

    def fn(first_fetch_occurred_event, worker_terminated_event):
      os.environ["GRPC_FAIL_FAST"] = "use_caller"

      cluster_resolver = TFConfigClusterResolver()
      if cluster_resolver.task_type != "chief":
        utils.start_server(cluster_resolver, "grpc")
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          cluster_resolver)
      ps_coordinator = coordinator_lib.ClusterCoordinator(strategy)

      with strategy.scope():
        v = variables.Variable(initial_value=0, dtype=dtypes.int32)

      @def_function.function
      def worker_fn():
        return v + 1, v - 1

      remote_value = ps_coordinator.schedule(worker_fn)
      logging.info("result (1st fetch): %r", remote_value.fetch())
      first_fetch_occurred_event.set()
      worker_terminated_event.wait()
      logging.info("result (2nd fetch): %r", remote_value.fetch())

    manager = multi_process_runner.manager()
    first_fetch_occurred_event = manager.Event()
    worker_terminated_event = manager.Event()
    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1, num_ps=1, has_eval=False),
        args=(first_fetch_occurred_event, worker_terminated_event),
        rpc_layer="grpc",
        return_output=True,
        use_dill_for_args=False)

    mpr.start()
    first_fetch_occurred_event.wait()
    mpr.terminate("worker", 0)
    worker_terminated_event.set()
    self.assertTrue(
        any("result (2nd fetch)" in msg for msg in mpr.join().stdout))


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
