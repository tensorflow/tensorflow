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
"""Multi-process runner tests for `Client` with `ParameterServerStrategyV2`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from absl import logging
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.client import client as client_lib
from tensorflow.python.distribute.client import utils
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class ClientMprTest(test.TestCase):

  def testScheduleTranslatePSFailureError(self):
    self._test_translate_ps_failure_error(test_schedule=True)

  def testJoinTranslatePSFailureError(self):
    self._test_translate_ps_failure_error(test_join=True)

  def _test_translate_ps_failure_error(self,
                                       test_schedule=False,
                                       test_join=False):

    def proc_func(functions_scheduled_event, test_finished_event):
      cluster_resolver = TFConfigClusterResolver()
      if cluster_resolver.task_type != "chief":
        utils.start_server(cluster_resolver, "grpc")
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          cluster_resolver)
      ps_client = client_lib.Client(strategy)

      with strategy.scope():
        v = variables.Variable(initial_value=0, dtype=dtypes.int32)

      @def_function.function
      def worker_fn():
        # An ever-running function.
        for _ in math_ops.range(100000):
          v.assign_add(1)

      # Keep the two workers occupied.
      ps_client.schedule(worker_fn)
      ps_client.schedule(worker_fn)
      # Now the main process can terminate.
      functions_scheduled_event.set()

      # Verified that join and schedule indeed raise
      # ParameterServerFailureError.
      try:
        if test_join:
          ps_client.join()
        if test_schedule:
          while ps_client.cluster._closure_queue._error is None:
            time.sleep(1)
          ps_client.schedule(worker_fn)
      except client_lib.ParameterServerFailureError:
        # The following verifies that after PS fails, continue executing
        # functions on workers should fail and indicate it's PS failure.
        for worker_id in range(3):
          with ops.device("/job:worker/replica:0/task:{}".format(worker_id)):
            try:
              # Executing a function after PS fails should result in a PS
              # failure.
              worker_fn()
            except Exception as e:  # pylint: disable=broad-except
              if client_lib._is_ps_failure(e):
                if worker_id < 2:
                  continue
                logging.info("_test_translate_ps_failure_error ends properly.")
                # Now we can safely exit the test.
                test_finished_event.set()
                return
            raise RuntimeError("Executing a function after PS fails, should "
                               "result in a PS failure.")

      raise RuntimeError("ParameterServerFailureError supposed to be raised.")

    manager = multi_process_runner.manager()
    functions_scheduled_event = manager.Event()
    test_finished_event = manager.Event()
    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=3, num_ps=1, has_eval=False),
        args=(functions_scheduled_event, test_finished_event),
        rpc_layer="grpc",
        list_stdout=True,
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


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
