# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Fault tolerance tests for coordination service-based failure handling."""

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.distribute.coordinator import fault_tolerance_test
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test


class BaseCoordinationServiceTest(fault_tolerance_test.BaseFaultToleranceTest):
  """Modify some tests to have stronger checks."""

  def setUp(self, num_workers, num_ps):
    super().setUp(num_workers=num_workers, num_ps=num_ps, use_cs=True)

  def testJoinRaisesUnavailableErrorAtPsFailure(self):
    self._run_and_kill_ps_task()
    with self.assertRaises(cluster_coordinator.PSUnavailableError):
      self.cluster_coord.join()

  def testScheduleRaisesUnavailableErrorAtPsFailure(self):
    self._run_and_kill_ps_task()
    with self.assertRaises(cluster_coordinator.PSUnavailableError):
      self.cluster_coord.schedule(def_function.function(lambda: None))


class SingleWorkerCoordinationServiceTest(
    BaseCoordinationServiceTest, test.TestCase
):

  def setUp(self):
    super().setUp(num_workers=1, num_ps=1)


class MultiWorkerCoordinationServiceTest(
    BaseCoordinationServiceTest, test.TestCase
):

  def setUp(self):
    super().setUp(num_workers=2, num_ps=2)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
