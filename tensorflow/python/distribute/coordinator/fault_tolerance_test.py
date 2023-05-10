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


from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.coordinator import fault_tolerance_test_base
from tensorflow.python.eager import test


class MultiWorkerFaultToleranceTest(
    fault_tolerance_test_base.BaseFaultToleranceTest, test.TestCase):
  """Multi worker fault tolerance tests.

  This covers the ordinary cases where multiple workers and PS are used.
  """

  def setUp(self):
    super(MultiWorkerFaultToleranceTest, self).setUp(2, 2)


class SingleWorkerFaultToleranceTest(
    fault_tolerance_test_base.BaseFaultToleranceTest, test.TestCase):
  """Single worker fault tolerance tests.

  This covers the cases that ensure training can continue in a single-worker
  cluster, even if the only worker can become unavailable at some point and
  recovered (if there are multiple workers, it is possible that the training
  succeeds with the workers that did not fail). Realistically single worker
  is very rarely used, but the tests are important to ensure the correct
  behaviors.
  """

  def setUp(self):
    super(SingleWorkerFaultToleranceTest, self).setUp(1, 1)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
