# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests tf.data service cluster with local and remote workers."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations


class MultiProcessClusterTest(data_service_test_base.TestBase,
                              parameterized.TestCase):
  """Verifies the local and remote workers are running and producing data."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[0, 1, 3], num_remote_workers=[0, 1, 3])))
  def testCluster(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    num_workers = num_local_workers + num_remote_workers
    if num_workers == 0:
      return
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(
        dataset,
        num_workers * list(range(num_elements)),
        assert_items_equal=True)


if __name__ == "__main__":
  multi_process_cluster.test_main()
