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
"""Multi-device tests for tf.data service ops."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.platform import test


class MultiDeviceTest(data_service_test_base.TestBase, parameterized.TestCase):

  def setUp(self):
    super(MultiDeviceTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(2)

  @combinations.generate(test_base.default_test_combinations())
  def testReadDatasetOnDifferentDevices(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    with ops.device(self._devices[0]):
      dataset = dataset_ops.Dataset.range(num_elements)
      element_spec = dataset.element_spec
      dataset_id = data_service_ops.register_dataset(
          cluster.dispatcher_address(), dataset)
      dataset = data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher_address(),
          dataset_id=dataset_id,
          element_spec=element_spec)
      self.assertDatasetProduces(dataset, list(range(num_elements)))

    with ops.device(self._devices[1]):
      dataset = data_service_ops.from_dataset_id(
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher_address(),
          dataset_id=dataset_id,
          element_spec=dataset.element_spec)
      self.assertDatasetProduces(dataset, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testResourceOnWrongDevice(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    with ops.device(self._devices[0]):
      initializer = self.lookupTableInitializer("keyvaluetensor", [10, 11])
      table = lookup_ops.StaticHashTable(initializer, -1)
      self.evaluate(lookup_ops.tables_initializer())
      dataset = dataset_ops.Dataset.range(3)
      dataset = dataset.map(table.lookup)
      dataset = self.make_distributed_dataset(dataset, cluster)
      self.assertDatasetProduces(
          dataset, [10, 11, -1], requires_initialization=True)

    with ops.device(self._devices[1]):
      dataset = dataset_ops.Dataset.range(3)
      dataset = dataset.map(table.lookup)
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Serialization error while trying to register a dataset"):
        dataset = self.make_distributed_dataset(dataset, cluster)
        self.getDatasetOutput(dataset, requires_initialization=True)


if __name__ == "__main__":
  test.main()
