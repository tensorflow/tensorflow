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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
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
  def testResourceOnWrongDevice(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    with ops.device(self._devices[0]):
      initializer = self.lookupTableInitializer("keyvaluetensor", [10, 11])
      table = lookup_ops.StaticHashTable(initializer, -1)
      self.evaluate(lookup_ops.tables_initializer())

    with ops.device(self._devices[1]):
      ds = dataset_ops.Dataset.range(3)
      ds = ds.map(table.lookup)
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Serialization error while trying to register a dataset"):
        ds = self.make_distributed_dataset(ds, cluster)
        self.getDatasetOutput(ds, requires_initialization=True)


if __name__ == "__main__":
  test.main()
