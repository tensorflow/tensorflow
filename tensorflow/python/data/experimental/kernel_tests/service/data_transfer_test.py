# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.data service alternate data transfer protocols."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class DataTransferTest(
    data_service_test_base.TestBase,
    parameterized.TestCase,
):

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(
              data_transfer_protocol=[
                  "good",
                  "bad_with_primary_fallback",
                  "bad_with_secondary_fallback",
              ]
          ),
      )
  )
  def testAltProtocolSucceeds(self, data_transfer_protocol):
    cluster = self.make_test_cluster(
        num_workers=1,
        data_transfer_protocol=data_transfer_protocol,
    )
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        data_transfer_protocol=data_transfer_protocol,
    )
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testAltProtocolFailsNotRegistered(self):
    cluster = self.make_test_cluster(
        num_workers=1,
        data_transfer_protocol="grpc",
    )
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        data_transfer_protocol="unknown",
    )
    with self.assertRaisesRegex(errors.NotFoundError, "not available"):
      self.evaluate(self.getNext(ds)())


if __name__ == "__main__":
  test.main()
