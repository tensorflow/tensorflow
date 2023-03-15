# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.data.experimental.distributed_save."""

import os
import shutil
import tempfile
import time

from absl.testing import parameterized

import numpy as np
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# TODO(mpcallanan): Restructure this and snapshot_ft_test.py to share more.


class DistributedSaveTest(
    data_service_test_base.TestBase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self._test_dir = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()),
        "distributed_save_test",
    )

  def tearDown(self):
    super().tearDown()
    try:
      shutil.rmtree(self._test_dir)
    except FileNotFoundError:
      pass

  @combinations.generate(test_base.eager_only_combinations())
  def testSimple(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    )
    self._wait_for_snapshot(cluster)

    dataset = dataset_ops.Dataset.load(self._test_dir)
    self.assertDatasetProduces(dataset, list(range(10)))

  # TODO(mpcallanan): Add test for multiple workers.

  @combinations.generate(test_base.eager_only_combinations())
  def testChooseFromDatasets(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(["a", "a", "a", "a", "a"]),
        dataset_ops.Dataset.from_tensor_slices(["b", "b", "b", "b", "b"]),
        dataset_ops.Dataset.from_tensor_slices(["c", "c", "c", "c", "c"]),
    ]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    )
    self._wait_for_snapshot(cluster)

    dataset = dataset_ops.Dataset.load(self._test_dir)
    self.assertDatasetProduces(dataset, ["a", "b", "c"] * 5)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributedLoad(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    )
    self._wait_for_snapshot(cluster)

    dataset = dataset_ops.Dataset.load(self._test_dir)
    dataset = dataset.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
        )
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(test_base.eager_only_combinations())
  def testWorkerFailure(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: array_ops.check_numerics(x, "message"))
    distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    )
    self._wait_for_error(cluster)

  @combinations.generate(test_base.eager_only_combinations())
  def testBadDispatcherAddress(self):
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "must be a string"):
      distributed_save_op.distributed_save(dataset, "", 1)
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      distributed_save_op.distributed_save(dataset, "", "")

  @combinations.generate(test_base.eager_only_combinations())
  def testBadCardinality(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Saving an infinite dataset is not allowed",
    ):
      distributed_save_op.distributed_save(
          dataset, self._test_dir, cluster.dispatcher_address()
      )

  def _wait_for_snapshot(self, cluster):
    while not os.path.exists(os.path.join(self._test_dir, "DONE")):
      time.sleep(0.1)

  def _wait_for_error(self, cluster):
    while not os.path.exists(os.path.join(self._test_dir, "ERROR")):
      time.sleep(0.1)


if __name__ == "__main__":
  test.main()
