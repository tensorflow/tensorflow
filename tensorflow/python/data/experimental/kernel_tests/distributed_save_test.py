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

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


# TODO(mpcallanan): Restructure this and snapshot_ft_test.py to share more.

# Enum value for `SnapshotStreamInfo::DONE`.
_DONE = 4


class DistributedSaveTest(test_base.DatasetTestBase, parameterized.TestCase):

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


class DistributedSaveTfDataServiceTest(data_service_test_base.TestBase,
                                       DistributedSaveTest):

  def save(self, dataset):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    distributed_save_op.distributed_save(dataset, self._test_dir,
                                         cluster.dispatcher_address())
    return cluster

  @combinations.generate(test_base.eager_only_combinations())
  def testSimple(self):
    dataset = dataset_ops.Dataset.range(10)
    cluster = self.save(dataset)
    # TODO(b/250921378) Test loading.
    self.assertTrue(
        os.path.exists(os.path.join(self._test_dir, "snapshot.metadata")))
    streams = lambda: cluster.snapshot_streams(self._test_dir)
    while len(streams()) != 1 or streams()[0].state != _DONE:
      time.sleep(0.1)
    self.assertTrue(os.path.exists(os.path.join(self._test_dir, "DONE")))

  @combinations.generate(test_base.eager_only_combinations())
  def testChooseFromDatasets(self):
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(["a", "a", "a", "a", "a"]),
        dataset_ops.Dataset.from_tensor_slices(["b", "b", "b", "b", "b"]),
        dataset_ops.Dataset.from_tensor_slices(["c", "c", "c", "c", "c"]),
    ]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    self.save(dataset)
    # TODO(b/250921378) Test loading.
    self.assertTrue(
        os.path.exists(os.path.join(self._test_dir, "snapshot.metadata"))
    )

  @combinations.generate(test_base.eager_only_combinations())
  def testBadDispatcherAddress(self):
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "must be a string"):
      distributed_save_op.distributed_save(dataset, "", 1)
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      distributed_save_op.distributed_save(dataset, "", "")

  @combinations.generate(test_base.eager_only_combinations())
  def testBadCardinality(self):
    dataset = dataset_ops.Dataset.range(10).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Saving an infinite dataset is not allowed",
    ):
      self.save(dataset)


if __name__ == "__main__":
  test.main()
