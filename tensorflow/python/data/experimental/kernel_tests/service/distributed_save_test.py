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
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# TODO(mpcallanan): Restructure this and snapshot_ft_test.py to share more.


class DistributedSaveTestBase:
  """Base class for setting up snapshot directories."""

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


class DistributedSaveTest(
    DistributedSaveTestBase,
    data_service_test_base.TestBase,
    parameterized.TestCase,
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_elements=[0, 10, 10000]),
      )
  )
  def testSaveLoad(self, num_workers, num_elements):
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()))
    _wait_for_snapshot(self._test_dir)

    dataset = dataset_ops.Dataset.load(self._test_dir)

    multiple_workers = num_workers > 1
    multiple_chunks = num_elements > 10
    ignore_order = multiple_workers or multiple_chunks
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=ignore_order
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[None, "AUTO", "GZIP"]),
      )
  )
  def testCompression(self, compression):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset,
        self._test_dir,
        cluster.dispatcher_address(),
        compression=compression,
    ))
    _wait_for_snapshot(self._test_dir)

    dataset = dataset_ops.Dataset.load(self._test_dir)
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasets(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(["a", "a", "a", "a", "a"]),
        dataset_ops.Dataset.from_tensor_slices(["b", "b", "b", "b", "b"]),
        dataset_ops.Dataset.from_tensor_slices(["c", "c", "c", "c", "c"]),
    ]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    ))
    _wait_for_snapshot(self._test_dir)

    dataset = dataset_ops.Dataset.load(self._test_dir)
    self.assertDatasetProduces(dataset, ["a", "b", "c"] * 5)

  @combinations.generate(test_base.default_test_combinations())
  def testLoadWithCustomReaderFunc(self):
    # TODO(b/250921378): Currently, all the unit tests only write one chunk
    # since the test dataset is small. The maximum chunk size is a C++ constant.
    # To test saving/loading multiple chunks in Python, we need a way to inject
    # the maximum chunk size. In this test, we simulate multiple chunks by
    # writing a snapshot and copying its output files.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    ))
    _wait_for_snapshot(self._test_dir)

    chunks_dir = os.path.join(self._test_dir, "chunks")
    files = os.listdir(chunks_dir)
    for i in range(2):
      for file in files:
        shutil.copy(
            os.path.join(chunks_dir, file),
            os.path.join(chunks_dir, f"{file}_{i}"),
        )

    def custom_reader_func(datasets):
      datasets = datasets.shuffle(3)
      return datasets.interleave(
          lambda x: x, num_parallel_calls=dataset_ops.AUTOTUNE
      )

    dataset = dataset_ops.Dataset.load(
        self._test_dir, reader_func=custom_reader_func
    )
    self.assertDatasetProduces(
        dataset, list(range(10)) * 3, assert_items_equal=True
    )

  @combinations.generate(test_base.default_test_combinations())
  def testDistributedLoad(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    ))
    _wait_for_snapshot(self._test_dir)

    dataset = dataset_ops.Dataset.load(self._test_dir)
    dataset = dataset.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
        )
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testDuplicateSnapshot(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "already started or completed"):
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, self._test_dir, cluster.dispatcher_address()))
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, self._test_dir, cluster.dispatcher_address()))

  @combinations.generate(test_base.default_test_combinations())
  def testWorkerFailure(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: array_ops.check_numerics(x, "message"))
    self.evaluate(distributed_save_op.distributed_save(
        dataset, self._test_dir, cluster.dispatcher_address()
    ))
    _wait_for_error(self._test_dir)

  @combinations.generate(test_base.default_test_combinations())
  def testBadDispatcherAddress(self):
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "must be a string"):
      self.evaluate(distributed_save_op.distributed_save(dataset, "", 1))
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      self.evaluate(distributed_save_op.distributed_save(dataset, "", ""))

  @combinations.generate(test_base.default_test_combinations())
  def testBadCardinality(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Saving an infinite dataset is not allowed",
    ):
      self.evaluate(distributed_save_op.distributed_save(
          dataset, self._test_dir, cluster.dispatcher_address()
      ))


class LoadCheckpointTest(
    DistributedSaveTestBase,
    data_service_test_base.TestBase,
    checkpoint_test_base.CheckpointTestBase,
    parameterized.TestCase,
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testLoadCheckpoint(self, verify_fn):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, self._test_dir, cluster.dispatcher_address()
        )
    )
    _wait_for_snapshot(self._test_dir)

    def _build_ds():
      return dataset_ops.Dataset.load(self._test_dir)

    verify_fn(self, _build_ds, num_outputs=10)


def _wait_for_snapshot(snapshot_path):
  while not os.path.exists(os.path.join(snapshot_path, "DONE")):
    time.sleep(0.1)


def _wait_for_error(snapshot_path):
  while not os.path.exists(os.path.join(snapshot_path, "ERROR")):
    time.sleep(0.1)


if __name__ == "__main__":
  test.main()
