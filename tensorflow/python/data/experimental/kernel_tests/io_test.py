# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `tf.data.experimental.{save,load}` operations."""
import os
import shutil

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import io
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IOTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):
    super(IOTest, self).setUp()
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "io_test")
    os.mkdir(tmpdir)
    self._test_dir = tmpdir

    self._checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    os.mkdir(self._checkpoint_prefix)
    self._save_dir = os.path.join(self.get_temp_dir(), "save")
    os.mkdir(self._save_dir)

  def tearDown(self):
    super(IOTest, self).tearDown()
    shutil.rmtree(self._test_dir)
    shutil.rmtree(self._checkpoint_prefix)
    shutil.rmtree(self._save_dir)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(compression=[None, "GZIP"])))
  def testBasic(self, compression):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._test_dir, compression=compression)
    dataset2 = io.load(
        self._test_dir, dataset.element_spec, compression=compression)
    self.assertDatasetProduces(dataset2, range(42))

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(pattern=["[1]", "[2", "3]", "?4", "5-6", "^7"])))
  def testDirContainsPattern(self, pattern):
    dataset = dataset_ops.Dataset.range(42)
    path = self._test_dir + "/inner" + pattern
    io.save(dataset, path)
    dataset2 = io.load(path, dataset.element_spec)
    self.assertDatasetProduces(dataset2, range(42))

  @combinations.generate(test_base.eager_only_combinations())
  def testCardinality(self):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._test_dir)
    dataset2 = io.load(self._test_dir, dataset.element_spec)
    self.assertEqual(self.evaluate(dataset2.cardinality()), 42)

  @combinations.generate(test_base.eager_only_combinations())
  def testCustomShardFunction(self):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._test_dir, shard_func=lambda x: x // 21)
    dataset2 = io.load(self._test_dir, dataset.element_spec)
    expected = []
    for i in range(21):
      expected.extend([i, i + 21])
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.eager_only_combinations())
  def testCustomReaderFunction(self):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._test_dir, shard_func=lambda x: x % 7)
    dataset2 = io.load(
        self._test_dir,
        dataset.element_spec,
        reader_func=lambda x: x.flat_map(lambda y: y))
    expected = []
    for i in range(7):
      expected.extend(range(i, 42, 7))
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(compression=[None, "GZIP"])))
  def testSaveInsideFunction(self, compression):

    dataset = dataset_ops.Dataset.range(42)

    @def_function.function
    def save_fn():
      io.save(dataset, self._test_dir, compression=compression)

    save_fn()
    dataset = io.load(
        self._test_dir, dataset.element_spec, compression=compression)
    self.assertDatasetProduces(dataset, range(42))

  @combinations.generate(test_base.eager_only_combinations())
  def testElementSpecOptional(self):
    range_dataset = dataset_ops.Dataset.range(42)
    dict_dataset = dataset_ops.Dataset.from_tensor_slices({"a": [1, 2],
                                                           "b": [3, 4]})
    tuple_dataset = dataset_ops.Dataset.from_tensor_slices(([1, 2], [3, 4]))
    dataset = dataset_ops.Dataset.zip((range_dataset, dict_dataset,
                                       tuple_dataset))
    io.save(dataset, self._test_dir)
    dataset_loaded = io.load(self._test_dir)
    self.assertDatasetsEqual(dataset, dataset_loaded)

  @combinations.generate(test_base.graph_only_combinations())
  def testElementSpecRequired(self):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._test_dir)
    with self.assertRaises(ValueError):
      _ = io.load(self._test_dir)

  @combinations.generate(test_base.eager_only_combinations())
  def testRepeatAndPrefetch(self):
    """This test reproduces github.com/tensorflow/tensorflow/issues/49165."""
    dataset1 = dataset_ops.Dataset.from_tensor_slices(np.random.rand(16, 32))
    io.save(dataset1, self._test_dir)
    dataset = io.load(self._test_dir)
    dataset = dataset.shuffle(buffer_size=16)
    dataset = dataset.batch(16)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    next_element = self.getNext(dataset)
    for _ in range(30):
      self.evaluate(next_element())


class LoadCheckpointTest(IOTest, checkpoint_test_base.CheckpointTestBase):

  def _build_ds(self):
    return io.load(self._save_dir)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._save_dir)
    verify_fn(self, self._build_ds, num_outputs=42)


class SaveCheckpointTest(IOTest, checkpoint_test_base.CheckpointTestBase):

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPI(self):
    dataset = dataset_ops.Dataset.range(40)
    checkpoint_args = {"directory": self._checkpoint_prefix, "max_to_keep": 50}
    io.save(dataset, self._save_dir, checkpoint_args=checkpoint_args)
    num_checkpoint_files = len(list(os.listdir(self._checkpoint_prefix)))
    # By default, we checkpoint every increment. Each checkpoint writes a
    # file containing the data and a file containing the index. There is
    # also an overall checkpoint file. Thus, we expect (2 * 40) + 1 files.
    self.assertEqual(81, num_checkpoint_files)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPICustomCheckpointInterval(self):
    dataset = dataset_ops.Dataset.range(40)
    step_counter = variables.Variable(0, trainable=False)
    checkpoint_args = {
        "checkpoint_interval": 5,
        "step_counter": step_counter,
        "directory": self._checkpoint_prefix,
        "max_to_keep": 10,
    }
    io.save(dataset, self._save_dir, checkpoint_args=checkpoint_args)
    num_checkpoint_files = len(list(os.listdir(self._checkpoint_prefix)))
    # We expect (2 * 8) + 1 files.
    self.assertEqual(17, num_checkpoint_files)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPIIncorrectArgs(self):
    dataset = dataset_ops.Dataset.range(42)
    checkpoint_args = {
        "directory": self._checkpoint_prefix,
        "incorrect_arg": "incorrect_arg"
    }
    with self.assertRaises(TypeError):
      io.save(dataset, self._save_dir, checkpoint_args=checkpoint_args)


if __name__ == "__main__":
  test.main()
