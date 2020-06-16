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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import io
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class IOTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):
    super(IOTest, self).setUp()
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "io_test")
    os.mkdir(tmpdir)
    self._test_dir = tmpdir

  def tearDown(self):
    super(IOTest, self).tearDown()
    shutil.rmtree(self._test_dir)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(compression=[None, "GZIP"])))
  def testBasic(self, compression):
    dataset = dataset_ops.Dataset.range(42)
    io.save(dataset, self._test_dir, compression=compression)
    dataset2 = io.load(
        self._test_dir, dataset.element_spec, compression=compression)
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


if __name__ == "__main__":
  test.main()
