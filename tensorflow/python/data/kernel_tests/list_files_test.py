# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.list_files()`."""

from os import path
import shutil
import tempfile

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ListFilesTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):
    super(ListFilesTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)
    super(ListFilesTest, self).tearDown()

  def _touchTempFiles(self, filenames):
    for filename in filenames:
      open(path.join(self.tmp_dir, filename), 'a').close()

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDirectory(self):
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             'No files matched'):
      dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
      # We need requires_initialization=True so that getNext uses
      # make_initializable_iterator instead of make_one_shot_iterator.
      # make_one_shot_iterator has an issue where it fails to capture control
      # dependencies when capturing the dataset, so it loses the assertion that
      # list_files matches at least one file.
      # TODO(b/140837601): Make this work with make_one_shot_iterator.
      self.getNext(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectory(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectoryNotShuffled(self):
    filenames = ['b', 'c', 'a']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=False)
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in sorted(filenames)
        ])

  def testFixedSeedResultsInRepeatableOrder(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    def dataset_fn():
      return dataset_ops.Dataset.list_files(
          path.join(self.tmp_dir, '*'), shuffle=True, seed=37)

    expected_filenames = [
        compat.as_bytes(path.join(self.tmp_dir, filename))
        for filename in filenames
    ]

    all_actual_filenames = []
    for _ in range(3):
      actual_filenames = []
      next_element = self.getNext(dataset_fn(), requires_initialization=True)
      try:
        while True:
          actual_filenames.append(self.evaluate(next_element()))
      except errors.OutOfRangeError:
        pass
      all_actual_filenames.append(actual_filenames)

    # Each run should produce the same set of filenames, which may be
    # different from the order of `expected_filenames`.
    self.assertCountEqual(expected_filenames, all_actual_filenames[0])
    # However, the different runs should produce filenames in the same order
    # as each other.
    self.assertEqual(all_actual_filenames[0], all_actual_filenames[1])
    self.assertEqual(all_actual_filenames[0], all_actual_filenames[2])

  @combinations.generate(test_base.default_test_combinations())
  def tesEmptyDirectoryInitializer(self):

    def dataset_fn():
      return dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))

    self.assertDatasetProduces(
        dataset_fn(),
        expected_error=(errors.InvalidArgumentError,
                        'No files matched pattern'),
        requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleDirectoryInitializer(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFileSuffixes(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*.py'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[1:-1]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFileMiddles(self):
    filenames = ['a.txt', 'b.py', 'c.pyc']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*.py*'))
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[1:]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNoShuffle(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    # Repeat the list twice and ensure that the order is the same each time.
    # NOTE(mrry): This depends on an implementation detail of `list_files()`,
    # which is that the list of files is captured when the iterator is
    # initialized. Otherwise, or if e.g. the iterator were initialized more than
    # once, it's possible that the non-determinism of `tf.matching_files()`
    # would cause this test to fail. However, it serves as a useful confirmation
    # that the `shuffle=False` argument is working as intended.
    # TODO(b/73959787): Provide some ordering guarantees so that this test is
    # more meaningful.
    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=False).repeat(2)
    next_element = self.getNext(dataset)

    expected_filenames = []
    actual_filenames = []
    for filename in filenames * 2:
      expected_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
      actual_filenames.append(compat.as_bytes(self.evaluate(next_element())))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    self.assertCountEqual(expected_filenames, actual_filenames)
    self.assertEqual(actual_filenames[:len(filenames)],
                     actual_filenames[len(filenames):])

  @combinations.generate(test_base.default_test_combinations())
  def testMultiplePatternsAsList(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    patterns = [path.join(self.tmp_dir, pat) for pat in ['*.py', '*.txt']]
    dataset = dataset_ops.Dataset.list_files(patterns)
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[:-1]
        ],
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testMultiplePatternsAsTensor(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(
        [path.join(self.tmp_dir, pat) for pat in ['*.py', '*.txt']])
    self.assertDatasetProduces(
        dataset,
        expected_output=[
            compat.as_bytes(path.join(self.tmp_dir, filename))
            for filename in filenames[:-1]
        ],
        assert_items_equal=True)


if __name__ == '__main__':
  test.main()
