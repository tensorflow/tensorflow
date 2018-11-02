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
"""Tests for the experimental input pipeline ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
from os import makedirs
import shutil
import tempfile
import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ListFilesDatasetOpTest(test_base.DatasetTestBase):

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)

  def _touchTempFiles(self, filenames):
    for filename in filenames:
      open(path.join(self.tmp_dir, filename), 'a').close()

  def testEmptyDirectory(self):
    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSimpleDirectory(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()

      expected_filenames = []
      actual_filenames = []
      for filename in filenames:
        expected_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))
      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testSimpleDirectoryNotShuffled(self):
    filenames = ['b', 'c', 'a']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=False)
    with self.cached_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()

      for filename in sorted(filenames):
        self.assertEqual(compat.as_bytes(path.join(self.tmp_dir, filename)),
                         sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFixedSeedResultsInRepeatableOrder(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = dataset_ops.Dataset.list_files(
        path.join(self.tmp_dir, '*'), shuffle=True, seed=37)
    with self.cached_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()

      expected_filenames = [compat.as_bytes(path.join(self.tmp_dir, filename))
                            for filename in filenames]

      all_actual_filenames = []
      for _ in range(3):
        actual_filenames = []
        sess.run(itr.initializer)
        try:
          while True:
            actual_filenames.append(sess.run(next_element))
        except errors.OutOfRangeError:
          pass
        all_actual_filenames.append(actual_filenames)

      # Each run should produce the same set of filenames, which may be
      # different from the order of `expected_filenames`.
      self.assertItemsEqual(expected_filenames, all_actual_filenames[0])
      # However, the different runs should produce filenames in the same order
      # as each other.
      self.assertEqual(all_actual_filenames[0], all_actual_filenames[1])
      self.assertEqual(all_actual_filenames[0], all_actual_filenames[2])

  def testEmptyDirectoryInitializer(self):
    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.cached_session() as sess:
      itr = dataset.make_initializable_iterator()
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError, 'No files matched pattern: '):
        sess.run(
            itr.initializer,
            feed_dict={filename_placeholder: path.join(self.tmp_dir, '*')})

  def testSimpleDirectoryInitializer(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.cached_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*')})

      expected_filenames = []
      actual_filenames = []
      for filename in filenames:
        expected_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFileSuffixes(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.cached_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*.py')})

      expected_filenames = []
      actual_filenames = []
      for filename in filenames[1:-1]:
        expected_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))
      self.assertItemsEqual(expected_filenames, actual_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFileMiddles(self):
    filenames = ['a.txt', 'b.py', 'c.pyc']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = dataset_ops.Dataset.list_files(filename_placeholder)

    with self.cached_session() as sess:
      itr = dataset.make_initializable_iterator()
      next_element = itr.get_next()
      sess.run(
          itr.initializer,
          feed_dict={filename_placeholder: path.join(self.tmp_dir, '*.py*')})

      expected_filenames = []
      actual_filenames = []
      for filename in filenames[1:]:
        expected_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

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
    with self.cached_session() as sess:
      itr = dataset.make_one_shot_iterator()
      next_element = itr.get_next()

      expected_filenames = []
      actual_filenames = []
      for filename in filenames * 2:
        expected_filenames.append(
            compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())
      self.assertItemsEqual(expected_filenames, actual_filenames)
      self.assertEqual(actual_filenames[:len(filenames)],
                       actual_filenames[len(filenames):])


class ListFilesDatasetBenchmark(test.Benchmark):

  def benchmarkNestedDirectories(self):
    tmp_dir = tempfile.mkdtemp()
    width = 1024
    depth = 16
    for i in range(width):
      for j in range(depth):
        new_base = path.join(tmp_dir, str(i),
                             *[str(dir_name) for dir_name in range(j)])
        makedirs(new_base)
        child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
        for f in child_files:
          filename = path.join(new_base, f)
          open(filename, 'w').close()

    patterns = [
        path.join(tmp_dir, path.join(*['**' for _ in range(depth)]), suffix)
        for suffix in ['*.txt', '*.log']
    ]

    deltas = []
    iters = 3
    for _ in range(iters):
      with ops.Graph().as_default():
        dataset = dataset_ops.Dataset.list_files(patterns)
        next_element = dataset.make_one_shot_iterator().get_next()

        with session.Session() as sess:
          sub_deltas = []
          while True:
            try:
              start = time.time()
              sess.run(next_element)
              end = time.time()
              sub_deltas.append(end - start)
            except errors.OutOfRangeError:
              break
          deltas.append(sub_deltas)

    median_deltas = np.median(deltas, axis=0)
    print('Nested directory size (width*depth): %d*%d Median wall time: '
          '%fs (read first filename), %fs (read second filename), avg %fs'
          ' (read %d more filenames)' %
          (width, depth, median_deltas[0], median_deltas[1],
           np.average(median_deltas[2:]), len(median_deltas) - 2))
    self.report_benchmark(
        iters=iters,
        wall_time=np.sum(median_deltas),
        extras={
            'read first file:':
                median_deltas[0],
            'read second file:':
                median_deltas[1],
            'avg time for reading %d more filenames:' %
            (len(median_deltas) - 2):
                np.average(median_deltas[2:])
        },
        name='benchmark_list_files_dataset_nesteddirectory(%d*%d)' %
        (width, depth))

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
  test.main()
