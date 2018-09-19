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
import shutil
import tempfile

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops.dataset_ops import MatchingFilesDataset
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops.gen_io_ops import matching_files
from tensorflow.python.framework import errors

import os
import time
from functools import partial


try:
  import psutil  # pylint: disable=g-import-not-at-top

  psutil_import_succeeded = True
except ImportError:
  psutil_import_succeeded = False


def timeit(fn, msg, N=0):
  start = time.time()
  res = fn()
  end = time.time()
  runtime = (end - start) * 1000
  msg = '{}: time: {:.2f} ms'.format(msg, runtime)
  if N:
    msg += ' ({:.2f} ms per iteration)'.format(runtime / N)
  print(msg)
  return res


width = 10
depth = 2


class MatchingFilesDatasetTest(test.TestCase):

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)

  def _touchTempFiles(self, filenames):
    for filename in filenames:
      open(path.join(self.tmp_dir, filename), 'a').close()

  def testEmptyDirectory(self):
    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      itr = iterator_ops.Iterator.from_structure(dataset.output_types)
      init_op = itr.make_initializer(dataset)
      next_element = itr.get_next()
      sess.run(init_op)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSimpleDirectory(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      itr = iterator_ops.Iterator.from_structure(dataset.output_types)
      init_op = itr.make_initializer(dataset)
      next_element = itr.get_next()
      sess.run(init_op)

      full_filenames = []
      produced_filenames = []
      for filename in filenames:
        full_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))
      self.assertItemsEqual(full_filenames, produced_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testSimpleDirectoryInitializer(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = MatchingFilesDataset(filename_placeholder)

    with self.cached_session() as sess:
      itr = iterator_ops.Iterator.from_structure(dataset.output_types)
      init_op = itr.make_initializer(dataset)
      next_element = itr.get_next()
      sess.run(
        init_op,
        feed_dict={filename_placeholder: path.join(self.tmp_dir, '*')})

      full_filenames = []
      produced_filenames = []
      for filename in filenames:
        full_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(full_filenames, produced_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFileSuffixes(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = MatchingFilesDataset(filename_placeholder)

    with self.cached_session() as sess:
      itr = iterator_ops.Iterator.from_structure(dataset.output_types)
      init_op = itr.make_initializer(dataset)
      next_element = itr.get_next()
      sess.run(
        init_op,
        feed_dict={filename_placeholder: path.join(self.tmp_dir, '*.py')})

      full_filenames = []
      produced_filenames = []
      for filename in filenames[1:-1]:
        full_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))
      self.assertItemsEqual(full_filenames, produced_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def testFileMiddles(self):
    filenames = ['a.txt', 'b.py', 'c.pyc']
    self._touchTempFiles(filenames)

    filename_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    dataset = MatchingFilesDataset(filename_placeholder)

    with self.cached_session() as sess:
      itr = iterator_ops.Iterator.from_structure(dataset.output_types)
      init_op = itr.make_initializer(dataset)
      next_element = itr.get_next()
      sess.run(
        init_op,
        feed_dict={filename_placeholder: path.join(self.tmp_dir, '*.py*')})

      full_filenames = []
      produced_filenames = []
      for filename in filenames[1:]:
        full_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        produced_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(full_filenames, produced_filenames)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(itr.get_next())

  def _load_data(self):
    new_files = []
    dir = "/tmp/test/"
    if not os.path.exists(dir):
      os.makedirs(dir)
    base = tempfile.mkdtemp(prefix=dir)
    print('saving files to dir: {}'.format(base))
    for i in range(width):
      new_base = os.path.join(base, str(i), *[str(j) for j in range(depth - 1)])
      if not os.path.exists(new_base):
        os.makedirs(new_base)
      f = os.path.join(new_base, 'stuff.txt')
      new_files.append(compat.as_bytes(f))
      open(f, 'w').close()
    return base, new_files

  def _read_data(self, data, sess, N=1):
    for _ in range(N):
      sess.run(data)

  def _read_data_with_result(self, data, sess, N=1):
    result = []
    for _ in range(N):
      result.append(sess.run(data))
    return result

  def testPerformance(self):
    base, test_filenames = self._load_data()
    test_filenames.sort(reverse=True)
    patterns = array_ops.placeholder(dtypes.string, shape=[None])
    dataset = MatchingFilesDataset(patterns)
    iterator = iterator_ops.Iterator.from_structure(dataset.output_types)
    init_op = iterator.make_initializer(dataset)
    get_next = iterator.get_next()
    result = []
    with self.cached_session() as sess:
      search_patterns = [base + "/*/*/*.txt"]
      sess.run(init_op, feed_dict={patterns: search_patterns})
      result.extend(timeit(partial(self._read_data_with_result, get_next, sess),
        "read first filename"))
      result.extend(timeit(partial(self._read_data_with_result, get_next, sess),
        "read second filename"))
      N = width * len(search_patterns) - 2
      filename = timeit(partial(self._read_data_with_result, get_next, sess, N),
        'read {} more filenames'.format(N), N)
      result.extend(filename)

    matched_filenames = [compat.as_bytes(x) for x in result]
    for file in matched_filenames:
      print(file)
    self.assertItemsEqual(matched_filenames, test_filenames)


if __name__ == "__main__":
  test.main()
