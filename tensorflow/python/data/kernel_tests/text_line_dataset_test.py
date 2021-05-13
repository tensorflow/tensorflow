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
"""Tests for `tf.data.TextLineDataset`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import pathlib
import zlib

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test
from tensorflow.python.util import compat


try:
  import psutil  # pylint: disable=g-import-not-at-top
  psutil_import_succeeded = True
except ImportError:
  psutil_import_succeeded = False


class TextLineDatasetTestBase(test_base.DatasetTestBase):
  """Base class for setting up and testing TextLineDataset."""

  def _lineText(self, f, l):
    return compat.as_bytes("%d: %d" % (f, l))

  def _createFiles(self,
                   num_files,
                   num_lines,
                   crlf=False,
                   compression_type=None):
    filenames = []
    for i in range(num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      contents = []
      for j in range(num_lines):
        contents.append(self._lineText(i, j))
        # Always include a newline after the record unless it is
        # at the end of the file, in which case we include it
        if j + 1 != num_lines or i == 0:
          contents.append(b"\r\n" if crlf else b"\n")
      contents = b"".join(contents)

      if not compression_type:
        with open(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "GZIP":
        with gzip.GzipFile(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "ZLIB":
        contents = zlib.compress(contents)
        with open(fn, "wb") as f:
          f.write(contents)
      else:
        raise ValueError("Unsupported compression_type", compression_type)

    return filenames


class TextLineDatasetTest(TextLineDatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression_type=[None, "GZIP", "ZLIB"])))
  def testTextLineDataset(self, compression_type):
    test_filenames = self._createFiles(
        2, 5, crlf=True, compression_type=compression_type)

    def dataset_fn(filenames, num_epochs, batch_size=None):
      repeat_dataset = readers.TextLineDataset(
          filenames, compression_type=compression_type).repeat(num_epochs)
      if batch_size:
        return repeat_dataset.batch(batch_size)
      return repeat_dataset

    # Basic test: read from file 0.
    expected_output = [self._lineText(0, i) for i in range(5)]
    self.assertDatasetProduces(
        dataset_fn([test_filenames[0]], 1), expected_output=expected_output)

    # Basic test: read from file 1.
    self.assertDatasetProduces(
        dataset_fn([test_filenames[1]], 1),
        expected_output=[self._lineText(1, i) for i in range(5)])

    # Basic test: read from both files.
    expected_output = [self._lineText(0, i) for i in range(5)]
    expected_output.extend(self._lineText(1, i) for i in range(5))
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 1), expected_output=expected_output)

    # Test repeated iteration through both files.
    expected_output = [self._lineText(0, i) for i in range(5)]
    expected_output.extend(self._lineText(1, i) for i in range(5))
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 10), expected_output=expected_output * 10)

    # Test batched and repeated iteration through both files.
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 10, 5),
        expected_output=[[self._lineText(0, i) for i in range(5)],
                         [self._lineText(1, i) for i in range(5)]] * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testTextLineDatasetParallelRead(self):
    test_filenames = self._createFiles(10, 10)
    files = dataset_ops.Dataset.from_tensor_slices(test_filenames).repeat(10)
    expected_output = []
    for j in range(10):
      expected_output.extend(self._lineText(j, i) for i in range(10))
    dataset = readers.TextLineDataset(files, num_parallel_reads=4)
    self.assertDatasetProduces(
        dataset, expected_output=expected_output * 10, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testTextLineDatasetBuffering(self):
    test_filenames = self._createFiles(2, 5, crlf=True)

    repeat_dataset = readers.TextLineDataset(test_filenames, buffer_size=10)
    expected_output = []
    for j in range(2):
      expected_output.extend([self._lineText(j, i) for i in range(5)])
    self.assertDatasetProduces(repeat_dataset, expected_output=expected_output)

  @combinations.generate(test_base.eager_only_combinations())
  def testIteratorResourceCleanup(self):
    filename = os.path.join(self.get_temp_dir(), "text.txt")
    with open(filename, "wt") as f:
      for i in range(3):
        f.write("%d\n" % (i,))
    first_iterator = iter(readers.TextLineDataset(filename))
    self.assertEqual(b"0", next(first_iterator).numpy())
    second_iterator = iter(readers.TextLineDataset(filename))
    self.assertEqual(b"0", next(second_iterator).numpy())
    # Eager kernel caching is based on op attributes, which includes the
    # Dataset's output shape. Create a different kernel to test that they
    # don't create resources with the same names.
    different_kernel_iterator = iter(
        readers.TextLineDataset(filename).repeat().batch(16))
    self.assertEqual([16], next(different_kernel_iterator).shape)
    # Remove our references to the Python Iterator objects, which (assuming no
    # reference cycles) is enough to trigger DestroyResourceOp and close the
    # partially-read files.
    del first_iterator
    del second_iterator
    del different_kernel_iterator
    if not psutil_import_succeeded:
      self.skipTest(
          "psutil is required to check that we've closed our files.")
    open_files = psutil.Process().open_files()
    self.assertNotIn(filename, [open_file.path for open_file in open_files])

  @combinations.generate(test_base.default_test_combinations())
  def testTextLineDatasetPathlib(self):
    files = self._createFiles(1, 5)
    files = [pathlib.Path(f) for f in files]

    expected_output = [self._lineText(0, i) for i in range(5)]
    ds = readers.TextLineDataset(files)
    self.assertDatasetProduces(
        ds, expected_output=expected_output, assert_items_equal=True)


class TextLineDatasetCheckpointTest(TextLineDatasetTestBase,
                                    checkpoint_test_base.CheckpointTestBase,
                                    parameterized.TestCase):

  def _build_iterator_graph(self, test_filenames, compression_type=None):
    return readers.TextLineDataset(
        test_filenames, compression_type=compression_type, buffer_size=10)

  @combinations.generate(test_base.default_test_combinations())
  def testTextLineCore(self):
    compression_types = [None, "GZIP", "ZLIB"]
    num_files = 5
    lines_per_file = 5
    num_outputs = num_files * lines_per_file
    for compression_type in compression_types:
      test_filenames = self._createFiles(
          num_files,
          lines_per_file,
          crlf=True,
          compression_type=compression_type)
      # pylint: disable=cell-var-from-loop
      self.run_core_tests(
          lambda: self._build_iterator_graph(test_filenames, compression_type),
          num_outputs)
      # pylint: enable=cell-var-from-loop


if __name__ == "__main__":
  test.main()
