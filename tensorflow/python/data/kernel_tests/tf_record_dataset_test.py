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
"""Tests for `tf.data.TFRecordDataset`."""
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
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class TFRecordDatasetTest(tf_record_test_base.TFRecordTestBase,
                          parameterized.TestCase):

  def _dataset_factory(self,
                       filenames,
                       compression_type="",
                       num_epochs=1,
                       batch_size=None):

    repeat_dataset = readers.TFRecordDataset(
        filenames, compression_type).repeat(num_epochs)
    if batch_size:
      return repeat_dataset.batch(batch_size)
    return repeat_dataset

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDatasetConstructorErrorsTensorInput(self):
    with self.assertRaisesRegex(TypeError,
                                "filenames.*must be.*Tensor.*string"):
      readers.TFRecordDataset([1, 2, 3])
    with self.assertRaisesRegex(TypeError,
                                "filenames.*must be.*Tensor.*string"):
      readers.TFRecordDataset(constant_op.constant([1, 2, 3]))
    # convert_to_tensor raises different errors in graph and eager
    with self.assertRaises(Exception):
      readers.TFRecordDataset(object())

  @combinations.generate(test_base.default_test_combinations())
  def testReadOneEpoch(self):
    # Basic test: read from file 0.
    dataset = self._dataset_factory(self._filenames[0])
    self.assertDatasetProduces(
        dataset,
        expected_output=[self._record(0, i) for i in range(self._num_records)])

    # Basic test: read from file 1.
    dataset = self._dataset_factory(self._filenames[1])
    self.assertDatasetProduces(
        dataset,
        expected_output=[self._record(1, i) for i in range(self._num_records)])

    # Basic test: read from both files.
    dataset = self._dataset_factory(self._filenames)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadTenEpochs(self):
    dataset = self._dataset_factory(self._filenames, num_epochs=10)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testReadTenEpochsOfBatches(self):
    dataset = self._dataset_factory(
        self._filenames, num_epochs=10, batch_size=self._num_records)
    expected_output = []
    for j in range(self._num_files):
      expected_output.append(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testReadZlibFiles(self):
    zlib_files = []
    for i, fn in enumerate(self._filenames):
      with open(fn, "rb") as f:
        cdata = zlib.compress(f.read())

        zfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.z" % i)
        with open(zfn, "wb") as f:
          f.write(cdata)
        zlib_files.append(zfn)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = self._dataset_factory(zlib_files, compression_type="ZLIB")
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadGzipFiles(self):
    gzip_files = []
    for i, fn in enumerate(self._filenames):
      with open(fn, "rb") as f:
        gzfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.gz" % i)
        with gzip.GzipFile(gzfn, "wb") as gzf:
          gzf.write(f.read())
        gzip_files.append(gzfn)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = self._dataset_factory(gzip_files, compression_type="GZIP")
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadWithBuffer(self):
    one_mebibyte = 2**20
    dataset = readers.TFRecordDataset(
        self._filenames, buffer_size=one_mebibyte)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadFromDatasetOfFiles(self):
    files = dataset_ops.Dataset.from_tensor_slices(self._filenames)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = readers.TFRecordDataset(files)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadTenEpochsFromDatasetOfFilesInParallel(self):
    files = dataset_ops.Dataset.from_tensor_slices(
        self._filenames).repeat(10)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = readers.TFRecordDataset(files, num_parallel_reads=4)
    self.assertDatasetProduces(
        dataset, expected_output=expected_output * 10, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetPathlib(self):
    files = [pathlib.Path(self._filenames[0])]

    expected_output = [self._record(0, i) for i in range(self._num_records)]
    ds = readers.TFRecordDataset(files)
    self.assertDatasetProduces(
        ds, expected_output=expected_output, assert_items_equal=True)


class TFRecordDatasetCheckpointTest(tf_record_test_base.TFRecordTestBase,
                                    checkpoint_test_base.CheckpointTestBase,
                                    parameterized.TestCase):

  def make_dataset(self,
                   num_epochs,
                   batch_size=1,
                   compression_type=None,
                   buffer_size=None):
    filenames = self._createFiles()
    if compression_type == "ZLIB":
      zlib_files = []
      for i, fn in enumerate(filenames):
        with open(fn, "rb") as f:
          cdata = zlib.compress(f.read())
          zfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.z" % i)
          with open(zfn, "wb") as f:
            f.write(cdata)
          zlib_files.append(zfn)
      filenames = zlib_files

    elif compression_type == "GZIP":
      gzip_files = []
      for i, fn in enumerate(self._filenames):
        with open(fn, "rb") as f:
          gzfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.gz" % i)
          with gzip.GzipFile(gzfn, "wb") as gzf:
            gzf.write(f.read())
          gzip_files.append(gzfn)
      filenames = gzip_files

    return readers.TFRecordDataset(
        filenames, compression_type,
        buffer_size=buffer_size).repeat(num_epochs).batch(batch_size)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(batch_size=[1, 5])))
  def testBatchSize(self, verify_fn, batch_size):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records // batch_size
    verify_fn(self,
              lambda: self.make_dataset(num_epochs, batch_size=batch_size),
              num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(buffer_size=[0, 5])))
  def testBufferSize(self, verify_fn, buffer_size):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    verify_fn(self,
              lambda: self.make_dataset(num_epochs, buffer_size=buffer_size),
              num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(compression_type=[None, "GZIP", "ZLIB"])))
  def testCompressionTypes(self, verify_fn, compression_type):
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    # pylint: disable=g-long-lambda
    verify_fn(
        self, lambda: self.make_dataset(
            num_epochs, compression_type=compression_type), num_outputs)


if __name__ == "__main__":
  test.main()
