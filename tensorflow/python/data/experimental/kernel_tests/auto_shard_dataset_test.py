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
"""Tests for the private `_AutoShardDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


def chunk(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]


class AutoShardDatasetTest(reader_dataset_ops_test_base.TFRecordDatasetTestBase,
                           parameterized.TestCase):

  def setUp(self):
    super(AutoShardDatasetTest, self).setUp()
    self._num_files = 10
    self._num_records = 10
    self.test_filenames = self._createFiles()

  def getAllDatasetElements(self, dataset):
    actual = []
    next_fn = self.getNext(dataset)
    while True:
      try:
        actual.append(self.evaluate(next_fn()))
      except errors.OutOfRangeError:
        break
    return actual

  def assertDatasetProducesWithShuffle(self, dataset, expected, batch,
                                       num_examples, shuffle):
    if shuffle:
      actual = []
      next_fn = self.getNext(dataset)
      for _ in range(num_examples):
        elem = self.evaluate(next_fn())
        if isinstance(elem, tuple):
          actual.extend(elem)
        else:
          actual.extend(elem.tolist())

      self.assertCountEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_fn())
    else:
      self.assertDatasetProduces(dataset, list(chunk(expected, batch)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testFlatMapReaderPipeline(self, shuffle):
    dataset = dataset_ops.Dataset.list_files(
        self.test_filenames, shuffle=shuffle)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in (3, 8)
        for r in range(0, 10)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)

  @combinations.generate(test_base.default_test_combinations())
  def testZipReaderPipeline(self):
    dataset1 = dataset_ops.Dataset.list_files(
        self.test_filenames, shuffle=False)
    dataset1 = dataset1.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset2 = dataset_ops.Dataset.list_files(
        self.test_filenames, shuffle=False)
    dataset2 = dataset2.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))

    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = distribute._AutoShardDataset(dataset, 5, 3)

    expected = [
        (b"Record %d of file %d" % (r, f), b"Record %d of file %d" % (r, f))  # pylint:disable=g-complex-comprehension
        for r in range(0, 10)
        for f in (3, 8)
    ]

    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testConcatenateReaderPipeline(self, shuffle):
    dataset1 = dataset_ops.Dataset.list_files(
        self.test_filenames, shuffle=shuffle)
    dataset1 = dataset1.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset1 = dataset1.batch(5)
    dataset2 = dataset_ops.Dataset.list_files(
        self.test_filenames, shuffle=shuffle)
    dataset2 = dataset2.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset2 = dataset2.batch(5)

    dataset = dataset1.concatenate(dataset2)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for r in range(0, 10)
        for f in (3, 8)
    ]
    expected += expected
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 8, shuffle)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testPipelineWithMap(self, shuffle):
    dataset = dataset_ops.Dataset.list_files(self.test_filenames, shuffle=False)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.map(lambda x: string_ops.substr_v2(x, 2, 1000))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)

    expected = [
        b"cord %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)

  @combinations.generate(test_base.default_test_combinations())
  def testDirectFilenameTFRecordReaderPipeline(self):
    dataset = core_readers.TFRecordDataset(self.test_filenames)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in (0, 5)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testValidPipelineWithRangeDataset(self, shuffle):
    dataset = dataset_ops.Dataset.range(self._num_files)
    dataset = dataset.map(lambda n: string_ops.string_join(  # pylint:disable=g-long-lambda
        [self.get_temp_dir(),
         string_ops.string_format("/tf_record.{}.txt", [n])]))
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.map(lambda x: string_ops.substr_v2(x, 2, 1000))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)

    expected = [
        b"cord %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(params=[(1, 0, 10, 10), (2, 1, 20, 5),
                                       (10, 1, 1, 10)])))
  def testStandardReaderPipeline(self, params):
    num_epochs, index, batch_size, parallel_reads = params
    dataset = readers.make_tf_record_dataset(
        file_pattern=self.test_filenames,
        num_epochs=num_epochs,
        batch_size=batch_size,
        parser_fn=None,
        num_parallel_reads=parallel_reads,
        drop_final_batch=True,
        shuffle=False)
    dataset = distribute._AutoShardDataset(dataset, 2, index)
    outputs = self.getNext(dataset)
    self._verify_records(
        outputs,
        batch_size=batch_size,
        file_index=[i for i in range(index, self._num_records, 2)],
        num_epochs=num_epochs,
        interleave_cycle_length=parallel_reads,
        drop_final_batch=True,
        use_parser_fn=None)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(outputs())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle=[True, False])))
  def testSampleResNetPipeline(self, shuffle):
    dataset = dataset_ops.Dataset.list_files(
        self.test_filenames, shuffle=shuffle)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for r in range(0, 10)
        for f in (3, 8)
    ]
    self.assertDatasetProducesWithShuffle(dataset, expected, 5, 4, shuffle)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.times(combinations.combine(
              sharding_policy=[distribute_options.AutoShardPolicy.DATA,
                               distribute_options.AutoShardPolicy.FILE]),
                             combinations.combine(shuffle=[True, False]))))
  def testReplicateAndShardProduceDisjointData(self, shuffle, sharding_policy):
    dataset = dataset_ops.Dataset.list_files(self.test_filenames,
                                             shuffle=shuffle)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)

    graph_def = dataset._as_serialized_graph(
        strip_device_assignment=True,
        external_state_policy=
        dataset.options().experimental_external_state_policy)

    options = dataset_ops.Options()
    options.experimental_distribute.auto_shard_policy = sharding_policy

    ds1 = distribute._RemoteDataset(graph_def, "/device:CPU:0",
                                    dataset.element_spec)
    ds2 = distribute._RemoteDataset(graph_def, "/device:CPU:0",
                                    dataset.element_spec)

    ds1 = ds1.with_options(options)
    ds2 = ds2.with_options(options)

    ds1 = distribute._AutoShardDataset(ds1, 2, 0)
    ds2 = distribute._AutoShardDataset(ds2, 2, 1)

    elems1 = set(self.getAllDatasetElements(ds1))
    elems2 = set(self.getAllDatasetElements(ds2))

    self.assertEmpty(elems1.intersection(elems2))

  @combinations.generate(test_base.default_test_combinations())
  def testWorkersGreaterThanNumFilesWithDataSharding(self):
    options = dataset_ops.Options()
    options.experimental_distribute.auto_shard_policy = (
        distribute_options.AutoShardPolicy.DATA)

    dataset = core_readers._TFRecordDataset(self.test_filenames)
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)

    # Should return "Record (0,5) of file (0 --> 9)" since we are sharding by
    # individual elements, we should be able to get some data from all files.
    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in (0, 5)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testAutoshardPolicyOff(self):
    options = dataset_ops.Options()
    options.experimental_distribute.auto_shard_policy = (
        distribute_options.AutoShardPolicy.OFF)

    dataset = core_readers._TFRecordDataset(self.test_filenames)
    dataset = dataset.with_options(options)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)

    # Should return every record in every file since autosharding is turned off.
    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testFileShardingWithoutReaderDatasetOp(self):
    options = dataset_ops.Options()
    options.experimental_distribute.auto_shard_policy = (
        distribute_options.AutoShardPolicy.FILE)

    dataset = dataset_ops.Dataset.range(1024)
    dataset = dataset.with_options(options)

    # We are specifying that we want a file sharding policy, and this pipeline
    # doesn't start with file reading, so we should error out.
    with self.assertRaises(errors.NotFoundError):
      dataset = distribute._AutoShardDataset(dataset, 10, 0)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testWorkersGreaterThanNumFiles(self):
    dataset = dataset_ops.Dataset.list_files(self.test_filenames)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(core_readers.TFRecordDataset, 10))
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 500, 499)
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordReaderWithDirectFileNames(self):
    # Using `_TFRecordDataset` creates a raw op rather than wrapping it around
    # a flat_map automatically.
    dataset = core_readers._TFRecordDataset(self.test_filenames)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in (0, 5)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordReaderWithDirectFileNamesAndShapes(self):
    # Using `_TFRecordDataset` creates a raw op rather than wrapping it around
    # a flat_map automatically.
    dataset = core_readers._TFRecordDataset(self.test_filenames)

    # BatchDataset contains `output_types` and `output_shapes`
    dataset = dataset.batch(5)
    dataset = distribute._AutoShardDataset(dataset, 2, 0)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 5)
    ]
    self.assertDatasetProduces(dataset, list(chunk(expected, 5)))

  @combinations.generate(test_base.default_test_combinations())
  def testShardOutOfRange(self):
    dataset = dataset_ops.Dataset.range(5)
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = distribute._AutoShardDataset(dataset, 10, 0)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testShardOutOfRangeEmptyDataset(self):
    dataset = dataset_ops.Dataset.range(0)
    with self.assertRaises(errors.OutOfRangeError):
      dataset = distribute._AutoShardDataset(dataset, 10, 0)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testShardWithRebatch(self):
    # Tests that Rebatch is a passthrough op.
    dataset = dataset_ops.Dataset.list_files(self.test_filenames, shuffle=False)
    dataset = dataset.apply(
        testing.assert_next(["Shard", "FlatMap", "BatchV2", "Rebatch"]))
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)
    dataset = distribute._RebatchDataset(dataset, num_replicas=1)
    dataset = distribute._AutoShardDataset(dataset, 5, 3)
    nxt = self.getNext(dataset)
    self.evaluate(nxt())

  @combinations.generate(test_base.default_test_combinations())
  def testNoReaderPipelines(self):
    dataset = dataset_ops.Dataset.range(1024)
    dataset = distribute._AutoShardDataset(dataset, 2, 0)
    self.assertDatasetProduces(dataset, [i for i in range(1024) if i % 2 == 0])

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownOpInPipelineStillShardsAtTheEnd(self):
    dataset = dataset_ops.Dataset.list_files(self.test_filenames, shuffle=False)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.apply(unique.unique())

    dataset = distribute._AutoShardDataset(dataset, 5, 0)

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in (0, 5)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidWorkerIndex(self):
    dataset = dataset_ops.Dataset.list_files(self.test_filenames)
    dataset = dataset.flat_map(core_readers.TFRecordDataset)
    dataset = dataset.batch(5)

    with self.assertRaises(errors.InvalidArgumentError):
      dataset = distribute._AutoShardDataset(dataset, 2, 2)
      self.evaluate(self.getNext(dataset)())


class AutoShardTextLineDatasetTest(
    reader_dataset_ops_test_base.TextLineDatasetTestBase,
    parameterized.TestCase):

  def setUp(self):
    super(AutoShardTextLineDatasetTest, self).setUp()
    self._num_files = 10
    self._num_records = 10
    self.test_filenames = self._createFiles(self._num_files, self._num_records)

  @combinations.generate(test_base.default_test_combinations())
  def testDirectFilenameTextLineReaderPipeline(self):
    dataset = core_readers.TextLineDataset(self.test_filenames)
    dataset = distribute._AutoShardDataset(dataset, 5, 0)

    expected = [
        b"%d: %d" % (f, r)  # pylint:disable=g-complex-comprehension
        for f in (0, 5)
        for r in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)


if __name__ == "__main__":
  test.main()
