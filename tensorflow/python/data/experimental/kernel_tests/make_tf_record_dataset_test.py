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
"""Tests for `tf.data.experimental.make_tf_record_dataset()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class MakeTFRecordDatasetTest(
    reader_dataset_ops_test_base.TFRecordDatasetTestBase,
    parameterized.TestCase):

  def _read_test(self, batch_size, num_epochs, file_index=None,
                 num_parallel_reads=1, drop_final_batch=False, parser_fn=False):
    if file_index is None:
      file_pattern = self.test_filenames
    else:
      file_pattern = self.test_filenames[file_index]

    if parser_fn:
      fn = lambda x: string_ops.substr(x, 1, 999)
    else:
      fn = None

    outputs = self.getNext(
        readers.make_tf_record_dataset(
            file_pattern=file_pattern,
            num_epochs=num_epochs,
            batch_size=batch_size,
            parser_fn=fn,
            num_parallel_reads=num_parallel_reads,
            drop_final_batch=drop_final_batch,
            shuffle=False))
    self._verify_records(
        outputs,
        batch_size,
        file_index,
        num_epochs=num_epochs,
        interleave_cycle_length=num_parallel_reads,
        drop_final_batch=drop_final_batch,
        use_parser_fn=parser_fn)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(outputs())

  @combinations.generate(test_base.default_test_combinations())
  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        # Basic test: read from file 0.
        self._read_test(batch_size, num_epochs, 0)

        # Basic test: read from file 1.
        self._read_test(batch_size, num_epochs, 1)

        # Basic test: read from both files.
        self._read_test(batch_size, num_epochs)

        # Basic test: read from both files, with parallel reads.
        self._read_test(batch_size, num_epochs, num_parallel_reads=8)

  @combinations.generate(test_base.default_test_combinations())
  def testDropFinalBatch(self):
    for batch_size in [1, 2, 10]:
      for num_epochs in [1, 3]:
        # Read from file 0.
        self._read_test(batch_size, num_epochs, 0, drop_final_batch=True)

        # Read from both files.
        self._read_test(batch_size, num_epochs, drop_final_batch=True)

        # Read from both files, with parallel reads.
        self._read_test(batch_size, num_epochs, num_parallel_reads=8,
                        drop_final_batch=True)

  @combinations.generate(test_base.default_test_combinations())
  def testParserFn(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        for drop_final_batch in [False, True]:
          self._read_test(batch_size, num_epochs, parser_fn=True,
                          drop_final_batch=drop_final_batch)
          self._read_test(batch_size, num_epochs, num_parallel_reads=8,
                          parser_fn=True, drop_final_batch=drop_final_batch)

  def _shuffle_test(self, batch_size, num_epochs, num_parallel_reads=1,
                    seed=None):

    def dataset_fn():
      return readers.make_tf_record_dataset(
          file_pattern=self.test_filenames,
          num_epochs=num_epochs,
          batch_size=batch_size,
          num_parallel_reads=num_parallel_reads,
          shuffle=True,
          shuffle_seed=seed)

    next_element = self.getNext(dataset_fn())
    first_batches = []
    try:
      while True:
        first_batches.append(self.evaluate(next_element()))
    except errors.OutOfRangeError:
      pass

    next_element = self.getNext(dataset_fn())
    second_batches = []
    try:
      while True:
        second_batches.append(self.evaluate(next_element()))
    except errors.OutOfRangeError:
      pass

    self.assertEqual(len(first_batches), len(second_batches))
    if seed is not None:
      # if you set a seed, should get the same results
      for i in range(len(first_batches)):
        self.assertAllEqual(first_batches[i], second_batches[i])

    expected = []
    for f in range(self._num_files):
      for r in range(self._num_records):
        expected.extend([self._record(f, r)] * num_epochs)

    for batches in (first_batches, second_batches):
      actual = []
      for b in batches:
        actual.extend(b)
      self.assertAllEqual(sorted(expected), sorted(actual))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffle(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        for num_parallel_reads in [1, 2]:
          # Test that all expected elements are produced
          self._shuffle_test(batch_size, num_epochs, num_parallel_reads)
          # Test that elements are produced in a consistent order if
          # you specify a seed.
          self._shuffle_test(batch_size, num_epochs, num_parallel_reads,
                             seed=21345)

  @combinations.generate(test_base.default_test_combinations())
  def testIndefiniteRepeatShapeInference(self):
    dataset = readers.make_tf_record_dataset(
        file_pattern=self.test_filenames, num_epochs=None, batch_size=32)
    for shape in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)):
      self.assertEqual(32, shape[0])


if __name__ == "__main__":
  test.main()
