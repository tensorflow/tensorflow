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
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class MakeTFRecordDatasetTest(tf_record_test_base.TFRecordTestBase,
                              parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              batch_size=[1, 2],
              num_epochs=[1, 3],
              file_index=[None, 1],
              num_parallel_reads=[1, 8],
              drop_final_batch=[False, True],
              parser_fn=[True, False])))
  def testRead(self, batch_size, num_epochs, file_index, num_parallel_reads,
               drop_final_batch, parser_fn):
    if file_index is None:
      file_pattern = self._filenames
    else:
      file_pattern = self._filenames[file_index]

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

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              batch_size=[1, 2],
              num_epochs=[1, 3],
              num_parallel_reads=[1, 2],
              seed=[None, 21345])))
  def testShuffle(self, batch_size, num_epochs, num_parallel_reads, seed):

    def dataset_fn():
      return readers.make_tf_record_dataset(
          file_pattern=self._filenames,
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
  def testIndefiniteRepeatShapeInference(self):
    dataset = readers.make_tf_record_dataset(
        file_pattern=self._filenames, num_epochs=None, batch_size=32)
    for shape in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)):
      self.assertEqual(32, shape[0])


if __name__ == "__main__":
  test.main()
