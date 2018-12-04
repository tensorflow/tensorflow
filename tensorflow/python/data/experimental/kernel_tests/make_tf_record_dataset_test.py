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

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class MakeTFRecordDatasetTest(
    reader_dataset_ops_test_base.TFRecordDatasetTestBase):

  def _interleave(self, iterators, cycle_length):
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
      if pending_iterators:
        open_iterators.append(pending_iterators.pop(0))
        num_open += 1

    while num_open:
      for i in range(min(cycle_length, len(open_iterators))):
        if open_iterators[i] is None:
          continue
        try:
          yield next(open_iterators[i])
        except StopIteration:
          if pending_iterators:
            open_iterators[i] = pending_iterators.pop(0)
          else:
            open_iterators[i] = None
            num_open -= 1

  def _next_expected_batch(self,
                           file_indices,
                           batch_size,
                           num_epochs,
                           cycle_length,
                           drop_final_batch,
                           use_parser_fn):

    def _next_record(file_indices):
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i

    def _next_record_interleaved(file_indices, cycle_length):
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for f, r in next_records:
        record = self._record(f, r)
        if use_parser_fn:
          record = record[1:]
        record_batch.append(record)
        batch_index += 1
        if len(record_batch) == batch_size:
          yield record_batch
          record_batch = []
          batch_index = 0
    if record_batch and not drop_final_batch:
      yield record_batch

  def _verify_records(self,
                      sess,
                      outputs,
                      batch_size,
                      file_index,
                      num_epochs,
                      interleave_cycle_length,
                      drop_final_batch,
                      use_parser_fn):
    if file_index is not None:
      file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices, batch_size, num_epochs, interleave_cycle_length,
        drop_final_batch, use_parser_fn):
      actual_batch = self.evaluate(outputs)
      self.assertAllEqual(expected_batch, actual_batch)

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

    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        outputs = dataset_ops.make_one_shot_iterator(
            readers.make_tf_record_dataset(
                file_pattern=file_pattern,
                num_epochs=num_epochs,
                batch_size=batch_size,
                parser_fn=fn,
                num_parallel_reads=num_parallel_reads,
                drop_final_batch=drop_final_batch,
                shuffle=False)).get_next()
        self._verify_records(
            sess, outputs, batch_size, file_index, num_epochs=num_epochs,
            interleave_cycle_length=num_parallel_reads,
            drop_final_batch=drop_final_batch, use_parser_fn=parser_fn)
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(outputs)

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
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        dataset = readers.make_tf_record_dataset(
            file_pattern=self.test_filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_parallel_reads=num_parallel_reads,
            shuffle=True,
            shuffle_seed=seed)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.evaluate(iterator.initializer)
        first_batches = []
        try:
          while True:
            first_batches.append(self.evaluate(next_element))
        except errors.OutOfRangeError:
          pass

        self.evaluate(iterator.initializer)
        second_batches = []
        try:
          while True:
            second_batches.append(self.evaluate(next_element))
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

  def testIndefiniteRepeatShapeInference(self):
    dataset = readers.make_tf_record_dataset(
        file_pattern=self.test_filenames, num_epochs=None, batch_size=32)
    for shape in nest.flatten(dataset.output_shapes):
      self.assertEqual(32, shape[0])


if __name__ == "__main__":
  test.main()
