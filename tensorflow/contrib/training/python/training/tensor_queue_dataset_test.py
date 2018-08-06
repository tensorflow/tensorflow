# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TensorQueueDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.contrib.training.python.training import tensor_queue_dataset as tqd
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class PrependFromQueueAndPaddedBatchDatasetTest(test.TestCase):

  def testNoEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=1))
    self.assertEqual((dtypes.variant, dtypes.int32), dataset.output_types)
    self.assertAllEqual(([None],) * 2,
                        [x.as_list() for x in dataset.output_shapes])
    iterator = dataset.make_one_shot_iterator()
    _, value = iterator.get_next()
    self.assertEqual([0], self.evaluate(value))
    self.assertEqual([1], self.evaluate(value))
    self.assertEqual([2], self.evaluate(value))
    with self.assertRaisesOpError("End of sequence"):
      self.evaluate(value)

  def testBatchedNoEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=2))
    iterator = dataset.make_one_shot_iterator()
    _, value = iterator.get_next()
    self.assertAllEqual([0, 1], self.evaluate(value))
    self.assertAllEqual([2], self.evaluate(value))
    with self.assertRaisesOpError("End of sequence"):
      self.evaluate(value)

  def testBatchedWithBiggerPaddingNoEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([[0], [1], [2]])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(
            batch_size=2, padded_shapes=[3]))
    iterator = dataset.make_one_shot_iterator()
    _, value = iterator.get_next()
    self.assertAllEqual([[0, 0, 0], [1, 0, 0]], self.evaluate(value))
    self.assertAllEqual([[2, 0, 0]], self.evaluate(value))
    with self.assertRaisesOpError("End of sequence"):
      self.evaluate(value)

  def testBatchedWithBiggerPaddingOneEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([[0], [1], [2]])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(
            batch_size=1, padded_shapes=[3]))
    iterator = dataset.make_one_shot_iterator()
    queue_handle, value = iterator.get_next()
    enqueue_negative = tqd.enqueue_in_queue_dataset(queue_handle, -value)
    with self.test_session() as sess:
      self.assertAllEqual([[0, 0, 0]], sess.run(value))
      value_1, _ = sess.run([value, enqueue_negative])
      self.assertAllEqual([[1, 0, 0]], value_1)
      value_2, _ = sess.run([value, enqueue_negative])
      self.assertAllEqual([[-1, 0, 0]], value_2)
      value_3 = sess.run(value)
      self.assertAllEqual([[1, 0, 0]], value_3)
      value_4, _ = sess.run([value, enqueue_negative])
      self.assertAllEqual([[2, 0, 0]], value_4)
      value_5 = sess.run(value)
      self.assertAllEqual([[-2, 0, 0]], value_5)
      with self.assertRaisesOpError("End of sequence"):
        sess.run(value)

  def testOneEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=1))
    iterator = dataset.make_one_shot_iterator()
    queue_handle, value = iterator.get_next()
    enqueue_negative = tqd.enqueue_in_queue_dataset(queue_handle, -value)
    with self.test_session() as sess:
      self.assertEqual([0], sess.run(value))
      value_1, _ = sess.run([value, enqueue_negative])
      self.assertEqual([1], value_1)
      value_2, _ = sess.run([value, enqueue_negative])
      self.assertEqual([-1], value_2)
      value_3 = sess.run(value)
      self.assertEqual([1], value_3)
      value_4, _ = sess.run([value, enqueue_negative])
      self.assertEqual([2], value_4)
      value_5 = sess.run(value)
      self.assertEqual([-2], value_5)
      with self.assertRaisesOpError("End of sequence"):
        sess.run(value)

  def testBatchedOneEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=2))
    iterator = dataset.make_one_shot_iterator()
    queue_handle, value = iterator.get_next()
    enqueue_negative = tqd.enqueue_in_queue_dataset(queue_handle, -value)
    enqueue_zeroth = tqd.enqueue_in_queue_dataset([queue_handle[0]],
                                                  array_ops.expand_dims(
                                                      value[0], axis=0))
    with self.test_session() as sess:
      value_0, _ = sess.run([value, enqueue_negative])
      self.assertAllEqual([0, 1], value_0)
      value_1, _ = sess.run([value, enqueue_zeroth])
      self.assertAllEqual([0, -1], value_1)
      value_2, _ = sess.run([value, enqueue_negative])
      self.assertAllEqual([0, 2], value_2)
      self.assertAllEqual([0, -2], sess.run(value))
      with self.assertRaisesOpError("End of sequence"):
        sess.run(value)

  def testManyEnqueue(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=1))
    iterator = dataset.make_one_shot_iterator()
    queue_handle, value = iterator.get_next()
    enqueue_many_more = [
        tqd.enqueue_in_queue_dataset(queue_handle, value + 100 + i)
        for i in range(1000)
    ]
    with self.test_session() as sess:
      value_0, _ = sess.run((value, enqueue_many_more))
      self.assertEqual([0], value_0)
      rest = []
      for _ in range(1000):
        rest.append(sess.run(value))
      self.assertEquals([[100 + i] for i in range(1000)], sorted(rest))
      # Going back to the original input.
      value_1, _ = sess.run((value, enqueue_many_more))
      self.assertEqual(1, value_1)
      rest = []
      for _ in range(1000):
        rest.append(sess.run(value))
      self.assertEquals([[100 + i + 1] for i in range(1000)], sorted(rest))
      with self.assertRaisesOpError("End of sequence"):
        sess.run(value)

  def testEnqueueWithPrefetch(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=1))
    # Prefetching will request additional values before they are
    # available to the queue.
    dataset = dataset.prefetch(buffer_size=3)
    iterator = dataset.make_one_shot_iterator()
    queue_handle, value = iterator.get_next()
    enqueue = tqd.enqueue_in_queue_dataset(queue_handle, value + 1)
    with self.test_session() as sess:
      i = 0
      while i < 4:
        received, _ = sess.run((value, enqueue))
        if received.size > 0:
          self.assertAllEqual([i], received)
          i += 1
      received_last = False
      while True:
        try:
          received = sess.run(value)
          if received.size > 0:
            self.assertAllEqual([4], received)
            received_last = True
        except errors.OutOfRangeError:
          break
      self.assertTrue(received_last)

  def testDatasetWithPaddedShapeSmallerThanInputFails(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([[0, 0, 0]]).repeat(None)
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(
            batch_size=1, padded_shapes=[2]))
    iterator = dataset.make_one_shot_iterator()
    _, value = iterator.get_next()
    with self.test_session() as sess:
      with self.assertRaisesOpError(
          r"Incompatible input shapes at component 0 between "
          r"input dataset this dataset: \[3\] vs. \[2\]"):
        sess.run(value)

  def testEnqueueWithIncompatibleInputsFailsWithInformativeError(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0]).repeat(None)
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=1))
    iterator = dataset.make_one_shot_iterator()
    queue_handle, value = iterator.get_next()

    enqueue_bad_structure = tqd.enqueue_in_queue_dataset(
        queue_handle, (value, value))
    enqueue_bad_dtype = tqd.enqueue_in_queue_dataset(queue_handle,
                                                     np.array(
                                                         [1.0],
                                                         dtype=np.float32))
    enqueue_bad_shape_no_batch_dim = tqd.enqueue_in_queue_dataset(
        queue_handle, ([1],))
    enqueue_bad_shape = tqd.enqueue_in_queue_dataset(queue_handle,
                                                     np.array(
                                                         [[1]], dtype=np.int32))

    with self.test_session() as sess:
      with self.assertRaisesOpError(
          "mismatched number of tensors.  Queue expects 1 tensors but "
          "tried to insert 2"):
        sess.run(enqueue_bad_structure)
      with self.assertRaisesOpError(r"Expected component 0 to have batched "
                                    r"shape \[1,...\], but saw shape: \[\]"):
        sess.run(enqueue_bad_shape_no_batch_dim)
      with self.assertRaisesOpError(
          r"mismatched shapes at component 0.  Attempted to insert tensor "
          r"with shape \[1\] but queue expected shape: \[\]"):
        sess.run(enqueue_bad_shape)
      with self.assertRaisesOpError(
          r"mismatched dtypes at component 0.  Attempted to insert tensor "
          r"of type float but queue expected type: int32"):
        sess.run(enqueue_bad_dtype)

  def testEnqueueWithPaddedBatchFailsWithInformativeError(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=1))
    with self.assertRaisesRegexp(
        TypeError, r"Unable to create padding for field of type 'variant'"):
      dataset.padded_batch(batch_size=10, padded_shapes=[1])

  def testOneEnqueueWithPadding(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 2, 4, 6])
    # Make a dataset of variable-length vectors and their lengths.
    dataset = dataset.map(
        lambda c: (c, c * array_ops.ones((c,), dtype=c.dtype)))
    # Emit a queue we can prepend to, and counts/values as padded
    # batch.
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=3))

    iterator = dataset.make_one_shot_iterator()
    queue, (count, padded_value) = iterator.get_next()

    # Split the padded_value into two pieces: head and rest
    rest_indices = array_ops.squeeze(array_ops.where(count > 2), axis=1)
    bound = math_ops.minimum(2, math_ops.reduce_max(count))
    value_head = padded_value[:, :bound]
    count_rest = array_ops.gather(count - 2, rest_indices)
    value_rest = array_ops.gather(padded_value, rest_indices)[:, bound:]
    queue_rest = array_ops.gather(queue, rest_indices)
    enqueue_rest_op = tqd.enqueue_in_queue_dataset(queue_rest,
                                                   (count_rest, value_rest))
    with ops.control_dependencies([enqueue_rest_op]):
      calc = array_ops.identity(value_head)

    with self.test_session() as sess:
      self.assertAllEqual([[0, 0], [2, 2], [4, 4]], sess.run(calc))
      self.assertAllEqual([[4, 4], [6, 6]], sess.run(calc))
      self.assertAllEqual([[6, 6]], sess.run(calc))
      self.assertAllEqual([[6, 6]], sess.run(calc))
      # Get some final batches due to prefetching.
      for _ in range(3):
        try:
          self.assertAllEqual(
              np.empty(shape=(0, 0), dtype=np.int32), sess.run(calc))
        except errors.OutOfRangeError as e:
          self.assertTrue(str(e).startswith("End of sequence"))

  def testNonstandardPadding(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 2, 4, 6])
    # Make a dataset of variable-length vectors and their lengths.
    dataset = dataset.map(
        lambda c: (c, c * array_ops.ones((c,), dtype=c.dtype)))
    # Emit a queue we can prepend to, and counts/values as padded
    # batch.
    dataset = dataset.apply(
        tqd.prepend_from_queue_and_padded_batch_dataset(
            batch_size=3, padding_values=(
                0,
                -1,
            )))

    iterator = dataset.make_one_shot_iterator()
    _, (unused_count, padded_value) = iterator.get_next()

    with self.test_session() as sess:
      self.assertAllEqual([[-1, -1, -1, -1], [2, 2, -1, -1], [4, 4, 4, 4]],
                          sess.run(padded_value))
      self.assertAllEqual([[6] * 6], sess.run(padded_value))
      with self.assertRaisesOpError("End of sequence"):
        sess.run(padded_value)


# TODO(ebrevdo): Figure out how to use run_core_tests to test state
# saving of an iterator that's had some tensors enqueued into its queue.
class PrependFromQueueAndPaddedBatchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testPrependFromQueueAndPaddedBatch(self):

    def build_dataset(seq_lens):
      return dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
          lambda x: array_ops.fill([x], x)).apply(
              tqd.prepend_from_queue_and_padded_batch_dataset(batch_size=4))

    seq_lens1 = np.random.randint(1, 20, size=(32,)).astype(np.int32)
    seq_lens2 = np.random.randint(21, 40, size=(32,)).astype(np.int32)
    self.run_core_tests(lambda: build_dataset(seq_lens1),
                        lambda: build_dataset(seq_lens2), 8)

  def testPrependFromQueueAndPaddedBatchNonDefaultPadding(self):

    def build_dataset(seq_lens):

      def fill_tuple(x):
        filled = array_ops.fill([x], x)
        return (filled, string_ops.as_string(filled))

      padded_shape = [-1]
      return dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
          fill_tuple).apply(
              tqd.prepend_from_queue_and_padded_batch_dataset(
                  batch_size=4,
                  padded_shapes=(padded_shape, padded_shape),
                  padding_values=(-1, "<end>")))

    seq_lens1 = np.random.randint(1, 20, size=(32,)).astype(np.int32)
    seq_lens2 = np.random.randint(21, 40, size=(32,)).astype(np.int32)
    self.run_core_tests(lambda: build_dataset(seq_lens1),
                        lambda: build_dataset(seq_lens2), 8)


if __name__ == "__main__":
  test.main()
