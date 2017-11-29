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
"""Tests for tf.contrib.training.python_input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.training.python.training import bucket_ops
from tensorflow.contrib.training.python.training import python_input
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import input as core_input
from tensorflow.python.training import queue_runner_impl


class PythonInputTest(test.TestCase):

  def testGenerator(self):
    def simple_generator():
      for i in range(2):
        yield {"value": i, "ignored": 3}

    simple_features = {
        "value": parsing_ops.FixedLenFeature(shape=[], dtype=dtypes.int32)
    }
    tensors = python_input.python_input(simple_generator, simple_features)
    self.assertEqual(["value"], tensors.keys())
    self.assertEqual(dtypes.int32, tensors["value"].dtype)
    self.assertEqual((), tensors["value"].shape)

    with self.test_session() as sess:
      self.assertEqual({"value": 0}, sess.run(tensors))
      self.assertEqual({"value": 1}, sess.run(tensors))
      with self.assertRaisesOpError("Iteration finished"):
        sess.run(tensors)

  def testInvalidGenerator(self):
    generator1 = lambda: iter([{"value": "a"}])
    int_features = {
        "value": parsing_ops.FixedLenFeature(shape=[], dtype=dtypes.int32)
    }
    tensors1 = python_input.python_input(generator1, int_features)

    with self.test_session() as sess:
      with self.assertRaisesOpError("invalid literal"):
        # Can't convert a string to an integer
        sess.run(tensors1)

    generator2 = lambda: iter([None])
    tensors2 = python_input.python_input(generator2, int_features)

    with self.test_session() as sess:
      with self.assertRaisesOpError("generator must return dict"):
        sess.run(tensors2)

    generator3 = lambda: iter([{"value": [1, 2]}])
    tensors3 = python_input.python_input(generator3, int_features)

    with self.test_session() as sess:
      with self.assertRaisesOpError("incompatible with declared shape"):
        sess.run(tensors3)

  def testGeneratorWorksWithBatching(self):
    def simple_generator():
      for i in range(5):
        yield {"value": i, "ignored": 3}

    simple_features = {
        "value": parsing_ops.FixedLenFeature(shape=[], dtype=dtypes.int32)
    }
    tensors = python_input.python_input(simple_generator, simple_features)

    # Request batches of size 4 at a time, the final batch may be smaller.
    batched_tensors = core_input.batch(tensors, batch_size=4,
                                       allow_smaller_final_batch=True)

    self.assertEqual(["value"], batched_tensors.keys())
    self.assertEqual(dtypes.int32, batched_tensors["value"].dtype)
    self.assertEqual([None], batched_tensors["value"].shape.as_list())

    with self.test_session() as sess:
      # The generator emits 5 items total.  The first 4 are returned in
      # the first session run; the final one is returned in the
      # second.  This works because allow_smaller_final_batch=True.
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
      r1 = sess.run(batched_tensors)
      r2 = sess.run(batched_tensors)
      self.assertAllEqual([0, 1, 2, 3], r1["value"])
      self.assertEqual([4], r2["value"])
      with self.assertRaisesOpError("Iteration finished"):
        sess.run(tensors)
      coord.request_stop()
      for thread in threads:
        thread.join()

  def testGeneratorWorksWithManyBatchingThreads(self):
    def simple_generator():
      for i in range(5000):
        yield {"value": i, "ignored": 3}

    simple_features = {
        "value": parsing_ops.FixedLenFeature(shape=[], dtype=dtypes.int32)
    }
    tensors = python_input.python_input(simple_generator, simple_features)

    # Request batches of size 20 at a time, the final batch may be smaller.
    _, batched_tensors = bucket_ops.bucket(
        tensors, which_bucket=tensors["value"] % 5,
        batch_size=20, num_buckets=5, num_threads=7, capacity=17,
        allow_smaller_final_batch=True)

    self.assertEqual(["value"], batched_tensors.keys())
    self.assertEqual(dtypes.int32, batched_tensors["value"].dtype)
    self.assertEqual([None], batched_tensors["value"].shape.as_list())

    with self.test_session() as sess:
      # The generator emits 5 items total.  The first 4 are returned in
      # the first session run; the final one is returned in the
      # second.  This works because allow_smaller_final_batch=True.
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
      results = []
      while True:
        try:
          r = sess.run(batched_tensors)
          results.extend(r["value"].tolist())
        except errors.OutOfRangeError:
          break
      coord.request_stop()
      for thread in threads:
        thread.join()
    self.assertEqual(sorted(results),
                     list(range(5000)))

  def testVaryingFieldsInGenerator(self):
    def simple_generator():
      for i in range(2):
        yield {"value": i,
               "seqlen_value": np.ones((i, 1))}

    simple_features = {
        "value": parsing_ops.FixedLenFeature(shape=[], dtype=dtypes.int32),
        "seqlen_value": parsing_ops.FixedLenSequenceFeature(
            shape=[1], dtype=dtypes.float32, allow_missing=True),
        "empty_value": parsing_ops.FixedLenFeature(
            default_value=[-1, -2], dtype=dtypes.int32, shape=[2])
    }
    tensors = python_input.python_input(simple_generator, simple_features)
    self.assertEqual(
        set(["value", "seqlen_value", "empty_value"]), set(tensors.keys()))
    self.assertEqual(dtypes.int32, tensors["value"].dtype)
    self.assertEqual((), tensors["value"].shape)
    self.assertEqual(dtypes.float32, tensors["seqlen_value"].dtype)
    self.assertEqual([None, 1], tensors["seqlen_value"].shape.as_list())
    self.assertEqual(dtypes.int32, tensors["empty_value"].dtype)
    self.assertEqual([2], tensors["empty_value"].shape)

    with self.test_session() as sess:
      r1 = sess.run(tensors)
      self.assertAllEqual(0, r1["value"])
      self.assertAllEqual(np.ones((0, 1)), r1["seqlen_value"])
      self.assertAllEqual([-1, -2], r1["empty_value"])

      r2 = sess.run(tensors)
      self.assertAllEqual(1, r2["value"])
      self.assertAllEqual([[1]], r2["seqlen_value"])
      self.assertAllEqual([-1, -2], r2["empty_value"])

      with self.assertRaisesOpError("Iteration finished"):
        sess.run(tensors)


if __name__ == "__main__":
  test.main()
