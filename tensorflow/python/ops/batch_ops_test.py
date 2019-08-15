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

"""Tests for the currently experimental in-graph batch ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import batch_ops
from tensorflow.python.ops import gen_batch_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


def delayed_plus1(x):
  """Sleeps for 100ms then returns x+1."""
  time.sleep(0.1)
  return x + 1


@test_util.run_all_in_graph_and_eager_modes
class BatchOpsTest(test.TestCase):
  """Tests for batch_ops.{un,}batch."""

  # Test for only non eager mode as batching in eager context as a functionality
  # is TBD.
  def testBasicBatch(self):
    """Tests that a single batched tensor executes together and only once."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, index, _ = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=36000000, grad_timeout_micros=0,
          batching_queue="")
      thread_results = []

      def worker():
        thread_results.extend(
            sess.run([batched, index], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([batched, index], feed_dict={inp: [2]})
      worker_thread.join()

      # At this point either the thread or the main did the batch and the other
      # should have empty results.
      if list(thread_results[0][0]):
        batch_t = thread_results[0][0]
        index_t = thread_results[1]
        empty_b = main_results[0][0]
        empty_m = main_results[1]
      else:
        batch_t = main_results[0][0]
        index_t = main_results[1]
        empty_b = thread_results[0][0]
        empty_m = thread_results[1]

      # Check that both the inputs made it out exactly once.
      self.assertAllEqual(sorted(batch_t), (1, 2))
      # Check that we get 2 rows in the index tensor.
      self.assertEqual(len(index_t), 2)
      # Check that the other ones are empty.
      self.assertEqual(len(empty_b), 0)
      self.assertEqual(len(empty_m), 0)

  def testBatchWithPadding(self):
    """Test that batching with padding up to an allowed batch size works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[2])
      batched, index, _ = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[5, 10],
          grad_timeout_micros=0, batching_queue="")
      thread_results = []

      def worker():
        thread_results.extend(
            sess.run([batched, index], feed_dict={inp: [1, 3]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([batched, index], feed_dict={inp: [2, 4]})
      worker_thread.join()

      # At this point either the thread or the main did the batch and the other
      # should have empty results.
      if list(thread_results[0][0]):
        batch_t = thread_results[0][0]
      else:
        batch_t = main_results[0][0]

      # Check that the batch tensor incorporates the padding.
      self.assertEqual(len(batch_t), 5)

  def testMultipleBatch(self):
    """Tests that multiple batched tensors execute together."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp0 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      inp1 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, _, _ = batch_ops.batch(
          [inp0, inp1],
          num_batch_threads=1,
          max_batch_size=2,
          batch_timeout_micros=36000000,
          grad_timeout_micros=0,
          batching_queue="")
      thread_results = []

      def worker():
        thread_results.extend(
            sess.run([batched], feed_dict={inp0: [1],
                                           inp1: [2]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([batched], feed_dict={inp0: [2], inp1: [3]})
      worker_thread.join()

      # At this point either the thread or the main did the batch and the other
      # should have empty results.
      if list(thread_results[0][0]):
        batch_t = thread_results[0]
        empty_t = main_results[0]
      else:
        batch_t = main_results[0]
        empty_t = thread_results[0]

      # Assert that the tensors were batched together.
      self.assertAllEqual(sorted(batch_t[0]), [1, 2])
      self.assertAllEqual(sorted(batch_t[1]), [2, 3])
      self.assertAllEqual(empty_t[0], [])
      self.assertAllEqual(empty_t[1], [])

  def testIllegalBatchDifferentDim0Sizes(self):
    """Tests illegally feeding tensors with different dim0 sizes."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp0 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      inp1 = array_ops.placeholder(dtype=dtypes.int32, shape=[2])
      batched, index, _ = batch_ops.batch(
          [inp0, inp1], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=0, grad_timeout_micros=0, batching_queue="")
      with self.assertRaises(Exception) as raised:
        _ = sess.run([batched, index], feed_dict={inp0: [0], inp1: [1, 2]})
      self.assertGreater(
          raised.exception.message.find("must have equal 0th-dimension size"),
          0)

  def testBasicUnbatch(self):
    """Tests that batch and unbatch work together."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, index, id_t = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[3, 10],
          grad_timeout_micros=0, batching_queue="")
      computation = batched[0] + 1
      result = batch_ops.unbatch(computation, index, id_t,
                                 timeout_micros=1000000, shared_name="unbatch")
      thread_results = []

      def worker():
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBasicUnbatchDecorated(self):
    """Tests that the batch_function decorator works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      # TODO(apassos): Removing this line causes test flakiness! Ideally should
      # be investigated.
      default_inp = array_ops.placeholder_with_default(2, shape=[])  # pylint: disable=unused-variable

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        self.assertTrue(in_t.shape is not None)
        return in_t + 1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      result = computation(inp)
      thread_results = []

      def worker():
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBatchDecoratedWithCapturedInput(self):
    """Tests that the batch_function decorator works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      captured_inp0 = array_ops.placeholder_with_default(2, shape=[])
      captured_inp1 = array_ops.placeholder_with_default(1, shape=[])

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        return in_t + captured_inp0 - captured_inp1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      result = computation(inp)
      thread_results = []

      def worker():
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBatchFunctionOp(self):
    """Tests that the batch_function op works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:

      @function.Defun(dtypes.int32)
      def computation(in_t):
        return in_t + 1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      result = gen_batch_ops.batch_function(
          [inp],
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,
          Tout=[dtypes.int32],
          f=computation,
          captured_tensors=computation.captured_inputs)
      thread_results = []

      def worker():
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBatchFunctionOpWithCapturedInput(self):
    """Tests that batch_function op works with captured input."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      captured_inp0 = array_ops.placeholder_with_default(2, shape=[])
      captured_inp1 = array_ops.placeholder_with_default(1, shape=[])
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])

      @function.Defun(dtypes.int32)
      def computation(inp):
        return inp + captured_inp0 - captured_inp1

      result = gen_batch_ops.batch_function(
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[3, 10],
          batching_queue="",
          f=computation,
          in_tensors=[inp],
          captured_tensors=computation.captured_inputs,
          Tout=[o.type for o in computation.definition.signature.output_arg])

      thread_results = []

      def worker():
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBatchFunctionOpWithInputError(self):
    """Tests that batch_function op works with error in the inputs."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])

      @function.Defun(dtypes.int32, dtypes.int32)
      def computation(in0, in1):
        return in0 + in1

      result = gen_batch_ops.batch_function(
          [inp],  # computation actually expects 2 inputs.
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          batching_queue="",
          f=computation,
          captured_tensors=computation.captured_inputs,
          Tout=[o.type for o in computation.definition.signature.output_arg])

      with self.assertRaisesRegexp(InvalidArgumentError,
                                   ".*2 arguments.*but 1.*"):
        sess.run([result], feed_dict={inp: [2]})

  def testBasicUnbatchDecoratedWithReshape(self):
    """Tests that the batch_function decorator works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        return array_ops.reshape(in_t, [-1]) + 1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1, 1])
      result = computation(inp)
      thread_results = []

      def worker():
        thread_results.extend(sess.run([result], feed_dict={inp: [[1]]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [[2]]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testUnbatchTimeout(self):
    """Tests that the unbatch timeout works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, index, id_t = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=36000000, grad_timeout_micros=0,
          batching_queue="")
      computation = batched[0] + 1
      timeout_micros = 10
      result = batch_ops.unbatch(computation, index, id_t, timeout_micros,
                                 shared_name="shared_unbatch")
      # Set up a parallel pipeline that delays the computation, but uses the
      # same unbatch resource object as the non-delayed pipeline.
      computation_delayed = script_ops.py_func(delayed_plus1,
                                               [batched[0]],
                                               dtypes.int32)
      result_delayed = batch_ops.unbatch(computation_delayed,
                                         index,
                                         id_t,
                                         timeout_micros,
                                         shared_name="shared_unbatch")

      thread_results = []
      def worker():
        # A first call using the non-delayed pipeline. The batcher will send an
        # empty tensor along the non-delayed pipeline.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      time.sleep(0.1)  # Ensure the thread's call starts first.
      # A second call using the delayed pipeline.  The batcher will send the
      # batched tensor along the delayed pipeline, thus delaying the arrival of
      # the batched tensor at the unbatch op, relative to the empty tensor.
      #
      # TODO(olston, apassos): Avoid relying on the order in which the batch op
      # emits the empty tensor versus the batched one.
      _ = sess.run([result_delayed], feed_dict={inp: [2]})
      worker_thread.join()
      # The thread's call should hit the timeout, and thus get 0 results.
      self.assertEqual(len(thread_results), 0)


if __name__ == "__main__":
  test.main()
