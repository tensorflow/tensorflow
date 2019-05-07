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

from tensorflow.contrib.batching.python.ops import batch_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test


def delayed_plus1(x):
  """Sleeps for 100ms then returns x+1."""
  time.sleep(0.1)
  return x + 1


class BatchOpsTest(test.TestCase):
  """Tests for batch_ops.{un,}batch."""

  def testBasicUnbatchV1Decorated(self):
    """Tests that the batch_function_v1 decorator works."""
    with self.cached_session() as sess:

      @batch_ops.batch_function_v1(1, 10, 100000)
      def computation(in_t):
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

  def testUnbatchGrad(self):
    """Tests that batch and unbatch are differentiable."""
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
      batched, index, id_t = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=36000000, grad_timeout_micros=1000000,
          batching_queue="")
      computation = batched[0] * batched[0]
      result = batch_ops.unbatch(computation, index, id_t,
                                 timeout_micros=1000000, shared_name="unbatch")
      grad = gradients_impl.gradients(result, inp)
      thread_results = []

      def worker():
        thread_results.extend(sess.run([grad], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([grad], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [4])

if __name__ == "__main__":
  test.main()
