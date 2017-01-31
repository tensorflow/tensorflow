# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for nccl ops. See also the cc test for nccl_communicator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import nccl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class AllReduceTest(test.TestCase):

  def testAllReduce(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    for dtype in [np.float32, np.int32, np.int64, np.float64]:
      # Create session inside outer loop to test use of
      # same communicator across multiple sessions.
      with self.test_session(use_gpu=True) as sess:
        self._testSingleAllReduce(sess, dtype, nccl.all_sum, lambda x, y: x + y)
        self._testSingleAllReduce(sess, dtype, nccl.all_prod,
                                  lambda x, y: x * y)
        self._testSingleAllReduce(sess, dtype, nccl.all_min, np.minimum)
        self._testSingleAllReduce(sess, dtype, nccl.all_max, np.maximum)

  def _testSingleAllReduce(self, sess, np_type, nccl_fn, numpy_accumulation_fn):
    for devices in [['/gpu:0', '/gpu:0', '/gpu:0'], ['/gpu:0', '/gpu:0']]:
      shape = (3, 4)
      np_ans = None
      tensors = []
      for d in devices:
        with ops.device(d):
          t = ((np.random.random_sample(shape) - .5) * 1024).astype(np_type)
          if np_ans is None:
            np_ans = t
          else:
            np_ans = numpy_accumulation_fn(np_ans, t)
          tensors.append(array_ops.identity(t))

      all_reduce_tensors = nccl_fn(tensors)

      # Test shape inference.
      for r in all_reduce_tensors:
        self.assertEqual(shape, r.get_shape())

      # Test execution and results.
      nccl_results = sess.run(all_reduce_tensors)
      for r in nccl_results:
        self.assertAllClose(r, np_ans)

  def testErrors(self):
    with self.assertRaisesRegexp(ValueError, 'Device assignment required'):
      nccl.all_sum([array_ops.identity(np.random.random_sample((3, 4)))])
    with self.assertRaisesRegexp(ValueError, 'Must pass >0 tensors'):
      nccl.all_sum([])


class BroadcastTest(test.TestCase):

  def testBroadcast(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    for dtype in [np.float32, np.int32, np.int64, np.float64]:
      # Create session inside outer loop to test use of
      # same communicator across multiple sessions.
      with self.test_session(use_gpu=True) as sess:
        for devices in [['/gpu:0', '/gpu:0', '/gpu:0'], ['/gpu:0', '/gpu:0']]:
          shape = (3, 4)
          sender = np.random.randint(0, len(devices) - 1)
          with ops.device(devices[sender]):
            np_ans = ((
                (np.random.random_sample(shape) - .5) * 1024).astype(dtype))
            t = array_ops.identity(np_ans)
          other_devices = devices[:sender] + devices[sender + 1:]
          send_op, received_tensors = nccl.broadcast(t, other_devices)

          # Verify shape inference.
          for r in received_tensors:
            self.assertEqual(shape, r.get_shape())

          # Run and verify results.
          nccl_results = sess.run(received_tensors + [send_op])
          for r in nccl_results[:-1]:
            self.assertAllClose(r, np_ans)


class CombinedTest(test.TestCase):
  """Tests using a mix of all-reduce ops in one session.run call."""

  def testCombined(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    for dtype in [np.float32, np.int32, np.int64, np.float64]:
      # Create session inside outer loop to test use of
      # same communicator across multiple sessions.
      with self.test_session(use_gpu=True) as sess:
        for devices in [['/gpu:0', '/gpu:0', '/gpu:0'], ['/gpu:0', '/gpu:0']]:
          shape = (3, 4)

          # all-reduce
          np_ans = np.zeros(shape=shape, dtype=dtype)
          tensors = []
          for d in devices:
            with ops.device(d):
              t = ((np.random.random_sample(shape) - .5) * 1024).astype(dtype)
              np_ans += t
              tensors.append(array_ops.identity(t))
          all_reduce_tensors = nccl.all_sum(tensors)

          sender = np.random.randint(0, len(devices) - 1)
          other_devices = devices[:sender] + devices[sender + 1:]
          send_op, received_tensors = nccl.broadcast(all_reduce_tensors[sender],
                                                     other_devices)

          # sender doesn't need to be fetched as part of outputs of session.run.
          del all_reduce_tensors[sender]

          # Verify shape inference.
          for r in received_tensors:
            self.assertEqual(shape, r.get_shape())

          # Run and verify results.
          nccl_results = sess.run(
              received_tensors + [send_op] + all_reduce_tensors)
          for r in nccl_results[:len(received_tensors)]:
            self.assertAllClose(r, np_ans)


if __name__ == '__main__':
  test.main()
