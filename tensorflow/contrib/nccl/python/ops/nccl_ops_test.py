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

from functools import partial
import numpy as np

from tensorflow.contrib import nccl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _DeviceTensors(tensors, devices):
  res = []
  for t, d in zip(tensors, devices):
    with ops.device(d):
      res.append(array_ops.identity(t))
  return res


def _NcclAllReduce(nccl_fun, tensors, devices):
  return nccl_fun(_DeviceTensors(tensors, devices)), []


def _NcclReduce(nccl_fun, tensors, devices):
  d_tensors = _DeviceTensors(tensors, devices)
  receiver = np.random.randint(0, len(devices))
  received_tensor, send_ops = nccl_fun(d_tensors, devices[receiver])
  return [received_tensor], send_ops


def _NcclBroadcast(tensors, devices):
  sender = np.random.randint(0, len(devices))
  d_tensor = _DeviceTensors(tensors[0:1], devices[sender:sender + 1])[0]
  other_devices = devices[:sender] + devices[sender + 1:]
  send_op, received_tensors = nccl.broadcast(d_tensor, other_devices)
  return received_tensors, [send_op]


class NcclTestCase(test.TestCase):

  def _Test(self, nccl_reduce, numpy_fn):
    """Tests that nccl_reduce does the same as reduction with numpy_fn.

    Args:
      nccl_reduce: A function taking a list of tensors and a list of devices,
          and returns a list of reduced tensors and a list of ops to perform the
          reduction.
      numpy_fn: A function taking two tensors and returning the reduction of the
          two.
    """
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    for dtype in [np.float32, np.int32, np.int64, np.float64]:
      # Create session inside outer loop to test use of
      # same communicator across multiple sessions.
      with self.test_session(use_gpu=True) as sess:

        for devices in [['/device:GPU:1', '/device:GPU:2', '/device:GPU:0'],
                        ['/device:GPU:1', '/device:GPU:0']]:
          shape = (3, 4)
          random = (np.random.random_sample(shape) - .5) * 1024
          tensors = [random.astype(dtype)] * len(devices)
          np_ans = tensors[0]
          for t in tensors[1:]:
            np_ans = numpy_fn(np_ans, t)

          reduce_tensors, reduce_ops = nccl_reduce(tensors, devices)
          self.assertNotEmpty(reduce_tensors)

          # Test shape inference.
          for r in reduce_tensors:
            self.assertEqual(shape, r.get_shape())

          # Test execution and results.
          nccl_results = sess.run(reduce_tensors + reduce_ops)
          for r in nccl_results[:len(reduce_tensors)]:
            self.assertAllClose(r, np_ans)

  def _TestGradient(self, nccl_reduce, numpy_fn):
    """Tests the gradient of nccl_reduce.

    Args:
      nccl_reduce: A function taking a list of tensors and a list of devices,
          and returns a list of reduced tensors and a list of ops to perform the
          reduction.
      numpy_fn: A function taking two tensors and returning the gradient of the
          reduction of the two.
    """
    def _Gradient(tensors, devices):
      reduce_tensors, _ = nccl_reduce(tensors, devices)
      tensor_ops = [t.op for t in reduce_tensors]
      d_tensors = _DeviceTensors(tensors, devices)
      grad_tensors = [
          ops.get_gradient_function(op)(op, loss)
          for op, loss in zip(tensor_ops, d_tensors)
      ]
      return grad_tensors, []

    self._Test(_Gradient, numpy_fn)


class AllReduceTest(NcclTestCase):

  def testAllReduce(self):
    self._Test(partial(_NcclAllReduce, nccl.all_sum), lambda x, y: x + y)
    self._Test(partial(_NcclAllReduce, nccl.all_prod), lambda x, y: x * y)
    self._Test(partial(_NcclAllReduce, nccl.all_min), np.minimum)
    self._Test(partial(_NcclAllReduce, nccl.all_max), np.maximum)

  def testAllSumGrad(self):
    self._TestGradient(
        partial(_NcclAllReduce, nccl.all_sum), lambda x, y: x + y)

  def testErrors(self):
    with self.assertRaisesRegexp(ValueError, 'Device assignment required'):
      nccl.all_sum([array_ops.identity(np.random.random_sample((3, 4)))])
    with self.assertRaisesRegexp(ValueError, 'Must pass >0 tensors'):
      nccl.all_sum([])


class SingleReduceTest(NcclTestCase):

  def testSum(self):
    self._Test(partial(_NcclReduce, nccl.reduce_sum), lambda x, y: x + y)


class BroadcastTest(NcclTestCase):

  def testBroadcast(self):
    self._Test(_NcclBroadcast, lambda x, y: x)


class CombinedTest(NcclTestCase):
  """Test all-reduce vs. single-reduce plus broadcast in one session.run."""

  def _combined(self, tensors, devices):
    all_reduce_tensors = _NcclAllReduce(nccl.all_sum, tensors, devices)[0]
    single_reduce_tensors, single_reduce_ops = _NcclReduce(
        nccl.reduce_sum, tensors, devices)
    broadcast_tensors, broadcast_ops = _NcclBroadcast(single_reduce_tensors,
                                                      devices)
    all_tensors = all_reduce_tensors + single_reduce_tensors + broadcast_tensors
    return all_tensors, single_reduce_ops + broadcast_ops

  def testCombined(self):
    self._Test(self._combined, lambda x, y: x + y)


if __name__ == '__main__':
  test.main()
