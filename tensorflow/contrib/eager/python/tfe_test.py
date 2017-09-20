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
"""TensorFlow Eager Execution: Sanity tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.eager.python import tfe
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TFETest(test_util.TensorFlowTestCase):

  def testMatmul(self):
    x = [[2.]]
    y = math_ops.matmul(x, x)  # tf.matmul
    self.assertAllEqual([[4.]], y.numpy())

  def testInstantError(self):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r'indices = 7 is not in \[0, 3\)'):
      array_ops.gather([0, 1, 2], 7)

  def testGradients(self):

    def square(x):
      return math_ops.multiply(x, x)

    grad = tfe.gradients_function(square)
    self.assertEquals([6], [x.numpy() for x in grad(3)])

  def testGradOfGrad(self):

    def square(x):
      return math_ops.multiply(x, x)

    grad = tfe.gradients_function(square)
    gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
    self.assertEquals([2], [x.numpy() for x in gradgrad(3)])

  def testCustomGrad(self):

    @tfe.custom_gradient
    def f(x):
      y = math_ops.multiply(x, x)

      def grad_fn(_):
        return [x + y]

      return y, grad_fn

    # TODO(ashankar): This [0] should ideally not be needed.
    grad = tfe.gradients_function(f, [0])
    self.assertEquals([12], [x.numpy() for x in grad(3)])

  def testGPU(self):
    if tfe.num_gpus() <= 0:
      self.skipTest('No GPUs available')

    # tf.Tensor.as_gpu_device() moves a tensor to GPU.
    x = constant_op.constant([[1., 2.], [3., 4.]]).as_gpu_tensor()
    # Alternatively, tfe.device() as a context manager places tensors and
    # operations.
    with tfe.device('gpu:0'):
      x += 1.
    # Without a device context, heuristics are used to place ops.
    # In this case, ops.reduce_mean runs on the GPU.
    reduction_indices = range(x.shape.ndims)
    m = math_ops.reduce_mean(x, reduction_indices)
    # m is on GPU, bring it back to CPU and compare.
    self.assertEqual(3.5, m.as_cpu_tensor().numpy())

  def testListDevices(self):
    # Expect at least one device.
    self.assertTrue(tfe.list_devices())

  def testNumGPUs(self):
    devices = tfe.list_devices()
    self.assertEqual(len(devices) - 1, tfe.num_gpus())

  def testCallingEnableEagerExecutionMoreThanOnce(self):
    # Note that eager.test.main() has already invoked enable_eager_exceution().
    with self.assertRaisesRegexp(
        ValueError, r'Do not call tfe\.%s more than once in the same process' %
        tfe.enable_eager_execution.__name__):
      tfe.enable_eager_execution()


if __name__ == '__main__':
  tfe.enable_eager_execution()
  test.main()
