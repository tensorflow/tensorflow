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
"""Tests for operations in eager execution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import tensor
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops


class TargetTest(test_util.TensorFlowTestCase):

  def testExecuteBasic(self):
    three = tensor.Tensor(3)
    five = tensor.Tensor(5)
    product = three * five
    self.assertEqual(15, product.numpy())

  def testMatMulGPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    three = tensor.Tensor([[3.]]).as_gpu_tensor()
    five = tensor.Tensor([[5.]]).as_gpu_tensor()
    product = math_ops.matmul(three, five)
    self.assertEqual([[15.0]], product.numpy())

  def testExecuteStringAttr(self):
    three = tensor.Tensor(3.0)
    checked_three = array_ops.check_numerics(three,
                                             message='just checking')
    self.assertEqual([[3]], checked_three.numpy())

  def testExecuteFloatAttr(self):
    three = tensor.Tensor(3.0)
    almost_three = tensor.Tensor(2.8)
    almost_equal = math_ops.approximate_equal(
        three, almost_three, tolerance=0.3)
    self.assertTrue(almost_equal.numpy())

  def testExecuteIntAttr(self):
    three = tensor.Tensor(3)
    four = tensor.Tensor(4)
    total = math_ops.add_n([three, four])
    self.assertEqual(7, total.numpy())

  def testExecuteBoolAttr(self):
    three = tensor.Tensor([[3]])
    five = tensor.Tensor([[5]])
    product = math_ops.matmul(three, five, transpose_a=True)
    self.assertEqual([[15]], product.numpy())

  def testExecuteOneListOutput(self):
    split_dim = tensor.Tensor(1)
    value = tensor.Tensor([[0, 1, 2], [3, 4, 5]])
    x1, x2, x3 = array_ops.split(value, 3, axis=split_dim)
    self.assertAllEqual([[0], [3]], x1.numpy())
    self.assertAllEqual([[1], [4]], x2.numpy())
    self.assertAllEqual([[2], [5]], x3.numpy())

  def testGraphMode(self):
    graph = ops.Graph()
    with graph.as_default(), context.graph_mode():
      array_ops.placeholder(dtypes.int32)
    self.assertEqual(1, len(graph.get_operations()))

  # Almost all TensorFlow kernels for GPU devices keep int32 tensors in host
  # memory.  This change approximates the same behavior for eager execution -
  # keeping int32 tensors in host memory.
  #
  # We do so to preclude the need for callers into such kernels from having to
  # explicitly place the int32 tensors in host memory. For example, prior to
  # this change one needed:
  #
  # with tfe.device('/gpu:0'):
  #   ...  # code here
  #   with tfe.device('/cpu:0'):
  #     shape = tfe.Tensor(...)
  #   y = tfe.ops.random_uniform(.., shape)
  #
  # Without the CPU device block tfe.ops.random_uniform would fail since the
  # kernel expects the shape in host memory.
  #
  # After this change, we simplify the code:
  #
  # with tfe.device('/gpu:0'):
  #   y = tfe.ops.random_uniform(, tfe.Tensor(...))
  #
  # The approximation is not exact since if there are GPU kernels which do not
  # require host memory for int32 tensors, there will be a discrepancy between
  # eager execution and TensorFlow graphs. However, as of July 2017, there
  # were no known GPU kernels that kept int32 tensors in device memory.
  def testInt32CPUDefault(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    with context.device('/gpu:0'):
      r = tensor.Tensor(1) + tensor.Tensor(2)
    self.assertEqual(r.numpy(), 3)

  def testExecuteListOutputLen1(self):
    split_dim = tensor.Tensor(1)
    value = tensor.Tensor([[0, 1, 2], [3, 4, 5]])
    result = array_ops.split(value, 1, axis=split_dim)
    self.assertTrue(isinstance(result, list))
    self.assertEqual(1, len(result))
    self.assertAllEqual([[0, 1, 2], [3, 4, 5]], result[0].numpy())

  def testExecuteListOutputLen0(self):
    empty = tensor.Tensor([], dtype=dtypes.int32)
    result = array_ops.unstack(empty, 0)
    self.assertTrue(isinstance(result, list))
    self.assertEqual(0, len(result))

  def testExecuteMultipleNonListOutput(self):
    x = tensor.Tensor([1, 2, 3, 4, 5, 6])
    y = tensor.Tensor([1, 3, 5])
    result = array_ops.listdiff(x, y)
    out, idx = result
    self.assertTrue(out is result.out)
    self.assertTrue(idx is result.idx)
    self.assertAllEqual([2, 4, 6], out.numpy())
    self.assertAllEqual([1, 3, 5], idx.numpy())

  def testExecuteMultipleListOutput(self):
    split_dim = tensor.Tensor(1, dtype=dtypes.int64)
    indices = tensor.Tensor([[0, 2], [0, 4], [0, 5], [1, 0], [1, 1]],
                            dtype=dtypes.int64)
    values = tensor.Tensor([2, 3, 5, 7, 11])
    shape = tensor.Tensor([2, 7], dtype=dtypes.int64)
    result = sparse_ops.gen_sparse_ops._sparse_split(  # pylint: disable=protected-access
        split_dim, indices, values, shape, num_split=2)
    output_indices, output_values, output_shape = result
    self.assertEqual(2, len(output_indices))
    self.assertEqual(2, len(output_values))
    self.assertEqual(2, len(output_shape))
    self.assertEqual(output_indices, result.output_indices)
    self.assertEqual(output_values, result.output_values)
    self.assertEqual(output_shape, result.output_shape)
    self.assertAllEqual([[0, 2], [1, 0], [1, 1]], output_indices[0].numpy())
    self.assertAllEqual([[0, 0], [0, 1]], output_indices[1].numpy())
    self.assertAllEqual([2, 7, 11], output_values[0].numpy())
    self.assertAllEqual([3, 5], output_values[1].numpy())
    self.assertAllEqual([2, 4], output_shape[0].numpy())
    self.assertAllEqual([2, 3], output_shape[1].numpy())

  # TODO(josh11b): Test an op that has multiple outputs, some but not
  # all of which are lists. Examples: barrier_take_many (currently
  # unsupported since it uses a type list) or sdca_optimizer (I don't
  # have an example of legal inputs & outputs).

  def testComposition(self):
    x = tensor.Tensor(1, dtype=dtypes.int32)
    three_x = x + x + x
    self.assertEquals(dtypes.int32, three_x.dtype)
    self.assertEquals(3, three_x.numpy())

  def testOperatorOverrides(self):
    a = tensor.Tensor([1])
    b = tensor.Tensor([-2])

    self.assertAllEqual((-a).numpy(), [-1])
    self.assertAllEqual(abs(b).numpy(), [2])

    self.assertAllEqual((a + b).numpy(), [-1])
    self.assertAllEqual((a - b).numpy(), [3])
    self.assertAllEqual((a * b).numpy(), [-2])
    # TODO(apassos) figure out why the line below hangs
    # self.assertAllEqual((a**b).numpy(), [1])
    self.assertAllEqual((a / b).numpy(), [1 / (-2)])
    self.assertAllEqual((a % b).numpy(), [-1])

    self.assertAllEqual((a < b).numpy(), [False])
    self.assertAllEqual((a <= b).numpy(), [False])
    self.assertAllEqual((a > b).numpy(), [True])
    self.assertAllEqual((a >= b).numpy(), [True])
    self.assertAllEqual((a == b), False)
    self.assertAllEqual((a != b), True)

    self.assertEqual(1, a[tensor.Tensor(0)].numpy())

  def test_basic_slice(self):
    npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
    t = tensor.Tensor(npt)

    self.assertAllEqual(npt[:, :, :], t[:, :, :].numpy())
    self.assertAllEqual(npt[::, ::, ::], t[::, ::, ::].numpy())
    self.assertAllEqual(npt[::1, ::1, ::1], t[::1, ::1, ::1].numpy())
    self.assertAllEqual(npt[::1, ::5, ::2], t[::1, ::5, ::2].numpy())
    self.assertAllEqual(npt[::-1, :, :], t[::-1, :, :].numpy())
    self.assertAllEqual(npt[:, ::-1, :], t[:, ::-1, :].numpy())
    self.assertAllEqual(npt[:, :, ::-1], t[:, :, ::-1].numpy())
    self.assertAllEqual(npt[-2::-1, :, ::1], t[-2::-1, :, ::1].numpy())
    self.assertAllEqual(npt[-2::-1, :, ::2], t[-2::-1, :, ::2].numpy())

  def testDegenerateSlices(self):
    npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
    t = tensor.Tensor(npt)
    # degenerate by offering a forward interval with a negative stride
    self.assertAllEqual(npt[0:-1:-1, :, :], t[0:-1:-1, :, :].numpy())
    # degenerate with a reverse interval with a positive stride
    self.assertAllEqual(npt[-1:0, :, :], t[-1:0, :, :].numpy())
    # empty interval in every dimension
    self.assertAllEqual(npt[-1:0, 2:2, 2:3:-1], t[-1:0, 2:2, 2:3:-1].numpy())

  def testEllipsis(self):
    npt = np.array(
        [[[[[1, 2], [3, 4], [5, 6]]], [[[7, 8], [9, 10], [11, 12]]]]])
    t = tensor.Tensor(npt)

    self.assertAllEqual(npt[0:], t[0:].numpy())
    # implicit ellipsis
    self.assertAllEqual(npt[0:, ...], t[0:, ...].numpy())
    # ellipsis alone
    self.assertAllEqual(npt[...], t[...].numpy())
    # ellipsis at end
    self.assertAllEqual(npt[0:1, ...], t[0:1, ...].numpy())
    # ellipsis at begin
    self.assertAllEqual(npt[..., 0:1], t[..., 0:1].numpy())
    # ellipsis at middle
    self.assertAllEqual(npt[0:1, ..., 0:1], t[0:1, ..., 0:1].numpy())

  def testShrink(self):
    npt = np.array([[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                     [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]])
    t = tensor.Tensor(npt)
    self.assertAllEqual(npt[:, :, :, :, 3], t[:, :, :, :, 3].numpy())
    self.assertAllEqual(npt[..., 3], t[..., 3].numpy())
    self.assertAllEqual(npt[:, 0], t[:, 0].numpy())
    self.assertAllEqual(npt[:, :, 0], t[:, :, 0].numpy())

  def testOpWithInputsOnDifferentDevices(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # The GPU kernel for the Reshape op requires that the
    # shape input be on CPU.
    value = tensor.Tensor([1., 2.]).as_gpu_tensor()
    shape = tensor.Tensor([2, 1])
    reshaped = array_ops.reshape(value, shape)
    self.assertAllEqual([[1], [2]], reshaped.as_cpu_tensor().numpy())

    # And if the shape is in device memory, it should complain
    # TODO(ashankar): Revisit this - perhaps instead of complaining,
    # it should implicitly copy the tensor to host memory?
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        'cannot compute Reshape as input #1 was expected to be on'):
      reshaped = array_ops.reshape(value, shape.as_gpu_tensor())

  def testInvalidInputDataType(self):
    # Fill requires the first input to be an int32 tensor.
    with self.assertRaisesRegexp(ValueError, 'int64'):
      array_ops.fill(tensor.Tensor([2], dtype=dtypes.int64), tensor.Tensor(1))

  def testOutputOnHostMemory(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')
    # The Shape op kernel on GPU places the output in host memory.
    value = tensor.Tensor([1.]).as_gpu_tensor()
    shape = array_ops.shape(value)
    self.assertEquals([1], shape.numpy())

  def testRandomUniform(self):
    scalar_shape = tensor.Tensor([], dtype=dtypes.int32)

    x = random_ops.random_uniform(scalar_shape)
    self.assertEquals(0, x.shape.ndims)
    self.assertEquals(dtypes.float32, x.dtype)

    x = random_ops.random_uniform(
        scalar_shape, minval=tensor.Tensor(5.), maxval=tensor.Tensor(6.))
    self.assertLess(x.numpy(), 6)
    self.assertGreaterEqual(x.numpy(), 5)


if __name__ == '__main__':
  test.main()
