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
"""Functional tests for XLA TensorArray Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _make_converter(dtype):
  def _converter(x):
    return np.asarray(x).astype(dtype.as_numpy_dtype)
  return _converter


# This lets me define `fn` repeatedly to pass to xla.compile.
#
# pylint: disable=function-redefined
@test_util.run_v1_only("b/")  # Support TF2 list operations
@test_util.with_control_flow_v2
class TensorArrayTest(xla_test.XLATestCase):

  @test_util.disable_control_flow_v2("Tries to evaluate flow")
  def testTensorArrayWriteRead(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0, 5.0]])
        w1 = w0.write(1, [[1.0, 3.0]])
        w2 = w1.write(2, [[7.0, -8.5]])

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)
        flow = w2.flow
        return [r0, r1, r2, flow]

      d0, d1, d2, flow_val = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0, 3.0]], d1)
      self.assertAllEqual([[7.0, -8.5]], d2)
      self.assertAllEqual([], flow_val.shape)

  def _testTensorArrayWritePack(self, tf_dtype):
    with self.session(), self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        w0 = ta.write(0, convert([[4.0, 5.0]]))
        w1 = w0.write(1, convert([[6.0, 7.0]]))
        w2 = w1.write(2, convert([[8.0, 9.0]]))

        return w2.stack()

      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]),
          self.evaluate(xla.compile(fn)[0]))

  def testTensorArrayWritePack(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWritePack(dtype)

  def testEmptyTensorArrayPack(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        empty_element = np.zeros((0, 1), dtype=np.float32)
        w0 = ta.write(0, empty_element)
        w1 = w0.write(1, empty_element)
        w2 = w1.write(2, empty_element)

        return w2.stack()

      self.assertAllEqual([3, 0, 1], self.evaluate(xla.compile(fn)[0]).shape)

  def _testTensorArrayWriteConcat(self, tf_dtype):
    with self.session(), self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0]]))
        w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
        w2 = w1.write(2, convert([[8.0, 9.0], [204.0, 205.0]]))

        return w2.concat()

      self.assertAllEqual(
          convert([[4.0, 5.0], [104.0, 105.0], [6.0, 7.0], [106.0, 107.0],
                   [8.0, 9.0], [204.0, 205.0]]),
          self.evaluate(xla.compile(fn)[0]))

  @test_util.disable_control_flow_v2("b/122315751 (concat)")
  def testTensorArrayWriteConcat(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWriteConcat(dtype)

  def _testTensorArrayUnpackRead(self, tf_dtype):
    with self.session() as session, self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Unpack a vector into scalars
        w0 = ta.unstack(convert([1.0, 2.0, 3.0]))
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)

        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Unpack a matrix into vectors.
        w1 = ta.unstack(
            convert([[1.0, 1.03125], [2.0, 2.03125], [3.0, 3.03125]]))
        r0 = w1.read(0)
        r1 = w1.read(1)
        r2 = w1.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))

      self.assertAllEqual(convert([1.0, 1.03125]), d0)
      self.assertAllEqual(convert([2.0, 2.03125]), d1)
      self.assertAllEqual(convert([3.0, 3.03125]), d2)

      def fn():
        # Reset ta because we're going to change the shape, else shape
        # inference will throw an error.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Try unpacking an empty matrix, which should not cause an error.
        w2 = ta.unstack(convert([[], [], []]))
        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

  def _testTensorArrayUnpackReadMaybeLegacy(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayUnpackRead(dtype)

  def testTensorArrayUnpackRead(self):
    self._testTensorArrayUnpackReadMaybeLegacy()

  def _testTensorArraySplitRead(self, tf_dtype):
    with self.session() as session, self.test_scope():
      convert = _make_converter(tf_dtype)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)

        # Split an empty vector.
        lengths = constant_op.constant([0, 0, 0])
        w0 = ta.split(convert([]), lengths=lengths)
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))

      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

      def fn():
        # Split a vector.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)
        lengths = constant_op.constant([1, 1, 1])
        w0 = ta.split(convert([1.0, 2.0, 3.0]), lengths=lengths)
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))

      self.assertAllEqual(convert([1.0]), d0)
      self.assertAllEqual(convert([2.0]), d1)
      self.assertAllEqual(convert([3.0]), d2)

      def fn():
        # Split a matrix.
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=3)
        lengths = constant_op.constant([1, 1, 1])
        w0 = ta.split(
            convert([[1.0, 101.0], [2.0, 201.0], [3.0, 301.0]]),
            lengths=lengths)
        r0 = w0.read(0)
        r1 = w0.read(1)
        r2 = w0.read(2)
        return [r0, r1, r2]

      d0, d1, d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual(convert([[1.0, 101.0]]), d0)
      self.assertAllEqual(convert([[2.0, 201.0]]), d1)
      self.assertAllEqual(convert([[3.0, 301.0]]), d2)

  @test_util.disable_control_flow_v2("b/122315872 (split)")
  def testTensorArraySplitRead(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArraySplitRead(dtype)

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorGradArrayWriteRead(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0]])
        w1 = w0.write(1, [[1.0]])
        w2 = w1.write(2, [[-3.0]])

        g_ta = w2.grad("grad")

        g_w0 = g_ta.write(0, [[5.0]])
        g_w1 = g_w0.write(1, [[2.0]])
        g_w2 = g_w1.write(2, [[-2.0]])

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)

        g_r0 = g_w2.read(0)
        g_r1 = g_w2.read(1)
        g_r2 = g_w2.read(2)

        return [r0, r1, r2, g_r0, g_r1, g_r2]

      d0, d1, d2, g_d0, g_d1, g_d2 = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual([[-3.0]], d2)
      self.assertAllEqual([[5.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual([[-2.0]], g_d2)

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorGradArrayDynamicWriteRead(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0]])
        w1 = w0.write(1, [[1.0]])
        w2 = w1.write(2, [[-3.0]])

        g_ta = w2.grad("grad")  # Get gradient array here so we know the shape

        s = w2.size()
        g_s = g_ta.size()

        g_w0 = g_ta.write(0, [[5.0]])
        g_w1 = g_w0.write(1, [[2.0]])
        g_w2 = g_w1.write(2, [[-2.0]])

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)

        g_r0 = g_w2.read(0)
        g_r1 = g_w2.read(1)
        g_r2 = g_w2.read(2)

        return [r0, r1, r2, g_r0, g_r1, g_r2, s, g_s]

      d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual([[-3.0]], d2)
      self.assertAllEqual([[5.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual([[-2.0]], g_d2)
      self.assertAllEqual(3, vs)
      self.assertAllEqual(3, g_vs)

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorGradAccessTwiceReceiveSameObject(self):
    with self.session() as session, self.test_scope():
      ta_out = {}

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            element_shape=[1, 2])

        g_ta_0 = ta.grad("grad")
        g_ta_1 = ta.grad("grad")

        ta_out[0] = g_ta_0.handle
        ta_out[1] = g_ta_1.handle

        with ops.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
          # Write with one gradient handle, read with another copy of it
          r1_0 = g_ta_1.read(0)

        with ops.control_dependencies([g_ta_0.handle.op, g_ta_1.handle.op]):
          return [r1_0]

      [d_r1_0] = self.evaluate(xla.compile(fn))
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

      # Can't assert this because adding a side output like we have here fails
      # as follows:
      #
      # ValueError: Operation u'TensorArrayGrad/TensorArrayGradV3' has been
      # marked as not fetchable.
      #
      # On the other hand, legitimately returning the handle from the
      # xla.compile function fails because we don't support DT_RESOURCE outputs
      # from XLA clusters.
      #
      # self.assertAllEqual(ta_out[0], ta_out[1])

  @test_util.disable_control_flow_v2("b/124334470")
  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        return ta.write(-1, constant_op.constant(7)).flow

      # Test writing the wrong datatype.
      # TODO(b/129870929): Remove InvalidArgumentError/second regexp after all
      # callers provide proper init dtype.
      with self.assertRaisesRegexp(
          (ValueError, errors.InvalidArgumentError),
          r"("
          r"conversion requested dtype float32 for Tensor with dtype int32"
          r"|"
          r"TensorArray dtype is float but op has dtype int32"
          r")"):
        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("b/124334096 verify dtype")
  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    # Find two different floating point types, create an array of
    # the first type, but try to read the other type.
    if len(self.float_types) > 1:
      dtype1, dtype2 = list(self.float_types)[:2]
      with self.session(), self.test_scope():

        def fn():
          ta = tensor_array_ops.TensorArray(
              dtype=dtype1, tensor_array_name="foo", size=3)

          w0 = ta.write(0, math_ops.cast([[4.0, 5.0]], dtype1))

          # Test reading wrong datatype.
          return gen_data_flow_ops.tensor_array_read_v3(
              handle=w0.handle, index=0, dtype=dtype2, flow_in=w0.flow)

        with self.assertRaisesOpError("TensorArray dtype is "):
          self.evaluate(xla.compile(fn))

        def fn():
          ta = tensor_array_ops.TensorArray(
              dtype=dtype1, tensor_array_name="foo", size=3)

          w0 = ta.write(0, math_ops.cast([[4.0, 5.0]], dtype1))

          # Test reading from a different index than the one we wrote to
          with ops.control_dependencies([w0.read(1)]):
            return 1.0

        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("b/122315872 (split)")
  def testTensorArraySplitIncompatibleShapesFails(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        return ta.split([1.0, 2.0, 3.0], 1).flow

      with self.assertRaisesWithPredicateMatch(
          ValueError, r"Shape must be rank 1 but is rank 0"):
        xla.compile(fn)[0].eval()

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        return ta.split([1.0, 2.0, 3.0], [1, 2, 3]).flow

      with self.assertRaisesOpError(
          r"lengths must be equal: 1 vs. 2"):
        xla.compile(fn)[0].eval()

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        return ta.split(1.0, [1]).flow

      with self.assertRaisesOpError(
          r"value must have rank >= 1"):
        xla.compile(fn)[0].eval()

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            infer_shape=False)

        return ta.split([1.0], [1]).flow

      with self.assertRaisesOpError(
          r"TensorArray's size is not equal to the size of lengths "
          r"\(1 vs. 2\)"):
        xla.compile(fn)[0].eval()

  def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
    with self.session(), self.test_scope():
      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)

        w0 = ta.write(2, c(3.0))
        w1 = w0.write(2, c(4.0))

        ta_grad = w1.grad("grad")

        w0_grad = ta_grad.write(2, c(3.0))
        w1_grad = w0_grad.write(2, c(4.0))
        w2_grad = w1_grad.write(2, c(5.0))

        return w2_grad.read(2)

      # Assert that aggregation works correctly
      self.assertAllEqual(c(12.00), xla.compile(fn)[0].eval())

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)

        w0 = ta.write(2, c(3.0))
        w1 = w0.write(2, c(4.0))

        ta_grad = w1.grad("grad")
        # Using differing shapes causes an exception
        wb0_grad = ta_grad.write(1, c(1.0))
        wb1_grad = wb0_grad.write(1, c([1.0]))

        return wb1_grad.flow

      with self.assertRaisesOpError(
          r"Mismatched TensorArray sizes"):
        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("TensorArray.grad is not supported in v2")
  def testTensorArrayWriteGradientAddMultipleAdds(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

  def testMultiTensorArray(self):
    with self.session(), self.test_scope():

      def fn():
        h1 = tensor_array_ops.TensorArray(
            size=1, dtype=dtypes.float32, tensor_array_name="foo")
        w1 = h1.write(0, 4.0)
        r1 = w1.read(0)

        h2 = tensor_array_ops.TensorArray(
            size=1, dtype=dtypes.float32, tensor_array_name="bar")

        w2 = h2.write(0, 5.0)
        r2 = w2.read(0)
        return r1 + r2

      self.assertAllClose(9.0, self.evaluate(xla.compile(fn)[0]))

  def _testTensorArrayGradientWriteReadType(self, dtype):
    with self.session() as session, self.test_scope():
      c = lambda x: np.array(x, dtype=dtype)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.as_dtype(dtype),
            tensor_array_name="foo",
            size=3,
            infer_shape=False)

        value_0 = constant_op.constant(c([[4.0, 5.0]]))
        value_1 = constant_op.constant(c([[3.0, 3.5]]))

        w0 = ta.write(0, value_0)
        w1 = w0.write(1, value_1)
        r0 = w1.read(0)
        r1 = w1.read(1)
        r0_2 = w1.read(0)

        # Test individual components' gradients
        grad_just_r0 = gradients_impl.gradients(
            ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
        grad_r0_r0_2 = gradients_impl.gradients(
            ys=[r0, r0_2],
            xs=[value_0],
            grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
        grad_just_r1 = gradients_impl.gradients(
            ys=[r1], xs=[value_1], grad_ys=[c([[-2.0, -4.0]])])
        # Test combined gradients
        grad = gradients_impl.gradients(
            ys=[r0, r0_2, r1],
            xs=[value_0, value_1],
            grad_ys=[c([[2.0, 3.0]]),
                     c([[1.0, -1.0]]),
                     c([[-2.0, -10.0]])])

        return [grad_just_r0, grad_r0_r0_2, grad_just_r1, grad]

      [grad_just_r0_vals, grad_r0_r0_2_vals, grad_just_r1_vals,
       grad_vals] = self.evaluate(xla.compile(fn))

      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      self.assertAllEqual(c([[-2.0, -4.0]]), grad_just_r1_vals[0])

      self.assertEqual(len(grad_vals), 2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
      self.assertAllEqual(c([[-2.0, -10.0]]), grad_vals[1])

  def testTensorArrayGradientWriteRead(self):
    for dtype in self.float_types:
      self._testTensorArrayGradientWriteReadType(dtype)
    for dtype in self.complex_types:
      self._testTensorArrayGradientWriteReadType(dtype)

  def _testTensorArrayGradientWritePackConcatAndRead(self):
    with self.session() as sess, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            clear_after_read=False)

        value_0 = constant_op.constant([-1.0, 1.0])
        value_1 = constant_op.constant([-10.0, 10.0])

        w0 = ta.write(0, value_0)
        w1 = w0.write(1, value_1)
        p0 = w1.stack()
        r0 = w1.read(0)
        s0 = w1.concat()

        # Test gradient accumulation between read(0), pack(), and concat().
        with ops.control_dependencies([p0, r0, s0]):
          return gradients_impl.gradients(
              ys=[p0, r0, s0],
              xs=[value_0, value_1],
              grad_ys=[
                  [[2.0, 3.0], [4.0, 5.0]],  # stack gradient
                  [-0.5, 1.5],  # read(0) gradient
                  [20.0, 30.0, 40.0, 50.0],  # concat gradient
              ])

      grad_vals = self.evaluate(xla.compile(fn))  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
      self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

  @test_util.disable_control_flow_v2("b/122315751 (concat)")
  def testTensorArrayGradientWritePackConcatAndRead(self):
    self._testTensorArrayGradientWritePackConcatAndRead()

  def testTensorArrayReadTwice(self):
    with self.session(), self.test_scope():

      def fn():
        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

        ta_readtwice = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            clear_after_read=False)
        w_readtwice = ta_readtwice.unstack(value)
        r0_readtwice = w_readtwice.read(0)
        with ops.control_dependencies([r0_readtwice]):
          r1_readtwice = w_readtwice.read(0)

        return [r0_readtwice, r1_readtwice]

      self.assertAllEqual([1.0, -1.0], self.evaluate(xla.compile(fn))[0])

  def _testTensorArrayGradientUnpackRead(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=2,
            clear_after_read=False)

        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

        w = ta.unstack(value)
        r0 = w.read(0)
        r0_1 = w.read(0)
        r1 = w.read(1)

        # Test combined gradients + aggregation of read(0).
        return gradients_impl.gradients(
            ys=[r0, r0_1, r1],
            xs=[value],
            grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])

      grad_vals = self.evaluate(xla.compile(fn))

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientUnpackRead(self):
    self._testTensorArrayGradientUnpackRead()

  @test_util.disable_control_flow_v2("b/122315751(concat), b/122315872(split)")
  def testTensorArrayGradientSplitConcat(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=2)

        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0],
                                      [100.0, -100.0], [1000.0, -1000.0]])

        w = ta.split(value, [2, 2])
        r = w.concat()

        # Test combined gradients
        return gradients_impl.gradients(
            ys=[r],
            xs=[value],
            grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0],
                      [2000.0, -2000.0]]])

      grad_vals = self.evaluate(xla.compile(fn))

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, -2.0], [20.0, -20.0], [200.0, -200.0],
                           [2000.0, -2000.0]],
                          grad_vals[0])

  def testCloseTensorArray(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        with ops.control_dependencies([ta.close()]):
          return 1.0

      self.evaluate(xla.compile(fn)[0])

  def testSizeTensorArray(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        return ta.size()

      self.assertAllEqual(3, self.evaluate(xla.compile(fn))[0])

  def testWriteCloseTensorArray(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            tensor_array_name="foo",
            size=3,
            infer_shape=False)
        w0 = ta.write(0, [[4.0, 5.0]])
        w1 = w0.write(1, [[3.0, 1.0]])
        with ops.control_dependencies([w1.close()]):
          return 1.0

      self.evaluate(xla.compile(fn))

  # TODO(phawkins): implement while loops.
  # def _testWhileLoopWritePackGradients(self, dynamic_size, dtype):
  #   np_dtype = dtype.as_numpy_dtype
  #   with self.session() as session, self.test_scope():
  #     v0 = array_ops.identity(np.arange(3 * 5, dtype=np_dtype).reshape(3, 5))
  #     var = variables.Variable(np.arange(100, 105, dtype=np_dtype))
  #     state0 = array_ops.identity(np.array([1] * 5, dtype=np_dtype))
  #     ta = tensor_array_ops.TensorArray(
  #         dtype=dtype,
  #         tensor_array_name="foo",
  #         size=0 if dynamic_size else 3,
  #         dynamic_size=dynamic_size)
  #     time_0 = array_ops.identity(0)

  #     def body(time, ta_t, state):
  #       sliced = array_ops.slice(
  #           v0, begin=array_ops.stack([time, 0]), size=[1, -1])
  #       sliced = array_ops.squeeze(sliced)
  #       out = sliced + var + state
  #       state += sliced
  #       ta_t = ta_t.write(time, out)
  #       return (time + 1, ta_t, state)

  #     (unused_0, h_final, unused_2) = control_flow_ops.while_loop(
  #         cond=lambda time, unused_1, unused_2: time < 3,
  #         body=body,
  #         loop_vars=(time_0, ta, state0),
  #         shape_invariants=(time_0.get_shape(), tensor_shape.unknown_shape(),
  #                           tensor_shape.unknown_shape()),
  #         parallel_iterations=3)
  #     vout = h_final.stack()

  #     grad_val = -np.arange(3 * 5, dtype=np_dtype).reshape(3, 5)
  #     v0_grad = gradients_impl.gradients([vout], [v0], [grad_val])[0]
  #     state0_grad = gradients_impl.gradients([vout], [state0], [grad_val])[0]
  #     var_grad = gradients_impl.gradients([vout], [var], [grad_val])[0]

  #     variables.global_variables_initializer().run()
  #     state0_t, var_t, v0_t, vout_t, v0_grad_t, var_grad_t, state0_grad_t = (
  #         self.evaluate([state0, var, v0, vout, v0_grad, var_grad, state0_grad])
  #     )
  #     just_v0_grad_t, = self.evaluate([v0_grad])

  #     # state = [ state0 | state0 + v0[0] | state0 + v0[0] + v0[1] ]
  #     # vout = [ v0[0] + var + state[0] |
  #     #          v0[1] + var + state[1] |
  #     #          v0[2] + var + state[2] ]
  #     #      = [ v0[0] + var + state0 |
  #     #          v0[1] + var + state0 + v0[0] |
  #     #          v0[2] + var + state0 + v0[0] + v0[1] ]
  #     #
  #     # d(vout[0])/d(v0) = [1 | 0 | 0 ]
  #     # d(vout[1])/d(v0) = [1 | 1 | 0 ]
  #     # d(vout[2])/d(v0) = [1 | 1 | 1 ]
  #     # d(vout)/d(var) = [1 | 1 | 1]
  #     # d(vout)/d(state0) = [ 1 | 1 | 1 ]

  #     state_per_time = np.array(
  #         [state0_t, state0_t + v0_t[0, :],
  #         state0_t + v0_t[0, :] + v0_t[1, :]])

  #     # Compare forward prop
  #     self.assertAllClose(v0_t + var_t + state_per_time, vout_t)

  #     # Compare backward prop
  #     expected_v0_grad_t = np.array([
  #         grad_val[0, :] + grad_val[1, :] + grad_val[2, :],
  #         grad_val[1, :] + grad_val[2, :], grad_val[2, :]
  #     ])

  #     self.assertAllEqual(expected_v0_grad_t, v0_grad_t)
  #     self.assertAllEqual(expected_v0_grad_t, just_v0_grad_t)
  #     self.assertAllClose(grad_val.sum(axis=0), var_grad_t)
  #     self.assertAllClose(grad_val.sum(axis=0), state0_grad_t)

  # def testWhileLoopWritePackGradients(self):
  #   self._testWhileLoopWritePackGradients(
  #       dynamic_size=False, dtype=dtypes.float32)
  #   # TODO(ebrevdo): re-enable when While supports non-float32 gradients.
  #   # self._testWhileLoopWritePackGradients(
  #   #     dynamic_size=False, dtype=tf.int64)

  # def testWhileLoopDynamicWritePackGradients(self):
  #   self._testWhileLoopWritePackGradients(
  #       dynamic_size=True, dtype=dtypes.float32)

  # def testGradSerialTwoLoops(self):
  #   with self.session(), self.test_scope():
  #     num_steps = 100
  #     acc = tensor_array_ops.TensorArray(
  #         dtype=dtypes.float32,
  #         size=num_steps,
  #         clear_after_read=False,
  #         element_shape=tensor_shape.scalar())
  #     i = constant_op.constant(0, name="i")
  #     x = constant_op.constant(2.0, name="x")

  #     c = lambda i, acc: i < 5

  #     def b(i, acc):
  #       x1 = control_flow_ops.cond(
  #           math_ops.equal(i, 0), lambda: x,
  #           lambda: math_ops.multiply(acc.read(i - 1), 2.0))
  #       return i + 1, acc.write(i, x1)

  #     i1, acc1 = control_flow_ops.while_loop(c, b, [i, acc])

  #     z = constant_op.constant(0.0)

  #     def fn(i, acc):
  #       return i + 1, acc.write(i, z)

  #     _, acc2 = control_flow_ops.while_loop(lambda i, acc: i < num_steps, fn,
  #                                           [i1, acc1])

  #     r = acc2.stack()
  #     grad = gradients_impl.gradients(r, [x])[0]
  #     self.assertAllClose(31.0, self.evaluate(grad))

  def testSumOfTwoReadVariablesWithoutRepeatGrad(self):
    with self.session() as session, self.test_scope():
      g0 = -(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)

      def fn():
        a = array_ops.identity(
            np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)
        b = array_ops.identity(
            np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1 + 3 * 5)
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
        ta = ta.write(0, a, name="write_a")
        ta = ta.write(1, b, name="write_b")
        c = (
            ta.read(0, name="read_a_0") +  # a + b
            ta.read(1, name="read_b_0"))
        grad_a = gradients_impl.gradients([c], [a], [g0])[0]  # d(a+b)/da = 1
        grad_b = gradients_impl.gradients([c], [b], [g0])[0]  # d(a+b)/db = 1

        return [grad_a, grad_b]

      grad_a, grad_b = xla.compile(fn)

      # Test gradients calculated individually
      grad_a_t, = self.evaluate([grad_a])
      self.assertAllEqual(grad_a_t, g0)

      grad_b_t, = self.evaluate([grad_b])
      self.assertAllEqual(grad_b_t, g0)

      # Test gradients calculated jointly.
      joint_grad_a_t, joint_grad_b_t = self.evaluate([grad_a, grad_b])
      self.assertAllEqual(joint_grad_a_t, g0)
      self.assertAllEqual(joint_grad_b_t, g0)

  def testWriteShape(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        c0 = constant_op.constant([4.0, 5.0])
        w0 = ta.write(0, c0)
        r0 = w0.read(0)

        return [c0, r0]

      c0, r0 = xla.compile(fn)

      self.assertAllEqual(c0.get_shape(), r0.get_shape())

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        c1 = constant_op.constant([6.0, 7.0])
        w0 = ta.write(0, c0)
        w1 = w0.write(1, c1)
        r0 = w1.read(0)
        r1 = w1.read(1)

        return [r0, c1, r1]

      [r0, c1, r1] = xla.compile(fn)

      self.assertAllEqual(c0.get_shape(), r0.get_shape())
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=3)
        w0 = ta.write(0, c0)
        c2 = constant_op.constant([4.0, 5.0, 6.0])
        return w0.write(0, c2).flow

      with self.assertRaises(ValueError):
        self.evaluate(xla.compile(fn))

  def _testGradientWhenNotAllComponentsRead(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
        x = constant_op.constant([2.0, 3.0])
        w = ta.unstack(x)
        r0 = w.read(0)
        # Calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
        return gradients_impl.gradients(ys=[r0], xs=[x], grad_ys=[1.0])

      grad_r0_vals = self.evaluate(xla.compile(fn))[0]
      self.assertAllEqual(grad_r0_vals, [1.0, 0.0])

  def testGradientWhenNotAllComponentsRead(self):
    self._testGradientWhenNotAllComponentsRead()

  def _testTensorArrayEvalEmpty(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=0, infer_shape=False)
        return ta.stack()

      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError, "Uninitialized TensorArray passed to "
          "TensorArrayStack/TensorArrayGatherV3"):
        xla.compile(fn)[0].eval()

  @test_util.disable_control_flow_v2("b/124335246")
  def testTensorArrayEvalEmpty(self):
    self._testTensorArrayEvalEmpty()

  def _testTensorArrayEvalEmptyWithDefault(self):
    with self.session(), self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=0, infer_shape=True)
        size = ta.size()
        ta = ta.unstack(array_ops.zeros([0, 3, 5]))
        return [size, ta.stack()]

      [size, stack] = self.evaluate(xla.compile(fn))
      self.assertEqual(0, size)
      self.assertAllEqual([0, 3, 5], stack.shape)
      # Concatenating zero tensors along their first dimension gives a
      # first dimension of zero
      if not control_flow_util.ENABLE_CONTROL_FLOW_V2:

        def fn():
          ta = tensor_array_ops.TensorArray(
              dtype=dtypes.float32, size=0, infer_shape=True)
          ta = ta.unstack(array_ops.zeros([0, 3, 5]))
          return ta.concat()

        # TODO(b/122315751): Enable this.
        self.assertAllEqual([0, 5], self.evaluate(xla.compile(fn))[0].shape)

  def testTensorArrayEvalEmptyWithDefault(self):
    self._testTensorArrayEvalEmptyWithDefault()

  def _testTensorArrayScatterRead(self, tf_dtype):
    with self.session() as session, self.test_scope():
      convert = _make_converter(tf_dtype)
      id0 = array_ops.placeholder(dtypes.int32)
      id1 = array_ops.placeholder(dtypes.int32)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=tf_dtype, tensor_array_name="foo", size=10)

        indices = constant_op.constant([1, 8])
        value = constant_op.constant(convert([[1.0, -1.0], [10.0, -10.0]]))

        w = ta.scatter(indices, value)
        r0 = w.read(id0)
        r1 = w.read(id1)

        return [r0, r1]

      # Test aggregation of read
      read_vals = session.run(xla.compile(fn), feed_dict={id0: 1, id1: 8})
      self.assertAllEqual(convert([1.0, -1.0]), read_vals[0])
      self.assertAllEqual(convert([10.0, -10.0]), read_vals[1])

  @test_util.disable_control_flow_v2("b/122315734 (scatter)")
  def testTensorArrayScatterRead(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayScatterRead(dtype)
    self._testTensorArrayScatterRead(dtypes.bool)

  @test_util.disable_control_flow_v2("b/122315734 (scatter)")
  def testTensorArrayScatterReadAndGradients(self):
    with self.session() as session, self.test_scope():
      id0 = array_ops.placeholder(dtypes.int32)
      id1 = array_ops.placeholder(dtypes.int32)

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=10)

        indices = constant_op.constant([1, 8])
        value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

        w = ta.scatter(indices, value)
        r0 = w.read(id0)
        r1 = w.read(id1)

        # Test combined gradients + aggregation of read(0).
        grad = gradients_impl.gradients(
            ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
        return [[r0, r1], grad]

      read_vals, grad_vals = session.run(
          xla.compile(fn), feed_dict={
              id0: 1,
              id1: 8
          })

      self.assertEqual(len(read_vals), 2)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([1.0, -1.0], read_vals[0])
      self.assertAllEqual([10.0, -10.0], read_vals[1])
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  @test_util.disable_control_flow_v2("b/122315378 (gather)")
  def testTensorArrayWriteGatherAndGradients(self):
    with self.session() as session, self.test_scope():

      def fn():
        ta = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, tensor_array_name="foo", size=10)

        values = constant_op.constant([[1.0 * x, -1.0 * x] for x in range(10)])
        indices = constant_op.constant([1, 8])

        w = ta.unstack(values)
        g = w.gather(indices)

        # Test combined gradients + aggregation of read(0).
        grad = gradients_impl.gradients(
            ys=[g], xs=[values], grad_ys=[[[2.0, 3.0], [4.0, 5.0]]])
        return [[g], grad]

      g_vals, grad_vals = self.evaluate(xla.compile(fn))

      # Gradients for 8 of the 10 unread components are zero.
      expected_grad = np.zeros((10, 2))
      expected_grad[1] = [2.0, 3.0]
      expected_grad[8] = [4.0, 5.0]

      self.assertEqual(len(g_vals), 1)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[1.0, -1.0], [8.0, -8.0]], g_vals[0])
      self.assertAllEqual(expected_grad, grad_vals[0])

  def testTensorArrayIdentity(self):
    with self.session() as session, self.test_scope():
      tensor_arrays = {}

      v0 = resource_variable_ops.ResourceVariable(0.0)
      v1 = resource_variable_ops.ResourceVariable(0.0)

      def fn():
        ta0 = tensor_array_ops.TensorArray(
            dtype=dtypes.float32, size=2, infer_shape=False)
        ta1 = tensor_array_ops.TensorArray(
            dtype=dtypes.int32, size=4, infer_shape=True)

        ta0 = ta0.write(0, 0.)
        ta1 = ta1.write(0, 1)

        with ops.control_dependencies([v0.assign_add(1.0)]):
          ta0 = ta0.identity()

        with ops.control_dependencies([v1.assign_add(1.0)]):
          ta1 = ta1.identity()

        read0 = ta0.read(0)
        read1 = ta1.read(0)

        size0 = ta0.size()
        size1 = ta1.size()

        tensor_arrays[0] = ta0
        tensor_arrays[1] = ta1

        return [read0, read1, size0, size1, v0, v1]

      variables.global_variables_initializer().run()

      read0_v, read1_v, size0_v, size1_v, v0, v1 = self.evaluate(
          xla.compile(fn))

      # Tests correct properties on new TensorArrays.
      self.assertEqual(dtypes.float32, tensor_arrays[0].dtype)
      self.assertEqual(dtypes.int32, tensor_arrays[1].dtype)

      # Tests that the control dependencies was added and executed.
      self.assertEqual(1.0, v0)
      self.assertEqual(1.0, v1)

      # Tests correct TensorArray.
      self.assertEqual(read0_v, 0)
      self.assertEqual(read1_v, 1)
      self.assertEqual(size0_v, 2)
      self.assertEqual(size1_v, 4)

if __name__ == "__main__":
  test.main()
