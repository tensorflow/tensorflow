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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _make_converter(dtype):
  def _converter(x):
    return np.asarray(x).astype(dtype.as_numpy_dtype)
  return _converter


class TensorArrayTest(xla_test.XLATestCase):

  def testTensorArrayWriteRead(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0, 3.0]])
      w2 = w1.write(2, [[7.0, -8.5]])

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)
      flow = w2.flow

      d0, d1, d2, flow_val = session.run([r0, r1, r2, flow])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0, 3.0]], d1)
      self.assertAllEqual([[7.0, -8.5]], d2)
      self.assertAllEqual([], flow_val.shape)

  def _testTensorArrayWritePack(self, tf_dtype):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      convert = _make_converter(tf_dtype)

      w0 = ta.write(0, convert([[4.0, 5.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.stack()

      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]), c0.eval())

  def testTensorArrayWritePack(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWritePack(dtype)

  def testEmptyTensorArrayPack(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      empty_element = np.zeros((0, 1), dtype=np.float32)
      w0 = ta.write(0, empty_element)
      w1 = w0.write(1, empty_element)
      w2 = w1.write(2, empty_element)

      c0 = w2.stack()

      self.assertAllEqual([3, 0, 1], c0.eval().shape)

  def _testTensorArrayWriteConcat(self, tf_dtype):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      convert = _make_converter(tf_dtype)

      w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0], [204.0, 205.0]]))

      c0 = w2.concat()

      self.assertAllEqual(
          convert([[4.0, 5.0], [104.0, 105.0], [6.0, 7.0],
                   [106.0, 107.0], [8.0, 9.0], [204.0, 205.0]]), c0.eval())

  def testTensorArrayWriteConcat(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWriteConcat(dtype)

  def _testTensorArrayUnpackRead(self, tf_dtype):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      convert = _make_converter(tf_dtype)

      # Unpack a vector into scalars
      w0 = ta.unstack(convert([1.0, 2.0, 3.0]))
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      # Unpack a matrix into vectors.
      w1 = ta.unstack(convert([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]))
      r0 = w1.read(0)
      r1 = w1.read(1)
      r2 = w1.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 1.1]), d0)
      self.assertAllEqual(convert([2.0, 2.1]), d1)
      self.assertAllEqual(convert([3.0, 3.1]), d2)

      # Reset ta because we're going to change the shape, else shape
      # inference will throw an error.
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      # Try unpacking an empty matrix, which should not cause an error.
      w2 = ta.unstack(convert([[], [], []]))
      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

  def _testTensorArrayUnpackReadMaybeLegacy(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayUnpackRead(dtype)

  def testTensorArrayUnpackRead(self):
    self._testTensorArrayUnpackReadMaybeLegacy()

  def _testTensorArraySplitRead(self, tf_dtype):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      convert = _make_converter(tf_dtype)

      # Split an empty vector.
      lengths = constant_op.constant([0, 0, 0])
      w0 = ta.split(convert([]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

      # Split a vector.
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)
      lengths = constant_op.constant([1, 1, 1])
      w0 = ta.split(convert([1.0, 2.0, 3.0]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0]), d0)
      self.assertAllEqual(convert([2.0]), d1)
      self.assertAllEqual(convert([3.0]), d2)

      # Split a matrix.
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)
      lengths = constant_op.constant([1, 1, 1])
      w0 = ta.split(
          convert([[1.0, 101.0], [2.0, 201.0], [3.0, 301.0]]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([[1.0, 101.0]]), d0)
      self.assertAllEqual(convert([[2.0, 201.0]]), d1)
      self.assertAllEqual(convert([[3.0, 301.0]]), d2)

  def testTensorArraySplitRead(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArraySplitRead(dtype)

  def testTensorGradArrayWriteRead(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3)

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

      d0, d1, d2, g_d0, g_d1, g_d2 = session.run([r0, r1, r2, g_r0, g_r1, g_r2])
      self.assertAllEqual([[4.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual([[-3.0]], d2)
      self.assertAllEqual([[5.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual([[-2.0]], g_d2)

  def testTensorGradArrayDynamicWriteRead(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3)

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

      d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs = session.run(
          [r0, r1, r2, g_r0, g_r1, g_r2, s, g_s])
      self.assertAllEqual([[4.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual([[-3.0]], d2)
      self.assertAllEqual([[5.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual([[-2.0]], g_d2)
      self.assertAllEqual(3, vs)
      self.assertAllEqual(3, g_vs)

  def testTensorGradAccessTwiceReceiveSameObject(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3,
          element_shape=[1, 2])
      g_ta_0 = ta.grad("grad")
      g_ta_1 = ta.grad("grad")

      with ops.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
        # Write with one gradient handle, read with another copy of it
        r1_0 = g_ta_1.read(0)

      t_g_ta_0, t_g_ta_1, d_r1_0 = session.run(
          [g_ta_0.handle.op, g_ta_1.handle.op, r1_0])
      self.assertAllEqual(t_g_ta_0, t_g_ta_1)
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      # Test writing the wrong datatype.
      with self.assertRaisesOpError(
          "TensorArray dtype is float but op has dtype int32"):
        ta.write(-1, np.int32(7)).flow.eval()

  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    # Find two different floating point types, create an array of
    # the first type, but try to read the other type.
    if len(self.float_types) > 1:
      dtype1, dtype2 = list(self.float_types)[:2]
      with self.test_session(), self.test_scope():
        ta = tensor_array_ops.TensorArray(
            dtype=dtype1, tensor_array_name="foo", size=3)

        w0 = ta.write(0, [[4.0, 5.0]])

        # Test reading wrong datatype.
        r0_bad = gen_data_flow_ops.tensor_array_read_v3(
            handle=w0.handle, index=0, dtype=dtype2, flow_in=w0.flow)
        with self.assertRaisesOpError("TensorArray dtype is "):
          r0_bad.eval()

        # Test reading from a different index than the one we wrote to
        w0.read(1)

  def testTensorArraySplitIncompatibleShapesFails(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      with self.assertRaisesOpError(
          r"value is not 1D"):
        lengths = array_ops.placeholder(dtypes.int64)
        ta.split([1.0, 2.0, 3.0], lengths).flow.eval(feed_dict={lengths: 1})

      with self.assertRaisesOpError(
          r"lengths must be equal: 1 vs. 2"):
        ta.split([1.0, 2.0, 3.0], [1, 2, 3]).flow.eval()

      with self.assertRaisesOpError(
          r"value must have rank >= 1"):
        ta.split(1.0, [1]).flow.eval()

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          infer_shape=False)

      with self.assertRaisesOpError(
          r"TensorArray's size is not equal to the size of lengths "
          r"\(1 vs. 2\)"):
        ta.split([1.0], [1]).flow.eval()

  def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)

      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      w0 = ta.write(2, c(3.0))
      w1 = w0.write(2, c(4.0))

      ta_grad = w1.grad("grad")

      w0_grad = ta_grad.write(2, c(3.0))
      w1_grad = w0_grad.write(2, c(4.0))
      w2_grad = w1_grad.write(2, c(5.0))

      # Assert that aggregation works correctly
      self.assertAllEqual(c(12.00), w2_grad.read(2).eval())

      # Using differing shapes causes an exception
      wb0_grad = ta_grad.write(1, c(1.0))
      wb1_grad = wb0_grad.write(1, c([1.0]))

      with self.assertRaisesOpError(
          r"Mismatched TensorArray sizes"):
        wb1_grad.flow.eval()

  def testTensorArrayWriteGradientAddMultipleAdds(self):
    for dtype in self.numeric_tf_types:
      self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

  def testMultiTensorArray(self):
    with self.test_session(), self.test_scope():
      h1 = tensor_array_ops.TensorArray(
          size=1, dtype=dtypes.float32, tensor_array_name="foo")
      w1 = h1.write(0, 4.0)
      r1 = w1.read(0)

      h2 = tensor_array_ops.TensorArray(
          size=1, dtype=dtypes.float32, tensor_array_name="bar")

      w2 = h2.write(0, 5.0)
      r2 = w2.read(0)
      r = r1 + r2
      self.assertAllClose(9.0, r.eval())

  def _testTensorArrayGradientWriteReadType(self, dtype):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.as_dtype(dtype),
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      c = lambda x: np.array(x, dtype=dtype)

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
      grad_just_r0_vals = session.run(grad_just_r0)
      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      grad_r0_r0_2 = gradients_impl.gradients(
          ys=[r0, r0_2],
          xs=[value_0],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
      grad_r0_r0_2_vals = session.run(grad_r0_r0_2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      grad_just_r1 = gradients_impl.gradients(
          ys=[r1], xs=[value_1], grad_ys=[c([[-2.0, -4.0]])])
      grad_just_r1_vals = session.run(grad_just_r1)
      self.assertAllEqual(c([[-2.0, -4.0]]), grad_just_r1_vals[0])

      # Test combined gradients
      grad = gradients_impl.gradients(
          ys=[r0, r0_2, r1],
          xs=[value_0, value_1],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]]), c([[-2.0, -10.0]])])
      grad_vals = session.run(grad)
      self.assertEqual(len(grad_vals), 2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
      self.assertAllEqual(c([[-2.0, -10.0]]), grad_vals[1])

  def testTensorArrayGradientWriteRead(self):
    for dtype in self.float_types:
      self._testTensorArrayGradientWriteReadType(dtype)
    for dtype in self.complex_types:
      self._testTensorArrayGradientWriteReadType(dtype)

  def _testTensorArrayGradientWritePackConcatAndRead(self):
    with self.test_session() as sess, self.test_scope():
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
        grad_r = gradients_impl.gradients(
            ys=[p0, r0, s0],
            xs=[value_0, value_1],
            grad_ys=[
                [[2.0, 3.0], [4.0, 5.0]],  # stack gradient
                [-0.5, 1.5],  # read(0) gradient
                [20.0, 30.0, 40.0, 50.0],  # concat gradient
            ])
      grad_vals = sess.run(grad_r)  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
      self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

  def testTensorArrayGradientWritePackConcatAndRead(self):
    self._testTensorArrayGradientWritePackConcatAndRead()

  def testTensorArrayReadTwice(self):
    with self.test_session(), self.test_scope():
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

      self.assertAllEqual([1.0, -1.0], r1_readtwice.eval())

  def _testTensorArrayGradientUnpackRead(self):
    with self.test_session() as session, self.test_scope():
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
      grad = gradients_impl.gradients(
          ys=[r0, r0_1, r1],
          xs=[value],
          grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientUnpackRead(self):
    self._testTensorArrayGradientUnpackRead()

  def testTensorArrayGradientSplitConcat(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=2)

      value = constant_op.constant(
          [[1.0, -1.0], [10.0, -10.0], [100.0, -100.0], [1000.0, -1000.0]])

      w = ta.split(value, [2, 2])
      r = w.concat()

      # Test combined gradients
      grad = gradients_impl.gradients(
          ys=[r],
          xs=[value],
          grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0],
                    [2000.0, -2000.0]]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, -2.0], [20.0, -20.0], [200.0, -200.0],
                           [2000.0, -2000.0]],
                          grad_vals[0])

  def testCloseTensorArray(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c1 = ta.close()
      session.run(c1)

  def testSizeTensorArray(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      s = ta.size()
      self.assertAllEqual(3, s.eval())

  def testWriteCloseTensorArray(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)
      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [3.0])
      w1.close().run()  # Expected to run without problems

  # TODO(phawkins): implement while loops.
  # def _testWhileLoopWritePackGradients(self, dynamic_size, dtype):
  #   np_dtype = dtype.as_numpy_dtype
  #   with self.test_session() as session, self.test_scope():
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
  #         session.run([state0, var, v0, vout, v0_grad, var_grad, state0_grad])
  #     )
  #     just_v0_grad_t, = session.run([v0_grad])

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
  #   with self.test_session(), self.test_scope():
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
  #     self.assertAllClose(31.0, grad.eval())

  def testSumOfTwoReadVariablesWithoutRepeatGrad(self):
    with self.test_session() as session, self.test_scope():
      a = array_ops.identity(
          np.arange(
              3 * 5, dtype=np.float32).reshape(3, 5) + 1)
      b = array_ops.identity(
          np.arange(
              3 * 5, dtype=np.float32).reshape(3, 5) + 1 + 3 * 5)
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      ta = ta.write(0, a, name="write_a")
      ta = ta.write(1, b, name="write_b")
      c = (
          ta.read(
              0, name="read_a_0") +  # a + b
          ta.read(
              1, name="read_b_0"))
      g0 = -(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)
      grad_a = gradients_impl.gradients([c], [a], [g0])[0]  # d(a+b)/da = 1
      grad_b = gradients_impl.gradients([c], [b], [g0])[0]  # d(a+b)/db = 1

      # Test gradients calculated individually
      grad_a_t, = session.run([grad_a])
      self.assertAllEqual(grad_a_t, g0)

      grad_b_t, = session.run([grad_b])
      self.assertAllEqual(grad_b_t, g0)

      # Test gradients calculated jointly.
      joint_grad_a_t, joint_grad_b_t = session.run([grad_a, grad_b])
      self.assertAllEqual(joint_grad_a_t, g0)
      self.assertAllEqual(joint_grad_b_t, g0)

  def testWriteShape(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c0 = constant_op.constant([4.0, 5.0])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c1 = constant_op.constant([6.0, 7.0])
      w1 = w0.write(1, c1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c2 = constant_op.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w0.write(0, c2)

  def testPartlyUnknownShape(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=6)

      c0 = array_ops.placeholder(dtypes.float32, [None, None, None, 3])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual([None, None, None, 3], r0.get_shape().as_list())

      c1 = array_ops.placeholder(dtypes.float32, [None, None, None, 3])
      w1 = w0.write(1, c1)
      r1 = w1.read(0)
      self.assertAllEqual([None, None, None, 3], r1.get_shape().as_list())

      # Writing less specific shape (doesn't change type.)
      c2 = array_ops.placeholder(dtypes.float32, [None, None, None, None])
      w2 = w1.write(2, c2)
      r2 = w2.read(0)
      self.assertAllEqual([None, None, None, 3], r2.get_shape().as_list())

      # Writing more specific shape in one dimension and less specific in
      # another.
      c3 = array_ops.placeholder(dtypes.float32, [None, None, 2, None])
      w3 = w2.write(3, c3)
      r3 = w3.read(0)
      self.assertAllEqual([None, None, 2, 3], r3.get_shape().as_list())

      # Writing partly defined shape using TensorArray.scatter.
      c4 = array_ops.placeholder(dtypes.float32, [2, None, 4, 2, 3])
      w4 = w3.scatter([4, 5], c4)
      r4 = w4.read(0)
      self.assertAllEqual([None, 4, 2, 3], r4.get_shape().as_list())

      # Writing fully defined shape using TensorArray.split.
      c5 = array_ops.placeholder(dtypes.float32, [10, 4, 2, 3])
      w5 = w4.split(c5, constant_op.constant([5, 5]))
      r5 = w5.read(0)
      self.assertAllEqual([5, 4, 2, 3], r5.get_shape().as_list())

  def _testUnpackShape(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          infer_shape=True)
      value = constant_op.constant(
          [[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])
      w0 = ta.unstack(value)
      r0 = w0.read(0)
      self.assertAllEqual((2,), r0.get_shape())

      c1 = constant_op.constant([4.0, 5.0])
      w1 = w0.write(3, c1)
      r1 = w1.read(0)
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      c2 = constant_op.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w1.write(4, c2)

  def testUnpackShape(self):
    self._testUnpackShape()

  def testSplitShape(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          infer_shape=True)
      value = constant_op.constant([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
      w0 = ta.split(value, [1, 1, 1])
      r0 = w0.read(0)
      self.assertAllEqual((1, 2), r0.get_shape())

      ta1 = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo1",
          size=0,
          infer_shape=True)
      w0 = ta1.split(value, [1, 2])
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def testWriteUnknownShape(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=True)
      c0 = array_ops.placeholder(dtypes.float32)
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def _testGradientWhenNotAllComponentsRead(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      x = constant_op.constant([2.0, 3.0])
      w = ta.unstack(x)
      r0 = w.read(0)
      # Calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
      grad_r0 = gradients_impl.gradients(ys=[r0], xs=[x], grad_ys=[1.0])
      grad_r0_vals = session.run(grad_r0)[0]
      self.assertAllEqual(grad_r0_vals, [1.0, 0.0])

  def testGradientWhenNotAllComponentsRead(self):
    self._testGradientWhenNotAllComponentsRead()

  def _testTensorArrayEvalEmpty(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=0, infer_shape=False)
      with self.assertRaisesOpError(
          "TensorArray has size zero, but element shape <unknown> is not fully "
          "defined. Currently only static shapes are supported when packing "
          "zero-size TensorArrays."):
        ta.stack().eval()

  def testTensorArrayEvalEmpty(self):
    self._testTensorArrayEvalEmpty()

  def _testTensorArrayEvalEmptyWithDefault(self):
    with self.test_session(), self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=0, infer_shape=True)
      self.assertEqual(0, ta.size().eval())
      ta = ta.unstack(array_ops.zeros([0, 3, 5]))
      packed = ta.stack()
      self.assertAllEqual([0, 3, 5], packed.eval().shape)
      # Concatenating zero tensors along their first dimension gives a
      # first dimension of zero
      self.assertAllEqual([0, 5], ta.concat().eval().shape)

  def testTensorArrayEvalEmptyWithDefault(self):
    self._testTensorArrayEvalEmptyWithDefault()

  def testTensorArrayScatterReadAndGradients(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=10)

      indices = constant_op.constant([1, 8])
      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.scatter(indices, value)
      r0 = w.read(1)
      r1 = w.read(8)

      # Test combined gradients + aggregation of read(0).
      grad = gradients_impl.gradients(
          ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
      read_vals, grad_vals = session.run([[r0, r1], grad])

      self.assertEqual(len(read_vals), 2)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([1.0, -1.0], read_vals[0])
      self.assertAllEqual([10.0, -10.0], read_vals[1])
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayWriteGatherAndGradients(self):
    with self.test_session() as session, self.test_scope():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=10)

      values = constant_op.constant([[1.0 * x, -1.0 * x] for x in range(10)])
      indices = constant_op.constant([1, 8])

      w = ta.unstack(values)
      g = w.gather(indices)

      # Test combined gradients + aggregation of read(0).
      grad = gradients_impl.gradients(
          ys=[g], xs=[values], grad_ys=[[[2.0, 3.0], [4.0, 5.0]]])
      g_vals, grad_vals = session.run([[g], grad])

      # Gradients for 8 of the 10 unread components are zero.
      expected_grad = np.zeros((10, 2))
      expected_grad[1] = [2.0, 3.0]
      expected_grad[8] = [4.0, 5.0]

      self.assertEqual(len(g_vals), 1)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[1.0, -1.0], [8.0, -8.0]], g_vals[0])
      self.assertAllEqual(expected_grad, grad_vals[0])

  def testTensorArrayIdentity(self):
    with self.test_session() as session, self.test_scope():
      ta0 = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2,
                                         infer_shape=False)
      ta1 = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=4,
                                         infer_shape=True)

      ta0 = ta0.write(0, 0.)
      ta1 = ta1.write(0, 1)

      v0 = resource_variable_ops.ResourceVariable(0)
      v1 = resource_variable_ops.ResourceVariable(0)

      with ops.control_dependencies([v0.assign_add(1)]):
        ta0 = ta0.identity()

      with ops.control_dependencies([v1.assign_add(1)]):
        ta1 = ta1.identity()

      read0 = ta0.read(0)
      read1 = ta1.read(0)

      size0 = ta0.size()
      size1 = ta1.size()

      # Tests correct properties on new TensorArrays.
      self.assertEqual(dtypes.float32, ta0.dtype)
      self.assertEqual(dtypes.int32, ta1.dtype)
      self.assertEqual(tensor_shape.unknown_shape(), read0.get_shape())
      self.assertEqual(tensor_shape.scalar(), read1.get_shape())

      variables.global_variables_initializer().run()

      read0_v, read1_v, size0_v, size1_v = session.run(
          (read0, read1, size0, size1))

      # Tests that the control dependencies was added and executed.
      self.assertEqual(1, v0.eval())
      self.assertEqual(1, v1.eval())

      # Tests correct TensorArray.
      self.assertEqual(read0_v, 0)
      self.assertEqual(read1_v, 1)
      self.assertEqual(size0_v, 2)
      self.assertEqual(size1_v, 4)

if __name__ == "__main__":
  test.main()
