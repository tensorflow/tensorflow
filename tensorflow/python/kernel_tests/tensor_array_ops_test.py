# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.ops.tensor_array_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.ops import tensor_array_ops


class TensorArrayCPUTest(tf.test.TestCase):
  _use_gpu = False

  def testTensorArrayWriteRead(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)

  def _testTensorArrayWritePack(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      w0 = ta.write(0, convert([[4.0, 5.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.pack()

      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]), c0.eval())

  def testTensorArrayWritePack(self):
    self._testTensorArrayWritePack(tf.float32)
    self._testTensorArrayWritePack(tf.float64)
    self._testTensorArrayWritePack(tf.int32)
    self._testTensorArrayWritePack(tf.int64)
    self._testTensorArrayWritePack(tf.complex64)
    self._testTensorArrayWritePack(tf.string)

  def _testTensorArrayWriteConcat(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0], [204.0, 205.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.concat()

      self.assertAllEqual(
          convert([[4.0, 5.0],
                   [104.0, 105.0],
                   [204.0, 205.0],
                   [6.0, 7.0],
                   [106.0, 107.0],
                   [8.0, 9.0]]), c0.eval())

  def testTensorArrayWriteConcat(self):
    self._testTensorArrayWriteConcat(tf.float32)
    self._testTensorArrayWriteConcat(tf.float64)
    self._testTensorArrayWriteConcat(tf.int32)
    self._testTensorArrayWriteConcat(tf.int64)
    self._testTensorArrayWriteConcat(tf.complex64)
    self._testTensorArrayWriteConcat(tf.string)

  def testTensorArrayUnpackWrongMajorSizeFails(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          r"Input value must have first dimension "
          r"equal to the array size \(2 vs. 3\)"):
        ta.unpack([1.0, 2.0]).flow.eval()

  def testTensorArrayPackNotAllValuesAvailableFails(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not read from TensorArray index 1 "
          "because it has not yet been written to."):
        ta.write(0, [[4.0, 5.0]]).pack().eval()

  def _testTensorArrayUnpackRead(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype is tf.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      # Unpack a vector into scalars
      w0 = ta.unpack(convert([1.0, 2.0, 3.0]))
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      # Unpack a matrix into vectors
      w1 = ta.unpack(convert([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]))
      r0 = w1.read(0)
      r1 = w1.read(1)
      r2 = w1.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 1.1]), d0)
      self.assertAllEqual(convert([2.0, 2.1]), d1)
      self.assertAllEqual(convert([3.0, 3.1]), d2)

  def testTensorArrayUnpackRead(self):
    self._testTensorArrayUnpackRead(tf.float32)
    self._testTensorArrayUnpackRead(tf.float64)
    self._testTensorArrayUnpackRead(tf.int32)
    self._testTensorArrayUnpackRead(tf.int64)
    self._testTensorArrayUnpackRead(tf.complex64)
    self._testTensorArrayUnpackRead(tf.string)

  def _testTensorArraySplitRead(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      # Split an empty vector
      lengths = tf.constant([0, 0, 0])
      w0 = ta.split(convert([]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

      # Split a vector
      lengths = tf.constant([2, 0, 1])
      w0 = ta.split(
          convert([1.0, 2.0, 3.0]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 2.0]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([3.0]), d2)

      # Split a matrix
      lengths = tf.constant([2, 0, 1])
      w0 = ta.split(
          convert([[1.0, 101.0], [2.0, 201.0], [3.0, 301.0]]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([[1.0, 101.0], [2.0, 201.0]]), d0)
      self.assertAllEqual(convert([]).reshape(0, 2), d1)
      self.assertAllEqual(convert([[3.0, 301.0]]), d2)

  def testTensorArraySplitRead(self):
    self._testTensorArraySplitRead(tf.float32)
    self._testTensorArraySplitRead(tf.float64)
    self._testTensorArraySplitRead(tf.int32)
    self._testTensorArraySplitRead(tf.int64)
    self._testTensorArraySplitRead(tf.complex64)
    self._testTensorArraySplitRead(tf.string)

  def testTensorGradArrayWriteRead(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      g_ta = ta.grad("grad")

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_w0 = g_ta.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2 = session.run([r0, r1, r2, g_r0, g_r1, g_r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)

  def testTensorGradArrayDynamicWriteRead(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=0, dynamic_size=True)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_ta = w2.grad("grad")  # Get gradient array here so we know the shape

      s = w2.size()
      g_s = g_ta.size()

      g_w0 = g_ta.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs = session.run([
          r0, r1, r2, g_r0, g_r1, g_r2, s, g_s])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)
      self.assertAllEqual(3, vs)
      self.assertAllEqual(3, g_vs)

  def testTensorGradAccessTwiceReceiveSameObject(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      g_ta_0 = ta.grad("grad")
      g_ta_1 = ta.grad("grad")

      with tf.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
        # Write with one gradient handle, read with another copy of it
        r1_0 = g_ta_1.read(0)

      t_g_ta_0, t_g_ta_1, d_r1_0 = session.run(
          [g_ta_0.handle, g_ta_1.handle, r1_0])
      self.assertAllEqual(t_g_ta_0, t_g_ta_1)
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      # Test writing the wrong datatype
      with self.assertRaisesOpError(
          "TensorArray dtype is float but Op is trying to write dtype string"):
        ta.write(-1, "wrong_type_scalar").flow.eval()

      # Test writing to a negative index
      with self.assertRaisesOpError(
          "Tried to write to index -1 but array is not "
          "resizeable and size is: 3"):
        ta.write(-1, 3.0).flow.eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to write to index 3 but array is not "
          "resizeable and size is: 3"):
        ta.write(3, 3.0).flow.eval()

  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      w0 = ta.write(0, [[4.0, 5.0]])

      # Test reading wrong datatype
      r0_bad = gen_data_flow_ops._tensor_array_read(
          handle=w0.handle, index=0, dtype=tf.int64, flow_in=w0.flow)
      with self.assertRaisesOpError(
          "TensorArray dtype is float but Op requested dtype int64."):
        r0_bad.eval()

      # Test reading from a different index than the one we wrote to
      r1 = w0.read(1)
      with self.assertRaisesOpError(
          "Could not read from TensorArray index 1 because "
          "it has not yet been written to."):
        r1.eval()

      # Test reading from a negative index
      with self.assertRaisesOpError(
          r"Tried to read from index -1 but array size is: 3"):
        ta.read(-1).eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to read from index 3 but array size is: 3"):
        ta.read(3).eval()

  def testTensorArrayWriteMultipleFails(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not write to TensorArray index 2 because "
          "it has already been written to."):
        ta.write(2, 3.0).write(2, 3.0).flow.eval()

  def testTensorArrayConcatIncompatibleShapesFails(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      w1 = ta.write(0, 3.0)
      w2 = w1.write(1, 4.0)
      w3 = w2.write(2, [3.0])

      with self.assertRaisesOpError(
          "Concat saw a scalar shape at index 0 but requires at least vectors"):
        w3.concat().eval()

      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      w1 = ta.write(0, [3.0])
      w2 = w1.write(1, [4.0])
      w3 = w2.write(2, [[3.0]])

      with self.assertRaisesOpError(
          r"TensorArray has inconsistent shapes.  Index 0 has "
          r"\(excepting dimension 0\) shape: \[\] but index 2 has \(excepting "
          r"dimension 0\) shape: \[1\]"):
        w3.concat().eval()

  def testTensorArraySplitIncompatibleShapesFails(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          r"Expected lengths to be a vector, received shape: \[\]"):
        lengths = tf.placeholder(tf.int64)
        ta.split([1.0, 2.0, 3.0], lengths).flow.eval(feed_dict={lengths: 1})

      with self.assertRaisesOpError(
          r"Expected sum of lengths to be equal to values.shape\[0\], "
          r"but sum of lengths is 1 and value's shape is: \[3\]"):
        ta.split([1.0, 2.0, 3.0], [1]).flow.eval()

      with self.assertRaisesOpError(
          r"Expected value to be at least a vector, but received shape: \[\]"):
        ta.split(1.0, [1]).flow.eval()

      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2)

      with self.assertRaisesOpError(
          r"TensorArray's size is not equal to the size of lengths "
          r"\(2 vs. 1\), and the TensorArray is not marked as "
          r"dynamically resizeable"):
        ta.split([1.0], [1]).flow.eval()

  def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=dtype, tensor_array_name="foo", size=3)
      ta_grad = ta.grad("grad")

      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      w0 = ta.write(2, c(3.0))
      w1 = w0.write(2, c(4.0))

      w0_grad = ta_grad.write(2, c(3.0))
      w1_grad = w0_grad.write(2, c(4.0))
      w2_grad = w1_grad.write(2, c(5.0))

      # Assert that aggregation works correctly
      self.assertAllEqual(c(12.00), w2_grad.read(2).eval())

      # Assert that if multiple_writes_aggregate is not enabled,
      # multiple writes raise an exception.
      with self.assertRaisesOpError(
          r"TensorArray foo: Could not write to TensorArray index 2 because "
          r"it has already been written to."):
        w1.flow.eval()

      # Using differing shapes causes an exception
      wb0_grad = ta_grad.write(1, c(1.0))
      wb1_grad = wb0_grad.write(1, c([1.0]))

      with self.assertRaisesOpError(
          r"Could not aggregate to TensorArray index 1 because the "
          r"existing shape is \[\] but the new input shape is \[1\]"):
        wb1_grad.flow.eval()

  def testTensorArrayWriteGradientAddMultipleAdds(self):
    for dtype in [tf.int32, tf.int64, tf.float32, tf.float64, tf.complex64]:
      self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

  def testMultiTensorArray(self):
    with self.test_session(use_gpu=self._use_gpu):
      h1 = tensor_array_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      w1 = h1.write(0, 4.0)
      r1 = w1.read(0)

      h2 = tensor_array_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="bar")

      w2 = h2.write(0, 5.0)
      r2 = w2.read(0)
      r = r1 + r2
      self.assertAllClose(9.0, r.eval())

  def testDuplicateTensorArrayFails(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      h1 = tensor_array_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      c1 = h1.write(0, 4.0)
      h2 = tensor_array_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      c2 = h2.write(0, 5.0)
      with self.assertRaises(errors.AlreadyExistsError):
        session.run([c1.flow, c2.flow])

  def _testTensorArrayGradientWriteReadType(self, dtype):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.as_dtype(dtype), tensor_array_name="foo", size=3)

      c = lambda x: np.array(x, dtype=dtype)

      value_0 = tf.constant(c([[4.0, 5.0]]))
      value_1 = tf.constant(c(3.0))

      w0 = ta.write(0, value_0)
      w1 = w0.write(1, value_1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      r0_2 = w1.read(0)

      # Test individual components' gradients
      grad_just_r0 = tf.gradients(
          ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
      grad_just_r0_vals = session.run(grad_just_r0)
      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      grad_r0_r0_2 = tf.gradients(
          ys=[r0, r0_2], xs=[value_0],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
      grad_r0_r0_2_vals = session.run(grad_r0_r0_2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      grad_just_r1 = tf.gradients(
          ys=[r1], xs=[value_1], grad_ys=[c(-2.0)])
      grad_just_r1_vals = session.run(grad_just_r1)
      self.assertAllEqual(c(-2.0), grad_just_r1_vals[0])

      # Test combined gradients
      grad = tf.gradients(
          ys=[r0, r0_2, r1], xs=[value_0, value_1],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]]), c(-2.0)])
      grad_vals = session.run(grad)
      self.assertEqual(len(grad_vals), 2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
      self.assertAllEqual(c(-2.0), grad_vals[1])

  def testTensorArrayGradientWriteRead(self):
    for dtype in (np.float32, np.float64, np.int32, np.int64, np.complex64):
      self._testTensorArrayGradientWriteReadType(dtype)

  def testTensorArrayGradientWritePackConcatAndRead(self):
    with self.test_session(use_gpu=self._use_gpu) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2,
          clear_after_read=False)

      value_0 = tf.constant([-1.0, 1.0])
      value_1 = tf.constant([-10.0, 10.0])

      w0 = ta.write(0, value_0)
      w1 = w0.write(1, value_1)
      p0 = w1.pack()
      r0 = w1.read(0)
      s0 = w1.concat()

      # Test gradient accumulation between read(0), pack(), and concat()
      with tf.control_dependencies([p0, r0, s0]):
        grad_r = tf.gradients(
            ys=[p0, r0, s0], xs=[value_0, value_1],
            grad_ys=[
                [[2.0, 3.0], [4.0, 5.0]],  # pack gradient
                [-0.5, 1.5],  # read(0) gradient
                [20.0, 30.0, 40.0, 50.0]])  # concat gradient
      grad_vals = sess.run(grad_r)  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
      self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

  def testTensorArrayReadTwice(self):
    with self.test_session(use_gpu=self._use_gpu):
      value = tf.constant([[1.0, -1.0], [10.0, -10.0]])

      ta_readonce = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2)

      w_readonce = ta_readonce.unpack(value)
      r0_readonce = w_readonce.read(0)
      with tf.control_dependencies([r0_readonce]):
        r1_readonce = w_readonce.read(0)

      with self.assertRaisesOpError(
          r"Could not read index 0 twice because it was cleared after a "
          r"previous read \(perhaps try setting clear_after_read = false\?\)"):
        r1_readonce.eval()

      ta_readtwice = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2,
          clear_after_read=False)
      w_readtwice = ta_readtwice.unpack(value)
      r0_readtwice = w_readtwice.read(0)
      with tf.control_dependencies([r0_readtwice]):
        r1_readtwice = w_readtwice.read(0)

      self.assertAllEqual([1.0, -1.0], r1_readtwice.eval())

  def testTensorArrayGradientUnpackRead(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2,
          clear_after_read=False)

      value = tf.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.unpack(value)
      r0 = w.read(0)
      r0_1 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = tf.gradients(
          ys=[r0, r0_1, r1], xs=[value],
          grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientSplitConcat(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2)

      value = tf.constant([[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])

      w = ta.split(value, [2, 1])
      r = w.concat()

      # Test combined gradients
      grad = tf.gradients(
          ys=[r], xs=[value],
          grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0]]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual(
          [[2.0, -2.0], [20.0, -20.0], [200.0, -200.0]], grad_vals[0])

  def testTensorArrayGradientDynamicUnpackRead(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=0, dynamic_size=True)

      value = tf.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.unpack(value)
      r0 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = tf.gradients(
          ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  def testCloseTensorArray(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      c1 = ta.close()
      session.run(c1)

  def testSizeTensorArray(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      s = ta.size()
      self.assertAllEqual(3, s.eval())

  def testWriteCloseTensorArray(self):
    with self.test_session(use_gpu=self._use_gpu):
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [3.0])
      w1.close().run()  # Expected to run without problems

      with self.assertRaisesOpError(
          r"TensorArray foo has already been closed."):
        with tf.control_dependencies([w1.close()]):
          w1.write(2, 3.0).flow.eval()

  def _testWhileLoopWritePackGradients(self, dynamic_size, dtype):
    np_dtype = dtype.as_numpy_dtype
    with self.test_session(use_gpu=self._use_gpu) as session:
      v0 = tf.identity(np.arange(3*5, dtype=np_dtype).reshape(3, 5))
      var = tf.Variable(np.arange(100, 105, dtype=np_dtype))
      state0 = tf.identity(np.array([1] * 5, dtype=np_dtype))
      ta = tensor_array_ops.TensorArray(
          dtype=dtype, tensor_array_name="foo",
          size=0 if dynamic_size else 3, dynamic_size=dynamic_size)
      time_0 = tf.identity(0)

      def body(time, ta_t, state):
        sliced = tf.slice(v0, begin=tf.pack([time, 0]), size=[1, -1])
        sliced = tf.squeeze(sliced)
        out = sliced + var + state
        state += sliced
        ta_t = ta_t.write(time, out)
        return (time+1, ta_t, state)

      (unused_0, h_final, unused_2) = tf.while_loop(
          cond=lambda time, unused_1, unused_2: time < 3,
          body=body,
          loop_vars=(time_0, ta, state0),
          parallel_iterations=3)
      vout = h_final.pack()

      grad_val = -np.arange(3*5, dtype=np_dtype).reshape(3, 5)
      v0_grad = tf.gradients([vout], [v0], [grad_val])[0]
      state0_grad = tf.gradients([vout], [state0], [grad_val])[0]
      var_grad = tf.gradients([vout], [var], [grad_val])[0]

      tf.initialize_all_variables().run()
      state0_t, var_t, v0_t, vout_t, v0_grad_t, var_grad_t, state0_grad_t = (
          session.run([state0, var, v0, vout, v0_grad, var_grad, state0_grad]))
      just_v0_grad_t, = session.run([v0_grad])

      # state = [ state0 | state0 + v0[0] | state0 + v0[0] + v0[1] ]
      # vout = [ v0[0] + var + state[0] |
      #          v0[1] + var + state[1] |
      #          v0[2] + var + state[2] ]
      #      = [ v0[0] + var + state0 |
      #          v0[1] + var + state0 + v0[0] |
      #          v0[2] + var + state0 + v0[0] + v0[1] ]
      #
      # d(vout[0])/d(v0) = [1 | 0 | 0 ]
      # d(vout[1])/d(v0) = [1 | 1 | 0 ]
      # d(vout[2])/d(v0) = [1 | 1 | 1 ]
      # d(vout)/d(var) = [1 | 1 | 1]
      # d(vout)/d(state0) = [ 1 | 1 | 1 ]

      state_per_time = np.array([
          state0_t,
          state0_t + v0_t[0, :],
          state0_t + v0_t[0, :] + v0_t[1, :]])

      # Compare forward prop
      self.assertAllClose(v0_t + var_t + state_per_time, vout_t)

      # Compare backward prop
      expected_v0_grad_t = np.array([
          grad_val[0, :] + grad_val[1, :] + grad_val[2, :],
          grad_val[1, :] + grad_val[2, :],
          grad_val[2, :]])

      self.assertAllEqual(expected_v0_grad_t, v0_grad_t)
      self.assertAllEqual(expected_v0_grad_t, just_v0_grad_t)
      self.assertAllClose(grad_val.sum(axis=0), var_grad_t)
      self.assertAllClose(grad_val.sum(axis=0), state0_grad_t)

  def testWhileLoopWritePackGradients(self):
    self._testWhileLoopWritePackGradients(
        dynamic_size=False, dtype=tf.float32)
    # TODO(ebrevdo): re-enable when While supports non-float32 gradients.
    # self._testWhileLoopWritePackGradients(
    #     dynamic_size=False, dtype=tf.int64)

  def testWhileLoopDynamicWritePackGradients(self):
    self._testWhileLoopWritePackGradients(
        dynamic_size=True, dtype=tf.float32)

  def testSumOfTwoReadVariablesWithoutRepeatGrad(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      a = tf.identity(np.arange(3*5, dtype=np.float32).reshape(3, 5) + 1)
      b = tf.identity(np.arange(3*5, dtype=np.float32).reshape(3, 5) + 1 + 3*5)
      ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=2)
      ta = ta.write(0, a, name="write_a")
      ta = ta.write(1, b, name="write_b")
      c = (ta.read(0, name="read_a_0") +  # a + b
           ta.read(1, name="read_b_0"))
      g0 = -(np.arange(3*5, dtype=np.float32).reshape(3, 5) + 1)
      grad_a = tf.gradients([c], [a], [g0])[0]  # d(a+b)/da = 1
      grad_b = tf.gradients([c], [b], [g0])[0]  # d(a+b)/db = 1

      # Test gradients calculated individually
      grad_a_t, = session.run([grad_a])
      self.assertAllEqual(grad_a_t, g0)

      grad_b_t, = session.run([grad_b])
      self.assertAllEqual(grad_b_t, g0)

      # Test gradients calculated jointly
      joint_grad_a_t, joint_grad_b_t = session.run([grad_a, grad_b])
      self.assertAllEqual(joint_grad_a_t, g0)
      self.assertAllEqual(joint_grad_b_t, g0)

  def _grad_source_for_name(self, name):
    return tensor_array_grad._GetGradSource(tf.constant(0, name=name))

  def testGetGradSource_Invalid(self):
    with self.assertRaises(ValueError):
      self._grad_source_for_name("")
    with self.assertRaises(ValueError):
      self._grad_source_for_name("foo")
    with self.assertRaises(ValueError):
      self._grad_source_for_name("foo/bar")

  def testGetGradSource_NoEnclosingScope(self):
    self.assertEqual("gradients:0", self._grad_source_for_name("gradients"))
    self.assertEqual("gradients_0:0", self._grad_source_for_name("gradients_0"))
    self.assertEqual("gradients", self._grad_source_for_name("gradients/foo"))
    self.assertEqual(
        "gradients_0", self._grad_source_for_name("gradients_0/foo"))
    self.assertEqual(
        "gradients", self._grad_source_for_name("gradients/foo/bar"))
    self.assertEqual(
        "gradients_0", self._grad_source_for_name("gradients_0/foo/bar"))

  def testGetGradSource_EnclosingScope(self):
    self.assertEqual(
        "foo/gradients:0", self._grad_source_for_name("foo/gradients"))
    self.assertEqual(
        "foo/gradients_0:0", self._grad_source_for_name("foo/gradients_0"))
    self.assertEqual(
        "foo/gradients", self._grad_source_for_name("foo/gradients/bar"))
    self.assertEqual(
        "foo/gradients_0", self._grad_source_for_name("foo/gradients_0/bar"))
    self.assertEqual(
        "foo/bar/gradients",
        self._grad_source_for_name("foo/bar/gradients/baz"))
    self.assertEqual(
        "foo/bar/gradients_0",
        self._grad_source_for_name("foo/bar/gradients_0/baz"))

  def testGetGradSource_NestedUsesInnermost(self):
    self.assertEqual(
        "foo/gradients/bar/gradients_0",
        self._grad_source_for_name("foo/gradients/bar/gradients_0/baz"))

  def testWriteShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3, infer_shape=True)
      c0 = tf.constant([4.0, 5.0])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())

      c1 = tf.constant([6.0, 7.0])
      w1 = w0.write(1, c1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      c2 = tf.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w0.write(0, c2)

  def testUnpackShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo",
          size=0, dynamic_size=True, infer_shape=True)
      value = tf.constant([[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])
      w0 = ta.unpack(value)
      r0 = w0.read(0)
      self.assertAllEqual((2,), r0.get_shape())

      c1 = tf.constant([4.0, 5.0])
      w1 = w0.write(3, c1)
      r1 = w1.read(0)
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      c2 = tf.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w1.write(4, c2)

  def testSplitShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo",
          size=0, dynamic_size=True, infer_shape=True)
      value = tf.constant([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
      w0 = ta.split(value, [1, 1, 1])
      r0 = w0.read(0)
      self.assertAllEqual((1, 2), r0.get_shape())

      ta1 = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo1",
          size=0, dynamic_size=True, infer_shape=True)
      w0 = ta1.split(value, [1, 2])
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def testWriteUnknownShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3, infer_shape=True)
      c0 = tf.placeholder(tf.float32)
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def testGradientWhenNotAllComponentsRead(self):
    with self.test_session(use_gpu=self._use_gpu) as session:
      ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=2)
      x = tf.constant([2.0, 3.0])
      w = ta.unpack(x)
      r0 = w.read(0)
      # calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
      grad_r0 = tf.gradients(ys=[r0], xs=[x], grad_ys=[1.0])
      grad_r0_vals = session.run(grad_r0)[0]
      self.assertAllEqual(grad_r0_vals, [1.0, 0.0])


class TensorArrayGPUTest(TensorArrayCPUTest):
  _use_gpu = True


if __name__ == "__main__":
  tf.test.main()
