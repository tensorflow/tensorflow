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

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
# pylint: enable=unused-import,g-bad-import-order

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_data_flow_ops


class TensorArrayTest(tf.test.TestCase):

  def _testTensorArrayWriteRead(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      w0 = h.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = sess.run([r0, r1, r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)

  def testTensorArrayWriteRead(self):
    self._testTensorArrayWriteRead(use_gpu=False)
    self._testTensorArrayWriteRead(use_gpu=True)

  def _testTensorArrayWritePack(self, tf_dtype, use_gpu):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=use_gpu):
      h = data_flow_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        convert = lambda x: np.asarray(x).astype(np.str)
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      w0 = h.write(0, convert([[4.0, 5.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.pack()

      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]), c0.eval())

  def _testTensorArrayWritePackWithType(self, tf_dtype):
    self._testTensorArrayWritePack(tf_dtype=tf_dtype, use_gpu=False)
    self._testTensorArrayWritePack(tf_dtype=tf_dtype, use_gpu=True)

  def testTensorArrayWritePack(self):
    self._testTensorArrayWritePackWithType(tf.float32)
    self._testTensorArrayWritePackWithType(tf.float64)
    self._testTensorArrayWritePackWithType(tf.int32)
    self._testTensorArrayWritePackWithType(tf.int64)
    self._testTensorArrayWritePackWithType(tf.complex64)
    self._testTensorArrayWritePackWithType(tf.string)

  def testTensorArrayUnpackWrongMajorSizeFails(self):
    with self.test_session():
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          r"Input value must have first dimension "
          r"equal to the array size \(2 vs. 3\)"):
        h.unpack([1.0, 2.0]).flow.eval()

  def testTensorArrayPackNotAllValuesAvailableFails(self):
    with self.test_session():
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not read from TensorArray index 1 "
          "because it has not yet been written to."):
        h.write(0, [[4.0, 5.0]]).pack().eval()

  def _testTensorArrayUnpackRead(self, tf_dtype, use_gpu):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        convert = lambda x: np.asarray(x).astype(np.str)
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      # Unpack a vector into scalars
      w0 = h.unpack(convert([1.0, 2.0, 3.0]))
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = sess.run([r0, r1, r2])
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      # Unpack a matrix into vectors
      w1 = h.unpack(convert([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]))
      r0 = w1.read(0)
      r1 = w1.read(1)
      r2 = w1.read(2)

      d0, d1, d2 = sess.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 1.1]), d0)
      self.assertAllEqual(convert([2.0, 2.1]), d1)
      self.assertAllEqual(convert([3.0, 3.1]), d2)

  def _testTensorArrayUnpackReadWithType(self, tf_dtype):
    self._testTensorArrayUnpackRead(tf_dtype=tf_dtype, use_gpu=False)
    self._testTensorArrayUnpackRead(tf_dtype=tf_dtype, use_gpu=True)

  def testTensorArrayUnpackRead(self):
    self._testTensorArrayUnpackReadWithType(tf.float32)
    self._testTensorArrayUnpackReadWithType(tf.float64)
    self._testTensorArrayUnpackReadWithType(tf.int32)
    self._testTensorArrayUnpackReadWithType(tf.int64)
    self._testTensorArrayUnpackReadWithType(tf.complex64)
    self._testTensorArrayUnpackReadWithType(tf.string)

  def _testTensorGradArrayWriteRead(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      g_h = h.grad()

      w0 = h.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_w0 = g_h.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2 = sess.run([r0, r1, r2, g_r0, g_r1, g_r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)

  def testTensorGradArrayWriteRead(self):
    self._testTensorGradArrayWriteRead(use_gpu=False)
    self._testTensorGradArrayWriteRead(use_gpu=True)

  def _testTensorGradAccessTwiceReceiveSameObject(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      g_h_0 = h.grad()
      g_h_1 = h.grad()

      with tf.control_dependencies([g_h_0.write(0, [[4.0, 5.0]]).flow]):
        # Write with one gradient handle, read with another copy of it
        r1_0 = g_h_1.read(0)

      t_g_h_0, t_g_h_1, d_r1_0 = sess.run([g_h_0.handle, g_h_1.handle, r1_0])
      self.assertAllEqual(t_g_h_0, t_g_h_1)
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

  def testTensorGradAccessTwiceReceiveSameObject(self):
    self._testTensorGradAccessTwiceReceiveSameObject(False)
    self._testTensorGradAccessTwiceReceiveSameObject(True)

  def _testTensorArrayWriteWrongIndexOrDataTypeFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      # Test writing the wrong datatype
      with self.assertRaisesOpError(
          "TensorArray dtype is float but Op is trying to write dtype string"):
        h.write(-1, "wrong_type_scalar").flow.eval()

      # Test writing to a negative index
      with self.assertRaisesOpError(
          "Tried to write to index -1 but array size is: 3"):
        h.write(-1, 3.0).flow.eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to write to index 3 but array size is: 3"):
        h.write(3, 3.0).flow.eval()

  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    self._testTensorArrayWriteWrongIndexOrDataTypeFails(use_gpu=False)
    self._testTensorArrayWriteWrongIndexOrDataTypeFails(use_gpu=True)

  def _testTensorArrayReadWrongIndexOrDataTypeFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      w0 = h.write(0, [[4.0, 5.0]])

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
        h.read(-1).eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to read from index 3 but array size is: 3"):
        h.read(3).eval()

  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    self._testTensorArrayReadWrongIndexOrDataTypeFails(use_gpu=False)
    self._testTensorArrayReadWrongIndexOrDataTypeFails(use_gpu=True)

  def _testTensorArrayWriteMultipleFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not write to TensorArray index 2 because "
          "it has already been written to."):
        h.write(2, 3.0).write(2, 3.0).flow.eval()

  def testTensorArrayWriteMultipleFails(self):
    self._testTensorArrayWriteMultipleFails(use_gpu=False)
    self._testTensorArrayWriteMultipleFails(use_gpu=True)

  def _testTensorArrayWriteGradientAddMultipleAddsType(self, use_gpu, dtype):
    with self.test_session(use_gpu=use_gpu):
      h = data_flow_ops.TensorArray(
          dtype=dtype, tensor_array_name="foo", size=3)
      h._gradient_add = True

      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      w0 = h.write(2, c(3.0))
      w1 = w0.write(2, c(4.0))

      self.assertAllEqual(c(7.00), w1.read(2).eval())

  def _testTensorArrayWriteGradientAddMultipleAdds(self, use_gpu):
    for dtype in [tf.int32, tf.int64, tf.float32, tf.float64, tf.complex64]:
      self._testTensorArrayWriteGradientAddMultipleAddsType(use_gpu, dtype)

  def testTensorArrayWriteGradientAddMultipleAdds(self):
    self._testTensorArrayWriteGradientAddMultipleAdds(use_gpu=False)
    self._testTensorArrayWriteGradientAddMultipleAdds(use_gpu=True)

  def _testMultiTensorArray(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h1 = data_flow_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      w1 = h1.write(0, 4.0)
      r1 = w1.read(0)

      h2 = data_flow_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="bar")

      w2 = h2.write(0, 5.0)
      r2 = w2.read(0)
      r = r1 + r2
      self.assertAllClose(9.0, r.eval())

  def testMultiTensorArray(self):
    self._testMultiTensorArray(use_gpu=False)
    self._testMultiTensorArray(use_gpu=True)

  def _testDuplicateTensorArrayFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h1 = data_flow_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      c1 = h1.write(0, 4.0)
      h2 = data_flow_ops.TensorArray(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      c2 = h2.write(0, 5.0)
      with self.assertRaises(errors.AlreadyExistsError):
        sess.run([c1.flow, c2.flow])

  def testDuplicateTensorArrayFails(self):
    self._testDuplicateTensorArrayFails(use_gpu=False)
    self._testDuplicateTensorArrayFails(use_gpu=True)

  def _testTensorArrayGradientWriteReadType(self, use_gpu, dtype):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.as_dtype(dtype), tensor_array_name="foo", size=3)

      c = lambda x: np.array(x, dtype=dtype)

      value_0 = tf.constant(c([[4.0, 5.0]]))
      value_1 = tf.constant(c(3.0))

      w0 = h.write(0, value_0)
      w1 = w0.write(1, value_1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      r0_2 = w1.read(0)

      # Test individual components' gradients
      grad_just_r0 = tf.gradients(
          ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
      grad_just_r0_vals = sess.run(grad_just_r0)
      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      grad_r0_r0_2 = tf.gradients(
          ys=[r0, r0_2], xs=[value_0],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
      grad_r0_r0_2_vals = sess.run(grad_r0_r0_2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      grad_just_r1 = tf.gradients(
          ys=[r1], xs=[value_1], grad_ys=[c(-2.0)])
      grad_just_r1_vals = sess.run(grad_just_r1)
      self.assertAllEqual(c(-2.0), grad_just_r1_vals[0])

      # Test combined gradients
      grad = tf.gradients(
          ys=[r0, r0_2, r1], xs=[value_0, value_1],
          grad_ys=[c(-1.0), c(-2.0), c([[2.0, 3.0]])])
      grad_vals = sess.run(grad)
      self.assertEqual(len(grad_vals), 2)
      self.assertAllClose(c(-3.0), grad_vals[0])
      self.assertAllEqual(c([[2.0, 3.0]]), grad_vals[1])

  def _testTensorArrayGradientWriteRead(self, use_gpu):
    for dtype in (np.float32, np.float64, np.int32, np.int64, np.complex64):
      self._testTensorArrayGradientWriteReadType(use_gpu, dtype)

  def testTensorArrayGradientWriteRead(self):
    self._testTensorArrayGradientWriteRead(False)
    self._testTensorArrayGradientWriteRead(True)

  def _testTensorArrayGradientWritePackAndRead(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2)

      value_0 = tf.constant([-1.0, 1.0])
      value_1 = tf.constant([-10.0, 10.0])

      w0 = h.write(0, value_0)
      w1 = w0.write(1, value_1)
      p0 = w1.pack()
      r0 = w1.read(0)

      # Test gradient accumulation between read(0) and pack()
      grad_r = tf.gradients(
          ys=[p0, r0], xs=[value_0, value_1],
          grad_ys=[
              [[2.0, 3.0], [4.0, 5.0]],
              [-0.5, 1.5]])
      grad_vals = sess.run(grad_r)  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5, 3.0 + 1.5], grad_vals[0])
      self.assertAllEqual([4.0, 5.0], grad_vals[1])

  def testTensorArrayGradientWritePackAndRead(self):
    self._testTensorArrayGradientWritePackAndRead(False)
    self._testTensorArrayGradientWritePackAndRead(True)

  def _testTensorArrayGradientUnpackRead(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=2)

      value = tf.constant([[1.0, -1.0], [10.0, -10.0]])

      w = h.unpack(value)
      r0 = w.read(0)
      r0_1 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = tf.gradients(
          ys=[r0, r0_1, r1], xs=[value], grad_ys=
          [[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])
      grad_vals = sess.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllClose([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientUnpackRead(self):
    self._testTensorArrayGradientUnpackRead(False)
    self._testTensorArrayGradientUnpackRead(True)

  def _testCloseTensorArray(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      c1 = h.close()
      sess.run(c1)

  def testCloseTensorArray(self):
    self._testCloseTensorArray(use_gpu=False)
    self._testCloseTensorArray(use_gpu=True)

  def _testWriteCloseTensorArray(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = data_flow_ops.TensorArray(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      w0 = h.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [3.0])
      w1.close().run()  # Expected to run without problems

      with self.assertRaisesOpError(r"Tensor foo has already been closed."):
        with tf.control_dependencies([w1.close()]):
          w1.write(2, 3.0).flow.eval()

  def testWriteCloseTensorArray(self):
    self._testWriteCloseTensorArray(use_gpu=False)
    self._testWriteCloseTensorArray(use_gpu=True)


if __name__ == "__main__":
  tf.test.main()
