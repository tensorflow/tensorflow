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
from tensorflow.python.ops import gen_data_flow_ops


class TensorArrayOpTest(tf.test.TestCase):

  def _testTensorArrayWriteRead(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      writes = [
          gen_data_flow_ops._tensor_array_write(h, 0, [[4.0, 5.0]]),
          gen_data_flow_ops._tensor_array_write(h, 1, [[1.0]]),
          gen_data_flow_ops._tensor_array_write(h, 2, -3.0)]

      with tf.control_dependencies(writes):
        r0 = gen_data_flow_ops._tensor_array_read(h, 0, tf.float32)
        r1 = gen_data_flow_ops._tensor_array_read(h, 1, tf.float32)
        r2 = gen_data_flow_ops._tensor_array_read(h, 2, tf.float32)

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
      h = gen_data_flow_ops._tensor_array(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        convert = lambda x: np.asarray(x).astype(np.str)
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      writes = [
          gen_data_flow_ops._tensor_array_write(h, 0, convert([[4.0, 5.0]])),
          gen_data_flow_ops._tensor_array_write(h, 1, convert([[6.0, 7.0]])),
          gen_data_flow_ops._tensor_array_write(h, 2, convert([[8.0, 9.0]]))]

      with tf.control_dependencies(writes):
        c0 = gen_data_flow_ops._tensor_array_pack(h, tf_dtype)

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
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          r"Input value must have first dimension "
          r"equal to the array size \(2 vs. 3\)"):
        gen_data_flow_ops._tensor_array_unpack(h, [1.0, 2.0]).run()

  def testTensorArrayPackNotAllValuesAvailableFails(self):
    with self.test_session():
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not read from TensorArray index 1 "
          "because it has not yet been written to."):
        with tf.control_dependencies([
            gen_data_flow_ops._tensor_array_write(h, 0, [[4.0, 5.0]])]):
          gen_data_flow_ops._tensor_array_pack(h, tf.float32).eval()

  def _testTensorArrayUnpackRead(self, tf_dtype, use_gpu):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._tensor_array(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == tf.string:
        convert = lambda x: np.asarray(x).astype(np.str)
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      # Unpack a vector into scalars
      with tf.control_dependencies([
          gen_data_flow_ops._tensor_array_unpack(
              h, convert([1.0, 2.0, 3.0]))]):
        r0 = gen_data_flow_ops._tensor_array_read(h, 0, tf_dtype)
        r1 = gen_data_flow_ops._tensor_array_read(h, 1, tf_dtype)
        r2 = gen_data_flow_ops._tensor_array_read(h, 2, tf_dtype)

      d0, d1, d2 = sess.run([r0, r1, r2])
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      # Unpack a matrix into vectors
      with tf.control_dependencies([
          gen_data_flow_ops._tensor_array_unpack(
              h, convert([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]))]):
        r0 = gen_data_flow_ops._tensor_array_read(h, 0, tf_dtype)
        r1 = gen_data_flow_ops._tensor_array_read(h, 1, tf_dtype)
        r2 = gen_data_flow_ops._tensor_array_read(h, 2, tf_dtype)

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
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      g_h = gen_data_flow_ops._tensor_array_grad(h)

      writes = [
          gen_data_flow_ops._tensor_array_write(h, 0, [[4.0, 5.0]]),
          gen_data_flow_ops._tensor_array_write(h, 1, [[1.0]]),
          gen_data_flow_ops._tensor_array_write(h, 2, -3.0)]

      grad_writes = [
          gen_data_flow_ops._tensor_array_write(g_h, 0, [[5.0, 6.0]]),
          gen_data_flow_ops._tensor_array_write(g_h, 1, [[2.0]]),
          gen_data_flow_ops._tensor_array_write(g_h, 2, -2.0)]

      with tf.control_dependencies(writes):
        r0 = gen_data_flow_ops._tensor_array_read(h, 0, tf.float32)
        r1 = gen_data_flow_ops._tensor_array_read(h, 1, tf.float32)
        r2 = gen_data_flow_ops._tensor_array_read(h, 2, tf.float32)

      with tf.control_dependencies(grad_writes):
        g_r0 = gen_data_flow_ops._tensor_array_read(g_h, 0, tf.float32)
        g_r1 = gen_data_flow_ops._tensor_array_read(g_h, 1, tf.float32)
        g_r2 = gen_data_flow_ops._tensor_array_read(g_h, 2, tf.float32)

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
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      g_h_0 = gen_data_flow_ops._tensor_array_grad(h)
      g_h_1 = gen_data_flow_ops._tensor_array_grad(h)

      with tf.control_dependencies([
          gen_data_flow_ops._tensor_array_write(g_h_0, 0, [[4.0, 5.0]])]):
        # Write with one gradient handle, read with another copy of it
        r1_0 = gen_data_flow_ops._tensor_array_read(g_h_1, 0, tf.float32)

      t_g_h_0, t_g_h_1, d_r1_0 = sess.run([g_h_0, g_h_1, r1_0])
      self.assertAllEqual(t_g_h_0, t_g_h_1)
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

  def testTensorGradAccessTwiceReceiveSameObject(self):
    self._testTensorGradAccessTwiceReceiveSameObject(False)
    self._testTensorGradAccessTwiceReceiveSameObject(True)

  def _testTensorArrayWriteWrongIndexOrDataTypeFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      # Test writing the wrong datatype
      with self.assertRaisesOpError(
          "TensorArray dtype is float but Op is trying to write dtype string"):
        gen_data_flow_ops._tensor_array_write(h, -1, "wrong_type_scalar").run()

      # Test writing to a negative index
      with self.assertRaisesOpError(
          "Tried to write to index -1 but array size is: 3"):
        gen_data_flow_ops._tensor_array_write(h, -1, 3.0).run()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to write to index 3 but array size is: 3"):
        gen_data_flow_ops._tensor_array_write(h, 3, 3.0).run()

  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    self._testTensorArrayWriteWrongIndexOrDataTypeFails(use_gpu=False)
    self._testTensorArrayWriteWrongIndexOrDataTypeFails(use_gpu=True)

  def _testTensorArrayReadWrongIndexOrDataTypeFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with tf.control_dependencies([
          gen_data_flow_ops._tensor_array_write(h, 0, [[4.0, 5.0]])]):

        # Test reading wrong datatype
        r0_bad = gen_data_flow_ops._tensor_array_read(h, 0, tf.int64)
        with self.assertRaisesOpError(
            "TensorArray dtype is float but Op requested dtype int64."):
          r0_bad.eval()

        # Test reading from a different index than the one we wrote to
        r1 = gen_data_flow_ops._tensor_array_read(h, 1, tf.float32)
        with self.assertRaisesOpError(
            "Could not read from TensorArray index 1 because "
            "it has not yet been written to."):
          r1.eval()

      # Test reading from a negative index
      with self.assertRaisesOpError(
          "Tried to read from index -1 but array size is: 3"):
        gen_data_flow_ops._tensor_array_read(h, -1, tf.float32).eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to read from index 3 but array size is: 3"):
        gen_data_flow_ops._tensor_array_read(h, 3, tf.float32).eval()

  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    self._testTensorArrayReadWrongIndexOrDataTypeFails(use_gpu=False)
    self._testTensorArrayReadWrongIndexOrDataTypeFails(use_gpu=True)

  def _testTensorArrayWriteMultipleFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not write to TensorArray index 2 because "
          "it has already been written to."):
        with tf.control_dependencies([
            gen_data_flow_ops._tensor_array_write(h, 2, 3.0)]):
          gen_data_flow_ops._tensor_array_write(h, 2, 3.0).run()

  def testTensorArrayWriteMultipleFails(self):
    self._testTensorArrayWriteMultipleFails(use_gpu=False)
    self._testTensorArrayWriteMultipleFails(use_gpu=True)

  def _testTensorArrayWriteGradientAddMultipleAddsType(self, use_gpu, dtype):
    with self.test_session(use_gpu=use_gpu):
      h = gen_data_flow_ops._tensor_array(
          dtype=dtype, tensor_array_name="foo", size=3)

      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      writes = [
          gen_data_flow_ops._tensor_array_write(
              h, 2, c(3.0), gradient_add=True),
          gen_data_flow_ops._tensor_array_write(
              h, 2, c(4.0), gradient_add=True)]

      with tf.control_dependencies(writes):
        self.assertAllEqual(
            c(7.00), gen_data_flow_ops._tensor_array_read(h, 2, dtype).eval())

  def _testTensorArrayWriteGradientAddMultipleAdds(self, use_gpu):
    for dtype in [tf.int32, tf.int64, tf.float32, tf.float64, tf.complex64]:
      self._testTensorArrayWriteGradientAddMultipleAddsType(use_gpu, dtype)

  def testTensorArrayWriteGradientAddMultipleAdds(self):
    self._testTensorArrayWriteGradientAddMultipleAdds(use_gpu=False)
    self._testTensorArrayWriteGradientAddMultipleAdds(use_gpu=True)

  def _testMultiTensorArray(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      h1 = gen_data_flow_ops._tensor_array(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      with tf.control_dependencies([
          gen_data_flow_ops._tensor_array_write(h1, 0, 4.0)]):
        r1 = gen_data_flow_ops._tensor_array_read(h1, 0, tf.float32)

      h2 = gen_data_flow_ops._tensor_array(
          size=1, dtype=tf.float32, tensor_array_name="bar")

      with tf.control_dependencies([
          gen_data_flow_ops._tensor_array_write(h2, 0, 5.0)]):
        r2 = gen_data_flow_ops._tensor_array_read(h2, 0, tf.float32)
      r = r1 + r2
      self.assertAllClose(9.0, r.eval())

  def testMultiTensorArray(self):
    self._testMultiTensorArray(use_gpu=False)
    self._testMultiTensorArray(use_gpu=True)

  def _testDuplicateTensorArrayFails(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h1 = gen_data_flow_ops._tensor_array(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      c1 = gen_data_flow_ops._tensor_array_write(h1, 0, 4.0)
      h2 = gen_data_flow_ops._tensor_array(
          size=1, dtype=tf.float32, tensor_array_name="foo")
      c2 = gen_data_flow_ops._tensor_array_write(h2, 0, 5.0)
      with self.assertRaises(errors.AlreadyExistsError):
        sess.run([c1, c2])

  def testDuplicateTensorArrayFails(self):
    self._testDuplicateTensorArrayFails(use_gpu=False)
    self._testDuplicateTensorArrayFails(use_gpu=True)

  def _testCloseTensorArray(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      c1 = gen_data_flow_ops._tensor_array_close(h)
      sess.run(c1)

  def testCloseTensorArray(self):
    self._testCloseTensorArray(use_gpu=False)
    self._testCloseTensorArray(use_gpu=True)

  def _testWriteCloseTensorArray(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      h = gen_data_flow_ops._tensor_array(
          dtype=tf.float32, tensor_array_name="foo", size=3)
      writes = [
          gen_data_flow_ops._tensor_array_write(h, 0, [[4.0, 5.0]]),
          gen_data_flow_ops._tensor_array_write(h, 1, [3.0]),
          gen_data_flow_ops._tensor_array_write(h, 2, -1.0)]
      with tf.control_dependencies(writes):
        close = gen_data_flow_ops._tensor_array_close(h)
      sess.run(close)

  def testWriteCloseTensorArray(self):
    self._testWriteCloseTensorArray(use_gpu=False)
    self._testWriteCloseTensorArray(use_gpu=True)


if __name__ == "__main__":
  tf.test.main()
