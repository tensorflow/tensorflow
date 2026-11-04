# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.reverse_sequence_op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ReverseSequenceTest(xla_test.XLATestCase):

  def _testReverseSequence(self,
                           x,
                           batch_axis,
                           seq_axis,
                           seq_lengths,
                           truth,
                           expected_err_re=None):
    with self.session():
      p = array_ops.placeholder(dtypes.as_dtype(x.dtype))
      lengths = array_ops.placeholder(dtypes.as_dtype(seq_lengths.dtype))
      with self.test_scope():
        ans = array_ops.reverse_sequence(
            p, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=lengths)
      if expected_err_re is None:
        tf_ans = ans.eval(feed_dict={p: x, lengths: seq_lengths})
        self.assertAllClose(tf_ans, truth, atol=1e-10)
      else:
        with self.assertRaisesOpError(expected_err_re):
          ans.eval(feed_dict={p: x, lengths: seq_lengths})

  def testSimple(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    expected = np.array([[1, 2, 3], [6, 5, 4], [8, 7, 9]], dtype=np.int32)
    self._testReverseSequence(
        x,
        batch_axis=0,
        seq_axis=1,
        seq_lengths=np.array([1, 3, 2], np.int32),
        truth=expected)

  def _testBasic(self, dtype, len_dtype):
    x = np.asarray(
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=dtype)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_lengths = np.asarray([3, 0, 4], dtype=len_dtype)

    truth_orig = np.asarray(
        [
            [[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
            [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
            [[20, 19, 18, 17], [24, 23, 22, 21]]
        ],  # reverse 0:4 (all)
        dtype=dtype)
    truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
    truth = truth_orig.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    seq_axis = 0  # permute seq_axis and batch_axis (originally 2 and 0, resp.)
    batch_axis = 2
    self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth)

  def testSeqLength(self):
    for dtype in self.all_types:
      for seq_dtype in self.all_types & {np.int32, np.int64}:
        self._testBasic(dtype, seq_dtype)

  def _reverseSelected(self, x, seq_lengths, axis, seq_axis, batch_axis):
    # Gathers the entries of x along axis where mask is true, which gives that
    # dimension a size only known at run time, and reverses the result.
    @def_function.function(jit_compile=True)
    def f(x, seq_lengths, mask):
      indices = array_ops.reshape(array_ops.where(mask), [-1])
      x = array_ops.gather(x, indices, axis=axis)
      if axis == batch_axis:
        seq_lengths = array_ops.gather(seq_lengths, indices)
      return array_ops.reverse_sequence(
          x, seq_lengths, seq_axis=seq_axis, batch_axis=batch_axis
      )

    x = np.array(x, dtype=np.int32)
    seq_lengths = np.array(seq_lengths, dtype=np.int32)
    mask = np.array([True, False, True, True, False])
    with self.session() as session:
      with self.device_scope():
        x_ph = array_ops.placeholder(dtypes.int32, x.shape)
        seq_lengths_ph = array_ops.placeholder(dtypes.int32, seq_lengths.shape)
        mask_ph = array_ops.placeholder(dtypes.bool, mask.shape)
        output = f(x_ph, seq_lengths_ph, mask_ph)
      result = session.run(
          output, {x_ph: x, seq_lengths_ph: seq_lengths, mask_ph: mask}
      )
    # Eager execution of the compiled function gives the same result, also
    # when the test itself runs in graph mode.
    with context.eager_mode(), self.device_scope():
      self.assertAllEqual(f(x, seq_lengths, mask), result)
    return result

  def testDynamicDimensions(self):
    if "TPU" in self.device:
      self.skipTest(
          "XLA:TPU fails to compile these dynamic dimensions with a "
          "bitcast shape size mismatch."
      )
    # Regression test for GitHub issue 128953: the output had the bound of a
    # dynamic input dimension instead of its size. The mask selects 3 of 5
    # entries, so the selected dimension has size 3 and bound 5. (XLA turns a
    # gather from 4 or fewer entries into selects, which XLA:GPU doesn't
    # support with dynamic shapes.)
    self.assertAllEqual(
        self._reverseSelected(
            [[1, 2, 3, 4, 0], [5, 6, 7, 8, 0]],
            [3, 2],
            axis=1,
            seq_axis=1,
            batch_axis=0,
        ),
        [[4, 3, 1], [7, 5, 8]],
    )
    self.assertAllEqual(
        self._reverseSelected(
            [[1, 5], [2, 6], [3, 7], [4, 8], [0, 0]],
            [3, 2],
            axis=0,
            seq_axis=0,
            batch_axis=1,
        ),
        [[4, 7], [3, 5], [1, 8]],
    )
    self.assertAllEqual(
        self._reverseSelected(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [0, 0, 0]],
            [3, 1, 2, 3, 1],
            axis=0,
            seq_axis=1,
            batch_axis=0,
        ),
        [[3, 2, 1], [8, 7, 9], [12, 11, 10]],
    )


if __name__ == "__main__":
  test.main()
