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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


class ReverseSequenceTest(test.TestCase):

  def _testReverseSequence(self,
                           x,
                           batch_axis,
                           seq_axis,
                           seq_lengths,
                           truth,
                           use_gpu=False,
                           expected_err_re=None):
    with self.cached_session(use_gpu=use_gpu):
      ans = array_ops.reverse_sequence(
          x, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=seq_lengths)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        self.assertAllClose(tf_ans, truth, atol=1e-10)
        self.assertShapeEqual(truth, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothReverseSequence(self,
                               x,
                               batch_axis,
                               seq_axis,
                               seq_lengths,
                               truth,
                               expected_err_re=None):
    self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth, True,
                              expected_err_re)
    self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth,
                              False, expected_err_re)

  def _testBasic(self, dtype, len_dtype=np.int64):
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
    self._testBothReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth)

  def testSeqLengthInt32(self):
    self._testBasic(np.float32, np.int32)

  def testFloatBasic(self):
    self._testBasic(np.float32)

  def testDoubleBasic(self):
    self._testBasic(np.float64)

  def testInt32Basic(self):
    self._testBasic(np.int32)

  def testInt64Basic(self):
    self._testBasic(np.int64)

  def testComplex64Basic(self):
    self._testBasic(np.complex64)

  def testComplex128Basic(self):
    self._testBasic(np.complex128)

  def testFloatReverseSequenceGrad(self):
    x = np.asarray(
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]],
         [[17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=np.float)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # transpose axes 0 <=> 2

    # reverse dim 0 up to (0:3, none, 0:4) along dim=2
    seq_axis = 0
    batch_axis = 2
    seq_lengths = np.asarray([3, 0, 4], dtype=np.int64)

    def reverse_sequence(x):
      seq_lengths_t = constant_op.constant(seq_lengths, shape=seq_lengths.shape)
      return array_ops.reverse_sequence(
          x,
          batch_axis=batch_axis,
          seq_axis=seq_axis,
          seq_lengths=seq_lengths_t)

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(reverse_sequence, [x]))
      self.assertLess(err, 1e-8)

  def testShapeFunctionEdgeCases(self):
    # Enter graph mode since we want to test partial shapes
    with context.graph_mode():
      t = array_ops.reverse_sequence(
          array_ops.placeholder(dtypes.float32, shape=None),
          seq_lengths=array_ops.placeholder(dtypes.int64, shape=(32,)),
          batch_axis=0,
          seq_axis=1)
      self.assertIs(t.get_shape().ndims, None)

  def testInvalidArguments(self):
    # Batch size mismatched between input and seq_lengths.
    # seq_length too long
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                (r"Dimensions must be equal|"
                                 r"Length of seq_lengths != input.dims\(0\)")):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2, 2], seq_axis=1)

    # seq_length too short
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                (r"Dimensions must be equal|"
                                 r"Length of seq_lengths != input.dims\(0\)")):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2], seq_axis=1)

    # Invalid seq_length shape
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                ("Shape must be rank 1 but is rank 2|"
                                 "seq_lengths must be 1-dim")):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [[2, 2]], seq_axis=1)

    # seq_axis out of bounds.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "seq_dim must be < input rank"):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2], seq_axis=2)

    # batch_axis out of bounds.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "batch_dim must be < input rank"):
      array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2],
                                 seq_axis=1,
                                 batch_axis=3)

    with self.assertRaisesRegex((errors.OpError, errors.InvalidArgumentError),
                                "batch_dim == seq_dim == 0"):
      output = array_ops.reverse_sequence([[1, 2], [3, 4]], [2, 2], seq_axis=0)
      self.evaluate(output)


if __name__ == "__main__":
  test.main()
