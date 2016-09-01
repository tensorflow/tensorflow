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

"""Tests for tensorflow.kernels.edit_distance_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def ConstantOf(x):
  x = np.asarray(x)
  # Convert to int64 if it's not a string or unicode
  if x.dtype.char not in "SU": x = np.asarray(x, dtype=np.int64)
  return tf.constant(x)


class EditDistanceTest(tf.test.TestCase):

  def _testEditDistanceST(
      self, hypothesis_st, truth_st, normalize, expected_output,
      expected_shape, expected_err_re=None):
    edit_distance = tf.edit_distance(
        hypothesis=hypothesis_st, truth=truth_st, normalize=normalize)

    if expected_err_re is None:
      self.assertEqual(edit_distance.get_shape(), expected_shape)
      output = edit_distance.eval()
      self.assertAllClose(output, expected_output)
    else:
      with self.assertRaisesOpError(expected_err_re):
        edit_distance.eval()

  def _testEditDistance(self, hypothesis, truth, normalize,
                        expected_output, expected_err_re=None):
    # Shape inference figures out the shape from the shape variables
    # Explicit tuple() needed since zip returns an iterator in Python 3.
    expected_shape = [
        max(h, t) for h, t in tuple(zip(hypothesis[2], truth[2]))[:-1]]

    # SparseTensorValue inputs.
    with tf.Graph().as_default() as g, self.test_session(g):
      # hypothesis and truth are (index, value, shape) tuples
      self._testEditDistanceST(
          hypothesis_st=tf.SparseTensorValue(
              *[ConstantOf(x) for x in hypothesis]),
          truth_st=tf.SparseTensorValue(*[ConstantOf(x) for x in truth]),
          normalize=normalize,
          expected_output=expected_output,
          expected_shape=expected_shape,
          expected_err_re=expected_err_re)

    # SparseTensor inputs.
    with tf.Graph().as_default() as g, self.test_session(g):
      # hypothesis and truth are (index, value, shape) tuples
      self._testEditDistanceST(
          hypothesis_st=tf.SparseTensor(*[ConstantOf(x) for x in hypothesis]),
          truth_st=tf.SparseTensor(*[ConstantOf(x) for x in truth]),
          normalize=normalize,
          expected_output=expected_output,
          expected_shape=expected_shape,
          expected_err_re=expected_err_re)

  def testEditDistanceNormalized(self):
    hypothesis_indices = [[0, 0], [0, 1],
                          [1, 0], [1, 1]]
    hypothesis_values = [0, 1,
                         1, -1]
    hypothesis_shape = [2, 2]
    truth_indices = [[0, 0],
                     [1, 0], [1, 1]]
    truth_values = [0,
                    1, 1]
    truth_shape = [2, 2]
    expected_output = [1.0, 0.5]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceUnnormalized(self):
    hypothesis_indices = [[0, 0],
                          [1, 0], [1, 1]]
    hypothesis_values = [10,
                         10, 11]
    hypothesis_shape = [2, 2]
    truth_indices = [[0, 0], [0, 1],
                     [1, 0], [1, 1]]
    truth_values = [1, 2,
                    1, -1]
    truth_shape = [2, 3]
    expected_output = [2.0, 2.0]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_output)

  def testEditDistanceProperDistance(self):
    # In this case, the values are individual characters stored in the
    # SparseTensor (type DT_STRING)
    hypothesis_indices = ([[0, i] for i, _ in enumerate("algorithm")] +
                          [[1, i] for i, _ in enumerate("altruistic")])
    hypothesis_values = [x for x in "algorithm"] + [x for x in "altruistic"]
    hypothesis_shape = [2, 11]
    truth_indices = ([[0, i] for i, _ in enumerate("altruistic")] +
                     [[1, i] for i, _ in enumerate("algorithm")])
    truth_values = [x for x in "altruistic"] + [x for x in "algorithm"]
    truth_shape = [2, 11]
    expected_unnormalized = [6.0, 6.0]
    expected_normalized = [6.0/len("altruistic"),
                           6.0/len("algorithm")]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=False,
        expected_output=expected_unnormalized)

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_normalized)

  def testEditDistance3D(self):
    hypothesis_indices = [[0, 0, 0],
                          [1, 0, 0]]
    hypothesis_values = [0, 1]
    hypothesis_shape = [2, 1, 1]
    truth_indices = [[0, 1, 0],
                     [1, 0, 0],
                     [1, 1, 0]]
    truth_values = [0, 1, 1]
    truth_shape = [2, 2, 1]
    expected_output = [[np.inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
                       [0.0, 1.0]]     # (1,0): match,    (1,1): no hypothesis

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceZeroLengthHypothesis(self):
    hypothesis_indices = np.empty((0, 2), dtype=np.int64)
    hypothesis_values = []
    hypothesis_shape = [1, 0]
    truth_indices = [[0, 0]]
    truth_values = [0]
    truth_shape = [1, 1]
    expected_output = [1.0]

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceZeroLengthTruth(self):
    hypothesis_indices = [[0, 0]]
    hypothesis_values = [0]
    hypothesis_shape = [1, 1]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    expected_output = [np.inf]  # Normalized, loss is 1/0 = inf

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)

  def testEditDistanceZeroLengthHypothesisAndTruth(self):
    hypothesis_indices = np.empty((0, 2), dtype=np.int64)
    hypothesis_values = []
    hypothesis_shape = [1, 0]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    expected_output = [0]  # Normalized is 0 because of exact match

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)


if __name__ == "__main__":
  tf.test.main()
