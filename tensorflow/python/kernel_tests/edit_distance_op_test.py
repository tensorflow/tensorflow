"""Tests for tensorflow.kernels.edit_distance_op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


def ConstantOf(x):
  x = np.asarray(x)
  # Convert to int64 if it's not a string
  if x.dtype.char != "S": x = np.asarray(x, dtype=np.int64)
  return tf.constant(x)


class EditDistanceTest(tf.test.TestCase):

  def _testEditDistance(self, hypothesis, truth, normalize,
                        expected_output, expected_err_re=None):
    # hypothesis and truth are (index, value, shape) tuples
    hypothesis_st = tf.SparseTensor(*[ConstantOf(x) for x in hypothesis])
    truth_st = tf.SparseTensor(*[ConstantOf(x) for x in truth])
    edit_distance = tf.edit_distance(
        hypothesis=hypothesis_st, truth=truth_st, normalize=normalize)

    with self.test_session():
      if expected_err_re is None:
        # Shape inference figures out the shape from the shape variables
        expected_shape = [
            max(h, t) for h, t in zip(hypothesis[2], truth[2])[:-1]]
        self.assertEqual(edit_distance.get_shape(), expected_shape)
        output = edit_distance.eval()
        self.assertAllClose(output, expected_output)
      else:
        with self.assertRaisesOpError(expected_err_re):
          edit_distance.eval()

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

  def testEditDistanceMissingHypothesis(self):
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

  def testEditDistanceMissingTruth(self):
    hypothesis_indices = [[0, 0]]
    hypothesis_values = [0]
    hypothesis_shape = [1, 1]
    truth_indices = np.empty((0, 2), dtype=np.int64)
    truth_values = []
    truth_shape = [1, 0]
    expected_output = [np.inf]  # Normalized, divide by zero

    self._testEditDistance(
        hypothesis=(hypothesis_indices, hypothesis_values, hypothesis_shape),
        truth=(truth_indices, truth_values, truth_shape),
        normalize=True,
        expected_output=expected_output)


if __name__ == "__main__":
  tf.test.main()
