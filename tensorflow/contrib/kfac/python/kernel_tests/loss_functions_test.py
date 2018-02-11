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
"""Tests for tf.contrib.kfac.loss_functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.kfac.python.ops import loss_functions
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class InsertSliceInZerosTest(test.TestCase):

  def testBadShape(self):
    bad_shaped_ones = array_ops.ones(shape=[1, 3])  # n.b. shape[1] != 1
    with self.assertRaises(ValueError):
      loss_functions.insert_slice_in_zeros(bad_shaped_ones, 1, 42, 17)

  def test3d(self):
    input_tensor = constant_op.constant([[[1, 2]], [[3, 4]]])
    expected_output_array = [[[1, 2], [0, 0]], [[3, 4], [0, 0]]]
    op = loss_functions.insert_slice_in_zeros(input_tensor, 1, 2, 0)
    with self.test_session() as sess:
      actual_output_array = sess.run(op)
    self.assertAllEqual(expected_output_array, actual_output_array)


class CategoricalLogitsNegativeLogProbLossTest(test.TestCase):

  def testSample(self):
    """Ensure samples can be drawn."""
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.asarray([
          [0., 0., 0.],  #
          [1., -1., 0.]
      ]).astype(np.float32)
      loss = loss_functions.CategoricalLogitsNegativeLogProbLoss(
          array_ops.constant(logits))
      sample = loss.sample(42)
      sample = sess.run(sample)
      self.assertEqual(sample.shape, (2,))

  def testEvaluateOnTargets(self):
    """Ensure log probability can be evaluated correctly."""
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.asarray([
          [0., 0., 0.],  #
          [1., -1., 0.]
      ]).astype(np.float32)
      targets = np.asarray([2, 1]).astype(np.int32)
      loss = loss_functions.CategoricalLogitsNegativeLogProbLoss(
          array_ops.constant(logits), targets=array_ops.constant(targets))
      neg_log_prob = loss.evaluate()
      neg_log_prob = sess.run(neg_log_prob)

      # Calculate explicit log probability of targets.
      probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
      log_probs = np.log([
          probs[0, targets[0]],  #
          probs[1, targets[1]]
      ])
      expected_log_prob = np.sum(log_probs)

      self.assertAllClose(neg_log_prob, -expected_log_prob)

  def testEvaluateOnSample(self):
    """Ensure log probability of a sample can be drawn."""
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.asarray([
          [0., 0., 0.],  #
          [1., -1., 0.]
      ]).astype(np.float32)
      loss = loss_functions.CategoricalLogitsNegativeLogProbLoss(
          array_ops.constant(logits))
      neg_log_prob = loss.evaluate_on_sample(42)

      # Simply ensure this doesn't crash. As the output is random, it's
      # difficult to say if the output is correct or not...
      neg_log_prob = sess.run(neg_log_prob)

  def testMultiMinibatchRegistration(self):
    """Ensure this loss function supports registering multiple minibatches."""
    with ops.Graph().as_default():
      tower_logits = []
      loss = None
      num_towers = 5
      for _ in range(num_towers):
        logits = random_ops.random_uniform(shape=[2, 3])
        tower_logits.append(logits)
        if loss is None:
          loss = loss_functions.CategoricalLogitsNegativeLogProbLoss(logits)
        else:
          loss.register_additional_minibatch(logits)
      self.assertListEqual(loss.input_minibatches, tower_logits)
      self.assertEqual(loss.num_registered_minibatches, num_towers)

  def testMultiplyFisherSingleVector(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.array([1., 2., 3.])
      loss = loss_functions.CategoricalLogitsNegativeLogProbLoss(logits)

      # the LossFunction.multiply_fisher docstring only says it supports the
      # case where the vector is the same shape as the input natural parameters
      # (i.e. the logits here), but here we also test leading dimensions
      vector = np.array([1., 2., 3.])
      vectors = [vector, vector.reshape(1, -1), np.stack([vector] * 4)]

      probs = np.exp(logits - np.logaddexp.reduce(logits))
      fisher = np.diag(probs) - np.outer(probs, probs)

      for vector in vectors:
        result = loss.multiply_fisher(vector)
        expected_result = np.dot(vector, fisher)
        self.assertAllClose(expected_result, sess.run(result))

  def testMultiplyFisherBatch(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.array([[1., 2., 3.], [4., 6., 8.]])
      loss = loss_functions.CategoricalLogitsNegativeLogProbLoss(logits)

      vector = np.array([[1., 2., 3.], [5., 3., 1.]])

      na = np.newaxis
      probs = np.exp(logits - np.logaddexp.reduce(logits, axis=-1,
                                                  keepdims=True))
      fishers = probs[..., na] * np.eye(3) - probs[..., na] * probs[..., na, :]

      result = loss.multiply_fisher(vector)
      expected_result = np.matmul(vector[..., na, :], fishers)[..., 0, :]
      self.assertEqual(sess.run(result).shape, logits.shape)
      self.assertAllClose(expected_result, sess.run(result))


class OnehotCategoricalLogitsNegativeLogProbLossTest(test.TestCase):

  def testSample(self):
    """Ensure samples can be drawn."""
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.asarray([
          [0., 0., 0.],  #
          [1., -1., 0.]
      ]).astype(np.float32)
      loss = loss_functions.OnehotCategoricalLogitsNegativeLogProbLoss(
          array_ops.constant(logits))
      sample = loss.sample(42)
      sample = sess.run(sample)
      self.assertEqual(sample.shape, (2, 3))

  def testEvaluateOnTargets(self):
    """Ensure log probability can be evaluated correctly."""
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.asarray([
          [0., 0., 0.],  #
          [1., -1., 0.]
      ]).astype(np.float32)
      targets = np.asarray([2, 1]).astype(np.int32)
      loss = loss_functions.OnehotCategoricalLogitsNegativeLogProbLoss(
          array_ops.constant(logits), targets=array_ops.one_hot(targets, 3))
      neg_log_prob = loss.evaluate()
      neg_log_prob = sess.run(neg_log_prob)

      # Calculate explicit log probability of targets.
      probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
      log_probs = np.log([
          probs[0, targets[0]],  #
          probs[1, targets[1]]
      ])
      expected_log_prob = np.sum(log_probs)

      self.assertAllClose(neg_log_prob, -expected_log_prob)

  def testEvaluateOnSample(self):
    """Ensure log probability of a sample can be drawn."""
    with ops.Graph().as_default(), self.test_session() as sess:
      logits = np.asarray([
          [0., 0., 0.],  #
          [1., -1., 0.]
      ]).astype(np.float32)
      loss = loss_functions.OnehotCategoricalLogitsNegativeLogProbLoss(
          array_ops.constant(logits))
      neg_log_prob = loss.evaluate_on_sample(42)

      # Simply ensure this doesn't crash. As the output is random, it's
      # difficult to say if the output is correct or not...
      neg_log_prob = sess.run(neg_log_prob)

  def testMultiMinibatchRegistration(self):
    """Ensure this loss function supports registering multiple minibatches."""
    with ops.Graph().as_default():
      tower_logits = []
      loss = None
      num_towers = 5
      for _ in range(num_towers):
        logits = random_ops.random_uniform(shape=[2, 3])
        tower_logits.append(logits)
        if loss is None:
          loss = loss_functions.OnehotCategoricalLogitsNegativeLogProbLoss(
              logits)
        else:
          loss.register_additional_minibatch(logits)
      self.assertListEqual(loss.input_minibatches, tower_logits)
      self.assertEqual(loss.num_registered_minibatches, num_towers)


if __name__ == "__main__":
  test.main()
