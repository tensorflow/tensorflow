# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for metrics.classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.metrics.python.metrics import classification
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ClassificationTest(test.TestCase):

  def testAccuracy1D(self):
    with self.test_session() as session:
      pred = array_ops.placeholder(dtypes.int32, shape=[None])
      labels = array_ops.placeholder(dtypes.int32, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={pred: [1, 0, 1, 0],
                                      labels: [1, 1, 0, 0]})
      self.assertEqual(result, 0.5)

  def testAccuracy1DBool(self):
    with self.test_session() as session:
      pred = array_ops.placeholder(dtypes.bool, shape=[None])
      labels = array_ops.placeholder(dtypes.bool, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={pred: [1, 0, 1, 0],
                                      labels: [1, 1, 0, 0]})
      self.assertEqual(result, 0.5)

  def testAccuracy1DInt64(self):
    with self.test_session() as session:
      pred = array_ops.placeholder(dtypes.int64, shape=[None])
      labels = array_ops.placeholder(dtypes.int64, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={pred: [1, 0, 1, 0],
                                      labels: [1, 1, 0, 0]})
      self.assertEqual(result, 0.5)

  def testAccuracy1DString(self):
    with self.test_session() as session:
      pred = array_ops.placeholder(dtypes.string, shape=[None])
      labels = array_ops.placeholder(dtypes.string, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(
          acc,
          feed_dict={pred: ['a', 'b', 'a', 'c'],
                     labels: ['a', 'c', 'b', 'c']})
      self.assertEqual(result, 0.5)

  def testAccuracyDtypeMismatch(self):
    with self.assertRaises(ValueError):
      pred = array_ops.placeholder(dtypes.int32, shape=[None])
      labels = array_ops.placeholder(dtypes.int64, shape=[None])
      classification.accuracy(pred, labels)

  def testAccuracyFloatLabels(self):
    with self.assertRaises(ValueError):
      pred = array_ops.placeholder(dtypes.int32, shape=[None])
      labels = array_ops.placeholder(dtypes.float32, shape=[None])
      classification.accuracy(pred, labels)

  def testAccuracy1DWeighted(self):
    with self.test_session() as session:
      pred = array_ops.placeholder(dtypes.int32, shape=[None])
      labels = array_ops.placeholder(dtypes.int32, shape=[None])
      weights = array_ops.placeholder(dtypes.float32, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 1],
                               labels: [1, 1, 0, 1],
                               weights: [3.0, 1.0, 2.0, 0.0]
                           })
      self.assertEqual(result, 0.5)

  def testAccuracy1DWeightedBroadcast(self):
    with self.test_session() as session:
      pred = array_ops.placeholder(dtypes.int32, shape=[None])
      labels = array_ops.placeholder(dtypes.int32, shape=[None])
      weights = array_ops.placeholder(dtypes.float32, shape=[])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 0],
                               labels: [1, 1, 0, 0],
                               weights: 3.0,
                           })
      self.assertEqual(result, 0.5)


class F1ScoreTest(test.TestCase):

  def setUp(self):
    super(F1ScoreTest, self).setUp()
    np.random.seed(1)

  def testVars(self):
    classification.f1_score(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        num_thresholds=3)
    expected = {'f1/true_positives:0', 'f1/false_positives:0',
                'f1/false_negatives:0'}
    self.assertEquals(
        expected, set(v.name for v in variables.local_variables()))
    self.assertEquals(
        set(expected), set(v.name for v in variables.local_variables()))
    self.assertEquals(
        set(expected),
        set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    f1, _ = classification.f1_score(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        num_thresholds=3,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [f1])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, f1_op = classification.f1_score(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        num_thresholds=3,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [f1_op])

  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=2, dtype=dtypes.int64, seed=2)
    f1, f1_op = classification.f1_score(predictions, labels, num_thresholds=3)

    with self.test_session() as sess:
      sess.run(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run([f1_op])

      # Then verify idempotency.
      initial_f1 = f1.eval()
      for _ in range(10):
        self.assertAllClose(initial_f1, f1.eval())

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = constant_op.constant(inputs, dtype=dtypes.float32)
      labels = constant_op.constant(inputs)
      f1, f1_op = classification.f1_score(predictions, labels, num_thresholds=3)

      sess.run(variables.local_variables_initializer())
      sess.run([f1_op])

      self.assertEqual(1, f1.eval())

  def testSomeCorrect(self):
    predictions = constant_op.constant(
        [1, 0, 1, 0], shape=(1, 4), dtype=dtypes.float32)
    labels = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
    f1, f1_op = classification.f1_score(predictions, labels, num_thresholds=1)
    with self.test_session() as sess:
      sess.run(variables.local_variables_initializer())
      sess.run([f1_op])
      # Threshold 0 will have around 0.5 precision and 1 recall yielding an F1
      # score of 2 * 0.5 * 1 / (1 + 0.5).
      self.assertAlmostEqual(2 * 0.5 * 1 / (1 + 0.5), f1.eval())

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(10000, 1))

    with self.test_session() as sess:
      predictions = constant_op.constant(inputs, dtype=dtypes.float32)
      labels = constant_op.constant(1 - inputs, dtype=dtypes.float32)
      f1, f1_op = classification.f1_score(predictions, labels, num_thresholds=3)

      sess.run(variables.local_variables_initializer())
      sess.run([f1_op])

      # Threshold 0 will have around 0.5 precision and 1 recall yielding an F1
      # score of 2 * 0.5 * 1 / (1 + 0.5).
      self.assertAlmostEqual(2 * 0.5 * 1 / (1 + 0.5), f1.eval(), places=2)

  def testWeights1d(self):
    with self.test_session() as sess:
      predictions = constant_op.constant(
          [[1, 0], [1, 0]], shape=(2, 2), dtype=dtypes.float32)
      labels = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = constant_op.constant(
          [[0], [1]], shape=(2, 1), dtype=dtypes.float32)
      f1, f1_op = classification.f1_score(predictions, labels, weights,
                                          num_thresholds=3)
      sess.run(variables.local_variables_initializer())
      sess.run([f1_op])

      self.assertAlmostEqual(1.0, f1.eval(), places=5)

  def testWeights2d(self):
    with self.test_session() as sess:
      predictions = constant_op.constant(
          [[1, 0], [1, 0]], shape=(2, 2), dtype=dtypes.float32)
      labels = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = constant_op.constant(
          [[0, 0], [1, 1]], shape=(2, 2), dtype=dtypes.float32)
      f1, f1_op = classification.f1_score(predictions, labels, weights,
                                          num_thresholds=3)
      sess.run(variables.local_variables_initializer())
      sess.run([f1_op])

      self.assertAlmostEqual(1.0, f1.eval(), places=5)

  def testZeroLabelsPredictions(self):
    with self.test_session() as sess:
      predictions = array_ops.zeros([4], dtype=dtypes.float32)
      labels = array_ops.zeros([4])
      f1, f1_op = classification.f1_score(predictions, labels, num_thresholds=3)
      sess.run(variables.local_variables_initializer())
      sess.run([f1_op])

      self.assertAlmostEqual(0.0, f1.eval(), places=5)

  def testWithMultipleUpdates(self):
    num_samples = 1000
    batch_size = 10
    num_batches = int(num_samples / batch_size)

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    noise = np.random.normal(0.0, scale=0.2, size=(num_samples, 1))
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0
    thresholds = [-0.01, 0.5, 1.01]

    expected_max_f1 = -1.0
    for threshold in thresholds:
      tp = 0
      fp = 0
      fn = 0
      tn = 0
      for i in range(num_samples):
        if predictions[i] >= threshold:
          if labels[i] == 1:
            tp += 1
          else:
            fp += 1
        else:
          if labels[i] == 1:
            fn += 1
          else:
            tn += 1
      epsilon = 1e-7
      expected_prec = tp / (epsilon + tp + fp)
      expected_rec = tp / (epsilon + tp + fn)
      expected_f1 = (2 * expected_prec * expected_rec /
                     (epsilon + expected_prec + expected_rec))
      if expected_f1 > expected_max_f1:
        expected_max_f1 = expected_f1

    labels = labels.astype(np.float32)
    predictions = predictions.astype(np.float32)
    tf_predictions, tf_labels = (dataset_ops.Dataset
                                 .from_tensor_slices((predictions, labels))
                                 .repeat()
                                 .batch(batch_size)
                                 .make_one_shot_iterator()
                                 .get_next())
    f1, f1_op = classification.f1_score(tf_labels, tf_predictions,
                                        num_thresholds=3)

    with self.test_session() as sess:
      sess.run(variables.local_variables_initializer())
      for _ in range(num_batches):
        sess.run([f1_op])
      # Since this is only approximate, we can't expect a 6 digits match.
      # Although with higher number of samples/thresholds we should see the
      # accuracy improving
      self.assertAlmostEqual(expected_max_f1, f1.eval(), 2)


if __name__ == '__main__':
  test.main()
