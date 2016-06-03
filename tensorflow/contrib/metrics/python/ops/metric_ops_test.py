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
"""Tests for metric_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


NAN = float('nan')


def _enqueue_vector(sess, queue, values, shape=None):
  if not shape:
    shape = (1, len(values))
  dtype = queue.dtypes[0]
  sess.run(queue.enqueue(tf.constant(values, dtype=dtype, shape=shape)))


def _binary_2d_label_to_sparse_value(labels):
  """Convert dense 2D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensorValue` whose values are indices along the last dimension of
    `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch, xi])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  shape = [len(labels), len(labels[0])]
  return tf.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64),
      np.array(shape, np.int64))


def _binary_2d_label_to_sparse(labels):
  """Convert dense 2D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensor` whose values are indices along the last dimension of
    `labels`.
  """
  v = _binary_2d_label_to_sparse_value(labels)
  return tf.SparseTensor(tf.constant(v.indices, tf.int64),
                         tf.constant(v.values, tf.int64),
                         tf.constant(v.shape, tf.int64))


def _binary_3d_label_to_sparse_value(labels):
  """Convert dense 3D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensorValue` whose values are indices along the last dimension of
    `labels`.
  """
  indices = []
  values = []
  for d0, labels_d0 in enumerate(labels):
    for d1, labels_d1 in enumerate(labels_d0):
      d2 = 0
      for class_id, label in enumerate(labels_d1):
        if label == 1:
          values.append(class_id)
          indices.append([d0, d1, d2])
          d2 += 1
        else:
          assert label == 0
  shape = [len(labels), len(labels[0]), len(labels[0][0])]
  return tf.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64),
      np.array(shape, np.int64))


def _binary_3d_label_to_sparse(labels):
  """Convert dense 3D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensor` whose values are indices along the last dimension of
    `labels`.
  """
  v = _binary_3d_label_to_sparse_value(labels)
  return tf.SparseTensor(tf.constant(v.indices, tf.int64),
                         tf.constant(v.values, tf.int64),
                         tf.constant(v.shape, tf.int64))


class StreamingMeanTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_mean(
        tf.ones([4, 3]),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_mean(
        tf.ones([4, 3]),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testAllValuesPresent(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = tf.contrib.metrics.streaming_mean(values)

      sess.run(tf.initialize_local_variables())
      for _ in range(4):
        sess.run(update_op)
      self.assertAlmostEqual(1.65, sess.run(mean), 5)

  def testUpdateOpsReturnsCurrentValue(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = tf.contrib.metrics.streaming_mean(values)

      sess.run(tf.initialize_local_variables())

      self.assertAlmostEqual(0.5, sess.run(update_op), 5)
      self.assertAlmostEqual(1.475, sess.run(update_op), 5)
      self.assertAlmostEqual(12.4/6.0, sess.run(update_op), 5)
      self.assertAlmostEqual(1.65, sess.run(update_op), 5)

      self.assertAlmostEqual(1.65, sess.run(mean), 5)

  def testMissingValues(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the missing labels.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [1, 1])
      _enqueue_vector(sess, weights_queue, [1, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = tf.contrib.metrics.streaming_mean(values, weights)

      sess.run(tf.initialize_local_variables())
      for _ in range(4):
        sess.run(update_op)
      self.assertAlmostEqual(-0.8, sess.run(mean), 5)


class StreamingAccuracyTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_accuracy(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_accuracy(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = tf.ones((10, 3))
    labels = tf.ones((10, 4))
    with self.assertRaises(ValueError):
      tf.contrib.metrics.streaming_accuracy(predictions, labels)

  def testPredictionsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = tf.ones((10, 3))
    labels = tf.ones((10, 3))
    weights = tf.ones((9, 3))
    with self.assertRaises(ValueError):
      tf.contrib.metrics.streaming_accuracy(predictions, labels, weights)

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=3, dtype=tf.int64, seed=1)
    labels = tf.random_uniform((10, 3), maxval=3, dtype=tf.int64, seed=1)
    accuracy, update_op = tf.contrib.metrics.streaming_accuracy(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_accuracy = accuracy.eval()
      for _ in range(10):
        self.assertEqual(initial_accuracy, accuracy.eval())

  def testMultipleUpdates(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      accuracy, update_op = tf.contrib.metrics.streaming_accuracy(
          predictions, labels)

      sess.run(tf.initialize_local_variables())
      for _ in range(4):
        sess.run(update_op)
      self.assertEqual(0.5, accuracy.eval())

  def testEffectivelyEquivalentSizes(self):
    predictions = tf.ones((40, 1))
    labels = tf.ones((40,))
    with self.test_session() as sess:
      accuracy, update_op = tf.contrib.metrics.streaming_accuracy(
          predictions, labels)

      sess.run(tf.initialize_local_variables())
      self.assertEqual(1.0, update_op.eval())
      self.assertEqual(1.0, accuracy.eval())

  def testMultipleUpdatesWithWeightedValues(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      # Create the queue that populates the missing labels.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.int64, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      weights = weights_queue.dequeue()

      accuracy, update_op = tf.contrib.metrics.streaming_accuracy(
          predictions, labels, weights)

      sess.run(tf.initialize_local_variables())
      for _ in range(4):
        sess.run(update_op)
      self.assertEqual(1.0, accuracy.eval())


class StreamingPrecisionTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_precision(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_precision(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    precision, update_op = tf.contrib.metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_precision = precision.eval()
      for _ in range(10):
        self.assertEqual(initial_precision, precision.eval())

  def testEffectivelyEquivalentShapes(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs, shape=(100, 1))
    labels = tf.constant(inputs, shape=(100,))
    precision, update_op = tf.contrib.metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertAlmostEqual(1, sess.run(update_op))
      self.assertAlmostEqual(1, precision.eval())

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs)
    labels = tf.constant(inputs)
    precision, update_op = tf.contrib.metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertAlmostEqual(1, sess.run(update_op))
      self.assertAlmostEqual(1, precision.eval())

  def testSomeCorrect(self):
    predictions = tf.constant([1, 0, 1, 0], shape=(1, 4))
    labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
    precision, update_op = tf.contrib.metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertAlmostEqual(0.5, update_op.eval())
      self.assertAlmostEqual(0.5, precision.eval())

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs)
    labels = tf.constant(1 - inputs)
    precision, update_op = tf.contrib.metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertAlmostEqual(0, precision.eval())

  def testZeroTrueAndFalsePositivesGivesZeroPrecision(self):
    predictions = tf.constant([0, 0, 0, 0])
    labels = tf.constant([0, 0, 0, 0])
    precision, update_op = tf.contrib.metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0.0, precision.eval())


class StreamingRecallTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_recall(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_recall(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    recall, update_op = tf.contrib.metrics.streaming_recall(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_recall = recall.eval()
      for _ in range(10):
        self.assertEqual(initial_recall, recall.eval())

  def testEffectivelyEquivalentShapes(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(np_inputs, shape=(100,))
    labels = tf.constant(np_inputs, shape=(100, 1))
    recall, update_op = tf.contrib.metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(1, recall.eval())

  def testAllCorrect(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(np_inputs)
    labels = tf.constant(np_inputs)
    recall, update_op = tf.contrib.metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(1, recall.eval())

  def testSomeCorrect(self):
    predictions = tf.constant([1, 0, 1, 0], shape=(1, 4))
    labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
    recall, update_op = tf.contrib.metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertAlmostEqual(0.5, update_op.eval())
      self.assertAlmostEqual(0.5, recall.eval())

  def testAllIncorrect(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(np_inputs)
    labels = tf.constant(1 - np_inputs)
    recall, update_op = tf.contrib.metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0, recall.eval())

  def testZeroTruePositivesAndFalseNegativesGivesZeroRecall(self):
    predictions = tf.zeros((1, 4))
    labels = tf.zeros((1, 4))
    recall, update_op = tf.contrib.metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0, recall.eval())


class StreamingAUCTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_auc(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_auc(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.float32, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    auc, update_op = tf.contrib.metrics.streaming_auc(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_auc = auc.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_auc, auc.eval(), 5)

  def testEffectivelyEquivalentShapes(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32, shape=(100,))
      labels = tf.constant(inputs, shape=(100, 1))
      auc, update_op = tf.contrib.metrics.streaming_auc(predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertEqual(1, auc.eval())

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(inputs)
      auc, update_op = tf.contrib.metrics.streaming_auc(predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertEqual(1, auc.eval())

  def testSomeCorrect(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
      auc, update_op = tf.contrib.metrics.streaming_auc(predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertAlmostEqual(0.5, auc.eval())

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(1 - inputs, dtype=tf.float32)
      auc, update_op = tf.contrib.metrics.streaming_auc(predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertAlmostEqual(0, auc.eval())

  def testZeroTruePositivesAndFalseNegativesGivesOneAUC(self):
    with self.test_session() as sess:
      predictions = tf.zeros([4], dtype=tf.float32)
      labels = tf.zeros([4])
      auc, update_op = tf.contrib.metrics.streaming_auc(predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertAlmostEqual(1, auc.eval(), 6)

  def np_auc(self, predictions, labels):
    """Computes the AUC explicitely using Numpy.

    Args:
      predictions: an ndarray with shape [N, 1].
      labels: an ndarray with shape [N, 1].

    Returns:
      the area under the ROC curve.
    """
    num_positives = np.count_nonzero(labels)
    num_negatives = labels.size - num_positives

    # Sort descending:
    inds = np.argsort(-predictions.transpose())

    predictions = predictions[inds].squeeze()
    labels = labels[inds].squeeze()

    tp = np.cumsum(labels > 0) / float(num_positives)

    return np.sum(tp[labels == 0] / num_negatives)

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
    expected_auc = self.np_auc(predictions, labels)

    labels = labels.astype(np.float32)
    predictions = predictions.astype(np.float32)

    with self.test_session() as sess:
      # Reshape the data so its easy to queue up:
      predictions_batches = predictions.reshape((batch_size, num_batches))
      labels_batches = labels.reshape((batch_size, num_batches))

      # Enqueue the data:
      predictions_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                                       shapes=(batch_size,))
      labels_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                                  shapes=(batch_size,))

      for i in range(int(num_batches)):
        tf_prediction = tf.constant(predictions_batches[:, i])
        tf_label = tf.constant(labels_batches[:, i])
        sess.run([predictions_queue.enqueue(tf_prediction),
                  labels_queue.enqueue(tf_label)])

      tf_predictions = predictions_queue.dequeue()
      tf_labels = labels_queue.dequeue()

      auc, update_op = tf.contrib.metrics.streaming_auc(
          tf_predictions, tf_labels, num_thresholds=500)

      sess.run(tf.initialize_local_variables())
      for _ in range(int(num_samples / batch_size)):
        sess.run(update_op)

      # Since this is only approximate, we can't expect a 6 digits match.
      # Although with higher number of samples/thresholds we should see the
      # accuracy improving
      self.assertAlmostEqual(expected_auc, auc.eval(), 2)


class StreamingPrecisionRecallThresholdsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    prec, _ = tf.contrib.metrics.streaming_precision_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    rec, _ = tf.contrib.metrics.streaming_recall_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [prec, rec])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, precision_op = tf.contrib.metrics.streaming_precision_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    _, recall_op = tf.contrib.metrics.streaming_recall_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name),
                         [precision_op, recall_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.float32, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    thresholds = [0, 0.5, 1.0]
    prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
        predictions, labels, thresholds)
    rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
        predictions, labels, thresholds)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates, then verify idempotency.
      sess.run([prec_op, rec_op])
      initial_prec = prec.eval()
      initial_rec = rec.eval()
      for _ in range(10):
        sess.run([prec_op, rec_op])
        self.assertAllClose(initial_prec, prec.eval())
        self.assertAllClose(initial_rec, rec.eval())

  def testEffectivelyEquivalentShapes(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32, shape=(100,))
      labels = tf.constant(inputs, shape=(100, 1))
      thresholds = [0.5]
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertEqual(1, prec.eval())
      self.assertEqual(1, rec.eval())

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(inputs)
      thresholds = [0.5]
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertEqual(1, prec.eval())
      self.assertEqual(1, rec.eval())

  def testSomeCorrect(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
      thresholds = [0.5]
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0.5, prec.eval())
      self.assertAlmostEqual(0.5, rec.eval())

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(1 - inputs, dtype=tf.float32)
      thresholds = [0.5]
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0, prec.eval())
      self.assertAlmostEqual(0, rec.eval())

  def testIgnoreMask(self):
    with self.test_session() as sess:
      predictions = tf.constant([[1, 0], [1, 0]], shape=(2, 2),
                                dtype=tf.float32)
      labels = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
      ignore_mask = tf.constant([[True, True], [False, False]], shape=(2, 2))
      thresholds = [0.5, 1.1]
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds, ignore_mask=ignore_mask)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds, ignore_mask=ignore_mask)

      [prec_low, prec_high] = tf.split(0, 2, prec)
      prec_low = tf.reshape(prec_low, shape=())
      prec_high = tf.reshape(prec_high, shape=())
      [rec_low, rec_high] = tf.split(0, 2, rec)
      rec_low = tf.reshape(rec_low, shape=())
      rec_high = tf.reshape(rec_high, shape=())

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(1.0, prec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, prec_high.eval(), places=5)
      self.assertAlmostEqual(1.0, rec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, rec_high.eval(), places=5)

  def testExtremeThresholds(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 1], shape=(1, 4))
      thresholds = [-1.0, 2.0]  # lower/higher than any values
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      [prec_low, prec_high] = tf.split(0, 2, prec)
      [rec_low, rec_high] = tf.split(0, 2, rec)

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0.75, prec_low.eval())
      self.assertAlmostEqual(0.0, prec_high.eval())
      self.assertAlmostEqual(1.0, rec_low.eval())
      self.assertAlmostEqual(0.0, rec_high.eval())

  def testZeroLabelsPredictions(self):
    with self.test_session() as sess:
      predictions = tf.zeros([4], dtype=tf.float32)
      labels = tf.zeros([4])
      thresholds = [0.5]
      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.initialize_local_variables())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0, prec.eval(), 6)
      self.assertAlmostEqual(0, rec.eval(), 6)

  def testWithMultipleUpdates(self):
    num_samples = 1000
    batch_size = 10
    num_batches = num_samples / batch_size

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    noise = np.random.normal(0.0, scale=0.2, size=(num_samples, 1))
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0
    thresholds = [0.3]

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(num_samples):
      if predictions[i] > thresholds[0]:
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

    labels = labels.astype(np.float32)
    predictions = predictions.astype(np.float32)

    with self.test_session() as sess:
      # Reshape the data so its easy to queue up:
      predictions_batches = predictions.reshape((batch_size, num_batches))
      labels_batches = labels.reshape((batch_size, num_batches))

      # Enqueue the data:
      predictions_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                                       shapes=(batch_size,))
      labels_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                                  shapes=(batch_size,))

      for i in range(int(num_batches)):
        tf_prediction = tf.constant(predictions_batches[:, i])
        tf_label = tf.constant(labels_batches[:, i])
        sess.run([predictions_queue.enqueue(tf_prediction),
                  labels_queue.enqueue(tf_label)])

      tf_predictions = predictions_queue.dequeue()
      tf_labels = labels_queue.dequeue()

      prec, prec_op = tf.contrib.metrics.streaming_precision_at_thresholds(
          tf_predictions, tf_labels, thresholds)
      rec, rec_op = tf.contrib.metrics.streaming_recall_at_thresholds(
          tf_predictions, tf_labels, thresholds)

      sess.run(tf.initialize_local_variables())
      for _ in range(int(num_samples / batch_size)):
        sess.run([prec_op, rec_op])
      # Since this is only approximate, we can't expect a 6 digits match.
      # Although with higher number of samples/thresholds we should see the
      # accuracy improving
      self.assertAlmostEqual(expected_prec, prec.eval(), 2)
      self.assertAlmostEqual(expected_rec, rec.eval(), 2)


class StreamingRecallAtKTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

    self._batch_size = 4
    self._num_classes = 3
    self._np_predictions = np.matrix(('0.1 0.2 0.7;'
                                      '0.6 0.2 0.2;'
                                      '0.0 0.9 0.1;'
                                      '0.2 0.0 0.8'))
    self._np_labels = [0, 0, 0, 0]

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_recall_at_k(
        predictions=tf.ones((self._batch_size, self._num_classes)),
        labels=tf.ones((self._batch_size,), dtype=tf.int32),
        k=1,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_recall_at_k(
        predictions=tf.ones((self._batch_size, self._num_classes)),
        labels=tf.ones((self._batch_size,), dtype=tf.int32),
        k=1,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testSingleUpdateAllPresentKIs1(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(self._np_labels, shape=(self._batch_size,))
    recall, update_op = tf.contrib.metrics.streaming_recall_at_k(
        predictions, labels, k=1)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0.25, recall.eval())

  def testSingleUpdateAllPresentKIs2(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(self._np_labels, shape=(self._batch_size,))
    recall, update_op = tf.contrib.metrics.streaming_recall_at_k(
        predictions, labels, k=2)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0.5, recall.eval())

  def testSingleUpdateAllPresentKIs3(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(self._np_labels, shape=(self._batch_size,))
    recall, update_op = tf.contrib.metrics.streaming_recall_at_k(
        predictions, labels, k=3)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(1.0, recall.eval())

  def testSingleUpdateSomeMissingKIs2(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(self._np_labels, shape=(self._batch_size,))
    ignore_mask = tf.constant([True, False, True, False],
                              shape=(self._batch_size,), dtype=tf.bool)
    recall, update_op = tf.contrib.metrics.streaming_recall_at_k(
        predictions, labels, k=2, ignore_mask=ignore_mask)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(1.0, recall.eval())


class StreamingSparsePrecisionTest(tf.test.TestCase):

  def _assert_precision_at_k(self,
                             predictions,
                             labels,
                             k,
                             expected,
                             class_id=None,
                             ignore_mask=None):
    loss, loss_update = tf.contrib.metrics.streaming_sparse_precision_at_k(
        predictions=tf.constant(predictions, tf.float32), labels=labels,
        k=k, class_id=class_id, ignore_mask=ignore_mask)

    # Fails without initialized vars.
    self.assertRaises(tf.OpError, loss.eval)
    self.assertRaises(tf.OpError, loss_update.eval)
    tf.initialize_variables(tf.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      self.assertTrue(math.isnan(loss_update.eval()))
      self.assertTrue(math.isnan(loss.eval()))
    else:
      self.assertEqual(expected, loss_update.eval())
      self.assertEqual(expected, loss.eval())

  def test_one_label_at_k1_no_predictions(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]

    # Classes 0,1,2 have 0 predictions, class 4 is out of range.
    for class_id in [0, 1, 2, 4]:
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_2d_label_to_sparse(labels), k=1, expected=NAN,
            class_id=class_id)

  def test_one_label_at_k1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]

    # Class 3: 1 label, 2 predictions, 1 correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_2d_label_to_sparse(labels), k=1,
          expected=1.0 / 2.0, class_id=3)

    # All classes: 2 labels, 2 predictions, 1 correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_2d_label_to_sparse(labels), k=1,
          expected=1.0 / 2.0)

  def test_three_labels_at_k5_no_predictions(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    labels = [
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ]

    # Classes 1,3,8 have 0 predictions, class 10 is out of range.
    for class_id in [1, 3, 8, 10]:
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_2d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id)

  def test_three_labels_at_k5_no_labels(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    labels = [
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ]

    # Classes 0,4,6,9: 0 labels, >=1 prediction.
    for class_id in [0, 4, 6, 9]:
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_2d_label_to_sparse(labels), k=5, expected=0.0,
            class_id=class_id)

  def test_three_labels_at_k5(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    labels = [
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ]

    # Class 2: 2 labels, 2 correct predictions.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_2d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=2)

    # Class 5: 1 label, 1 correct prediction.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_2d_label_to_sparse(labels), k=5,
          expected=1.0 / 1.0, class_id=5)

    # Class 7: 1 label, 1 incorrect prediction.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_2d_label_to_sparse(labels), k=5,
          expected=0.0 / 1.0, class_id=7)

    # All classes: 10 predictions, 3 correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_2d_label_to_sparse(labels), k=5,
          expected=3.0 / 10.0)

  def test_3d_no_predictions(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    # Classes 1,3,8 have 0 predictions, class 10 is out of range.
    for class_id in [1, 3, 8, 10]:
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id)

  def test_3d_no_labels(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    # Classes 0,4,6,9: 0 labels, >=1 prediction.
    for class_id in [0, 4, 6, 9]:
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=0.0,
            class_id=class_id)

  def test_3d(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    # Class 2: 4 predictions, all correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=4.0 / 4.0, class_id=2)

    # Class 5: 2 predictions, both correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=5)

    # Class 7: 2 predictions, 1 correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=1.0 / 2.0, class_id=7)

    # All classes: 20 predictions, 7 correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=7.0 / 20.0)

  def test_3d_ignore_all(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    for class_id in xrange(10):
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id, ignore_mask=[True, True])
      with self.test_session():
        self._assert_precision_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id, ignore_mask=[[True, True], [True, True]])
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
          ignore_mask=[True, True])
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
          ignore_mask=[[True, True], [True, True]])

  def test_3d_ignore_some(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    # Class 2: 2 predictions, both correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=2, ignore_mask=[False, True])

    # Class 2: 2 predictions, both correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=2, ignore_mask=[True, False])

    # Class 7: 1 incorrect prediction.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=0.0 / 1.0, class_id=7, ignore_mask=[False, True])

    # Class 7: 1 correct prediction.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=1.0 / 1.0, class_id=7, ignore_mask=[True, False])

    # Class 7: no predictions.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=NAN, class_id=7, ignore_mask=[[False, True], [True, False]])

    # Class 7: 2 predictions, 1 correct.
    with self.test_session():
      self._assert_precision_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=1.0 / 2.0, class_id=7,
          ignore_mask=[[True, False], [False, True]])

  def test_sparse_tensor_value(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]
    expected_precision = 0.5
    with self.test_session():
      _, precision = tf.contrib.metrics.streaming_sparse_precision_at_k(
          predictions=tf.constant(predictions, tf.float32),
          labels=_binary_2d_label_to_sparse_value(labels), k=1)

      tf.initialize_variables(tf.local_variables()).run()

      self.assertEqual(expected_precision, precision.eval())


class StreamingSparseRecallTest(tf.test.TestCase):

  def _assert_recall_at_k(self,
                          predictions,
                          labels,
                          k,
                          expected,
                          class_id=None,
                          ignore_mask=None):
    loss, loss_update = tf.contrib.metrics.streaming_sparse_recall_at_k(
        predictions=tf.constant(predictions, tf.float32),
        labels=labels, k=k, class_id=class_id, ignore_mask=ignore_mask)

    # Fails without initialized vars.
    self.assertRaises(tf.OpError, loss.eval)
    self.assertRaises(tf.OpError, loss_update.eval)
    tf.initialize_variables(tf.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      self.assertTrue(math.isnan(loss_update.eval()))
      self.assertTrue(math.isnan(loss.eval()))
    else:
      self.assertEqual(expected, loss_update.eval())
      self.assertEqual(expected, loss.eval())

  def test_one_label_at_k1_empty_classes(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]

    # Classes 0,1 have 0 labels, 0 predictions, class 4 is out of range.
    for class_id in [0, 1, 4]:
      with self.test_session():
        self._assert_recall_at_k(
            predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
            k=1, expected=NAN, class_id=class_id)

  def test_one_label_at_k1_no_predictions(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]

    # Class 2: 0 predictions.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=1, expected=0.0, class_id=2)

  def test_one_label_at_k1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]

    # Class 3: 1 label, 2 predictions, 1 correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=1, expected=1.0 / 1.0, class_id=3)

    # All classes: 2 labels, 2 predictions, 1 correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=1, expected=1.0 / 2.0)

  def test_three_labels_at_k5_no_labels(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    labels = [
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]

    # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
    for class_id in [0, 3, 4, 6, 9, 10]:
      with self.test_session():
        self._assert_recall_at_k(
            predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
            k=5, expected=NAN, class_id=class_id)

  def test_three_labels_at_k5_no_predictions(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    labels = [
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]

    # Class 8: 1 label, no predictions.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=5, expected=0.0 / 1.0, class_id=8)

  def test_three_labels_at_k5(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    labels = [
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]

    # Class 2: 2 labels, both correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=5, expected=2.0 / 2.0, class_id=2)

    # Class 5: 1 label, incorrect.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=5, expected=1.0 / 1.0, class_id=5)

    # Class 7: 1 label, incorrect.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=5, expected=0.0 / 1.0, class_id=7)

    # All classes: 6 labels, 3 correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions=predictions, labels=_binary_2d_label_to_sparse(labels),
          k=5, expected=3.0 / 6.0)

  def test_3d_no_labels(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    ]]

    # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
    for class_id in [0, 3, 4, 6, 9, 10]:
      with self.test_session():
        self._assert_recall_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id)

  def test_3d_no_predictions(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    ]]

    # Classes 1,8 have 0 predictions, >=1 label.
    for class_id in [1, 8]:
      with self.test_session():
        self._assert_recall_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=0.0,
            class_id=class_id)

  def test_3d(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    # Class 2: 4 labels, all correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=4.0 / 4.0, class_id=2)

    # Class 5: 2 labels, both correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=5)

    # Class 7: 2 labels, 1 incorrect.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=1.0 / 2.0, class_id=7)

    # All classes: 12 labels, 7 correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=7.0 / 12.0)

  def test_3d_ignore_all(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    for class_id in xrange(10):
      with self.test_session():
        self._assert_recall_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id, ignore_mask=[True, True])
      with self.test_session():
        self._assert_recall_at_k(
            predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
            class_id=class_id, ignore_mask=[[True, True], [True, True]])
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
          ignore_mask=[True, True])
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5, expected=NAN,
          ignore_mask=[[True, True], [True, True]])

  def test_3d_ignore_some(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = [[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]]

    # Class 2: 2 labels, both correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=2, ignore_mask=[False, True])

    # Class 2: 2 labels, both correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=2.0 / 2.0, class_id=2, ignore_mask=[True, False])

    # Class 7: 1 label, correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=1.0 / 1.0, class_id=7, ignore_mask=[True, False])

    # Class 7: 1 label, incorrect.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=0.0 / 1.0, class_id=7, ignore_mask=[False, True])

    # Class 7: 2 labels, 1 correct.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=1.0 / 2.0, class_id=7,
          ignore_mask=[[False, True], [False, True]])

    # Class 7: No labels.
    with self.test_session():
      self._assert_recall_at_k(
          predictions, _binary_3d_label_to_sparse(labels), k=5,
          expected=NAN, class_id=7,
          ignore_mask=[[True, False], [True, False]])

  def test_sparse_tensor_value(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 1, 0], [0, 0, 0, 1]]
    expected_recall = 0.5
    with self.test_session():
      _, recall = tf.contrib.metrics.streaming_sparse_recall_at_k(
          predictions=tf.constant(predictions, tf.float32),
          labels=_binary_2d_label_to_sparse_value(labels), k=1)

      tf.initialize_variables(tf.local_variables()).run()

      self.assertEqual(expected_recall, recall.eval())


class StreamingMeanAbsoluteErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_mean_absolute_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_mean_absolute_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testEffectivelyEquivalentShapes(self):
    predictions = tf.ones((10, 3, 1))
    labels = tf.ones((10, 3,))
    error, update_op = tf.contrib.metrics.streaming_mean_absolute_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertEqual(0.0, update_op.eval())
      self.assertEqual(0.0, error.eval())

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = tf.contrib.metrics.streaming_mean_absolute_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateWithErrorAndMissing(self):
    predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)
    weights = tf.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = tf.contrib.metrics.streaming_mean_absolute_error(
        predictions, labels, weights)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(3, error.eval())


class StreamingMeanRelativeErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_mean_relative_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        normalizer=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_mean_relative_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        normalizer=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    normalizer = tf.random_normal((10, 3), seed=3)
    error, update_op = tf.contrib.metrics.streaming_mean_relative_error(
        predictions, labels, normalizer)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testEffectivelyEquivalentShapes(self):
    predictions = tf.ones((10, 3, 1))
    labels = tf.ones((10, 3,))
    normalizer = tf.ones((10, 3, 1))
    error, update_op = tf.contrib.metrics.streaming_mean_relative_error(
        predictions, labels, normalizer)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertEqual(0.0, update_op.eval())
      self.assertEqual(0.0, error.eval())

  def testSingleUpdateNormalizedByLabels(self):
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_labels = np.asarray([1, 3, 2, 3], dtype=np.float32)
    expected_error = np.mean(
        np.divide(np.absolute(np_predictions - np_labels),
                  np_labels))

    predictions = tf.constant(np_predictions, shape=(1, 4), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(1, 4))

    error, update_op = tf.contrib.metrics.streaming_mean_relative_error(
        predictions, labels, normalizer=labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(expected_error, error.eval())

  def testSingleUpdateNormalizedByZeros(self):
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)

    predictions = tf.constant(np_predictions, shape=(1, 4), dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)

    error, update_op = tf.contrib.metrics.streaming_mean_relative_error(
        predictions, labels, normalizer=tf.zeros_like(labels))

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0.0, error.eval())


class StreamingMeanSquaredErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testEffectivelyEquivalentShapes(self):
    predictions = tf.ones((10, 3, 1))
    labels = tf.ones((10, 3,))
    error, update_op = tf.contrib.metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertEqual(0.0, update_op.eval())
      self.assertEqual(0.0, error.eval())

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = tf.contrib.metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateZeroError(self):
    predictions = tf.zeros((1, 3), dtype=tf.float32)
    labels = tf.zeros((1, 3), dtype=tf.float32)

    error, update_op = tf.contrib.metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0, error.eval())

  def testSingleUpdateWithError(self):
    predictions = tf.constant([2, 4, 6], shape=(1, 3), dtype=tf.float32)
    labels = tf.constant([1, 3, 2], shape=(1, 3), dtype=tf.float32)

    error, update_op = tf.contrib.metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(6, error.eval())

  def testSingleUpdateWithErrorAndWeights(self):
    predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)
    weights = tf.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = tf.contrib.metrics.streaming_mean_squared_error(
        predictions, labels, weights)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(13, error.eval())

  def testMultipleBatchesOfSizeOne(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      error, update_op = tf.contrib.metrics.streaming_mean_squared_error(
          predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      sess.run(update_op)

      self.assertAlmostEqual(208 / 6.0, error.eval(), 5)

  def testMetricsComputedConcurrently(self):
    with self.test_session() as sess:
      # Create the queue that populates one set of predictions.
      preds_queue0 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue0, [10, 8, 6])
      _enqueue_vector(sess, preds_queue0, [-4, 3, -1])
      predictions0 = preds_queue0.dequeue()

      # Create the queue that populates one set of predictions.
      preds_queue1 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue1, [0, 1, 1])
      _enqueue_vector(sess, preds_queue1, [1, 1, 0])
      predictions1 = preds_queue1.dequeue()

      # Create the queue that populates one set of labels.
      labels_queue0 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue0, [1, 3, 2])
      _enqueue_vector(sess, labels_queue0, [2, 4, 6])
      labels0 = labels_queue0.dequeue()

      # Create the queue that populates another set of labels.
      labels_queue1 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue1, [-5, -3, -1])
      _enqueue_vector(sess, labels_queue1, [5, 4, 3])
      labels1 = labels_queue1.dequeue()

      mse0, update_op0 = tf.contrib.metrics.streaming_mean_squared_error(
          predictions0, labels0, name='msd0')
      mse1, update_op1 = tf.contrib.metrics.streaming_mean_squared_error(
          predictions1, labels1, name='msd1')

      sess.run(tf.initialize_local_variables())
      sess.run([update_op0, update_op1])
      sess.run([update_op0, update_op1])

      mse0, mse1 = sess.run([mse0, mse1])
      self.assertAlmostEqual(208 / 6.0, mse0, 5)
      self.assertAlmostEqual(79 / 6.0, mse1, 5)

  def testMultipleMetricsOnMultipleBatchesOfSizeOne(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      mae, ma_update_op = tf.contrib.metrics.streaming_mean_absolute_error(
          predictions, labels)
      mse, ms_update_op = tf.contrib.metrics.streaming_mean_squared_error(
          predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run([ma_update_op, ms_update_op])
      sess.run([ma_update_op, ms_update_op])

      self.assertAlmostEqual(32 / 6.0, mae.eval(), 5)
      self.assertAlmostEqual(208 / 6.0, mse.eval(), 5)


class StreamingRootMeanSquaredErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_root_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_root_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testEffectivelyEquivalentShapes(self):
    predictions = tf.ones((10, 3,))
    labels = tf.ones((10, 3, 1))
    error, update_op = tf.contrib.metrics.streaming_root_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertEqual(0.0, update_op.eval())
      self.assertEqual(0.0, error.eval())

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = tf.contrib.metrics.streaming_root_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateZeroError(self):
    with self.test_session() as sess:
      predictions = tf.constant(0.0, shape=(1, 3), dtype=tf.float32)
      labels = tf.constant(0.0, shape=(1, 3), dtype=tf.float32)

      rmse, update_op = tf.contrib.metrics.streaming_root_mean_squared_error(
          predictions, labels)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertEqual(0, rmse.eval())

  def testSingleUpdateWithError(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6], shape=(1, 3), dtype=tf.float32)
      labels = tf.constant([1, 3, 2], shape=(1, 3), dtype=tf.float32)

      rmse, update_op = tf.contrib.metrics.streaming_root_mean_squared_error(
          predictions, labels)

      sess.run(tf.initialize_local_variables())
      self.assertAlmostEqual(math.sqrt(6), update_op.eval(), 5)
      self.assertAlmostEqual(math.sqrt(6), rmse.eval(), 5)

  def testSingleUpdateWithErrorAndMissing(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)
      weights = tf.constant([0, 1, 0, 1], shape=(1, 4))

      rmse, update_op = tf.contrib.metrics.streaming_root_mean_squared_error(
          predictions, labels, weights)

      sess.run(tf.initialize_local_variables())
      sess.run(update_op)

      self.assertAlmostEqual(math.sqrt(13), rmse.eval(), 5)


class StreamingMeanCosineDistanceTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions=tf.ones((10, 3)),
        labels=tf.ones((10, 3)),
        dim=1,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions=tf.ones((10, 3)),
        labels=tf.ones((10, 3)),
        dim=1,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testEffectivelyEquivalentShapes(self):
    predictions = tf.nn.l2_normalize(tf.ones((10, 3,)), dim=1)
    labels = tf.nn.l2_normalize(tf.ones((10, 3, 1)), dim=1)
    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=1)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertAlmostEqual(0.0, update_op.eval(), 5)
      self.assertAlmostEqual(0.0, error.eval(), 5)

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=1)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateZeroError(self):
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))

    predictions = tf.constant(np_labels, shape=(1, 3, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(1, 3, 3), dtype=tf.float32)

    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0, error.eval())

  def testSingleUpdateWithError1(self):
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))
    np_predictions = np.matrix(('1 0 0;'
                                '0 0 -1;'
                                '1 0 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)

    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertAlmostEqual(1, error.eval(), 5)

  def testSingleUpdateWithError2(self):
    np_predictions = np.matrix((
        '0.819031913261206 0.567041924552012 0.087465312324590;'
        '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
        '0.707106781186548 -0.707106781186548 0'))
    np_labels = np.matrix((
        '0.819031913261206 0.567041924552012 0.087465312324590;'
        '0.665139432070255 0.739487441769973 0.103671883216994;'
        '0.707106781186548 0.707106781186548 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)
    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertAlmostEqual(1.0, error.eval(), 5)

  def testSingleUpdateWithErrorAndMissing1(self):
    np_predictions = np.matrix(('1 0 0;'
                                '0 0 -1;'
                                '1 0 0'))
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)
    weights = tf.constant([1, 0, 0], shape=(3, 1, 1))

    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2, weights=weights)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertEqual(0, error.eval())

  def testSingleUpdateWithErrorAndMissing2(self):
    np_predictions = np.matrix(('1 0 0;'
                                '0 0 -1;'
                                '1 0 0'))
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)
    weights = tf.constant([0, 1, 1], shape=(3, 1, 1))

    error, update_op = tf.contrib.metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2, weights=weights)

    with self.test_session() as sess:
      sess.run(tf.initialize_local_variables())
      self.assertEqual(1.5, update_op.eval())
      self.assertEqual(1.5, error.eval())


class PcntBelowThreshTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = tf.contrib.metrics.streaming_percentage_less(
        values=tf.ones((10,)),
        threshold=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = tf.contrib.metrics.streaming_percentage_less(
        values=tf.ones((10,)),
        threshold=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testAllPresentOneUpdate(self):
    with self.test_session() as sess:
      values = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
      ignore_mask = tf.constant([False, False, False, False], shape=(1, 4))

      pcnt0, update_op0 = tf.contrib.metrics.streaming_percentage_less(
          values, 100, ignore_mask, name='high')
      pcnt1, update_op1 = tf.contrib.metrics.streaming_percentage_less(
          values, 7, ignore_mask, name='medium')
      pcnt2, update_op2 = tf.contrib.metrics.streaming_percentage_less(
          values, 1, ignore_mask, name='low')

      sess.run(tf.initialize_local_variables())
      sess.run([update_op0, update_op1, update_op2])

      pcnt0, pcnt1, pcnt2 = sess.run([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.75, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)

  def testSomePresentOneUpdate(self):
    with self.test_session() as sess:
      values = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
      ignore_mask = tf.constant([False, True, True, False], shape=(1, 4))

      pcnt0, update_op0 = tf.contrib.metrics.streaming_percentage_less(
          values, 100, ignore_mask, name='high')
      pcnt1, update_op1 = tf.contrib.metrics.streaming_percentage_less(
          values, 7, ignore_mask, name='medium')
      pcnt2, update_op2 = tf.contrib.metrics.streaming_percentage_less(
          values, 1, ignore_mask, name='low')

      sess.run(tf.initialize_local_variables())
      self.assertListEqual([1.0, 0.5, 0.0],
                           sess.run([update_op0, update_op1, update_op2]))

      pcnt0, pcnt1, pcnt2 = sess.run([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.5, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)


if __name__ == '__main__':
  tf.test.main()
