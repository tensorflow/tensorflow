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
"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

NAN = float('nan')


def _enqueue_vector(sess, queue, values, shape=None):
  if not shape:
    shape = (1, len(values))
  dtype = queue.dtypes[0]
  sess.run(
      queue.enqueue(constant_op.constant(
          values, dtype=dtype, shape=shape)))


def _binary_2d_label_to_2d_sparse_value(labels):
  """Convert dense 2D binary indicator to sparse ID.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator, shape [batch_size, num_classes].

  Returns:
    `SparseTensorValue` of shape [batch_size, num_classes], where num_classes
    is the number of `1` values in each row of `labels`. Values are indices
    of `1` values along the last dimension of `labels`.
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
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _binary_2d_label_to_1d_sparse_value(labels):
  """Convert dense 2D binary indicator to sparse ID.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator, shape [batch_size, num_classes]. Each
    row must contain exactly 1 `1` value.

  Returns:
    `SparseTensorValue` of shape [batch_size]. Values are indices of `1` values
    along the last dimension of `labels`.

  Raises:
    ValueError: if there is not exactly 1 `1` value per row of `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  if indices != [[i] for i in range(len(labels))]:
    raise ValueError('Expected 1 label/example, got %s.' % indices)
  shape = [len(labels)]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


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
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _assert_nan(test_case, actual):
  test_case.assertTrue(math.isnan(actual), 'Expected NAN, got %s.' % actual)


def _assert_metric_variables(test_case, expected):
  test_case.assertEquals(
      set(expected), set(v.name for v in variables.local_variables()))
  test_case.assertEquals(
      set(expected),
      set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))


def _test_values(shape):
  return np.reshape(np.cumsum(np.ones(shape)), newshape=shape)


class MeanTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean(array_ops.ones([4, 3]))
    _assert_metric_variables(self, ('mean/count:0', 'mean/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean(
        array_ops.ones([4, 3]), metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean(
        array_ops.ones([4, 3]), updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testBasic(self):
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean(values)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAlmostEqual(1.65, self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testUpdateOpsReturnsCurrentValue(self):
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean(values)

      self.evaluate(variables.local_variables_initializer())

      self.assertAlmostEqual(0.5, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1.475, self.evaluate(update_op), 5)
      self.assertAlmostEqual(12.4 / 6.0, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1.65, self.evaluate(update_op), 5)

      self.assertAlmostEqual(1.65, self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    values = _test_values((3, 2, 4, 1))
    mean_results = (
        metrics.mean(values),
        metrics.mean(values, weights=1.0),
        metrics.mean(values, weights=np.ones((1, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 1, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 1, 1, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 1, 4))),
        metrics.mean(values, weights=np.ones((1, 1, 4, 1))),
        metrics.mean(values, weights=np.ones((1, 2, 1))),
        metrics.mean(values, weights=np.ones((1, 2, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 2, 4))),
        metrics.mean(values, weights=np.ones((1, 2, 4, 1))),
        metrics.mean(values, weights=np.ones((3, 1, 1))),
        metrics.mean(values, weights=np.ones((3, 1, 1, 1))),
        metrics.mean(values, weights=np.ones((3, 1, 4))),
        metrics.mean(values, weights=np.ones((3, 1, 4, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 1, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 4))),
        metrics.mean(values, weights=np.ones((3, 2, 4, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 4, 1, 1))),)
    expected = np.mean(values)
    with self.cached_session():
      variables.local_variables_initializer().run()
      for mean_result in mean_results:
        mean, update_op = mean_result
        self.assertAlmostEqual(expected, update_op.eval())
        self.assertAlmostEqual(expected, mean.eval())

  def _test_3d_weighted(self, values, weights):
    expected = (
        np.sum(np.multiply(weights, values)) /
        np.sum(np.multiply(weights, np.ones_like(values)))
    )
    mean, update_op = metrics.mean(values, weights=weights)
    with self.cached_session():
      variables.local_variables_initializer().run()
      self.assertAlmostEqual(expected, update_op.eval(), places=5)
      self.assertAlmostEqual(expected, mean.eval(), places=5)

  @test_util.run_deprecated_v1
  def test1x1x1Weighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5,)).reshape((1, 1, 1)))

  @test_util.run_deprecated_v1
  def test1x1xNWeighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 7, 11, 3)).reshape((1, 1, 4)))

  @test_util.run_deprecated_v1
  def test1xNx1Weighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 11)).reshape((1, 2, 1)))

  @test_util.run_deprecated_v1
  def test1xNxNWeighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1x1Weighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 7, 11)).reshape((3, 1, 1)))

  @test_util.run_deprecated_v1
  def testNx1xNWeighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4)))

  @test_util.run_deprecated_v1
  def testNxNxNWeighted(self):
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidWeights(self):
    values_placeholder = array_ops.placeholder(dtype=dtypes_lib.float32)
    values = _test_values((3, 2, 4, 1))
    invalid_weights = (
        (1,),
        (1, 1),
        (3, 2),
        (2, 4, 1),
        (4, 2, 4, 1),
        (3, 3, 4, 1),
        (3, 2, 5, 1),
        (3, 2, 4, 2),
        (1, 1, 1, 1, 1))
    expected_error_msg = 'weights can not be broadcast to values'
    for invalid_weight in invalid_weights:
      # Static shapes.
      with self.assertRaisesRegexp(ValueError, expected_error_msg):
        metrics.mean(values, invalid_weight)

      # Dynamic shapes.
      with self.assertRaisesRegexp(errors_impl.OpError, expected_error_msg):
        with self.cached_session():
          _, update_op = metrics.mean(values_placeholder, invalid_weight)
          variables.local_variables_initializer().run()
          update_op.eval(feed_dict={values_placeholder: values})


class MeanTensorTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_tensor(array_ops.ones([4, 3]))
    _assert_metric_variables(self,
                             ('mean/total_tensor:0', 'mean/count_tensor:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_tensor(
        array_ops.ones([4, 3]), metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_tensor(
        array_ops.ones([4, 3]), updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testBasic(self):
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[-0.9 / 4., 3.525]], self.evaluate(mean))

  @test_util.run_deprecated_v1
  def testMultiDimensional(self):
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(2, 2, 2))
      _enqueue_vector(
          sess,
          values_queue, [[[1, 2], [1, 2]], [[1, 2], [1, 2]]],
          shape=(2, 2, 2))
      _enqueue_vector(
          sess,
          values_queue, [[[1, 2], [1, 2]], [[3, 4], [9, 10]]],
          shape=(2, 2, 2))
      values = values_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(2):
        self.evaluate(update_op)
      self.assertAllClose([[[1, 2], [1, 2]], [[2, 3], [5, 6]]],
                          self.evaluate(mean))

  @test_util.run_deprecated_v1
  def testUpdateOpsReturnsCurrentValue(self):
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values)

      self.evaluate(variables.local_variables_initializer())

      self.assertAllClose([[0, 1]], self.evaluate(update_op), 5)
      self.assertAllClose([[-2.1, 5.05]], self.evaluate(update_op), 5)
      self.assertAllClose([[2.3 / 3., 10.1 / 3.]], self.evaluate(update_op), 5)
      self.assertAllClose([[-0.9 / 4., 3.525]], self.evaluate(update_op), 5)

      self.assertAllClose([[-0.9 / 4., 3.525]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testBinaryWeighted1d(self):
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [[1]])
      _enqueue_vector(sess, weights_queue, [[0]])
      _enqueue_vector(sess, weights_queue, [[1]])
      _enqueue_vector(sess, weights_queue, [[0]])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[3.25, 0.5]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [[0.0025]])
      _enqueue_vector(sess, weights_queue, [[0.005]])
      _enqueue_vector(sess, weights_queue, [[0.01]])
      _enqueue_vector(sess, weights_queue, [[0.0075]])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[0.8, 3.52]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testWeighted2d_1(self):
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [1, 1])
      _enqueue_vector(sess, weights_queue, [1, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[-2.1, 0.5]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testWeighted2d_2(self):
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[0, 0.5]], self.evaluate(mean), 5)


class AccuracyTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        name='my_accuracy')
    _assert_metric_variables(self,
                             ('my_accuracy/count:0', 'my_accuracy/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones((10, 3))
    labels = array_ops.ones((10, 4))
    with self.assertRaises(ValueError):
      metrics.accuracy(labels, predictions)

  @test_util.run_deprecated_v1
  def testPredictionsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones((10, 3))
    labels = array_ops.ones((10, 3))
    weights = array_ops.ones((9, 3))
    with self.assertRaises(ValueError):
      metrics.accuracy(labels, predictions, weights)

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=3, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=3, dtype=dtypes_lib.int64, seed=1)
    accuracy, update_op = metrics.accuracy(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_accuracy = accuracy.eval()
      for _ in range(10):
        self.assertEqual(initial_accuracy, accuracy.eval())

  @test_util.run_deprecated_v1
  def testMultipleUpdates(self):
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      accuracy, update_op = metrics.accuracy(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      for _ in xrange(3):
        self.evaluate(update_op)
      self.assertEqual(0.5, self.evaluate(update_op))
      self.assertEqual(0.5, accuracy.eval())

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizes(self):
    predictions = array_ops.ones((40, 1))
    labels = array_ops.ones((40,))
    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, update_op.eval())
      self.assertEqual(1.0, accuracy.eval())

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithScalarWeight(self):
    predictions = array_ops.ones((40, 1))
    labels = array_ops.ones((40,))
    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions, weights=2.0)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, update_op.eval())
      self.assertEqual(1.0, accuracy.eval())

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithStaticShapedWeight(self):
    predictions = ops.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = array_ops.expand_dims(ops.convert_to_tensor([1, 0, 0]),
                                   1)  # shape 3, 1
    weights = array_ops.expand_dims(ops.convert_to_tensor([100, 1, 1]),
                                    1)  # shape 3, 1

    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions, weights)

      self.evaluate(variables.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(update_op.eval(), .95)
      self.assertGreater(accuracy.eval(), .95)

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithDynamicallyShapedWeight(self):
    predictions = ops.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = array_ops.expand_dims(ops.convert_to_tensor([1, 0, 0]),
                                   1)  # shape 3, 1

    weights = [[100], [1], [1]]  # shape 3, 1
    weights_placeholder = array_ops.placeholder(
        dtype=dtypes_lib.int32, name='weights')
    feed_dict = {weights_placeholder: weights}

    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions,
                                             weights_placeholder)

      self.evaluate(variables.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(update_op.eval(feed_dict=feed_dict), .95)
      self.assertGreater(accuracy.eval(feed_dict=feed_dict), .95)

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeightedValues(self):
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.int64, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      weights = weights_queue.dequeue()

      accuracy, update_op = metrics.accuracy(labels, predictions, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in xrange(3):
        self.evaluate(update_op)
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, accuracy.eval())


class PrecisionTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.precision(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(self, ('precision/false_positives/count:0',
                                    'precision/true_positives/count:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.precision(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.precision(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_precision = precision.eval()
      for _ in range(10):
        self.assertEqual(initial_precision, precision.eval())

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs)
    labels = constant_op.constant(inputs)
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op), 6)
      self.assertAlmostEqual(1.0, precision.eval(), 6)

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleInputDtypes(self):
    for dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions = math_ops.cast(
          constant_op.constant([1, 0, 1, 0], shape=(1, 4)), dtype=dtype)
      labels = math_ops.cast(
          constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=dtype)
      precision, update_op = metrics.precision(labels, predictions)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.5, update_op.eval())
        self.assertAlmostEqual(0.5, precision.eval())

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    predictions = constant_op.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    precision, update_op = metrics.precision(
        labels, predictions, weights=constant_op.constant([[2], [5]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 2.0 + 5.0
      weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, precision.eval())

  @test_util.run_deprecated_v1
  def testWeightedScalar_placeholders(self):
    predictions = array_ops.placeholder(dtype=dtypes_lib.float32)
    labels = array_ops.placeholder(dtype=dtypes_lib.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.precision(labels, predictions, weights=2)

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 2.0 + 2.0
      weighted_positives = (2.0 + 2.0) + (2.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testWeighted1d_placeholders(self):
    predictions = array_ops.placeholder(dtype=dtypes_lib.float32)
    labels = array_ops.placeholder(dtype=dtypes_lib.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.precision(
        labels, predictions, weights=constant_op.constant([[2], [5]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 2.0 + 5.0
      weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    predictions = constant_op.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    precision, update_op = metrics.precision(
        labels,
        predictions,
        weights=constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 3.0 + 4.0
      weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, precision.eval())

  @test_util.run_deprecated_v1
  def testWeighted2d_placeholders(self):
    predictions = array_ops.placeholder(dtype=dtypes_lib.float32)
    labels = array_ops.placeholder(dtype=dtypes_lib.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.precision(
        labels,
        predictions,
        weights=constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 3.0 + 4.0
      weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs)
    labels = constant_op.constant(1 - inputs)
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertAlmostEqual(0, precision.eval())

  @test_util.run_deprecated_v1
  def testZeroTrueAndFalsePositivesGivesZeroPrecision(self):
    predictions = constant_op.constant([0, 0, 0, 0])
    labels = constant_op.constant([0, 0, 0, 0])
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertEqual(0.0, precision.eval())


class RecallTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.recall(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self,
        ('recall/false_negatives/count:0', 'recall/true_positives/count:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.recall(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.recall(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_recall = recall.eval()
      for _ in range(10):
        self.assertEqual(initial_recall, recall.eval())

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(np_inputs)
    labels = constant_op.constant(np_inputs)
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertAlmostEqual(1.0, recall.eval(), 6)

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleInputDtypes(self):
    for dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions = math_ops.cast(
          constant_op.constant([1, 0, 1, 0], shape=(1, 4)), dtype=dtype)
      labels = math_ops.cast(
          constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=dtype)
      recall, update_op = metrics.recall(labels, predictions)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.5, update_op.eval())
        self.assertAlmostEqual(0.5, recall.eval())

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    predictions = constant_op.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    weights = constant_op.constant([[2], [5]])
    recall, update_op = metrics.recall(labels, predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      weighted_tp = 2.0 + 5.0
      weighted_t = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_t
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, recall.eval())

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    predictions = constant_op.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    weights = constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]])
    recall, update_op = metrics.recall(labels, predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      weighted_tp = 3.0 + 1.0
      weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
      expected_precision = weighted_tp / weighted_t
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, recall.eval())

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(np_inputs)
    labels = constant_op.constant(1 - np_inputs)
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertEqual(0, recall.eval())

  @test_util.run_deprecated_v1
  def testZeroTruePositivesAndFalseNegativesGivesZeroRecall(self):
    predictions = array_ops.zeros((1, 4))
    labels = array_ops.zeros((1, 4))
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertEqual(0, recall.eval())


class AUCTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.auc(predictions=array_ops.ones((10, 1)),
                labels=array_ops.ones((10, 1)))
    _assert_metric_variables(self,
                             ('auc/true_positives:0', 'auc/false_negatives:0',
                              'auc/false_positives:0', 'auc/true_negatives:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.auc(predictions=array_ops.ones((10, 1)),
                          labels=array_ops.ones((10, 1)),
                          metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.auc(predictions=array_ops.ones((10, 1)),
                               labels=array_ops.ones((10, 1)),
                               updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    auc, update_op = metrics.auc(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_auc = auc.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_auc, auc.eval(), 5)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    self.allCorrectAsExpected('ROC')

  def allCorrectAsExpected(self, curve):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(inputs)
      auc, update_op = metrics.auc(labels, predictions, curve=curve)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1, self.evaluate(update_op))

      self.assertEqual(1, auc.eval())

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleLabelDtypes(self):
    with self.cached_session():
      for label_dtype in (
          dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
        predictions = constant_op.constant(
            [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
        labels = math_ops.cast(
            constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=label_dtype)
        auc, update_op = metrics.auc(labels, predictions)

        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.5, self.evaluate(update_op))

        self.assertAlmostEqual(0.5, auc.eval())

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
      weights = constant_op.constant([2], shape=(1, 1))
      auc, update_op = metrics.auc(labels, predictions, weights=weights)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.5, self.evaluate(update_op), 5)

      self.assertAlmostEqual(0.5, auc.eval(), 5)

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
      weights = constant_op.constant([1, 2, 3, 4], shape=(1, 4))
      auc, update_op = metrics.auc(labels, predictions, weights=weights)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.7, self.evaluate(update_op), 5)

      self.assertAlmostEqual(0.7, auc.eval(), 5)

  @test_util.run_deprecated_v1
  def testManualThresholds(self):
    with self.cached_session():
      # Verifies that thresholds passed in to the `thresholds` parameter are
      # used correctly.
      # The default thresholds do not split the second and third predictions.
      # Thus, when we provide manual thresholds which correctly split it, we get
      # an accurate AUC value.
      predictions = constant_op.constant(
          [0.12, 0.3001, 0.3003, 0.72], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 0, 1], shape=(1, 4))
      weights = constant_op.constant([1, 1, 1, 1], shape=(1, 4))
      thresholds = [0.0, 0.2, 0.3002, 0.6, 1.0]
      default_auc, default_update_op = metrics.auc(labels,
                                                   predictions,
                                                   weights=weights)
      manual_auc, manual_update_op = metrics.auc(labels,
                                                 predictions,
                                                 weights=weights,
                                                 thresholds=thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.875, self.evaluate(default_update_op), 3)
      self.assertAlmostEqual(0.875, default_auc.eval(), 3)

      self.assertAlmostEqual(0.75, self.evaluate(manual_update_op), 3)
      self.assertAlmostEqual(0.75, manual_auc.eval(), 3)

  # Regarding the AUC-PR tests: note that the preferred method when
  # calculating AUC-PR is summation_method='careful_interpolation'.
  @test_util.run_deprecated_v1
  def testCorrectAUCPRSpecialCase(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 1], shape=(1, 4))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='careful_interpolation')

      self.evaluate(variables.local_variables_initializer())
      # expected ~= 0.79726744594
      expected = 1 - math.log(1.5) / 2
      self.assertAlmostEqual(expected, self.evaluate(update_op), delta=1e-3)
      self.assertAlmostEqual(expected, auc.eval(), delta=1e-3)

  @test_util.run_deprecated_v1
  def testCorrectAnotherAUCPRSpecialCase(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8, 0.1, 0.135, 0.81],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 0, 1, 0, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='careful_interpolation')

      self.evaluate(variables.local_variables_initializer())
      # expected ~= 0.61350593198
      expected = (2.5 - 2 * math.log(4./3) - 0.25 * math.log(7./5)) / 3
      self.assertAlmostEqual(expected, self.evaluate(update_op), delta=1e-3)
      self.assertAlmostEqual(expected, auc.eval(), delta=1e-3)

  @test_util.run_deprecated_v1
  def testThirdCorrectAUCPRSpecialCase(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [0.0, 0.1, 0.2, 0.33, 0.3, 0.4, 0.5],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 0, 0, 1, 1, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='careful_interpolation')

      self.evaluate(variables.local_variables_initializer())
      # expected ~= 0.90410597584
      expected = 1 - math.log(4./3) / 3
      self.assertAlmostEqual(expected, self.evaluate(update_op), delta=1e-3)
      self.assertAlmostEqual(expected, auc.eval(), delta=1e-3)

  @test_util.run_deprecated_v1
  def testIncorrectAUCPRSpecialCase(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 1], shape=(1, 4))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='trapezoidal')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.79166, self.evaluate(update_op), delta=1e-3)

      self.assertAlmostEqual(0.79166, auc.eval(), delta=1e-3)

  @test_util.run_deprecated_v1
  def testAnotherIncorrectAUCPRSpecialCase(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8, 0.1, 0.135, 0.81],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 0, 1, 0, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='trapezoidal')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.610317, self.evaluate(update_op), delta=1e-3)

      self.assertAlmostEqual(0.610317, auc.eval(), delta=1e-3)

  @test_util.run_deprecated_v1
  def testThirdIncorrectAUCPRSpecialCase(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [0.0, 0.1, 0.2, 0.33, 0.3, 0.4, 0.5],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 0, 0, 1, 1, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='trapezoidal')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.90277, self.evaluate(update_op), delta=1e-3)

      self.assertAlmostEqual(0.90277, auc.eval(), delta=1e-3)

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(1 - inputs, dtype=dtypes_lib.float32)
      auc, update_op = metrics.auc(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0, self.evaluate(update_op))

      self.assertAlmostEqual(0, auc.eval())

  @test_util.run_deprecated_v1
  def testZeroTruePositivesAndFalseNegativesGivesOneAUC(self):
    with self.cached_session():
      predictions = array_ops.zeros([4], dtype=dtypes_lib.float32)
      labels = array_ops.zeros([4])
      auc, update_op = metrics.auc(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1, self.evaluate(update_op), 6)

      self.assertAlmostEqual(1, auc.eval(), 6)

  @test_util.run_deprecated_v1
  def testRecallOneAndPrecisionOneGivesOnePRAUC(self):
    with self.cached_session():
      predictions = array_ops.ones([4], dtype=dtypes_lib.float32)
      labels = array_ops.ones([4])
      auc, update_op = metrics.auc(labels, predictions, curve='PR')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1, self.evaluate(update_op), 6)

      self.assertAlmostEqual(1, auc.eval(), 6)

  def np_auc(self, predictions, labels, weights):
    """Computes the AUC explicitly using Numpy.

    Args:
      predictions: an ndarray with shape [N].
      labels: an ndarray with shape [N].
      weights: an ndarray with shape [N].

    Returns:
      the area under the ROC curve.
    """
    if weights is None:
      weights = np.ones(np.size(predictions))
    is_positive = labels > 0
    num_positives = np.sum(weights[is_positive])
    num_negatives = np.sum(weights[~is_positive])

    # Sort descending:
    inds = np.argsort(-predictions)

    sorted_labels = labels[inds]
    sorted_weights = weights[inds]
    is_positive = sorted_labels > 0

    tp = np.cumsum(sorted_weights * is_positive) / num_positives
    return np.sum((sorted_weights * tp)[~is_positive]) / num_negatives

  @test_util.run_deprecated_v1
  def testWithMultipleUpdates(self):
    num_samples = 1000
    batch_size = 10
    num_batches = int(num_samples / batch_size)

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=num_samples)
    noise = np.random.normal(0.0, scale=0.2, size=num_samples)
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0

    def _enqueue_as_batches(x, enqueue_ops):
      x_batches = x.astype(np.float32).reshape((num_batches, batch_size))
      x_queue = data_flow_ops.FIFOQueue(
          num_batches, dtypes=dtypes_lib.float32, shapes=(batch_size,))
      for i in range(num_batches):
        enqueue_ops[i].append(x_queue.enqueue(x_batches[i, :]))
      return x_queue.dequeue()

    for weights in (None, np.ones(num_samples), np.random.exponential(
        scale=1.0, size=num_samples)):
      expected_auc = self.np_auc(predictions, labels, weights)

      with self.cached_session() as sess:
        enqueue_ops = [[] for i in range(num_batches)]
        tf_predictions = _enqueue_as_batches(predictions, enqueue_ops)
        tf_labels = _enqueue_as_batches(labels, enqueue_ops)
        tf_weights = (_enqueue_as_batches(weights, enqueue_ops) if
                      weights is not None else None)

        for i in range(num_batches):
          sess.run(enqueue_ops[i])

        auc, update_op = metrics.auc(tf_labels,
                                     tf_predictions,
                                     curve='ROC',
                                     num_thresholds=500,
                                     weights=tf_weights)

        self.evaluate(variables.local_variables_initializer())
        for i in range(num_batches):
          self.evaluate(update_op)

        # Since this is only approximate, we can't expect a 6 digits match.
        # Although with higher number of samples/thresholds we should see the
        # accuracy improving
        self.assertAlmostEqual(expected_auc, auc.eval(), 2)


class SpecificityAtSensitivityTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.specificity_at_sensitivity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        sensitivity=0.7)
    _assert_metric_variables(self,
                             ('specificity_at_sensitivity/true_positives:0',
                              'specificity_at_sensitivity/false_negatives:0',
                              'specificity_at_sensitivity/false_positives:0',
                              'specificity_at_sensitivity/true_negatives:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.specificity_at_sensitivity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        sensitivity=0.7,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.specificity_at_sensitivity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        sensitivity=0.7,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=2, dtype=dtypes_lib.int64, seed=1)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_specificity = specificity.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_specificity, specificity.eval(), 5)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
    labels = constant_op.constant(inputs)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1, self.evaluate(update_op))
      self.assertEqual(1, specificity.eval())

  @test_util.run_deprecated_v1
  def testSomeCorrectHighSensitivity(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.45, 0.5, 0.8, 0.9]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.8)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op))
      self.assertAlmostEqual(1.0, specificity.eval())

  @test_util.run_deprecated_v1
  def testSomeCorrectLowSensitivity(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.4)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      self.assertAlmostEqual(0.6, self.evaluate(update_op))
      self.assertAlmostEqual(0.6, specificity.eval())

  @test_util.run_deprecated_v1
  def testWeighted1d_multipleLabelDtypes(self):
    for label_dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.2, 0.2, 0.26, 0.26]
      labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
      weights_values = [3]

      predictions = constant_op.constant(
          predictions_values, dtype=dtypes_lib.float32)
      labels = math_ops.cast(labels_values, dtype=label_dtype)
      weights = constant_op.constant(weights_values)
      specificity, update_op = metrics.specificity_at_sensitivity(
          labels, predictions, weights=weights, sensitivity=0.4)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())

        self.assertAlmostEqual(0.6, self.evaluate(update_op))
        self.assertAlmostEqual(0.6, specificity.eval())

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weights_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    weights = constant_op.constant(weights_values)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, weights=weights, sensitivity=0.4)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      self.assertAlmostEqual(8.0 / 15.0, self.evaluate(update_op))
      self.assertAlmostEqual(8.0 / 15.0, specificity.eval())


class SensitivityAtSpecificityTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.sensitivity_at_specificity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        specificity=0.7)
    _assert_metric_variables(self,
                             ('sensitivity_at_specificity/true_positives:0',
                              'sensitivity_at_specificity/false_negatives:0',
                              'sensitivity_at_specificity/false_positives:0',
                              'sensitivity_at_specificity/true_negatives:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.sensitivity_at_specificity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        specificity=0.7,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.sensitivity_at_specificity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        specificity=0.7,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=2, dtype=dtypes_lib.int64, seed=1)
    sensitivity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_sensitivity = sensitivity.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_sensitivity, sensitivity.eval(), 5)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
    labels = constant_op.constant(inputs)
    specificity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op), 6)
      self.assertAlmostEqual(1.0, specificity.eval(), 6)

  @test_util.run_deprecated_v1
  def testSomeCorrectHighSpecificity(self):
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.8)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.8, self.evaluate(update_op))
      self.assertAlmostEqual(0.8, specificity.eval())

  @test_util.run_deprecated_v1
  def testSomeCorrectLowSpecificity(self):
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.4)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.6, self.evaluate(update_op))
      self.assertAlmostEqual(0.6, specificity.eval())

  @test_util.run_deprecated_v1
  def testWeighted_multipleLabelDtypes(self):
    for label_dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions_values = [
          0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
      labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
      weights_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

      predictions = constant_op.constant(
          predictions_values, dtype=dtypes_lib.float32)
      labels = math_ops.cast(labels_values, dtype=label_dtype)
      weights = constant_op.constant(weights_values)
      specificity, update_op = metrics.sensitivity_at_specificity(
          labels, predictions, weights=weights, specificity=0.4)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.675, self.evaluate(update_op))
        self.assertAlmostEqual(0.675, specificity.eval())


# TODO(nsilberman): Break this up into two sets of tests.
class PrecisionRecallThresholdsTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.precision_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0])
    _assert_metric_variables(self, (
        'precision_at_thresholds/true_positives:0',
        'precision_at_thresholds/false_positives:0',
    ))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    prec, _ = metrics.precision_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    rec, _ = metrics.recall_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [prec, rec])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, precision_op = metrics.precision_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    _, recall_op = metrics.recall_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    self.assertListEqual(
        ops.get_collection(my_collection_name), [precision_op, recall_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    thresholds = [0, 0.5, 1.0]
    prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                    thresholds)
    rec, rec_op = metrics.recall_at_thresholds(labels, predictions, thresholds)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates, then verify idempotency.
      self.evaluate([prec_op, rec_op])
      initial_prec = prec.eval()
      initial_rec = rec.eval()
      for _ in range(10):
        self.evaluate([prec_op, rec_op])
        self.assertAllClose(initial_prec, prec.eval())
        self.assertAllClose(initial_rec, rec.eval())

  # TODO(nsilberman): fix tests (passing but incorrect).
  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(inputs)
      thresholds = [0.5]
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertEqual(1, prec.eval())
      self.assertEqual(1, rec.eval())

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleLabelDtypes(self):
    with self.cached_session():
      for label_dtype in (
          dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
        predictions = constant_op.constant(
            [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
        labels = math_ops.cast(
            constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=label_dtype)
        thresholds = [0.5]
        prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                        thresholds)
        rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                   thresholds)

        self.evaluate(variables.local_variables_initializer())
        self.evaluate([prec_op, rec_op])

        self.assertAlmostEqual(0.5, prec.eval())
        self.assertAlmostEqual(0.5, rec.eval())

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(1 - inputs, dtype=dtypes_lib.float32)
      thresholds = [0.5]
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(0, prec.eval())
      self.assertAlmostEqual(0, rec.eval())

  @test_util.run_deprecated_v1
  def testWeights1d(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [[1, 0], [1, 0]], shape=(2, 2), dtype=dtypes_lib.float32)
      labels = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = constant_op.constant(
          [[0], [1]], shape=(2, 1), dtype=dtypes_lib.float32)
      thresholds = [0.5, 1.1]
      prec, prec_op = metrics.precision_at_thresholds(
          labels, predictions, thresholds, weights=weights)
      rec, rec_op = metrics.recall_at_thresholds(
          labels, predictions, thresholds, weights=weights)

      [prec_low, prec_high] = array_ops.split(
          value=prec, num_or_size_splits=2, axis=0)
      prec_low = array_ops.reshape(prec_low, shape=())
      prec_high = array_ops.reshape(prec_high, shape=())
      [rec_low, rec_high] = array_ops.split(
          value=rec, num_or_size_splits=2, axis=0)
      rec_low = array_ops.reshape(rec_low, shape=())
      rec_high = array_ops.reshape(rec_high, shape=())

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(1.0, prec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, prec_high.eval(), places=5)
      self.assertAlmostEqual(1.0, rec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, rec_high.eval(), places=5)

  @test_util.run_deprecated_v1
  def testWeights2d(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [[1, 0], [1, 0]], shape=(2, 2), dtype=dtypes_lib.float32)
      labels = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = constant_op.constant(
          [[0, 0], [1, 1]], shape=(2, 2), dtype=dtypes_lib.float32)
      thresholds = [0.5, 1.1]
      prec, prec_op = metrics.precision_at_thresholds(
          labels, predictions, thresholds, weights=weights)
      rec, rec_op = metrics.recall_at_thresholds(
          labels, predictions, thresholds, weights=weights)

      [prec_low, prec_high] = array_ops.split(
          value=prec, num_or_size_splits=2, axis=0)
      prec_low = array_ops.reshape(prec_low, shape=())
      prec_high = array_ops.reshape(prec_high, shape=())
      [rec_low, rec_high] = array_ops.split(
          value=rec, num_or_size_splits=2, axis=0)
      rec_low = array_ops.reshape(rec_low, shape=())
      rec_high = array_ops.reshape(rec_high, shape=())

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(1.0, prec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, prec_high.eval(), places=5)
      self.assertAlmostEqual(1.0, rec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, rec_high.eval(), places=5)

  @test_util.run_deprecated_v1
  def testExtremeThresholds(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 1, 1], shape=(1, 4))
      thresholds = [-1.0, 2.0]  # lower/higher than any values
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      [prec_low, prec_high] = array_ops.split(
          value=prec, num_or_size_splits=2, axis=0)
      [rec_low, rec_high] = array_ops.split(
          value=rec, num_or_size_splits=2, axis=0)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(0.75, prec_low.eval())
      self.assertAlmostEqual(0.0, prec_high.eval())
      self.assertAlmostEqual(1.0, rec_low.eval())
      self.assertAlmostEqual(0.0, rec_high.eval())

  @test_util.run_deprecated_v1
  def testZeroLabelsPredictions(self):
    with self.cached_session():
      predictions = array_ops.zeros([4], dtype=dtypes_lib.float32)
      labels = array_ops.zeros([4])
      thresholds = [0.5]
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(0, prec.eval(), 6)
      self.assertAlmostEqual(0, rec.eval(), 6)

  @test_util.run_deprecated_v1
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

    with self.cached_session() as sess:
      # Reshape the data so its easy to queue up:
      predictions_batches = predictions.reshape((batch_size, num_batches))
      labels_batches = labels.reshape((batch_size, num_batches))

      # Enqueue the data:
      predictions_queue = data_flow_ops.FIFOQueue(
          num_batches, dtypes=dtypes_lib.float32, shapes=(batch_size,))
      labels_queue = data_flow_ops.FIFOQueue(
          num_batches, dtypes=dtypes_lib.float32, shapes=(batch_size,))

      for i in range(int(num_batches)):
        tf_prediction = constant_op.constant(predictions_batches[:, i])
        tf_label = constant_op.constant(labels_batches[:, i])
        sess.run([
            predictions_queue.enqueue(tf_prediction),
            labels_queue.enqueue(tf_label)
        ])

      tf_predictions = predictions_queue.dequeue()
      tf_labels = labels_queue.dequeue()

      prec, prec_op = metrics.precision_at_thresholds(tf_labels, tf_predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(tf_labels, tf_predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(int(num_samples / batch_size)):
        self.evaluate([prec_op, rec_op])
      # Since this is only approximate, we can't expect a 6 digits match.
      # Although with higher number of samples/thresholds we should see the
      # accuracy improving
      self.assertAlmostEqual(expected_prec, prec.eval(), 2)
      self.assertAlmostEqual(expected_rec, rec.eval(), 2)


def _test_precision_at_k(predictions,
                         labels,
                         k,
                         expected,
                         class_id=None,
                         weights=None,
                         test_case=None):
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.precision_at_k(
        predictions=constant_op.constant(predictions, dtypes_lib.float32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


def _test_precision_at_top_k(
    predictions_idx,
    labels,
    expected,
    k=None,
    class_id=None,
    weights=None,
    test_case=None):
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.precision_at_top_k(
        predictions_idx=constant_op.constant(predictions_idx, dtypes_lib.int32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      test_case.assertTrue(math.isnan(update.eval()))
      test_case.assertTrue(math.isnan(metric.eval()))
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


def _test_average_precision_at_k(predictions,
                                 labels,
                                 k,
                                 expected,
                                 weights=None,
                                 test_case=None):
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    predictions = constant_op.constant(predictions, dtypes_lib.float32)
    metric, update = metrics.average_precision_at_k(
        labels, predictions, k, weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertAlmostEqual(expected, update.eval())
      test_case.assertAlmostEqual(expected, metric.eval())


class SingleLabelPrecisionAtKTest(test.TestCase):

  def setUp(self):
    self._predictions = ((0.1, 0.3, 0.2, 0.4), (0.1, 0.2, 0.3, 0.4))
    self._predictions_idx = [[3], [3]]
    indicator_labels = ((0, 0, 0, 1), (0, 0, 1, 0))
    class_labels = (3, 2)
    # Sparse vs dense, and 1d vs 2d labels should all be handled the same.
    self._labels = (
        _binary_2d_label_to_1d_sparse_value(indicator_labels),
        _binary_2d_label_to_2d_sparse_value(indicator_labels), np.array(
            class_labels, dtype=np.int64), np.array(
                [[class_id] for class_id in class_labels], dtype=np.int64))
    self._test_precision_at_k = functools.partial(
        _test_precision_at_k, test_case=self)
    self._test_precision_at_top_k = functools.partial(
        _test_precision_at_top_k, test_case=self)
    self._test_average_precision_at_k = functools.partial(
        _test_average_precision_at_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_at_k1_nan(self):
    for labels in self._labels:
      # Classes 0,1,2 have 0 predictions, classes -1 and 4 are out of range.
      for class_id in (-1, 0, 1, 2, 4):
        self._test_precision_at_k(
            self._predictions, labels, k=1, expected=NAN, class_id=class_id)
        self._test_precision_at_top_k(
            self._predictions_idx, labels, k=1, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_at_k1(self):
    for labels in self._labels:
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_precision_at_k(
          self._predictions, labels, k=1, expected=1.0 / 2, class_id=3)
      self._test_precision_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 2, class_id=3)

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_precision_at_k(
          self._predictions, labels, k=1, expected=1.0 / 2)
      self._test_precision_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 2)
      self._test_average_precision_at_k(
          self._predictions, labels, k=1, expected=1.0 / 2)


class MultiLabelPrecisionAtKTest(test.TestCase):

  def setUp(self):
    self._test_precision_at_k = functools.partial(
        _test_precision_at_k, test_case=self)
    self._test_precision_at_top_k = functools.partial(
        _test_precision_at_top_k, test_case=self)
    self._test_average_precision_at_k = functools.partial(
        _test_average_precision_at_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_average_precision(self):
    # Example 1.
    # Matches example here:
    # fastml.com/what-you-wanted-to-know-about-mean-average-precision
    labels_ex1 = (0, 1, 2, 3, 4)
    labels = np.array([labels_ex1], dtype=np.int64)
    predictions_ex1 = (0.2, 0.1, 0.0, 0.4, 0.0, 0.5, 0.3)
    predictions = (predictions_ex1,)
    predictions_idx_ex1 = (5, 3, 6, 0, 1)
    precision_ex1 = (0.0 / 1, 1.0 / 2, 1.0 / 3, 2.0 / 4)
    avg_precision_ex1 = (0.0 / 1, precision_ex1[1] / 2, precision_ex1[1] / 3,
                         (precision_ex1[1] + precision_ex1[3]) / 4)
    for i in xrange(4):
      k = i + 1
      self._test_precision_at_k(
          predictions, labels, k, expected=precision_ex1[i])
      self._test_precision_at_top_k(
          (predictions_idx_ex1[:k],), labels, k=k, expected=precision_ex1[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex1[i])

    # Example 2.
    labels_ex2 = (0, 2, 4, 5, 6)
    labels = np.array([labels_ex2], dtype=np.int64)
    predictions_ex2 = (0.3, 0.5, 0.0, 0.4, 0.0, 0.1, 0.2)
    predictions = (predictions_ex2,)
    predictions_idx_ex2 = (1, 3, 0, 6, 5)
    precision_ex2 = (0.0 / 1, 0.0 / 2, 1.0 / 3, 2.0 / 4)
    avg_precision_ex2 = (0.0 / 1, 0.0 / 2, precision_ex2[2] / 3,
                         (precision_ex2[2] + precision_ex2[3]) / 4)
    for i in xrange(4):
      k = i + 1
      self._test_precision_at_k(
          predictions, labels, k, expected=precision_ex2[i])
      self._test_precision_at_top_k(
          (predictions_idx_ex2[:k],), labels, k=k, expected=precision_ex2[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex2[i])

    # Both examples, we expect both precision and average precision to be the
    # average of the 2 examples.
    labels = np.array([labels_ex1, labels_ex2], dtype=np.int64)
    predictions = (predictions_ex1, predictions_ex2)
    streaming_precision = [(ex1 + ex2) / 2
                           for ex1, ex2 in zip(precision_ex1, precision_ex2)]
    streaming_average_precision = [
        (ex1 + ex2) / 2
        for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)
    ]
    for i in xrange(4):
      k = i + 1
      predictions_idx = (predictions_idx_ex1[:k], predictions_idx_ex2[:k])
      self._test_precision_at_k(
          predictions, labels, k, expected=streaming_precision[i])
      self._test_precision_at_top_k(
          predictions_idx, labels, k=k, expected=streaming_precision[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=streaming_average_precision[i])

    # Weighted examples, we expect streaming average precision to be the
    # weighted average of the 2 examples.
    weights = (0.3, 0.6)
    streaming_average_precision = [
        (weights[0] * ex1 + weights[1] * ex2) / (weights[0] + weights[1])
        for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)
    ]
    for i in xrange(4):
      k = i + 1
      self._test_average_precision_at_k(
          predictions,
          labels,
          k,
          expected=streaming_average_precision[i],
          weights=weights)

  @test_util.run_deprecated_v1
  def test_average_precision_some_labels_out_of_range(self):
    """Tests that labels outside the [0, n_classes) range are ignored."""
    labels_ex1 = (-1, 0, 1, 2, 3, 4, 7)
    labels = np.array([labels_ex1], dtype=np.int64)
    predictions_ex1 = (0.2, 0.1, 0.0, 0.4, 0.0, 0.5, 0.3)
    predictions = (predictions_ex1,)
    predictions_idx_ex1 = (5, 3, 6, 0, 1)
    precision_ex1 = (0.0 / 1, 1.0 / 2, 1.0 / 3, 2.0 / 4)
    avg_precision_ex1 = (0.0 / 1, precision_ex1[1] / 2, precision_ex1[1] / 3,
                         (precision_ex1[1] + precision_ex1[3]) / 4)
    for i in xrange(4):
      k = i + 1
      self._test_precision_at_k(
          predictions, labels, k, expected=precision_ex1[i])
      self._test_precision_at_top_k(
          (predictions_idx_ex1[:k],), labels, k=k, expected=precision_ex1[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex1[i])

  @test_util.run_deprecated_v1
  def test_average_precision_different_num_labels(self):
    """Tests the case where the numbers of labels differ across examples."""
    predictions = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 1], [0, 0, 0, 1]])
    dense_labels = np.array([[2, 3], [3, -1]], dtype=np.int64)
    predictions_idx_ex1 = np.array(((0, 1, 2, 3), (3, 2, 1, 0)))
    precision_ex1 = ((0.0 / 1, 0.0 / 2, 1.0 / 3, 2.0 / 4),
                     (1.0 / 1, 1.0 / 2, 1.0 / 3, 1.0 / 4))
    mean_precision_ex1 = np.mean(precision_ex1, axis=0)
    avg_precision_ex1 = (
        (0.0 / 1, 0.0 / 2, 1.0 / 3 / 2, (1.0 / 3 + 2.0 / 4) / 2),
        (1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1))
    mean_avg_precision_ex1 = np.mean(avg_precision_ex1, axis=0)
    for labels in (sparse_labels, dense_labels):
      for i in xrange(4):
        k = i + 1
        self._test_precision_at_k(
            predictions, labels, k, expected=mean_precision_ex1[i])
        self._test_precision_at_top_k(
            predictions_idx_ex1[:, :k], labels, k=k,
            expected=mean_precision_ex1[i])
        self._test_average_precision_at_k(
            predictions, labels, k, expected=mean_avg_precision_ex1[i])

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5_no_predictions(self):
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 1,3,8 have 0 predictions, classes -1 and 10 are out of range.
      for class_id in (-1, 1, 3, 8, 10):
        self._test_precision_at_k(
            predictions, labels, k=5, expected=NAN, class_id=class_id)
        self._test_precision_at_top_k(
            predictions_idx, labels, k=5, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5_no_labels(self):
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 0,4,6,9: 0 labels, >=1 prediction.
      for class_id in (0, 4, 6, 9):
        self._test_precision_at_k(
            predictions, labels, k=5, expected=0.0, class_id=class_id)
        self._test_precision_at_top_k(
            predictions_idx, labels, k=5, expected=0.0, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5(self):
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 2: 2 labels, 2 correct predictions.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=2.0 / 2, class_id=2)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=2.0 / 2, class_id=2)

      # Class 5: 1 label, 1 correct prediction.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=1.0 / 1, class_id=5)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=1.0 / 1, class_id=5)

      # Class 7: 1 label, 1 incorrect prediction.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=0.0 / 1, class_id=7)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=0.0 / 1, class_id=7)

      # All classes: 10 predictions, 3 correct.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=3.0 / 10)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=3.0 / 10)

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5_some_out_of_range(self):
    """Tests that labels outside the [0, n_classes) range are ignored."""
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sp_labels = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2],
                 [1, 3]],
        # values -1 and 10 are outside the [0, n_classes) range and are ignored.
        values=np.array([2, 7, -1, 8, 1, 2, 5, 10], np.int64),
        dense_shape=[2, 4])

    # Class 2: 2 labels, 2 correct predictions.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=2.0 / 2, class_id=2)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=2.0 / 2, class_id=2)

    # Class 5: 1 label, 1 correct prediction.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=1.0 / 1, class_id=5)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=1.0 / 1, class_id=5)

    # Class 7: 1 label, 1 incorrect prediction.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=0.0 / 1, class_id=7)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=0.0 / 1, class_id=7)

    # All classes: 10 predictions, 3 correct.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=3.0 / 10)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=3.0 / 10)

  @test_util.run_deprecated_v1
  def test_3d_nan(self):
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Classes 1,3,8 have 0 predictions, classes -1 and 10 are out of range.
    for class_id in (-1, 1, 3, 8, 10):
      self._test_precision_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d_no_labels(self):
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Classes 0,4,6,9: 0 labels, >=1 prediction.
    for class_id in (0, 4, 6, 9):
      self._test_precision_at_k(
          predictions, labels, k=5, expected=0.0, class_id=class_id)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=0.0, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d(self):
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Class 2: 4 predictions, all correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=4.0 / 4, class_id=2)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=4.0 / 4, class_id=2)

    # Class 5: 2 predictions, both correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2, class_id=5)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=2.0 / 2, class_id=5)

    # Class 7: 2 predictions, 1 correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 2, class_id=7)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=1.0 / 2, class_id=7)

    # All classes: 20 predictions, 7 correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=7.0 / 20)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=7.0 / 20)

  @test_util.run_deprecated_v1
  def test_3d_ignore_some(self):
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Class 2: 2 predictions, both correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])

    # Class 2: 2 predictions, both correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])

    # Class 7: 1 incorrect prediction.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])

    # Class 7: 1 correct prediction.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])

    # Class 7: no predictions.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=NAN, class_id=7,
        weights=[[1, 0], [0, 1]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=NAN, class_id=7,
        weights=[[1, 0], [0, 1]])

    # Class 7: 2 predictions, 1 correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[0, 1], [1, 0]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[0, 1], [1, 0]])


def _test_recall_at_k(predictions,
                      labels,
                      k,
                      expected,
                      class_id=None,
                      weights=None,
                      test_case=None):
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.recall_at_k(
        predictions=constant_op.constant(predictions, dtypes_lib.float32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


def _test_recall_at_top_k(
    predictions_idx,
    labels,
    expected,
    k=None,
    class_id=None,
    weights=None,
    test_case=None):
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.recall_at_top_k(
        predictions_idx=constant_op.constant(predictions_idx, dtypes_lib.int32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


class SingleLabelRecallAtKTest(test.TestCase):

  def setUp(self):
    self._predictions = ((0.1, 0.3, 0.2, 0.4), (0.1, 0.2, 0.3, 0.4))
    self._predictions_idx = [[3], [3]]
    indicator_labels = ((0, 0, 0, 1), (0, 0, 1, 0))
    class_labels = (3, 2)
    # Sparse vs dense, and 1d vs 2d labels should all be handled the same.
    self._labels = (
        _binary_2d_label_to_1d_sparse_value(indicator_labels),
        _binary_2d_label_to_2d_sparse_value(indicator_labels), np.array(
            class_labels, dtype=np.int64), np.array(
                [[class_id] for class_id in class_labels], dtype=np.int64))
    self._test_recall_at_k = functools.partial(
        _test_recall_at_k, test_case=self)
    self._test_recall_at_top_k = functools.partial(
        _test_recall_at_top_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_at_k1_nan(self):
    # Classes 0,1 have 0 labels, 0 predictions, classes -1 and 4 are out of
    # range.
    for labels in self._labels:
      for class_id in (-1, 0, 1, 4):
        self._test_recall_at_k(
            self._predictions, labels, k=1, expected=NAN, class_id=class_id)
        self._test_recall_at_top_k(
            self._predictions_idx, labels, k=1, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_at_k1_no_predictions(self):
    for labels in self._labels:
      # Class 2: 0 predictions.
      self._test_recall_at_k(
          self._predictions, labels, k=1, expected=0.0, class_id=2)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=1, expected=0.0, class_id=2)

  @test_util.run_deprecated_v1
  def test_one_label_at_k1(self):
    for labels in self._labels:
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_recall_at_k(
          self._predictions, labels, k=1, expected=1.0 / 1, class_id=3)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3)

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_recall_at_k(self._predictions, labels, k=1, expected=1.0 / 2)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 2)

  @test_util.run_deprecated_v1
  def test_one_label_at_k1_weighted_class_id3(self):
    predictions = self._predictions
    predictions_idx = self._predictions_idx
    for labels in self._labels:
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3, weights=(0.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(2.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(2.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0, 1.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0, 1.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0, 0.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0, 0.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=2.0 / 2, class_id=3,
          weights=(2.0, 3.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=2.0 / 2, class_id=3,
          weights=(2.0, 3.0))

  @test_util.run_deprecated_v1
  def test_one_label_at_k1_weighted(self):
    predictions = self._predictions
    predictions_idx = self._predictions_idx
    for labels in self._labels:
      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_recall_at_k(
          predictions, labels, k=1, expected=NAN, weights=(0.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=NAN, weights=(0.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(1.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 2, weights=(1.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(2.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 2, weights=(2.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, weights=(1.0, 0.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, weights=(1.0, 0.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=0.0 / 1, weights=(0.0, 1.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=0.0 / 1, weights=(0.0, 1.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=2.0 / 5, weights=(2.0, 3.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=2.0 / 5, weights=(2.0, 3.0))


class MultiLabel2dRecallAtKTest(test.TestCase):

  def setUp(self):
    self._predictions = ((0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9),
                         (0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6))
    self._predictions_idx = ((9, 4, 6, 2, 0), (5, 7, 2, 9, 6))
    indicator_labels = ((0, 0, 1, 0, 0, 0, 0, 1, 1, 0),
                        (0, 1, 1, 0, 0, 1, 0, 0, 0, 0))
    class_labels = ((2, 7, 8), (1, 2, 5))
    # Sparse vs dense labels should be handled the same.
    self._labels = (_binary_2d_label_to_2d_sparse_value(indicator_labels),
                    np.array(
                        class_labels, dtype=np.int64))
    self._test_recall_at_k = functools.partial(
        _test_recall_at_k, test_case=self)
    self._test_recall_at_top_k = functools.partial(
        _test_recall_at_top_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_at_k5_nan(self):
    for labels in self._labels:
      # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
      for class_id in (0, 3, 4, 6, 9, 10):
        self._test_recall_at_k(
            self._predictions, labels, k=5, expected=NAN, class_id=class_id)
        self._test_recall_at_top_k(
            self._predictions_idx, labels, k=5, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_at_k5_no_predictions(self):
    for labels in self._labels:
      # Class 8: 1 label, no predictions.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=0.0 / 1, class_id=8)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=0.0 / 1, class_id=8)

  @test_util.run_deprecated_v1
  def test_at_k5(self):
    for labels in self._labels:
      # Class 2: 2 labels, both correct.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=2.0 / 2, class_id=2)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=2.0 / 2, class_id=2)

      # Class 5: 1 label, incorrect.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=1.0 / 1, class_id=5)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=1.0 / 1, class_id=5)

      # Class 7: 1 label, incorrect.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=0.0 / 1, class_id=7)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=0.0 / 1, class_id=7)

      # All classes: 6 labels, 3 correct.
      self._test_recall_at_k(self._predictions, labels, k=5, expected=3.0 / 6)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=3.0 / 6)

  @test_util.run_deprecated_v1
  def test_at_k5_some_out_of_range(self):
    """Tests that labels outside the [0, n_classes) count in denominator."""
    labels = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2],
                 [1, 3]],
        # values -1 and 10 are outside the [0, n_classes) range.
        values=np.array([2, 7, -1, 8, 1, 2, 5, 10], np.int64),
        dense_shape=[2, 4])

    # Class 2: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, labels, k=5, expected=2.0 / 2, class_id=2)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=2.0 / 2, class_id=2)

    # Class 5: 1 label, incorrect.
    self._test_recall_at_k(
        self._predictions, labels, k=5, expected=1.0 / 1, class_id=5)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=1.0 / 1, class_id=5)

    # Class 7: 1 label, incorrect.
    self._test_recall_at_k(
        self._predictions, labels, k=5, expected=0.0 / 1, class_id=7)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=0.0 / 1, class_id=7)

    # All classes: 8 labels, 3 correct.
    self._test_recall_at_k(self._predictions, labels, k=5, expected=3.0 / 8)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=3.0 / 8)


class MultiLabel3dRecallAtKTest(test.TestCase):

  def setUp(self):
    self._predictions = (((0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9),
                          (0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6)),
                         ((0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6),
                          (0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9)))
    self._predictions_idx = (((9, 4, 6, 2, 0), (5, 7, 2, 9, 6)),
                             ((5, 7, 2, 9, 6), (9, 4, 6, 2, 0)))
    # Note: We don't test dense labels here, since examples have different
    # numbers of labels.
    self._labels = _binary_3d_label_to_sparse_value(((
        (0, 0, 1, 0, 0, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 1, 0, 0, 0, 0)), (
            (0, 1, 1, 0, 0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 1, 0))))
    self._test_recall_at_k = functools.partial(
        _test_recall_at_k, test_case=self)
    self._test_recall_at_top_k = functools.partial(
        _test_recall_at_top_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_3d_nan(self):
    # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
    for class_id in (0, 3, 4, 6, 9, 10):
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=NAN, class_id=class_id)
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=NAN,
          class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d_no_predictions(self):
    # Classes 1,8 have 0 predictions, >=1 label.
    for class_id in (1, 8):
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=0.0, class_id=class_id)
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=0.0,
          class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d(self):
    # Class 2: 4 labels, all correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=4.0 / 4, class_id=2)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=4.0 / 4,
        class_id=2)

    # Class 5: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=2.0 / 2, class_id=5)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=2.0 / 2,
        class_id=5)

    # Class 7: 2 labels, 1 incorrect.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=1.0 / 2, class_id=7)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=1.0 / 2,
        class_id=7)

    # All classes: 12 labels, 7 correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=7.0 / 12)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=7.0 / 12)

  @test_util.run_deprecated_v1
  def test_3d_ignore_all(self):
    for class_id in xrange(10):
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0], [0]])
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=NAN,
          class_id=class_id, weights=[[0], [0]])
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0, 0], [0, 0]])
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=NAN,
          class_id=class_id, weights=[[0, 0], [0, 0]])
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=NAN, weights=[[0], [0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=NAN,
        weights=[[0], [0]])
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=NAN,
        weights=[[0, 0], [0, 0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=NAN,
        weights=[[0, 0], [0, 0]])

  @test_util.run_deprecated_v1
  def test_3d_ignore_some(self):
    # Class 2: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=2.0 / 2.0,
        class_id=2, weights=[[1], [0]])

    # Class 2: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=2.0 / 2.0,
        class_id=2, weights=[[0], [1]])

    # Class 7: 1 label, correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=1.0 / 1.0,
        class_id=7, weights=[[0], [1]])

    # Class 7: 1 label, incorrect.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=0.0 / 1.0,
        class_id=7, weights=[[1], [0]])

    # Class 7: 2 labels, 1 correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[1, 0], [1, 0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=1.0 / 2.0,
        class_id=7, weights=[[1, 0], [1, 0]])

    # Class 7: No labels.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=NAN, class_id=7,
        weights=[[0, 1], [0, 1]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=NAN, class_id=7,
        weights=[[0, 1], [0, 1]])


class MeanAbsoluteErrorTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_absolute_error(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self, ('mean_absolute_error/count:0', 'mean_absolute_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_absolute_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_absolute_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.mean_absolute_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights(self):
    predictions = constant_op.constant(
        [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)
    weights = constant_op.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = metrics.mean_absolute_error(labels, predictions, weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(3, self.evaluate(update_op))
      self.assertEqual(3, error.eval())


class MeanRelativeErrorTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_relative_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        normalizer=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self, ('mean_relative_error/count:0', 'mean_relative_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_relative_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        normalizer=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_relative_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        normalizer=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    normalizer = random_ops.random_normal((10, 3), seed=3)
    error, update_op = metrics.mean_relative_error(labels, predictions,
                                                   normalizer)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateNormalizedByLabels(self):
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_labels = np.asarray([1, 3, 2, 3], dtype=np.float32)
    expected_error = np.mean(
        np.divide(np.absolute(np_predictions - np_labels), np_labels))

    predictions = constant_op.constant(
        np_predictions, shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(np_labels, shape=(1, 4))

    error, update_op = metrics.mean_relative_error(
        labels, predictions, normalizer=labels)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(expected_error, self.evaluate(update_op))
      self.assertEqual(expected_error, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateNormalizedByZeros(self):
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)

    predictions = constant_op.constant(
        np_predictions, shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_relative_error(
        labels, predictions, normalizer=array_ops.zeros_like(labels))

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0.0, self.evaluate(update_op))
      self.assertEqual(0.0, error.eval())


class MeanSquaredErrorTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_squared_error(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self, ('mean_squared_error/count:0', 'mean_squared_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateZeroError(self):
    predictions = array_ops.zeros((1, 3), dtype=dtypes_lib.float32)
    labels = array_ops.zeros((1, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))
      self.assertEqual(0, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError(self):
    predictions = constant_op.constant(
        [2, 4, 6], shape=(1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2], shape=(1, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(6, self.evaluate(update_op))
      self.assertEqual(6, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights(self):
    predictions = constant_op.constant(
        [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)
    weights = constant_op.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = metrics.mean_squared_error(labels, predictions, weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(13, self.evaluate(update_op))
      self.assertEqual(13, error.eval())

  @test_util.run_deprecated_v1
  def testMultipleBatchesOfSizeOne(self):
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      error, update_op = metrics.mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertAlmostEqual(208.0 / 6, self.evaluate(update_op), 5)

      self.assertAlmostEqual(208.0 / 6, error.eval(), 5)

  @test_util.run_deprecated_v1
  def testMetricsComputedConcurrently(self):
    with self.cached_session() as sess:
      # Create the queue that populates one set of predictions.
      preds_queue0 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue0, [10, 8, 6])
      _enqueue_vector(sess, preds_queue0, [-4, 3, -1])
      predictions0 = preds_queue0.dequeue()

      # Create the queue that populates one set of predictions.
      preds_queue1 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue1, [0, 1, 1])
      _enqueue_vector(sess, preds_queue1, [1, 1, 0])
      predictions1 = preds_queue1.dequeue()

      # Create the queue that populates one set of labels.
      labels_queue0 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue0, [1, 3, 2])
      _enqueue_vector(sess, labels_queue0, [2, 4, 6])
      labels0 = labels_queue0.dequeue()

      # Create the queue that populates another set of labels.
      labels_queue1 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue1, [-5, -3, -1])
      _enqueue_vector(sess, labels_queue1, [5, 4, 3])
      labels1 = labels_queue1.dequeue()

      mse0, update_op0 = metrics.mean_squared_error(
          labels0, predictions0, name='msd0')
      mse1, update_op1 = metrics.mean_squared_error(
          labels1, predictions1, name='msd1')

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([update_op0, update_op1])
      self.evaluate([update_op0, update_op1])

      mse0, mse1 = self.evaluate([mse0, mse1])
      self.assertAlmostEqual(208.0 / 6, mse0, 5)
      self.assertAlmostEqual(79.0 / 6, mse1, 5)

  @test_util.run_deprecated_v1
  def testMultipleMetricsOnMultipleBatchesOfSizeOne(self):
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      mae, ma_update_op = metrics.mean_absolute_error(labels, predictions)
      mse, ms_update_op = metrics.mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([ma_update_op, ms_update_op])
      self.evaluate([ma_update_op, ms_update_op])

      self.assertAlmostEqual(32.0 / 6, mae.eval(), 5)
      self.assertAlmostEqual(208.0 / 6, mse.eval(), 5)


class RootMeanSquaredErrorTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.root_mean_squared_error(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self,
        ('root_mean_squared_error/count:0', 'root_mean_squared_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.root_mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.root_mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.root_mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateZeroError(self):
    with self.cached_session():
      predictions = constant_op.constant(
          0.0, shape=(1, 3), dtype=dtypes_lib.float32)
      labels = constant_op.constant(0.0, shape=(1, 3), dtype=dtypes_lib.float32)

      rmse, update_op = metrics.root_mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))

      self.assertEqual(0, rmse.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [2, 4, 6], shape=(1, 3), dtype=dtypes_lib.float32)
      labels = constant_op.constant(
          [1, 3, 2], shape=(1, 3), dtype=dtypes_lib.float32)

      rmse, update_op = metrics.root_mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(math.sqrt(6), update_op.eval(), 5)
      self.assertAlmostEqual(math.sqrt(6), rmse.eval(), 5)

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights(self):
    with self.cached_session():
      predictions = constant_op.constant(
          [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant(
          [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)
      weights = constant_op.constant([0, 1, 0, 1], shape=(1, 4))

      rmse, update_op = metrics.root_mean_squared_error(labels, predictions,
                                                        weights)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(math.sqrt(13), self.evaluate(update_op))

      self.assertAlmostEqual(math.sqrt(13), rmse.eval(), 5)


def _reweight(predictions, labels, weights):
  return (np.concatenate([[p] * int(w) for p, w in zip(predictions, weights)]),
          np.concatenate([[l] * int(w) for l, w in zip(labels, weights)]))


class MeanCosineDistanceTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_cosine_distance(
        predictions=array_ops.ones((10, 3)),
        labels=array_ops.ones((10, 3)),
        dim=1)
    _assert_metric_variables(self, (
        'mean_cosine_distance/count:0',
        'mean_cosine_distance/total:0',
    ))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_cosine_distance(
        predictions=array_ops.ones((10, 3)),
        labels=array_ops.ones((10, 3)),
        dim=1,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_cosine_distance(
        predictions=array_ops.ones((10, 3)),
        labels=array_ops.ones((10, 3)),
        dim=1,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=1)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateZeroError(self):
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))

    predictions = constant_op.constant(
        np_labels, shape=(1, 3, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(1, 3, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=2)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))
      self.assertEqual(0, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError1(self):
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))
    np_predictions = np.matrix(('1 0 0;' '0 0 -1;' '1 0 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=2)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1, error.eval(), 5)

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError2(self):
    np_predictions = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
         '0.707106781186548 -0.707106781186548 0'))
    np_labels = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '0.665139432070255 0.739487441769973 0.103671883216994;'
         '0.707106781186548 0.707106781186548 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=2)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1.0, error.eval(), 5)

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights1(self):
    np_predictions = np.matrix(('1 0 0;' '0 0 -1;' '1 0 0'))
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    weights = constant_op.constant(
        [1, 0, 0], shape=(3, 1, 1), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(
        labels, predictions, dim=2, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))
      self.assertEqual(0, error.eval())

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights2(self):
    np_predictions = np.matrix(('1 0 0;' '0 0 -1;' '1 0 0'))
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    weights = constant_op.constant(
        [0, 1, 1], shape=(3, 1, 1), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(
        labels, predictions, dim=2, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.5, update_op.eval())
      self.assertEqual(1.5, error.eval())


class PcntBelowThreshTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.percentage_below(values=array_ops.ones((10,)), threshold=2)
    _assert_metric_variables(self, (
        'percentage_below_threshold/count:0',
        'percentage_below_threshold/total:0',
    ))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.percentage_below(
        values=array_ops.ones((10,)),
        threshold=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.percentage_below(
        values=array_ops.ones((10,)),
        threshold=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testOneUpdate(self):
    with self.cached_session():
      values = constant_op.constant(
          [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)

      pcnt0, update_op0 = metrics.percentage_below(values, 100, name='high')
      pcnt1, update_op1 = metrics.percentage_below(values, 7, name='medium')
      pcnt2, update_op2 = metrics.percentage_below(values, 1, name='low')

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([update_op0, update_op1, update_op2])

      pcnt0, pcnt1, pcnt2 = self.evaluate([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.75, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)

  @test_util.run_deprecated_v1
  def testSomePresentOneUpdate(self):
    with self.cached_session():
      values = constant_op.constant(
          [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
      weights = constant_op.constant(
          [1, 0, 0, 1], shape=(1, 4), dtype=dtypes_lib.float32)

      pcnt0, update_op0 = metrics.percentage_below(
          values, 100, weights=weights, name='high')
      pcnt1, update_op1 = metrics.percentage_below(
          values, 7, weights=weights, name='medium')
      pcnt2, update_op2 = metrics.percentage_below(
          values, 1, weights=weights, name='low')

      self.evaluate(variables.local_variables_initializer())
      self.assertListEqual([1.0, 0.5, 0.0],
                           self.evaluate([update_op0, update_op1, update_op2]))

      pcnt0, pcnt1, pcnt2 = self.evaluate([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.5, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)


class MeanIOUTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_iou(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2)
    _assert_metric_variables(self, ('mean_iou/total_confusion_matrix:0',))

  @test_util.run_deprecated_v1
  def testMetricsCollections(self):
    my_collection_name = '__metrics__'
    mean_iou, _ = metrics.mean_iou(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean_iou])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_iou(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones([10, 3])
    labels = array_ops.ones([10, 4])
    with self.assertRaises(ValueError):
      metrics.mean_iou(labels, predictions, num_classes=2)

  @test_util.run_deprecated_v1
  def testLabelsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones([10])
    labels = array_ops.ones([10])
    weights = array_ops.zeros([9])
    with self.assertRaises(ValueError):
      metrics.mean_iou(labels, predictions, num_classes=2, weights=weights)

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    num_classes = 3
    predictions = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    mean_iou, update_op = metrics.mean_iou(
        labels, predictions, num_classes=num_classes)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_mean_iou = mean_iou.eval()
      for _ in range(10):
        self.assertEqual(initial_mean_iou, mean_iou.eval())

  @test_util.run_deprecated_v1
  def testMultipleUpdates(self):
    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0 / 2.0, 1.0 / 4.0, 0.])
      self.assertEqual(desired_output, miou.eval())

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeights(self):
    num_classes = 2
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      weights = weights_queue.dequeue()

      mean_iou, update_op = metrics.mean_iou(
          labels, predictions, num_classes, weights=weights)

      variables.local_variables_initializer().run()
      for _ in range(6):
        self.evaluate(update_op)
      desired_output = np.mean([2.0 / 3.0, 1.0 / 2.0])
      self.assertAlmostEqual(desired_output, mean_iou.eval())

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithMissingClass(self):
    # Test the case where there are no predictions and labels for
    # one class, and thus there is one row and one column with
    # zero entries in the confusion matrix.
    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      # There is no prediction for class 2.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      # There is label for class 2.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0 / 3.0, 2.0 / 4.0])
      self.assertAlmostEqual(desired_output, miou.eval())

  @test_util.run_deprecated_v1
  def testUpdateOpEvalIsAccumulatedConfusionMatrix(self):
    predictions = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[5]), constant_op.constant(
                    1, shape=[5])
        ],
        0)
    labels = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[3]), constant_op.constant(
                    1, shape=[7])
        ],
        0)
    num_classes = 2
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      confusion_matrix = update_op.eval()
      self.assertAllEqual([[3, 0], [2, 5]], confusion_matrix)
      desired_miou = np.mean([3. / 5., 5. / 7.])
      self.assertAlmostEqual(desired_miou, miou.eval())

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    predictions = array_ops.zeros([40])
    labels = array_ops.zeros([40])
    num_classes = 1
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(40, update_op.eval()[0])
      self.assertEqual(1.0, miou.eval())

  @test_util.run_deprecated_v1
  def testAllWrong(self):
    predictions = array_ops.zeros([40])
    labels = array_ops.ones([40])
    num_classes = 2
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[0, 0], [40, 0]], update_op.eval())
      self.assertEqual(0., miou.eval())

  @test_util.run_deprecated_v1
  def testResultsWithSomeMissing(self):
    predictions = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[5]), constant_op.constant(
                    1, shape=[5])
        ],
        0)
    labels = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[3]), constant_op.constant(
                    1, shape=[7])
        ],
        0)
    num_classes = 2
    weights = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[1]), constant_op.constant(
                    1, shape=[8]), constant_op.constant(
                        0, shape=[1])
        ],
        0)
    with self.cached_session():
      miou, update_op = metrics.mean_iou(
          labels, predictions, num_classes, weights=weights)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[2, 0], [2, 4]], update_op.eval())
      desired_miou = np.mean([2. / 4., 4. / 6.])
      self.assertAlmostEqual(desired_miou, miou.eval())

  @test_util.run_deprecated_v1
  def testMissingClassInLabels(self):
    labels = constant_op.constant([
        [[0, 0, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 1]],
        [[1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0]]])
    predictions = constant_op.constant([
        [[0, 0, 2, 1, 1, 0],
         [0, 1, 2, 2, 0, 1]],
        [[0, 0, 2, 1, 1, 1],
         [1, 1, 2, 0, 0, 0]]])
    num_classes = 3
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[7, 4, 3], [3, 5, 2], [0, 0, 0]], update_op.eval())
      self.assertAlmostEqual(
          1 / 3 * (7 / (7 + 3 + 7) + 5 / (5 + 4 + 5) + 0 / (0 + 5 + 0)),
          miou.eval())

  @test_util.run_deprecated_v1
  def testMissingClassOverallSmall(self):
    labels = constant_op.constant([0])
    predictions = constant_op.constant([0])
    num_classes = 2
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[1, 0], [0, 0]], update_op.eval())
      self.assertAlmostEqual(1, miou.eval())

  @test_util.run_deprecated_v1
  def testMissingClassOverallLarge(self):
    labels = constant_op.constant([
        [[0, 0, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 1]],
        [[1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0]]])
    predictions = constant_op.constant([
        [[0, 0, 1, 1, 0, 0],
         [1, 1, 0, 0, 1, 1]],
        [[0, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 0]]])
    num_classes = 3
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[9, 5, 0], [3, 7, 0], [0, 0, 0]], update_op.eval())
      self.assertAlmostEqual(
          1 / 2 * (9 / (9 + 3 + 5) + 7 / (7 + 5 + 3)), miou.eval())


class MeanPerClassAccuracyTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.mean_per_class_accuracy(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2)
    _assert_metric_variables(self, ('mean_accuracy/count:0',
                                    'mean_accuracy/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollections(self):
    my_collection_name = '__metrics__'
    mean_accuracy, _ = metrics.mean_per_class_accuracy(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(
        ops.get_collection(my_collection_name), [mean_accuracy])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_per_class_accuracy(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones([10, 3])
    labels = array_ops.ones([10, 4])
    with self.assertRaises(ValueError):
      metrics.mean_per_class_accuracy(labels, predictions, num_classes=2)

  @test_util.run_deprecated_v1
  def testLabelsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones([10])
    labels = array_ops.ones([10])
    weights = array_ops.zeros([9])
    with self.assertRaises(ValueError):
      metrics.mean_per_class_accuracy(
          labels, predictions, num_classes=2, weights=weights)

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    num_classes = 3
    predictions = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    mean_accuracy, update_op = metrics.mean_per_class_accuracy(
        labels, predictions, num_classes=num_classes)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_mean_accuracy = mean_accuracy.eval()
      for _ in range(10):
        self.assertEqual(initial_mean_accuracy, mean_accuracy.eval())

    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0, 1.0 / 3.0, 0.0])
      self.assertAlmostEqual(desired_output, mean_accuracy.eval())

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeights(self):
    num_classes = 2
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.5])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      weights = weights_queue.dequeue()

      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes, weights=weights)

      variables.local_variables_initializer().run()
      for _ in range(6):
        self.evaluate(update_op)
      desired_output = np.mean([2.0 / 2.0, 0.5 / 1.5])
      self.assertAlmostEqual(desired_output, mean_accuracy.eval())

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithMissingClass(self):
    # Test the case where there are no predictions and labels for
    # one class, and thus there is one row and one column with
    # zero entries in the confusion matrix.
    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      # There is no prediction for class 2.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      # There is label for class 2.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0 / 2.0, 2.0 / 3.0, 0.])
      self.assertAlmostEqual(desired_output, mean_accuracy.eval())

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    predictions = array_ops.zeros([40])
    labels = array_ops.zeros([40])
    num_classes = 1
    with self.cached_session():
      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, update_op.eval()[0])
      self.assertEqual(1.0, mean_accuracy.eval())

  @test_util.run_deprecated_v1
  def testAllWrong(self):
    predictions = array_ops.zeros([40])
    labels = array_ops.ones([40])
    num_classes = 2
    with self.cached_session():
      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([0.0, 0.0], update_op.eval())
      self.assertEqual(0., mean_accuracy.eval())

  @test_util.run_deprecated_v1
  def testResultsWithSomeMissing(self):
    predictions = array_ops.concat([
        constant_op.constant(0, shape=[5]), constant_op.constant(1, shape=[5])
    ], 0)
    labels = array_ops.concat([
        constant_op.constant(0, shape=[3]), constant_op.constant(1, shape=[7])
    ], 0)
    num_classes = 2
    weights = array_ops.concat([
        constant_op.constant(0, shape=[1]), constant_op.constant(1, shape=[8]),
        constant_op.constant(0, shape=[1])
    ], 0)
    with self.cached_session():
      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes, weights=weights)
      self.evaluate(variables.local_variables_initializer())
      desired_accuracy = np.array([2. / 2., 4. / 6.], dtype=np.float32)
      self.assertAllEqual(desired_accuracy, update_op.eval())
      desired_mean_accuracy = np.mean(desired_accuracy)
      self.assertAlmostEqual(desired_mean_accuracy, mean_accuracy.eval())


class FalseNegativesTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.false_negatives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('false_negatives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.false_negatives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(3., tn_update_op.eval())
      self.assertAllClose(3., tn.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.false_negatives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(5., tn_update_op.eval())
      self.assertAllClose(5., tn.eval())


class FalseNegativesAtThresholdsTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.false_negatives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('false_negatives/false_negatives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fn, fn_update_op = metrics.false_negatives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), fn.eval())
      self.assertAllEqual((0, 2, 3), fn_update_op.eval())
      self.assertAllEqual((0, 2, 3), fn.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fn, fn_update_op = metrics.false_negatives_at_thresholds(
        predictions=predictions,
        labels=labels,
        weights=((3.0,), (5.0,), (7.0,)),
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), fn.eval())
      self.assertAllEqual((0.0, 8.0, 11.0), fn_update_op.eval())
      self.assertAllEqual((0.0, 8.0, 11.0), fn.eval())


class FalsePositivesTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.false_positives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('false_positives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.false_positives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(7., tn_update_op.eval())
      self.assertAllClose(7., tn.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.false_positives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(14., tn_update_op.eval())
      self.assertAllClose(14., tn.eval())


class FalsePositivesAtThresholdsTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.false_positives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('false_positives/false_positives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fp, fp_update_op = metrics.false_positives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), fp.eval())
      self.assertAllEqual((7, 4, 2), fp_update_op.eval())
      self.assertAllEqual((7, 4, 2), fp.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fp, fp_update_op = metrics.false_positives_at_thresholds(
        predictions=predictions,
        labels=labels,
        weights=((1.0, 2.0, 3.0, 5.0),
                 (7.0, 11.0, 13.0, 17.0),
                 (19.0, 23.0, 29.0, 31.0)),
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), fp.eval())
      self.assertAllEqual((125.0, 42.0, 12.0), fp_update_op.eval())
      self.assertAllEqual((125.0, 42.0, 12.0), fp.eval())


class TrueNegativesTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.true_negatives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('true_negatives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.true_negatives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(3., tn_update_op.eval())
      self.assertAllClose(3., tn.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.true_negatives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(4., tn_update_op.eval())
      self.assertAllClose(4., tn.eval())


class TrueNegativesAtThresholdsTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.true_negatives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('true_negatives/true_negatives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tn, tn_update_op = metrics.true_negatives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), tn.eval())
      self.assertAllEqual((2, 5, 7), tn_update_op.eval())
      self.assertAllEqual((2, 5, 7), tn.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tn, tn_update_op = metrics.true_negatives_at_thresholds(
        predictions=predictions,
        labels=labels,
        weights=((0.0, 2.0, 3.0, 5.0),),
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), tn.eval())
      self.assertAllEqual((5.0, 15.0, 23.0), tn_update_op.eval())
      self.assertAllEqual((5.0, 15.0, 23.0), tn.eval())


class TruePositivesTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.true_positives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('true_positives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.true_positives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(7., tn_update_op.eval())
      self.assertAllClose(7., tn.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.true_positives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn.eval())
      self.assertAllClose(12., tn_update_op.eval())
      self.assertAllClose(12., tn.eval())


class TruePositivesAtThresholdsTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    metrics.true_positives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('true_positives/true_positives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tp, tp_update_op = metrics.true_positives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), tp.eval())
      self.assertAllEqual((3, 1, 0), tp_update_op.eval())
      self.assertAllEqual((3, 1, 0), tp.eval())

  @test_util.run_deprecated_v1
  def testWeighted(self):
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tp, tp_update_op = metrics.true_positives_at_thresholds(
        predictions=predictions, labels=labels, weights=37.0,
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), tp.eval())
      self.assertAllEqual((111.0, 37.0, 0.0), tp_update_op.eval())
      self.assertAllEqual((111.0, 37.0, 0.0), tp.eval())


if __name__ == '__main__':
  test.main()
