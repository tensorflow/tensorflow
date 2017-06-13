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
"""Tests for head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.core.framework import summary_pb2
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import queue_runner_impl


_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def _initialize_variables(test_case, scaffold):
  scaffold.finalize()
  test_case.assertIsNone(scaffold.init_feed_dict)
  test_case.assertIsNone(scaffold.init_fn)
  scaffold.init_op.run()
  scaffold.ready_for_local_init_op.eval()
  scaffold.local_init_op.run()
  scaffold.ready_op.eval()
  test_case.assertIsNotNone(scaffold.saver)


def _assert_simple_summaries(test_case, expected_summaries, summary_str,
                             tol=1e-6):
  """Assert summary the specified simple values.

  Args:
    test_case: test case.
    expected_summaries: Dict of expected tags and simple values.
    summary_str: Serialized `summary_pb2.Summary`.
    tol: Tolerance for relative and absolute.
  """
  summary = summary_pb2.Summary()
  summary.ParseFromString(summary_str)
  test_case.assertAllClose(expected_summaries, {
      v.tag: v.simple_value for v in summary.value
  }, rtol=tol, atol=tol)


def _assert_no_hooks(test_case, spec):
  test_case.assertAllEqual([], spec.training_chief_hooks)
  test_case.assertAllEqual([], spec.training_hooks)


def _sigmoid(logits):
  return 1 / (1 + np.exp(-logits))


# TODO(roumposg): Reuse the code from dnn_testing_utils.
def _assert_close(expected, actual, rtol=1e-04, message='',
                  name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs((expected - actual) / expected, 'diff')
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=(message, 'Condition expected =~ actual did not hold element-wise:'
              'expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff,
              'rtol = ', rtol,),
        summarize=expected.get_shape().num_elements(),
        name=scope)


class MultiClassHeadWithSoftmaxCrossEntropyLoss(test.TestCase):

  def test_n_classes_is_none(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=None)

  def test_n_classes_is_2(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=2)

  def test_invalid_logits_shape(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    logits_2x2 = np.array(((45., 44.), (41., 42.),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': np.array(((30.,), (42.,),))},
          mode=model_fn.ModeKeys.PREDICT,
          logits=logits_2x2)

    # Dynamic shape.
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((30.,), (42.,),))},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.predictions[prediction_keys.PredictionKeys.PROBABILITIES].eval({
            logits_placeholder: logits_2x2
        })

  def test_invalid_labels_shape(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    # Labels should be shape (batch_size, 1).
    labels_2x2 = np.array(((45, 44), (41, 42),), dtype=np.int)
    logits_2x3 = np.array(((1., 2., 3.), (1., 2., 3.),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'labels shape'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x3,
          labels=labels_2x2)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'labels shape'):
        spec.loss.eval({
            logits_placeholder: logits_2x3,
            labels_placeholder: labels_2x2
        })

  def test_invalid_labels_type(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    # Labels should be shape (batch_size, 1).
    labels_2x1 = np.array(((1.,), (1.,),))
    logits_2x3 = np.array(((1., 2., 3.), (1., 2., 3.),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x3,
          labels=labels_2x1)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_placeholder,
          labels=labels_placeholder)

  def test_invalid_labels_values(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    labels_2x1_with_large_id = np.array(((45,), (1,),), dtype=np.int)
    labels_2x1_with_negative_id = np.array(((-5,), (1,),), dtype=np.int)
    logits_2x3 = np.array(((1., 2., 4.), (1., 2., 3.),))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.test_session():
      with self.assertRaisesOpError('Label IDs must < n_classes'):
        spec.loss.eval({
            labels_placeholder: labels_2x1_with_large_id,
            logits_placeholder: logits_2x3
        })

    with self.test_session():
      with self.assertRaisesOpError('Label IDs must >= 0'):
        spec.loss.eval({
            labels_placeholder: labels_2x1_with_negative_id,
            logits_placeholder: logits_2x3
        })

  def test_invalid_labels_sparse_tensor(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    labels_2x1 = sparse_tensor.SparseTensor(
        values=['english', 'italian'],
        indices=[[0, 0], [1, 0]],
        dense_shape=[2, 1])
    logits_2x3 = np.array(((1., 2., 4.), (1., 2., 3.),))

    with self.assertRaisesRegexp(
        ValueError, 'SparseTensor labels are not supported.'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x3,
          labels=labels_2x1)

  def test_incompatible_labels_shape(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    # Labels should be shape (batch_size, 1).
    # Here batch sizes are different.
    values_3x1 = np.array(((1,), (1,), (1,),))
    values_2x3 = np.array(((1., 2., 3.), (1., 2., 3.),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Dimensions must be equal'):
      head.create_estimator_spec(
          features={'x': values_2x3},
          mode=model_fn.ModeKeys.EVAL,
          logits=values_2x3,
          labels=values_3x1)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': values_2x3},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.OpError,
          'logits and labels must have the same first dimension'):
        spec.loss.eval({
            labels_placeholder: values_3x1,
            logits_placeholder: values_2x3
        })

  def test_predict(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_probabilities = [[0.576117, 0.2119416, 0.2119416],
                              [0.2119416, 0.2119416, 0.576117]]
    expected_class_ids = [[0], [2]]
    expected_classes = [[b'0'], [b'2']]
    expected_export_classes = [[b'0', b'1', b'2']] * 2

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    self.assertItemsEqual(
        ('', _DEFAULT_SERVING_KEY), spec.export_outputs.keys())

    # Assert predictions and export_outputs.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits,
                          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(
          expected_probabilities,
          predictions[prediction_keys.PredictionKeys.PROBABILITIES])
      self.assertAllClose(expected_class_ids,
                          predictions[prediction_keys.PredictionKeys.CLASS_IDS])
      self.assertAllEqual(expected_classes,
                          predictions[prediction_keys.PredictionKeys.CLASSES])

      self.assertAllClose(
          expected_probabilities,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].scores))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))

  def test_predict_with_vocabulary_list(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_classes = [[b'aang'], [b'zuko']]
    expected_export_classes = [[b'aang', b'iroh', b'zuko']] * 2

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertAllEqual(
          expected_classes,
          sess.run(spec.predictions[prediction_keys.PredictionKeys.CLASSES]))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))

  def test_weight_should_not_impact_prediction(self):
    n_classes = 3
    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_probabilities = [[0.576117, 0.2119416, 0.2119416],
                              [0.2119416, 0.2119416, 0.576117]]
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, weight_column='label_weights')

    weights_2x1 = [[1.], [2.]]
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,),), dtype=np.int32),
            'label_weights': weights_2x1,
        },
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits,
                          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(
          expected_probabilities,
          predictions[prediction_keys.PredictionKeys.PROBABILITIES])

  def test_eval(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)

    # Create estimator spec.
    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss / 2,
        keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
    }

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval()
                             for k in value_ops},
          rtol=tol,
          atol=tol)

  def test_eval_with_label_vocabulary(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss / 2,
        keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
    }

    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol, atol=tol)

  def test_weighted_multi_example_eval(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((10, 0, 0), (0, 10, 0), (0, 0, 10),), dtype=np.float32)
    labels = np.array(((1,), (2,), (2,)), dtype=np.int64)
    weights_3x1 = np.array(((1.,), (2.,), (3.,)), dtype=np.float64)
    # loss = sum(cross_entropy(labels, logits) * [1, 2, 3])
    #      = sum([10, 10, 0] * [1, 2, 3]) = 30
    expected_loss = 30.
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,),), dtype=np.int32),
            'label_weights': weights_3x1,
        },
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss / np.sum(weights_3x1),
        # Weighted accuracy is 1 * 3.0 / sum weights = 0.5
        keys.ACCURACY: 0.5,
    }

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert loss, and metrics.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol, atol=tol)

  def test_train(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    expected_train_result = 'my_train_op'
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.

    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          metric_keys.MetricKeys.LOSS_MEAN: expected_loss / 2,
      }, summary_str, tol)

  def test_train_with_one_dim_label_and_weights(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, weight_column='label_weights')

    logits = np.array(((10, 0, 0), (0, 10, 0), (0, 0, 10),), dtype=np.float32)
    labels_rank_1 = np.array((1, 2, 2,), dtype=np.int64)
    weights_rank_1 = np.array((1., 2., 3.,), dtype=np.float64)

    self.assertEqual((3,), labels_rank_1.shape)
    self.assertEqual((3,), weights_rank_1.shape)

    expected_train_result = 'my_train_op'
    # loss = sum(cross_entropy(labels, logits) * [1, 2, 3])
    #      = sum([10, 10, 0] * [1, 2, 3]) = 30
    expected_loss = 30.

    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,),), dtype=np.float32),
            'label_weights': weights_rank_1,
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)

    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          metric_keys.MetricKeys.LOSS_MEAN: (
              expected_loss / np.sum(weights_rank_1)),
      }, summary_str, tol)

  def test_train_with_vocabulary(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss = sess.run(spec.loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

  def test_weighted_multi_example_train(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((10, 0, 0), (0, 10, 0), (0, 0, 10),), dtype=np.float32)
    labels = np.array(((1,), (2,), (2,)), dtype=np.int64)
    weights_3x1 = np.array(((1.,), (2.,), (3.,)), dtype=np.float64)
    expected_train_result = 'my_train_op'
    # loss = sum(cross_entropy(labels, logits) * [1, 2, 3])
    #      = sum([10, 10, 0] * [1, 2, 3]) = 30
    expected_loss = 30.

    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,),), dtype=np.float32),
            'label_weights': weights_3x1,
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss mean = sum(cross_entropy(labels, logits) * [1,2,3]) / (1+2+3)
          #      = sum([10, 10, 0] * [1, 2, 3]) / 6 = 30 / 6
          metric_keys.MetricKeys.LOSS_MEAN:
              expected_loss / np.sum(weights_3x1),
      }, summary_str, tol)


# TODO(ptucker): Add thresholds tests.
class BinaryLogisticHeadWithSigmoidCrossEntropyLossTest(test.TestCase):

  def test_threshold_too_small(self):
    with self.assertRaisesRegexp(ValueError, r'thresholds not in \(0, 1\)'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          thresholds=(0., 0.5))

  def test_threshold_too_large(self):
    with self.assertRaisesRegexp(ValueError, r'thresholds not in \(0, 1\)'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          thresholds=(0.5, 1.))

  def test_invalid_logits_shape(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    self.assertEqual(1, head.logits_dimension)

    # Logits should be shape (batch_size, 1).
    logits_2x2 = np.array(((45., 44.), (41., 42.),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.PREDICT,
          logits=logits_2x2)

    # Dynamic shape.
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.predictions[prediction_keys.PredictionKeys.PROBABILITIES].eval({
            logits_placeholder: logits_2x2
        })

  def test_invalid_labels_shape(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    self.assertEqual(1, head.logits_dimension)

    # Labels and logits should be shape (batch_size, 1).
    labels_2x2 = np.array(((45., 44.), (41., 42.),))
    logits_2x1 = np.array(((45.,), (41.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'labels shape'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x1,
          labels=labels_2x2)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'labels shape'):
        spec.loss.eval({
            logits_placeholder: logits_2x1,
            labels_placeholder: labels_2x2
        })

  def test_incompatible_labels_shape(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    self.assertEqual(1, head.logits_dimension)

    # Both logits and labels should be shape (batch_size, 1).
    values_2x1 = np.array(((0.,), (1.,),))
    values_3x1 = np.array(((0.,), (1.,), (0.,),))

    # Static shape.
    with self.assertRaisesRegexp(
        ValueError, 'logits and labels must have the same shape'):
      head.create_estimator_spec(
          features={'x': values_2x1},
          mode=model_fn.ModeKeys.EVAL,
          logits=values_2x1,
          labels=values_3x1)
    with self.assertRaisesRegexp(
        ValueError, 'logits and labels must have the same shape'):
      head.create_estimator_spec(
          features={'x': values_2x1},
          mode=model_fn.ModeKeys.EVAL,
          logits=values_3x1,
          labels=values_2x1)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': values_2x1},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'Incompatible shapes'):
        spec.loss.eval({
            labels_placeholder: values_2x1,
            logits_placeholder: values_3x1
        })
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'Incompatible shapes'):
        spec.loss.eval({
            labels_placeholder: values_3x1,
            logits_placeholder: values_2x1
        })

  def test_predict(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = [[0.3], [-0.4]]
    expected_logistics = [[0.574443], [0.401312]]
    expected_probabilities = [[0.425557, 0.574443], [0.598688, 0.401312]]
    expected_class_ids = [[1], [0]]
    expected_classes = [[b'1'], [b'0']]
    expected_export_classes = [[b'0', b'1']] * 2
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    # Assert spec contains expected tensors.
    self.assertIsNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNone(spec.train_op)
    self.assertItemsEqual(('', 'classification', 'regression',
                           _DEFAULT_SERVING_KEY), spec.export_outputs.keys())
    _assert_no_hooks(self, spec)

    # Assert predictions.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits,
                          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(expected_logistics,
                          predictions[prediction_keys.PredictionKeys.LOGISTIC])
      self.assertAllClose(
          expected_probabilities,
          predictions[prediction_keys.PredictionKeys.PROBABILITIES])
      self.assertAllClose(expected_class_ids,
                          predictions[prediction_keys.PredictionKeys.CLASS_IDS])
      self.assertAllEqual(expected_classes,
                          predictions[prediction_keys.PredictionKeys.CLASSES])
      self.assertAllClose(
          expected_probabilities,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].scores))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))
      self.assertAllClose(expected_logistics,
                          sess.run(spec.export_outputs['regression'].value))

  def test_predict_with_vocabulary_list(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        label_vocabulary=['aang', 'iroh'])

    logits = [[1.], [0.]]
    expected_classes = [[b'iroh'], [b'aang']]

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertAllEqual(
          expected_classes,
          sess.run(spec.predictions[prediction_keys.PredictionKeys.CLASSES]))

  def test_eval(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    # Create estimator spec.
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array(((1,), (1,),), dtype=np.int32))

    keys = metric_keys.MetricKeys
    expected_metrics = {
        # loss = sum(cross_entropy(labels, logits)) = sum(0, 41) = 41
        # loss_mean = loss/2 = 41./2 = 20.5
        keys.LOSS_MEAN: 20.5,
        keys.ACCURACY: 1./2,
        keys.PREDICTION_MEAN: 1./2,
        keys.LABEL_MEAN: 2./2,
        keys.ACCURACY_BASELINE: 2./2,
        keys.AUC: 0.,
        keys.AUC_PR: 1.,
    }

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(41., loss)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_eval_with_vocabulary_list(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        label_vocabulary=['aang', 'iroh'])

    # Create estimator spec.
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=[[b'iroh'], [b'iroh']])

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      sess.run(update_ops)
      self.assertAllClose(1. / 2,
                          value_ops[metric_keys.MetricKeys.ACCURACY].eval())

  def test_eval_with_thresholds(self):
    thresholds = [0.25, 0.5, 0.75]
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        thresholds=thresholds)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=np.array(((-1,), (1,),), dtype=np.float32),
        labels=np.array(((1,), (1,),), dtype=np.int32))

    # probabilities[i] = 1/(1 + exp(-logits[i])) =>
    # probabilities = [1/(1 + exp(1)), 1/(1 + exp(-1))] = [0.269, 0.731]
    # loss = -sum(ln(probabilities[label[i]])) = -ln(0.269) -ln(0.731)
    #      = 1.62652338
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: 1.62652338 / 2.,
        keys.ACCURACY: 1./2,
        keys.PREDICTION_MEAN: 1./2,
        keys.LABEL_MEAN: 2./2,
        keys.ACCURACY_BASELINE: 2./2,
        keys.AUC: 0.,
        keys.AUC_PR: 1.,
        keys.ACCURACY_AT_THRESHOLD % thresholds[0]: 1.,
        keys.PRECISION_AT_THRESHOLD % thresholds[0]: 1.,
        keys.RECALL_AT_THRESHOLD % thresholds[0]: 1.,
        keys.ACCURACY_AT_THRESHOLD % thresholds[1]: .5,
        keys.PRECISION_AT_THRESHOLD % thresholds[1]: 1.,
        keys.RECALL_AT_THRESHOLD % thresholds[1]: .5,
        keys.ACCURACY_AT_THRESHOLD % thresholds[2]: 0.,
        keys.PRECISION_AT_THRESHOLD % thresholds[2]: 0.,
        keys.RECALL_AT_THRESHOLD % thresholds[2]: 0.,
    }

    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())

  def test_train(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    # Create estimator spec.
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = sum(cross_entropy(labels, logits)) = sum(0, 41) = 41
    expected_loss = 41.
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=np.array(((1,), (1,),), dtype=np.float64),
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/2 = 41/2 = 20.5
          metric_keys.MetricKeys.LOSS_MEAN: 20.5,
      }, summary_str)

  def test_float_labels_train(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    # Create estimator spec.
    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = sum(cross_entropy(labels, logits))
    #      = sum(-label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i]))
    #      = -0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5))
    #        -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))
    #      = 1.2484322
    expected_loss = 1.2484322
    def _train_op_fn(loss):
      with ops.control_dependencies((_assert_close(
          math_ops.to_float(expected_loss), math_ops.to_float(loss)),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={'x': np.array([[42]], dtype=np.float32)},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=np.array([[0.8], [0.4]], dtype=np.float32),
        train_op_fn=_train_op_fn)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAlmostEqual(expected_loss, loss, delta=1.e-5)
      self.assertEqual(expected_train_result, train_result)

  def test_float_labels_eval(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    # Create estimator spec.
    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array([[42]], dtype=np.float32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array([[0.8], [0.4]], dtype=np.float32))

    # loss = sum(cross_entropy(labels, logits))
    #      = sum(-label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i]))
    #      = -0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5))
    #        -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))
    #      = 1.2484322
    expected_loss = 1.2484322

    # Assert loss.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAlmostEqual(expected_loss, loss, delta=1.e-5)
      self.assertAlmostEqual(
          expected_loss / 2., metrics[metric_keys.MetricKeys.LOSS_MEAN])

  def test_weighted_multi_example_predict(self):
    """3 examples, 1 batch."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.int32)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,), (43,), (44,)), dtype=np.int32),
            'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float32),
        },
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(
          logits.astype(np.float32),
          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(
          _sigmoid(logits).astype(np.float32),
          predictions[prediction_keys.PredictionKeys.LOGISTIC])
      self.assertAllClose(
          [[0., 1.], [1., 0.],
           [0., 1.]], predictions[prediction_keys.PredictionKeys.PROBABILITIES])
      self.assertAllClose([[1], [0], [1]],
                          predictions[prediction_keys.PredictionKeys.CLASS_IDS])
      self.assertAllEqual([[b'1'], [b'0'], [b'1']],
                          predictions[prediction_keys.PredictionKeys.CLASSES])

  def test_weighted_multi_example_eval(self):
    """3 examples, 1 batch."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.int32)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,), (43,), (44,)), dtype=np.int32),
            'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float32),
        },
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array(((1,), (1,), (0,)), dtype=np.int32))

    # label_mean = (1*1 + .1*1 + 1.5*0)/(1 + .1 + 1.5) = 1.1/2.6
    #            = .42307692307
    expected_label_mean = .42307692307
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # losses = label_weights*cross_entropy(labels, logits)
        #        = (1*0 + .1*41 + 1.5*44) = (1, 4.1, 66)
        # loss = sum(losses) = 1 + 4.1 + 66 = 70.1
        # loss_mean = loss/sum(label_weights) = 70.1/(1 + .1 + 1.5)
        #           = 70.1/2.6 = 26.9615384615
        keys.LOSS_MEAN: 26.9615384615,
        # accuracy = (1*1 + .1*0 + 1.5*0)/(1 + .1 + 1.5) = 1/2.6 = .38461538461
        keys.ACCURACY: .38461538461,
        # prediction_mean = (1*1 + .1*0 + 1.5*1)/(1 + .1 + 1.5) = 2.5/2.6
        #                 = .96153846153
        keys.PREDICTION_MEAN: .96153846153,
        keys.LABEL_MEAN: expected_label_mean,
        keys.ACCURACY_BASELINE: 1 - expected_label_mean,
        keys.AUC: .45454565,
        keys.AUC_PR: .6737757325172424,
    }

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(70.1, loss)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_train_with_one_dim_labels_and_weights(self):
    """3 examples, 1 batch."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    labels_rank_1 = np.array((1., 1., 0.,))
    weights_rank_1 = np.array(((1., .1, 1.5,)), dtype=np.float64)
    self.assertEqual((3,), labels_rank_1.shape)
    self.assertEqual((3,), weights_rank_1.shape)

    expected_train_result = b'my_train_op'
    # losses = label_weights*cross_entropy(labels, logits)
    #        = (1*0 + .1*41 + 1.5*44) = (1, 4.1, 66)
    # loss = sum(losses) = 1 + 4.1 + 66 = 70.1
    expected_loss = 70.1
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
            'label_weights': weights_rank_1,
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertIsNotNone(spec.train_op)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((
          spec.loss, spec.train_op, spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/sum(label_weights) = 70.1/(1 + .1 + 1.5)
          #           = 70.1/2.6 = 26.9615384615
          metric_keys.MetricKeys.LOSS_MEAN: 26.9615384615,
      }, summary_str)

  def test_weighted_multi_example_train(self):
    """3 examples, 1 batch."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # losses = label_weights*cross_entropy(labels, logits)
    #        = (1*0 + .1*41 + 1.5*44) = (1, 4.1, 66)
    # loss = sum(losses) = 1 + 4.1 + 66 = 70.1
    expected_loss = 70.1
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
            'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float64),
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=np.array(((1.,), (1.,), (0.,))),
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertIsNotNone(spec.train_op)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((
          spec.loss, spec.train_op, spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/sum(label_weights) = 70.1/(1 + .1 + 1.5)
          #           = 70.1/2.6 = 26.9615384615
          metric_keys.MetricKeys.LOSS_MEAN: 26.9615384615,
      }, summary_str)


class RegressionHeadWithMeanSquaredErrorLossTest(test.TestCase):

  def test_invalid_label_dimension(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid label_dimension'):
      head_lib._regression_head_with_mean_squared_error_loss(label_dimension=-1)
    with self.assertRaisesRegexp(ValueError, r'Invalid label_dimension'):
      head_lib._regression_head_with_mean_squared_error_loss(label_dimension=0)

  def test_invalid_logits(self):
    head = head_lib._regression_head_with_mean_squared_error_loss(
        label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    logits_1d = np.array(((45.,), (41.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.PREDICT,
          logits=logits_1d)

    # Dynamic shape.
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.predictions[prediction_keys.PredictionKeys.PREDICTIONS].eval({
            logits_placeholder: logits_1d
        })

  def test_incompatible_labels_eval(self):
    head = head_lib._regression_head_with_mean_squared_error_loss(
        label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    values_3d = np.array(((45., 46., 47.), (41., 42., 43.),))
    values_1d = np.array(((43.,), (44.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'labels shape'):
      head.create_estimator_spec(
          features={'x': values_1d},
          mode=model_fn.ModeKeys.EVAL,
          logits=values_3d,
          labels=values_1d)
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': values_3d}, labels=values_3d,
          mode=model_fn.ModeKeys.EVAL, logits=values_1d, train_op_fn=None)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': values_1d},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.loss.eval({
            labels_placeholder: values_3d,
            logits_placeholder: values_1d
        })
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'labels shape'):
        spec.loss.eval({
            labels_placeholder: values_1d,
            logits_placeholder: values_3d
        })

  def test_incompatible_labels_train(self):
    head = head_lib._regression_head_with_mean_squared_error_loss(
        label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    values_3d = np.array(((45., 46., 47.), (41., 42., 43.),))
    values_1d = np.array(((43.,), (44.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'labels shape'):
      head.create_estimator_spec(
          features={'x': values_1d},
          mode=model_fn.ModeKeys.TRAIN,
          logits=values_3d,
          labels=values_1d,
          train_op_fn=lambda x: x)
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': values_3d},
          mode=model_fn.ModeKeys.TRAIN,
          logits=values_1d,
          labels=values_3d,
          train_op_fn=lambda x: x)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': values_1d},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits_placeholder,
        labels=labels_placeholder,
        train_op_fn=lambda x: x)
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.loss.eval({
            labels_placeholder: values_3d,
            logits_placeholder: values_1d
        })
    with self.test_session():
      with self.assertRaisesRegexp(errors.OpError, 'labels shape'):
        spec.loss.eval({
            labels_placeholder: values_1d,
            logits_placeholder: values_3d
        })

  def test_predict(self):
    head = head_lib._regression_head_with_mean_squared_error_loss()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.int32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertIsNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNone(spec.train_op)
    self.assertItemsEqual(
        ('', signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY),
        spec.export_outputs.keys())
    _assert_no_hooks(self, spec)

    # Assert predictions.
    with self.test_session():
      _initialize_variables(self, spec.scaffold)
      self.assertAllClose(logits, spec.predictions[prediction_key].eval())

  def test_eval(self):
    head = head_lib._regression_head_with_mean_squared_error_loss()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array(((43,), (44,),), dtype=np.int32))

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertItemsEqual(
        (metric_keys.MetricKeys.LOSS_MEAN,), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
          metric_keys.MetricKeys.LOSS_MEAN]
      predictions, loss, loss_mean = sess.run((
          spec.predictions[prediction_key], spec.loss, loss_mean_update_op))
      self.assertAllClose(logits, predictions)
      # loss = (43-45)^2 + (44-41)^2 = 4+9 = 13
      self.assertAllClose(13., loss)
      # loss_mean = loss/2 = 13/2 = 6.5
      expected_loss_mean = 6.5
      # Check results of both update (in `loss_mean`) and value ops.
      self.assertAllClose(expected_loss_mean, loss_mean)
      self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

  def test_train(self):
    head = head_lib._regression_head_with_mean_squared_error_loss()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),), dtype=np.float32)},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=np.array(((43.,), (44.,),), dtype=np.float64),
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/2 = 13/2 = 6.5
          metric_keys.MetricKeys.LOSS_MEAN: 6.5,
      }, summary_str)

  def test_weighted_multi_example_eval(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,), (44,)), dtype=np.int32)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,), (43,), (44,)), dtype=np.int32),
            'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float32),
        },
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array(((35,), (42,), (45,)), dtype=np.int32))

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertItemsEqual(
        (metric_keys.MetricKeys.LOSS_MEAN,), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
          metric_keys.MetricKeys.LOSS_MEAN]
      predictions, loss, loss_mean = sess.run((
          spec.predictions[prediction_key], spec.loss, loss_mean_update_op))
      self.assertAllClose(logits, predictions)
      # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
      self.assertAllClose(101.6, loss)
      # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
      expected_loss_mean = 39.0769231
      # Check results of both update (in `loss_mean`) and value ops.
      self.assertAllClose(expected_loss_mean, loss_mean)
      self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

  def test_weight_with_numeric_column(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column=feature_column_lib.numeric_column(
            'label_weights', normalizer_fn=lambda x: x + 1.))

    # Create estimator spec.
    logits = np.array(((45,), (41,), (44,)), dtype=np.int32)
    spec = head.create_estimator_spec(
        features={
            'x':
                np.array(((42,), (43,), (44,)), dtype=np.int32),
            'label_weights':
                np.array(((0.,), (-0.9,), (0.5,)), dtype=np.float32),
        },
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array(((35,), (42,), (45,)), dtype=np.int32))

    # Assert loss.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss = sess.run(spec.loss)
      # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
      self.assertAllClose(101.6, loss)

  def test_weighted_multi_example_train(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42,), (43,), (44,)), dtype=np.float32),
            'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float64),
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=np.array(((35.,), (42.,), (45.,)), dtype=np.float32),
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
          metric_keys.MetricKeys.LOSS_MEAN: 39.0769231,
      }, summary_str)

  def test_with_one_dim_label_and_weight(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    x_feature_rank_1 = np.array((42., 43., 44.,), dtype=np.float32)
    weight_rank_1 = np.array((1., .1, 1.5,), dtype=np.float64)
    labels_rank_1 = np.array((35., 42., 45.,))
    self.assertEqual((3,), x_feature_rank_1.shape)
    self.assertEqual((3,), weight_rank_1.shape)
    self.assertEqual((3,), labels_rank_1.shape)

    spec = head.create_estimator_spec(
        features={
            'x': x_feature_rank_1,
            'label_weights': weight_rank_1,
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
          metric_keys.MetricKeys.LOSS_MEAN: 39.0769231,
      }, summary_str)

  def test_weighted_multi_value_eval(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45., 41., 44.),))
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42., 43., 44.),)),
            'label_weights': np.array(((1., .1, 1.5),)),
        },
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=np.array(((35., 42., 45.),)))

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertItemsEqual(
        (metric_keys.MetricKeys.LOSS_MEAN,), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
          metric_keys.MetricKeys.LOSS_MEAN]
      predictions, loss, loss_mean = sess.run((
          spec.predictions[prediction_key], spec.loss, loss_mean_update_op))
      self.assertAllClose(logits, predictions)
      # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
      self.assertAllClose(101.6, loss)
      # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
      expected_loss_mean = 39.076923
      # Check results of both update (in `loss_mean`) and value ops.
      self.assertAllClose(expected_loss_mean, loss_mean)
      self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

  def test_weighted_multi_value_train(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45., 41., 44.),))
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features={
            'x': np.array(((42., 43., 44.),)),
            'label_weights': np.array(((1., .1, 1.5),)),
        },
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=np.array(((35., 42., 45.),)),
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[prediction_key].dtype)
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Evaluate predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
          metric_keys.MetricKeys.LOSS_MEAN: 39.076923,
      }, summary_str)

  def test_weighted_multi_batch_eval(self):
    """1d label, 1 example, 3 batches."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45.,), (41.,), (44.,)))
    input_fn = numpy_io.numpy_input_fn(
        x={
            'x': np.array(((42.,), (43.,), (44.,))),
            'label_weights': np.array(((1.,), (.1,), (1.5,))),
            # 'logits' is not a feature, but we use `numpy_input_fn` to make a
            # batched version of it, and pop it off before passing to
            # `create_estimator_spec`.
            'logits': logits,
        },
        y=np.array(((35.,), (42.,), (45.,))),
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    batched_features, batched_labels = input_fn()
    batched_logits = batched_features.pop('logits')
    spec = head.create_estimator_spec(
        features=batched_features,
        mode=model_fn.ModeKeys.EVAL,
        logits=batched_logits,
        labels=batched_labels,
        train_op_fn=None)

    # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
    # loss = sum(losses) = 100+.1+1.5 = 101.6
    # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
    expected_metrics = {metric_keys.MetricKeys.LOSS_MEAN: 39.076923}

    # Assert spec contains expected tensors.
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    _assert_no_hooks(self, spec)

    with self.test_session() as sess:
      # Finalize graph and initialize variables.
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      queue_runner_impl.start_queue_runners()

      # Run tensors for `steps` steps.
      steps = len(logits)
      results = tuple([
          sess.run((
              spec.loss,
              # The `[1]` gives us the metric update op.
              {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
          )) for _ in range(steps)
      ])

      # Assert losses and metrics.
      self.assertAllClose((100, .1, 1.5), [r[0] for r in results])
      # For metrics, check results of both update (in `results`) and value ops.
      # Note: we only check the result of the last step for streaming metrics.
      self.assertAllClose(expected_metrics, results[steps - 1][1])
      self.assertAllClose(expected_metrics, {
          k: spec.eval_metric_ops[k][0].eval() for k in spec.eval_metric_ops
      })

  def test_weighted_multi_batch_train(self):
    """1d label, 1 example, 3 batches."""
    head = head_lib._regression_head_with_mean_squared_error_loss(
        weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45.,), (41.,), (44.,)))
    input_fn = numpy_io.numpy_input_fn(
        x={
            'x': np.array(((42.,), (43.,), (44.,))),
            'label_weights': np.array(((1.,), (.1,), (1.5,))),
            # 'logits' is not a feature, but we use `numpy_input_fn` to make a
            # batched version of it, and pop it off before passing to
            # `create_estimator_spec`.
            'logits': logits,
        },
        y=np.array(((35.,), (42.,), (45.,))),
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    batched_features, batched_labels = input_fn()
    batched_logits = batched_features.pop('logits')
    spec = head.create_estimator_spec(
        features=batched_features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=batched_logits,
        labels=batched_labels,
        train_op_fn=lambda loss: loss * -7.)

    # Assert spec contains expected tensors.
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertIsNotNone(spec.train_op)

    with self.test_session() as sess:
      # Finalize graph and initialize variables.
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      queue_runner_impl.start_queue_runners()

      results = tuple([
          sess.run((spec.loss, spec.train_op)) for _ in range(len(logits))
      ])

      # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
      expected_losses = np.array((100, .1, 1.5))
      self.assertAllClose(expected_losses, [r[0] for r in results])
      self.assertAllClose(expected_losses * -7., [r[1] for r in results])


if __name__ == '__main__':
  test.main()
