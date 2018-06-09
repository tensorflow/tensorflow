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
from tensorflow.python.estimator.canned import dnn_testing_utils
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
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import monitored_session
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


class CreateEstimatorSpecTest(test.TestCase):

  class _HeadWithTPUSupport(head_lib._Head):
    """Head that overrides _create_tpu_estimator_spec."""

    def name(self):
      return 'HeadWithTPUSupport'

    def logits_dimension(self):
      return None

    def create_loss(self, features, mode, logits, labels):
      return None

    def _create_tpu_estimator_spec(self, features, mode, logits, labels=None,
                                   optimizer=None, train_op_fn=None,
                                   regularization_losses=None):
      return model_fn._TPUEstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          loss=constant_op.constant(0.0, dtype=dtypes.float32))

  class _HeadWithOutTPUSupport(head_lib._Head):
    """Head that overrides create_estimator_spec."""

    def name(self):
      return 'HeadWithOutTPUSupport'

    def logits_dimension(self):
      return None

    def create_loss(self, features, mode, logits, labels):
      return None

    def create_estimator_spec(self, features, mode, logits, labels=None,
                              optimizer=None, train_op_fn=None,
                              regularization_losses=None):
      return model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          loss=constant_op.constant(0.0, dtype=dtypes.float32))

  class _InvalidHead(head_lib._Head):
    """Head that overrides neither estimator_spec functions."""

    def name(self):
      return 'InvalidHead'

    def logits_dimension(self):
      return None

    def create_loss(self, features, mode, logits, labels):
      return None

  def test_head_override_tpu_estimator_spec(self):
    """Test for `_Head` that overrides _create_tpu_estimator_spec."""
    head = self._HeadWithTPUSupport()

    tpu_spec = head._create_tpu_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(tpu_spec, model_fn._TPUEstimatorSpec))
    est_spec = head.create_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(est_spec, model_fn.EstimatorSpec))

  def test_head_override_estimator_spec(self):
    """Test for `_Head` that overrides create_estimator_spec."""
    head = self._HeadWithOutTPUSupport()

    with self.assertRaisesRegexp(
        NotImplementedError,
        'TPUEstimatorSpec not available for this model head.'):
      _ = head._create_tpu_estimator_spec(
          features=None, mode=None, logits=None)
    est_spec = head.create_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(est_spec, model_fn.EstimatorSpec))

  def test_invalid_head_class(self):
    head = self._InvalidHead()

    with self.assertRaisesRegexp(
        NotImplementedError,
        'TPUEstimatorSpec not available for this model head.'):
      _ = head._create_tpu_estimator_spec(
          features=None, mode=None, logits=None)
    with self.assertRaisesRegexp(
        NotImplementedError,
        r'Subclasses of _Head must implement `create_estimator_spec\(\)` or '
        r'_create_tpu_estimator_spec\(\).'):
      _ = head.create_estimator_spec(
          features=None, mode=None, logits=None)


class MultiClassHeadWithSoftmaxCrossEntropyLoss(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def test_n_classes_is_none(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=None)

  def test_n_classes_is_2(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=2)

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=3, loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: none'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=3, loss_reduction=losses.Reduction.NONE)

  def test_loss_fn_arg_labels_missing(self):
    def _loss_fn(logits):
      del logits  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):
    def _loss_fn(labels):
      del labels  # unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):
    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused
    head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):
    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes=3, loss_fn=_loss_fn)

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
    features = {'x': np.array(((42.,),))}

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
      head.create_loss(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x3,
          labels=labels_2x2)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[2 2\]'):
        training_loss.eval({
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
    features = {'x': np.array(((42.,),))}

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
      head.create_loss(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x3,
          labels=labels_2x1)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
      head.create_loss(
          features=features,
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
    training_loss = head.create_loss(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesOpError('Labels must <= n_classes - 1'):
        training_loss.eval({
            labels_placeholder: labels_2x1_with_large_id,
            logits_placeholder: logits_2x3
        })

    with self.test_session():
      with self.assertRaisesOpError('Labels must >= 0'):
        training_loss.eval({
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
      head.create_loss(
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
    features = {'x': values_2x3}

    # Static shape.
    with self.assertRaisesRegexp(
        ValueError,
        r'Shape mismatch: The shape of labels \(received \(3,\)\) should equal '
        r'the shape of logits except for the last dimension '
        r'\(received \(2, 3\)\)\.'
    ):
      head.create_loss(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=values_2x3,
          labels=values_3x1)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[3 1\]'):
        training_loss.eval({
            labels_placeholder: values_3x1,
            logits_placeholder: values_2x3
        })

  def test_name(self):
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, name='foo')
    self.assertEqual('foo', head.name)

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
        (_DEFAULT_SERVING_KEY, 'predict', 'classification'),
        spec.export_outputs.keys())

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

  def test_eval_create_loss(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = cross_entropy(labels, logits) = [10, 0].
    expected_training_loss = 10.
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_eval_create_loss_loss_fn(self):
    """Tests head.create_loss for eval mode and custom loss_fn."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    logits_input = np.array([[-10., 10., 0.], [-15., 10., 0]], dtype=np.float32)
    labels_input = np.array([[1], [2]], dtype=np.int64)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_fn=_loss_fn)

    actual_training_loss = head.create_loss(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_input,
        labels=labels_input)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(np.sum(loss), actual_training_loss.eval())

  def test_eval_create_loss_loss_fn_wrong_shape(self):
    """Tests custom loss_fn that returns Tensor of unexpected shape."""
    loss = np.array([1., 2.], dtype=np.float32)
    def _loss_fn(labels, logits):
      del labels, logits  # Unused
      return constant_op.constant(loss)
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_fn=_loss_fn)

    logits = np.array([[-10., 10., 0.], [-15., 10., 0.]], dtype=np.float32)
    labels = np.array([[1], [2]], dtype=np.int64)
    actual_training_loss = head.create_loss(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 1\]\. \] '
          r'\[logits_shape: \] \[2 3\] \[loss_shape: \] \[2\]'):
        actual_training_loss.eval()

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3)

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.EVAL,
          logits=np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32),
          labels=None)

  def test_eval(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)
    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
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

  def test_eval_metric_ops_with_head_name(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, name='some_multiclass_head')
    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    expected_metric_keys = [
        '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS_MEAN),
        '{}/some_multiclass_head'.format(metric_keys.MetricKeys.ACCURACY)
    ]
    self.assertItemsEqual(expected_metric_keys, spec.eval_metric_ops.keys())

  def test_eval_with_regularization_losses(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(10, 0) / 2 = 5.
    expected_unregularized_loss = 5.
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
        keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
    }

    # Assert predictions, loss, and metrics.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_regularized_loss, loss, rtol=tol, atol=tol)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval()
                             for k in value_ops},
          rtol=tol,
          atol=tol)

  def test_eval_with_label_vocabulary_create_loss(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])
    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = cross_entropy(labels, logits) = [10, 0].
    expected_training_loss = 10.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_eval_with_label_vocabulary(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    spec = head.create_estimator_spec(
        features=features,
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

  def test_train_create_loss(self):
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # unreduced_loss = cross_entropy(labels, logits) = [10, 0].
    expected_unreduced_loss = [[10.], [0.]]
    # Weights default to 1.
    expected_weights = 1.
    # training_loss = 1 * 10 + 1 * 0
    expected_training_loss = 10.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    tol = 1e-2
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # unreduced_loss = cross_entropy(labels, logits) = [10, 0].
    expected_unreduced_loss = [[10.], [0.]]
    # Weights default to 1.
    expected_weights = 1.
    # training_loss = 1 * 10 + 1 * 0 / num_nonzero_weights
    expected_training_loss = 10. / 2.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    tol = 1e-2
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3)
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32),
          labels=None,
          train_op_fn=_no_op_train_fn)

  def test_train(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    spec = head.create_estimator_spec(
        features=features,
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

  def test_train_with_optimizer(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    expected_train_result = 'my_train_op'

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        return string_ops.string_join(
            [constant_op.constant(expected_train_result),
             string_ops.as_string(loss, precision=2)])

    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_train_with_update_ops(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes)

    with ops.Graph().as_default():
      w = variables.Variable(1)
      update_op = w.assign_add(1)
      ops.add_to_collection(ops.GraphKeys.UPDATE_OPS, update_op)

      t = variables.Variable('')
      expected_train_result = b'my_train_op'
      def _train_op_fn(loss):
        del loss
        return t.assign(expected_train_result)

      spec = head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32),
          labels=np.array(((1,), (1,)), dtype=np.int64),
          train_op_fn=_train_op_fn)

      with self.test_session() as sess:
        _initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, name='some_multiclass_head')

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    features = {'x': np.array(((42,),), dtype=np.int32)}

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert summaries.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      _assert_simple_summaries(self, {
          '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS):
              expected_loss,
          '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS_MEAN):
              expected_loss / 2,
      }, summary_str, tol)

  def test_train_with_regularization_losses(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    logits = np.array(((10, 0, 0), (0, 10, 0),), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(10, 0) / 2 = 5.
    # loss = unregularized_loss + regularization_loss = 7.
    expected_loss = 7.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses)

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
          metric_keys.MetricKeys.LOSS_REGULARIZATION: (
              expected_regularization_loss),
      }, summary_str, tol)

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='label_weights')

    logits = np.array(((10, 0, 0), (0, 10, 0), (0, 0, 10),), dtype=np.float32)
    labels_rank_1 = np.array((1, 2, 2,), dtype=np.int64)
    weights_rank_1 = np.array((1., 2., 3.,), dtype=np.float64)
    features = {
        'x': np.array(((42,),), dtype=np.float32),
        'label_weights': weights_rank_1
    }

    # unreduced_loss = cross_entropy(labels, logits) = [10, 10, 0].
    expected_unreduced_loss = [[10.], [10.], [0.]]
    # weights are reshaped to [3, 1] to match logits.
    expected_weights = [[1.], [2.], [3.]]
    # training_loss = 1 * 10 + 2 * 10 + 3 * 0 = 30.
    expected_training_loss = 30.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1)
    tol = 1e-2
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights.eval())

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='label_weights')

    logits = np.array(((10, 0, 0), (0, 10, 0), (0, 0, 10),), dtype=np.float32)
    labels_rank_1 = np.array((1, 2, 2,), dtype=np.int64)
    weights_rank_1 = np.array((1., 2., 3.,), dtype=np.float64)

    self.assertEqual((3,), labels_rank_1.shape)
    self.assertEqual((3,), weights_rank_1.shape)

    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    # loss = sum(cross_entropy(labels, logits) * [1, 2, 3])
    #      = sum([10, 10, 0] * [1, 2, 3]) = 30
    expected_loss = 30.

    features = {
        'x': np.array(((42,),), dtype=np.float32),
        'label_weights': weights_rank_1
    }
    spec = head.create_estimator_spec(
        features=features,
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

  def test_train_with_vocabulary_create_loss(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = cross_entropy(labels, logits) = [10, 0].
    expected_training_loss = 10.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_train_with_vocabulary(self):
    n_classes = 3
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
    expected_loss = 10.
    spec = head.create_estimator_spec(
        features=features,
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

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='weights')

    logits = np.array([[[10, 0, 0], [12, 0, 0]],
                       [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)

    # unreduced_loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    expected_unreduced_loss = [[[0.], [12.]], [[0.], [15.]]]
    # weights are reshaped to [2, 2, 1] to match logits.
    expected_weights = [[[1.], [1.5]], [[2.], [2.5]]]
    # training_loss = 1*0 + 1.5*12 + 2*0 + 2.5*15 = 55.5
    expected_training_loss = 55.5
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    tol = 1e-2
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights.eval())

  def test_multi_dim_weighted_train(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='weights')

    logits = np.array([[[10, 0, 0], [12, 0, 0]],
                       [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    # loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weighted_sum_loss = 1*0 + 1.5*12 + 2*0 + 2.5*15 = 55.5
    expected_loss = 55.5
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert predictions, loss, train_op, and summaries.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_multi_dim_train_weights_wrong_inner_dim(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 1]."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]],
                       [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1.], [2.]], dtype=np.float32)
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2, 3]."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]],
                       [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[[1., 1.1, 1.2], [1.5, 1.6, 1.7]],
                        [[2., 2.1, 2.2], [2.5, 2.6, 2.7]]])
    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights_placeholder},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 3\]'):
        spec.loss.eval({weights_placeholder: weights})

  def test_multi_dim_weighted_eval(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
        n_classes=3, weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]],
                       [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weighted_sum_loss = 1*0 + 1.5*12 + 2*0 + 2.5*15 = 55.5
    expected_loss = 55.5
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss / np.sum(weights),
        keys.ACCURACY: (1.*1. + 1.5*0. + 2.*1. + 2.5*0.) / np.sum(weights),
    }

    # Assert predictions, loss, and metrics.
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


class BinaryLogisticHeadWithSigmoidCrossEntropyLossTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def test_threshold_too_small(self):
    with self.assertRaisesRegexp(ValueError, r'thresholds not in \(0, 1\)'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          thresholds=(0., 0.5))

  def test_threshold_too_large(self):
    with self.assertRaisesRegexp(ValueError, r'thresholds not in \(0, 1\)'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          thresholds=(0.5, 1.))

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: none'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          loss_reduction=losses.Reduction.NONE)

  def test_loss_fn_arg_labels_missing(self):
    def _loss_fn(logits):
      del logits  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):
    def _loss_fn(labels):
      del labels  # unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):
    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):
    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          loss_fn=_loss_fn)

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
    with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
      head.create_loss(
          features={'x': np.array(((42.,),))},
          mode=model_fn.ModeKeys.EVAL,
          logits=logits_2x1,
          labels=labels_2x2)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.create_loss(
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[2 2\]'):
        training_loss.eval({
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
      head.create_loss(
          features={'x': values_2x1},
          mode=model_fn.ModeKeys.EVAL,
          logits=values_2x1,
          labels=values_3x1)
    with self.assertRaisesRegexp(
        ValueError, 'logits and labels must have the same shape'):
      head.create_loss(
          features={'x': values_2x1},
          mode=model_fn.ModeKeys.EVAL,
          logits=values_3x1,
          labels=values_2x1)

    # Dynamic shape.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.create_loss(
        features={'x': values_2x1},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[3 1\] \[labels_shape: \] \[2 1\]'):
        training_loss.eval({
            labels_placeholder: values_2x1,
            logits_placeholder: values_3x1
        })
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[3 1\]'):
        training_loss.eval({
            labels_placeholder: values_3x1,
            logits_placeholder: values_2x1
        })

  def test_name(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        name='foo')
    self.assertEqual('foo', head.name)

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
    self.assertItemsEqual(('classification', 'regression', 'predict',
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

  def test_eval_create_loss(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # loss = cross_entropy(labels, logits) = [0, 41].
    expected_training_loss = 41.
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.EVAL,
          logits=np.array(((45,), (-41,),), dtype=np.float32),
          labels=None)

  def test_eval(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        # loss = sum(cross_entropy(labels, logits)) = sum(0, 41) = 41
        # loss_mean = loss/2 = 41./2 = 20.5
        keys.LOSS_MEAN: 20.5,
        keys.ACCURACY: 1./2,
        keys.PRECISION: 1.,
        keys.RECALL: 1./2,
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

  def test_eval_metric_ops_with_head_name(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        name='some_binary_head')
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    expected_metric_keys = [
        '{}/some_binary_head'.format(metric_keys.MetricKeys.LOSS_MEAN),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.ACCURACY),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.PRECISION),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.RECALL),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.PREDICTION_MEAN),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.LABEL_MEAN),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.ACCURACY_BASELINE),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.AUC),
        '{}/some_binary_head'.format(metric_keys.MetricKeys.AUC_PR),
    ]
    self.assertItemsEqual(expected_metric_keys, spec.eval_metric_ops.keys())

  def test_eval_with_regularization_losses(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(0, 41) / 2 = 20.5
    expected_unregularized_loss = 20.5
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
        keys.ACCURACY: 1./2,
        keys.PRECISION: 1.,
        keys.RECALL: 1./2,
        keys.PREDICTION_MEAN: 1./2,
        keys.LABEL_MEAN: 2./2,
        keys.ACCURACY_BASELINE: 2./2,
        keys.AUC: 0.,
        keys.AUC_PR: 1.,
    }

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_regularized_loss, loss)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_eval_with_vocabulary_list_create_loss(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        label_vocabulary=['aang', 'iroh'])
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(41., training_loss.eval())

  def test_eval_with_vocabulary_list(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        label_vocabulary=['aang', 'iroh'])
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      sess.run(update_ops)
      self.assertAllClose(1. / 2,
                          value_ops[metric_keys.MetricKeys.ACCURACY].eval())

  def test_eval_with_thresholds_create_loss(self):
    thresholds = [0.25, 0.5, 0.75]
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        thresholds=thresholds)
    logits = np.array(((-1,), (1,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # probabilities[i] = 1/(1 + exp(-logits[i])) =>
    # probabilities = [1/(1 + exp(1)), 1/(1 + exp(-1))] = [0.269, 0.731]
    # loss = -ln(probabilities[label[i]])) = [-ln(0.269), -ln(0.731)]
    #      = [1.31304389, 0.31334182]
    # weighted sum loss = 1.62638571
    expected_training_loss = 1.62638571
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_eval_with_thresholds(self):
    thresholds = [0.25, 0.5, 0.75]
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        thresholds=thresholds)
    logits = np.array(((-1,), (1,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    # probabilities[i] = 1/(1 + exp(-logits[i])) =>
    # probabilities = [1/(1 + exp(1)), 1/(1 + exp(-1))] = [0.269, 0.731]
    # loss = -sum(ln(probabilities[label[i]])) = -ln(0.269) -ln(0.731)
    #      = 1.62652338
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: 1.62652338 / 2.,
        keys.ACCURACY: 1./2,
        keys.PRECISION: 1.,
        keys.RECALL: .5,
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
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(1.62652338, loss)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval()
                             for k in value_ops},
          atol=tol,
          rtol=tol)

  def test_train_create_loss(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # unreduced_loss = cross_entropy(labels, logits) = [0, 41]
    expected_unreduced_loss = [[0.], [41.]]
    # weights default to 1.
    expected_weights = 1.
    # training loss = 1 * 0 + 1 * 41
    expected_training_loss = 41.
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())
      self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # unreduced_loss = cross_entropy(labels, logits) = [0, 41]
    expected_unreduced_loss = [[0.], [41.]]
    # weights default to 1.
    expected_weights = 1.
    # training loss = (1 * 0 + 1 * 41) / num_nonzero_weights
    expected_training_loss = 41. / 2.
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())
      self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
      self.assertAllClose(expected_weights, actual_weights)

  def test_eval_create_loss_loss_fn(self):
    """Tests head.create_loss for eval mode and custom loss_fn."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    logits_input = np.array([[-10.], [10.]], dtype=np.float32)
    labels_input = np.array([[1], [0]], dtype=np.int64)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_fn=_loss_fn)

    actual_training_loss = head.create_loss(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_input,
        labels=labels_input)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(np.sum(loss), actual_training_loss.eval())

  def test_eval_create_loss_loss_fn_wrong_shape(self):
    """Tests custom loss_fn that returns Tensor of unexpected shape."""
    loss = np.array([1., 2.], dtype=np.float32)
    def _loss_fn(labels, logits):
      del labels, logits  # Unused
      return constant_op.constant(loss)
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_fn=_loss_fn)

    logits = np.array([[-10.], [10.]], dtype=np.float32)
    labels = np.array([[1], [0]], dtype=np.int64)
    actual_training_loss = head.create_loss(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 1\]\. \] '
          r'\[logits_shape: \] \[2 1\] \[loss_shape: \] \[2\]'):
        actual_training_loss.eval()

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((45,), (-41,),), dtype=np.float32),
          labels=None,
          train_op_fn=_no_op_train_fn)

  def test_train(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) = sum(0, 41) = 41
    expected_loss = 41.
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
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

  def test_train_with_optimizer(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) = sum(0, 41) = 41
    expected_loss = 41.

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        with ops.control_dependencies((check_ops.assert_equal(
            math_ops.to_float(expected_loss), math_ops.to_float(loss),
            name='assert_loss'),)):
          return constant_op.constant(expected_train_result)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)

  def test_train_with_update_ops(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    with ops.Graph().as_default():
      w = variables.Variable(1)
      update_op = w.assign_add(1)
      ops.add_to_collection(ops.GraphKeys.UPDATE_OPS, update_op)

      t = variables.Variable('')
      expected_train_result = b'my_train_op'
      def _train_op_fn(loss):
        del loss
        return t.assign(expected_train_result)

      spec = head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((45,), (-41,),), dtype=np.float32),
          labels=np.array(((1,), (1,),), dtype=np.float64),
          train_op_fn=_train_op_fn)

      with self.test_session() as sess:
        _initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        name='some_binary_head')

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) = sum(0, 41) = 41
    expected_loss = 41.

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    # Assert summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      _assert_simple_summaries(
          self,
          {
              '{}/some_binary_head'.format(metric_keys.MetricKeys.LOSS):
                  expected_loss,
              # loss_mean = loss/2 = 41/2 = 20.5
              '{}/some_binary_head'.format(metric_keys.MetricKeys.LOSS_MEAN):
                  20.5,
          },
          summary_str)

  def test_train_with_regularization_losses(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42,),), dtype=np.float32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(0, 41) / 2 = 20.5
    # loss = unregularized_loss + regularization_loss = 7.
    expected_loss = 22.5
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses)

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
          metric_keys.MetricKeys.LOSS_REGULARIZATION: (
              expected_regularization_loss),
      }, summary_str)

  def test_float_labels_invalid_values(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[1.2], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)[0]
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r'Labels must <= n_classes - 1'):
      with self.test_session():
        _initialize_variables(self, monitored_session.Scaffold())
        training_loss.eval()

  def test_float_labels_train_create_loss(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    # loss = cross_entropy(labels, logits)
    #      = -label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i])
    #      = [-0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5)),
    #         -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))]
    #      = [0.57407698418, 0.67435524446]
    # weighted sum loss = 0.57407698418 + 0.67435524446
    expected_training_loss = 1.24843222864
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_float_labels_train(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    expected_train_result = b'my_train_op'
    features = {'x': np.array([[42]], dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits))
    #      = sum(-label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i]))
    #      = -0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5))
    #        -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))
    #      = 1.2484322
    expected_loss = 1.2484322
    def _train_op_fn(loss):
      with ops.control_dependencies((dnn_testing_utils.assert_close(
          math_ops.to_float(expected_loss), math_ops.to_float(loss)),)):
        return constant_op.constant(expected_train_result)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAlmostEqual(expected_loss, loss, delta=1.e-5)
      self.assertEqual(expected_train_result, train_result)

  def test_float_labels_eval_create_loss(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    # loss = cross_entropy(labels, logits)
    #      = -label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i])
    #      = [-0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5)),
    #         -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))]
    #      = [0.57407698418, 0.67435524446]
    # weighted sum loss = 0.57407698418 + 0.67435524446
    expected_training_loss = 1.24843222864
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_float_labels_eval(self):
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

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
        keys.PRECISION: 1./2.5,
        keys.RECALL: 1./1.1,
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

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    labels_rank_1 = np.array((1., 1., 0.,))
    weights_rank_1 = np.array(((1., .1, 1.5,)), dtype=np.float64)
    features = {
        'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
        'label_weights': weights_rank_1,
    }
    # unreduced_loss = cross_entropy(labels, logits) = [0, 41, 44]
    expected_unreduced_loss = [[0.], [41.], [44.]]
    # weights are reshaped to [3, 1] to match logits.
    expected_weights = [[1.], [.1], [1.5]]
    # training loss = 1 * 0 + .1 * 41 + 1.5 * 44
    expected_training_loss = 70.1
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(),
          rtol=1e-2, atol=1e-2)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(),
          rtol=1e-2, atol=1e-2)
      self.assertAllClose(expected_weights, actual_weights.eval())

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    labels_rank_1 = np.array((1., 1., 0.,))
    weights_rank_1 = np.array(((1., .1, 1.5,)), dtype=np.float64)
    self.assertEqual((3,), labels_rank_1.shape)
    self.assertEqual((3,), weights_rank_1.shape)
    features = {
        'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
        'label_weights': weights_rank_1,
    }
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
        features=features,
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

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2]."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # unreduced_loss = cross_entropy(labels, logits) = [[10, 0], [0, 12]].
    expected_unreduced_loss = [[[10.], [0.]], [[0.], [12.]]]
    # Weights are reshaped to [2, 2, 1] to match logits.
    expected_weights = [[[1.], [1.5]], [[2.], [2.5]]]
    # training_loss = 1*10 + 1.5*0 + 2*0 + 2.5*12 = 40
    expected_training_loss = 40.
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    tol = 1e-2
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(),
          rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(),
          rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights.eval())

  def test_multi_dim_weighted_train(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2]."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = cross_entropy(labels, logits) = [[10, 0], [0, 12]].
    # weighted_sum_loss = 1*10 + 1.5*0 + 2*0 + 2.5*12 = 40
    expected_loss = 40.
    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert predictions, loss, train_op, and summaries.
    tol = 1e-2
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_multi_dim_train_weights_wrong_inner_dim(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 1]."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1.], [2.]], dtype=np.float32)
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 1\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2, 2]."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights_placeholder},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 1\]\s\[weights_shape: \]\s\[2 2 2\]'):
        spec.loss.eval({
            weights_placeholder: np.array([[[1., 1.1], [1.5, 1.6]],
                                           [[2., 2.1], [2.5, 2.6]]])})

  def test_multi_dim_weighted_eval(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2]."""
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = cross_entropy(labels, logits) = [[10, 0], [0, 12]].
    # weighted_sum_loss = 1*10 + 1.5*0 + 2*0 + 2.5*12 = 40
    expected_loss = 40.

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss / np.sum(weights),
        keys.ACCURACY: (1.*0. + 1.5*1. + 2.*1. + 2.5*0.) / np.sum(weights),
        keys.PRECISION: 2.0/3.0,
        keys.RECALL: 2.0/4.5,
        keys.PREDICTION_MEAN: (1.*1 + 1.5*0 + 2.*1 + 2.5*0) / np.sum(weights),
        keys.LABEL_MEAN: (1.*0 + 1.5*0 + 2.*1 + 2.5*1) / np.sum(weights),
        keys.ACCURACY_BASELINE: (1.*0 + 1.5*0 + 2.*1 + 2.5*1) / np.sum(weights),
        # We cannot reliably calculate AUC with only 4 data points, but the
        # values should not change because of backwards-compatibility.
        keys.AUC: 0.5222,
        keys.AUC_PR: 0.7341,
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


class RegressionHead(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def test_invalid_label_dimension(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid label_dimension'):
      head_lib._regression_head(label_dimension=-1)
    with self.assertRaisesRegexp(ValueError, r'Invalid label_dimension'):
      head_lib._regression_head(label_dimension=0)

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib._regression_head(loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: none'):
      head_lib._regression_head(loss_reduction=losses.Reduction.NONE)

  def test_loss_fn_arg_labels_missing(self):
    def _loss_fn(logits):
      del logits  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):
    def _loss_fn(labels):
      del labels  # unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):
    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):
    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_invalid_logits(self):
    head = head_lib._regression_head(label_dimension=3)
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
    head = head_lib._regression_head(label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    values_3d = np.array(((45., 46., 47.), (41., 42., 43.),))
    values_1d = np.array(((43.,), (44.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
      head.create_loss(
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
    training_loss = head.create_loss(
        features={'x': values_1d},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 3\] \[labels_shape: \] \[2 1\]'):
        training_loss.eval({
            labels_placeholder: values_1d,
            logits_placeholder: values_3d
        })

  def test_incompatible_labels_train(self):
    head = head_lib._regression_head(label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    values_3d = np.array(((45., 46., 47.), (41., 42., 43.),))
    values_1d = np.array(((43.,), (44.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
      head.create_loss(
          features={'x': values_1d},
          mode=model_fn.ModeKeys.TRAIN,
          logits=values_3d,
          labels=values_1d)

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
    training_loss = head.create_loss(
        features={'x': values_1d},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits_placeholder,
        labels=labels_placeholder)[0]
    with self.test_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 3\] \[labels_shape: \] \[2 1\]'):
        training_loss.eval({
            labels_placeholder: values_1d,
            logits_placeholder: values_3d
        })

  def test_name(self):
    head = head_lib._regression_head(name='foo')
    self.assertEqual('foo', head.name)

  def test_predict(self):
    head = head_lib._regression_head()
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
    default_serving_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    self.assertItemsEqual(
        (default_serving_key, 'predict', 'regression'),
        spec.export_outputs.keys())
    _assert_no_hooks(self, spec)

    # Assert predictions.
    with self.test_session():
      _initialize_variables(self, spec.scaffold)
      self.assertAllClose(logits, spec.predictions[prediction_key].eval())
      self.assertAllClose(
          logits, spec.export_outputs[default_serving_key].value.eval())
      self.assertAllClose(
          logits, spec.export_outputs['regression'].value.eval())
      self.assertAllClose(
          logits, spec.export_outputs['predict'].outputs['predictions'].eval())

  def test_predict_with_inverse_link_fn(self):
    def _inverse_link_fn(logits):
      return logits - 10.
    head = head_lib._regression_head(inverse_link_fn=_inverse_link_fn)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.int32)
    expected_predictions = np.array(((35,), (31,),), dtype=np.int32)
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    # Assert spec contains expected tensors.
    keys = prediction_keys.PredictionKeys
    self.assertItemsEqual(
        (keys.PREDICTIONS, keys.LOGITS), spec.predictions.keys())
    self.assertEqual(dtypes.float32, spec.predictions[keys.PREDICTIONS].dtype)
    self.assertEqual(dtypes.float32, spec.predictions[keys.LOGITS].dtype)
    default_serving_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    self.assertItemsEqual(
        (default_serving_key, 'predict', 'regression'),
        spec.export_outputs.keys())

    # Assert predictions.
    with self.test_session():
      _initialize_variables(self, spec.scaffold)
      self.assertAllClose(
          expected_predictions, spec.predictions[keys.PREDICTIONS].eval())
      self.assertAllClose(logits, spec.predictions[keys.LOGITS].eval())
      self.assertAllClose(
          expected_predictions,
          spec.export_outputs[default_serving_key].value.eval())
      self.assertAllClose(
          expected_predictions, spec.export_outputs['regression'].value.eval())
      self.assertAllClose(
          expected_predictions,
          spec.export_outputs['predict'].outputs['predictions'].eval())
      self.assertAllClose(
          logits, spec.export_outputs['predict'].outputs['logits'].eval())

  def test_eval_create_loss(self):
    head = head_lib._regression_head()
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      # loss = [(43-45)^2, (44-41)] = [4, 9]
      self.assertAllClose(13., training_loss.eval())

  def test_eval_create_loss_loss_fn(self):
    """Tests head.create_loss for eval mode and custom loss_fn."""
    loss = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
    logits_input = np.array([[-1., 1.], [-2., 2.]], dtype=np.float32)
    labels_input = np.array([[1., 0.], [2., -1.]], dtype=np.float32)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)
    head = head_lib._regression_head(label_dimension=2, loss_fn=_loss_fn)

    actual_training_loss = head.create_loss(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_input,
        labels=labels_input)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(np.sum(loss), actual_training_loss.eval())

  def test_eval_create_loss_loss_fn_wrong_shape(self):
    """Tests custom loss_fn that returns Tensor of unexpected shape."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    def _loss_fn(labels, logits):
      del labels, logits  # Unused
      return constant_op.constant(loss)
    head = head_lib._regression_head(label_dimension=2, loss_fn=_loss_fn)

    logits = np.array([[-1., 1.], [-2., 2.]], dtype=np.float32)
    labels = np.array([[1., 0.], [2., -1.]], dtype=np.float32)
    actual_training_loss = head.create_loss(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 2\]\. \] '
          r'\[logits_shape: \] \[2 2\] \[loss_shape: \] \[2 1\]'):
        actual_training_loss.eval()

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._regression_head()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.EVAL,
          logits=np.array(((45,), (41,),), dtype=np.float32),
          labels=None)

  def test_eval(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

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

  def test_eval_metric_ops_with_head_name_for_regression(self):
    head = head_lib._regression_head(name='some_regression_head')
    logits = np.array(((1,), (9,)), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    expected_metric_keys = [
        '{}/some_regression_head'.format(metric_keys.MetricKeys.LOSS_MEAN),
    ]
    self.assertItemsEqual(expected_metric_keys, spec.eval_metric_ops.keys())

  def test_eval_with_regularization_losses(self):
    head = head_lib._regression_head(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = ((43-45)^2 + (44-41)^2) / batch_size
    #                    = (4 + 9) / 2 = 6.5
    expected_unregularized_loss = 6.5
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
    }

    # Assert predictions, loss, and metrics.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
      predictions, loss, metrics = sess.run((
          spec.predictions[prediction_key], spec.loss, update_ops))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_regularized_loss, loss)
      # Check results of both update (in `metrics`) and value ops.
      self.assertAllClose(expected_metrics, metrics)
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_train_create_loss(self):
    head = head_lib._regression_head()
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # unreduced_loss = [(43-45)^2, (44-41)] = [4, 9]
    expected_unreduced_loss = [[4.], [9.]]
    # weights default to 1.
    expected_weights = 1
    # training_loss = 1 * 4 + 1 * 9 = 13
    expected_training_loss = 13.
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())
      self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib._regression_head(
        loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # unreduced_loss = [(43-45)^2, (44-41)] = [4, 9]
    expected_unreduced_loss = [[4.], [9.]]
    # weights default to 1.
    expected_weights = 1
    # training_loss = (1 * 4 + 1 * 9) / num_nonzero_weights
    expected_training_loss = 13. / 2.
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())
      self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._regression_head()
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((45,), (41,),), dtype=np.float32),
          labels=None,
          train_op_fn=_no_op_train_fn)

  def test_train(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
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

  def test_train_with_optimizer(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        with ops.control_dependencies((check_ops.assert_equal(
            math_ops.to_float(expected_loss), math_ops.to_float(loss),
            name='assert_loss'),)):
          return constant_op.constant(expected_train_result)

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)

  def test_train_with_update_ops(self):
    head = head_lib._regression_head()

    with ops.Graph().as_default():
      w = variables.Variable(1)
      update_op = w.assign_add(1)
      ops.add_to_collection(ops.GraphKeys.UPDATE_OPS, update_op)

      t = variables.Variable('')
      expected_train_result = b'my_train_op'
      def _train_op_fn(loss):
        del loss
        return t.assign(expected_train_result)

      spec = head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((45,), (41,),), dtype=np.float32),
          labels=np.array(((43.,), (44.,),), dtype=np.float64),
          train_op_fn=_train_op_fn)

      with self.test_session() as sess:
        _initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    head = head_lib._regression_head(name='some_regression_head')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      _assert_simple_summaries(
          self,
          {
              '{}/some_regression_head'.format(metric_keys.MetricKeys.LOSS):
                  expected_loss,
              # loss_mean = loss/2 = 13/2 = 6.5
              '{}/some_regression_head'
              .format(metric_keys.MetricKeys.LOSS_MEAN):
                  6.5,
          },
          summary_str)

  def test_train_with_regularization_losses(self):
    head = head_lib._regression_head(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = ((43-45)^2 + (44-41)^2) / batch_size
    #                    = (4 + 9) / 2 = 6.5
    # loss = unregularized_loss + regularization_loss = 8.5
    expected_loss = 8.5
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses)

    # Assert predictions, loss, train_op, and summaries.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          metric_keys.MetricKeys.LOSS_REGULARIZATION: (
              expected_regularization_loss),
      }, summary_str)

  def test_weighted_multi_example_eval(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head(weight_column='label_weights')
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
    head = head_lib._regression_head(
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
    head = head_lib._regression_head(weight_column='label_weights')
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

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib._regression_head(weight_column='label_weights')
    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    x_feature_rank_1 = np.array((42., 43., 44.,), dtype=np.float32)
    weight_rank_1 = np.array((1., .1, 1.5,), dtype=np.float64)
    labels_rank_1 = np.array((35., 42., 45.,))
    # unreduced_loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
    expected_unreduced_loss = [[100.], [1.], [1.]]
    # weights are reshaped to [3, 1] to match logits.
    expected_weights = [[1.], [.1], [1.5]]
    # training_loss = 100 * 1 + 1 * .1 + 1.5 * 1 = 101.6
    expected_training_loss = 101.6
    features = {'x': x_feature_rank_1, 'label_weights': weight_rank_1}
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())
      self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
      self.assertAllClose(expected_weights, actual_weights.eval())

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib._regression_head(weight_column='label_weights')
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
    features = {'x': x_feature_rank_1, 'label_weights': weight_rank_1}
    self.assertEqual((3,), x_feature_rank_1.shape)
    self.assertEqual((3,), weight_rank_1.shape)
    self.assertEqual((3,), labels_rank_1.shape)

    spec = head.create_estimator_spec(
        features=features,
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

  def test_weighted_multi_value_eval_create_loss(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),))
    }
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      # loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
      # weighted sum loss = 1 * 100 + .1 * 1 + 1.5 * 1 = 101.6
      self.assertAllClose(101.6, training_loss.eval())

  def test_weighted_multi_value_eval(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),))
    }
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

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

  def test_weighted_multi_value_train_create_loss(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),))
    }
    # Create loss.
    training_loss = head.create_loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)[0]
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      # loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
      # weighted sum loss = 1 * 100 + .1 * 1 + 1.5 * 1 = 101.6
      self.assertAllClose(101.6, training_loss.eval())

  def test_weighted_multi_value_train(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),)),
    }
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
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
    head = head_lib._regression_head(weight_column='label_weights')
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
    head = head_lib._regression_head(weight_column='label_weights')
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

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 2]."""
    label_dimension = 3
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=label_dimension)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    weights = np.array([[1., 1.5], [2., 2.5]])
    expected_unreduced_loss = [[[1., 1., 1.], [4., 4., 4.]],
                               [[9., 9., 9.], [16., 16., 16.]]]
    expected_training_loss = np.sum(
        np.array([[[1. * x for x in [1., 1., 1.]],
                   [1.5 * x for x in [4., 4., 4.]]],
                  [[2. * x for x in [9., 9., 9.]],
                   [2.5 * x for x in [16., 16., 16.]]]]))
    # Weights are expanded to [2, 2, 1] to match logits.
    expected_weights = [[[1.], [1.5]], [[2.], [2.5]]]
    # Create loss.
    training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
        features={'label_weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())
      self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
      self.assertAllClose(expected_weights, actual_weights.eval())

  def test_multi_dim_weighted_train(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 2]."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    expected_train_result = b'my_train_op'
    features = {
        'label_weights': np.array([[1., 1.5], [2., 2.5]]),
    }
    # loss = 1*3*1^2 + 1.5*3*2^2 + 2*3*3^2 +2.5*3*4^2 = 195
    expected_loss = 195.
    # Create estimator spec.
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_loss, spec.loss.eval())

  def test_multi_dim_train_weights_wrong_inner_dim(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 1]."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    features = {
        'label_weights': np.array([[1.], [2]]),
    }
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 2, 2]."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    features = {
        'label_weights': weights_placeholder,
    }
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.test_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 2\]'):
        spec.loss.eval({
            weights_placeholder: np.array([[[1., 1.1], [1.5, 1.6]],
                                           [[2., 2.1], [2.5, 2.6]]])})


if __name__ == '__main__':
  test.main()
