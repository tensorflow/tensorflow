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
"""Tests for head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.contrib.estimator.python.estimator import head as head_lib
from tensorflow.contrib.estimator.python.estimator import multi_head as multi_head_lib
from tensorflow.core.framework import summary_pb2
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants


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


class MultiHeadTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def test_no_heads(self):
    with self.assertRaisesRegexp(
        ValueError, r'Must specify heads\. Given: \[\]'):
      multi_head_lib.multi_head(heads=[])

  def test_head_name_missing(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    head2 = head_lib.multi_label_head(n_classes=3)
    with self.assertRaisesRegexp(
        ValueError, r'All given heads must have head_name specified\.'):
      multi_head_lib.multi_head([head1, head2])

  def test_head_weights_wrong_size(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    head2 = head_lib.multi_label_head(n_classes=3, head_name='head2')
    with self.assertRaisesRegexp(
        ValueError,
        r'heads and head_weights must have the same size\. '
        r'Given len\(heads\): 2. Given len\(head_weights\): 1\.'):
      multi_head_lib.multi_head([head1, head2], head_weights=[1.])

  def test_predict_two_heads(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    head2 = head_lib.multi_label_head(n_classes=3, head_name='head2')
    multi_head = multi_head_lib.multi_head([head1, head2])

    logits = {
        'head1': np.array([[-1., 1.], [-1.5, 1.]], dtype=np.float32),
        'head2': np.array([[2., -2., 2.], [-3., 2., -2.]], dtype=np.float32)
    }
    expected_probabilities = {
        'head1': _sigmoid(logits['head1']),
        'head2': _sigmoid(logits['head2']),
    }

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    self.assertItemsEqual(
        (_DEFAULT_SERVING_KEY, _DEFAULT_SERVING_KEY + '/head1', 'head1',
         _DEFAULT_SERVING_KEY + '/head2', 'head2'),
        spec.export_outputs.keys())

    # Assert predictions and export_outputs.
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(
          logits['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          logits['head2'],
          predictions[('head2', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          expected_probabilities['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.PROBABILITIES)])
      self.assertAllClose(
          expected_probabilities['head2'],
          predictions[('head2', prediction_keys.PredictionKeys.PROBABILITIES)])

      self.assertAllClose(
          expected_probabilities['head1'],
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].scores))
      self.assertAllClose(
          expected_probabilities['head1'],
          sess.run(spec.export_outputs['head1'].scores))
      self.assertAllClose(
          expected_probabilities['head2'],
          sess.run(spec.export_outputs['head2'].scores))

  def test_eval_two_heads_with_weights(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    head2 = head_lib.multi_label_head(n_classes=3, head_name='head2')
    multi_head = multi_head_lib.multi_head(
        [head1, head2], head_weights=[1., 2.])

    logits = {
        'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2': np.array([[20., -20., 20.], [-30., 20., -20.]],
                          dtype=np.float32),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # head1: expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # head2: expected_unweighted_loss = [[20., 20., 20.], [30., 0., 0]]
    # Average over classes, weighted sum ober batch and heads.
    expected_loss_head1 = 17.5
    expected_loss_head2 = 30.0
    expected_loss = 1. * expected_loss_head1 + 2. * expected_loss_head2

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN + '/head1': expected_loss_head1 / 2,
        keys.LOSS_MEAN + '/head2': expected_loss_head2 / 2,
        # auc and auc_pr cannot be reliably calculated for only 4-6 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC + '/head1': 0.1667,
        keys.AUC + '/head2': 0.3333,
        keys.AUC_PR + '/head1': 0.6667,
        keys.AUC_PR + '/head2': 0.5000,
    }

    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    tol = 1e-3
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
          rtol=tol,
          atol=tol)

  def test_train_create_loss_one_head(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    multi_head = multi_head_lib.multi_head([head1])

    logits = {'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)}
    labels = {'head1': np.array([[1, 0], [1, 1]], dtype=np.int64)}
    with self.assertRaisesRegexp(
        NotImplementedError,
        r'create_loss not yet implemented for MultiHead\.'):
      multi_head.create_loss(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=logits,
          labels=labels)

  def test_train_create_loss_two_heads_with_weights(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    head2 = head_lib.multi_label_head(n_classes=3, head_name='head2')
    multi_head = multi_head_lib.multi_head(
        [head1, head2], head_weights=[1., 2.])

    logits = {
        'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2': np.array([[20., -20., 20.], [-30., 20., -20.]],
                          dtype=np.float32),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    with self.assertRaisesRegexp(
        NotImplementedError,
        r'create_loss not yet implemented for MultiHead\.'):
      multi_head.create_loss(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=logits,
          labels=labels)

  def test_train_one_head(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    multi_head = multi_head_lib.multi_head([head1])

    logits = {'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)}
    labels = {'head1': np.array([[1, 0], [1, 1]], dtype=np.int64)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, sum over weights.
    expected_loss = 17.5
    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=3)])

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
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
    tol = 1e-3
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS + '/head1': expected_loss,
          # Average loss over examples.
          metric_keys.MetricKeys.LOSS_MEAN + '/head1': expected_loss / 2,
      }, summary_str, tol)

  def test_train_two_heads_with_weights(self):
    head1 = head_lib.multi_label_head(n_classes=2, head_name='head1')
    head2 = head_lib.multi_label_head(n_classes=3, head_name='head2')
    multi_head = multi_head_lib.multi_head(
        [head1, head2], head_weights=[1., 2.])

    logits = {
        'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2': np.array([[20., -20., 20.], [-30., 20., -20.]],
                          dtype=np.float32),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # head1: expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # head2: expected_unweighted_loss = [[20., 20., 20.], [30., 0., 0]]
    # Average over classes, weighted sum ober batch and heads.
    expected_loss_head1 = 17.5
    expected_loss_head2 = 30.0
    expected_loss = 1. * expected_loss_head1 + 2. * expected_loss_head2
    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=3)])

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
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
    tol = 1e-3
    with self.test_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS + '/head1': expected_loss_head1,
          metric_keys.MetricKeys.LOSS + '/head2': expected_loss_head2,
          # Average loss over examples.
          metric_keys.MetricKeys.LOSS_MEAN + '/head1': expected_loss_head1 / 2,
          metric_keys.MetricKeys.LOSS_MEAN + '/head2': expected_loss_head2 / 2,
      }, summary_str, tol)


if __name__ == '__main__':
  test.main()
