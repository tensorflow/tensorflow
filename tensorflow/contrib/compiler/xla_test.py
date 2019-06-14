# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for contrib.compiler.xla."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl.testing import parameterized

from tensorflow.contrib.compiler import xla
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python import summary
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.training import training


_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_EXPECTED_LOSS = 1
_EXPECTED_FEATURE = 2
_EXPECTED_LABEL = 3


def _test_train_model_fn(features, labels, mode, params):
  """A dummy model_fn for testing purpose."""
  del features, labels, params
  loss = constant_op.constant(_EXPECTED_LOSS)
  return model_fn_lib.EstimatorSpec(
      mode=mode, loss=loss, train_op=array_ops.identity(loss))


@xla.estimator_model_fn
def decorated_model_fn(features, labels, mode, params):
  return _test_train_model_fn(features, labels, mode, params)


def make_dummy_features_labels():
  # XLA CPU/GPU backend doesn't support guaranteed constant, thus use dataset
  # container to work around.
  features_dataset = dataset_ops.Dataset.from_tensors(
      constant_op.constant(_EXPECTED_FEATURE)).repeat(10)
  features_op = features_dataset.make_one_shot_iterator().get_next()
  labels_dataset = dataset_ops.Dataset.from_tensors(
      constant_op.constant(_EXPECTED_LABEL)).repeat(10)
  labels_op = labels_dataset.make_one_shot_iterator().get_next()
  return features_op, labels_op


class XlaDecoratorTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('test_use_as_decorator', decorated_model_fn, None),
      ('test_use_as_function', xla.estimator_model_fn(_test_train_model_fn),
       None),
      ('test_use_tpu_false_hparams', decorated_model_fn,
       hparam.HParams(use_tpu=False)),
      ('test_use_tpu_false_dict_params', decorated_model_fn, {
          'use_tpu': False
      }),
  )
  def test_compile(self, model_fn, params):
    """Calls model_fn and verifies it is compiled."""
    with test.mock.patch.object(xla, 'compile') as mock_xla_compile:
      loss = constant_op.constant(_EXPECTED_LOSS)
      mock_xla_compile.return_value = [loss]

      features, labels = make_dummy_features_labels()
      estimator_spec = model_fn(
          features=features, labels=labels, mode=_TRAIN, params=params or {})

      self.assertEqual(mock_xla_compile.call_count, 1)
      self.assertEqual(estimator_spec.mode, _TRAIN)

      with self.test_session() as sess:
        self.assertEqual(sess.run(estimator_spec.loss), sess.run(loss))
        self.assertEqual(sess.run(estimator_spec.train_op), sess.run(loss))

  @parameterized.named_parameters(
      ('test_use_tpu_true_hparams', decorated_model_fn,
       hparam.HParams(use_tpu=True)),
      ('test_use_tpu_true_dict_params', decorated_model_fn, {
          'use_tpu': True
      }),
  )
  def test_not_compile(self, model_fn, params):
    """Calls model_fn and verifies it is NOT compiled."""
    with test.mock.patch.object(xla, 'compile') as mock_xla_compile:
      loss = constant_op.constant(_EXPECTED_LOSS)
      mock_xla_compile.return_value = [loss]

      features, labels = make_dummy_features_labels()
      estimator_spec = model_fn(
          features=features, labels=labels, mode=_TRAIN, params=params or {})

      mock_xla_compile.assert_not_called()
      self.assertEqual(estimator_spec.mode, _TRAIN)

      with self.test_session() as sess:
        self.assertEqual(sess.run(estimator_spec.loss), sess.run(loss))
        self.assertEqual(sess.run(estimator_spec.train_op), sess.run(loss))

  def test_model_with_summary(self):
    """Tests that summary ops are disabled."""

    @xla.estimator_model_fn
    def model_fn_with_summary(features, labels, mode, params):
      del features, labels, params
      loss = constant_op.constant(_EXPECTED_LOSS)
      summary.scalar('loss_scalar_summary', loss)
      summary.histogram('loss_histogram_summary', loss)
      summary.image('loss_image_summary', loss)
      return model_fn_lib.EstimatorSpec(
          mode=mode, loss=loss, train_op=array_ops.identity(loss))

    features, labels = make_dummy_features_labels()
    estimator_spec = model_fn_with_summary(
        features=features, labels=labels, mode=_TRAIN, params={})

    with self.test_session() as sess:
      self.assertEqual(sess.run(estimator_spec.loss), _EXPECTED_LOSS)


def _test_eval_metric_fn(eval_tensor_1, eval_tensor_2):
  return {
      'metric_1': (eval_tensor_1, eval_tensor_1),
      'metric_2': (eval_tensor_2, eval_tensor_2),
  }


class XlaDecoratorEvaluationTest(test.TestCase):

  def _verify_evaluation_result(self, eval_model_fn):
    features, labels = make_dummy_features_labels()
    estimator_spec = eval_model_fn(
        features=features, labels=labels, mode=_EVAL, params={})

    with self.test_session() as sess:
      self.assertEqual(sess.run(estimator_spec.loss), _EXPECTED_LOSS)
      self.assertEqual(
          sess.run(estimator_spec.eval_metric_ops['metric_1'][0]),
          _EXPECTED_FEATURE + _EXPECTED_LABEL)
      self.assertEqual(
          sess.run(estimator_spec.eval_metric_ops['metric_1'][1]),
          _EXPECTED_FEATURE + _EXPECTED_LABEL)
      self.assertEqual(
          sess.run(estimator_spec.eval_metric_ops['metric_2'][0]),
          _EXPECTED_FEATURE - _EXPECTED_LABEL)
      self.assertEqual(
          sess.run(estimator_spec.eval_metric_ops['metric_2'][1]),
          _EXPECTED_FEATURE - _EXPECTED_LABEL)

  def test_eval_base_estimator_spec_eval_metric_ops_disallowed(self):

    @xla.estimator_model_fn
    def eval_model_fn_return_estimator_spec(features, labels, mode, params):
      del features, labels, params
      loss = constant_op.constant(_EXPECTED_LOSS)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metric_ops={
              'metric': (array_ops.identity(loss), control_flow_ops.no_op())
          })

    with self.assertRaisesRegexp(
        ValueError, 'EstimatorSpec.eval_metric_ops is not supported with XLA '
        'compilation. Please use TPUEstimatorSpec.eval_metrics instead.'):
      self._verify_evaluation_result(eval_model_fn_return_estimator_spec)

  def test_eval_base_estimator_spec_no_eval_metric_ops(self):

    @xla.estimator_model_fn
    def eval_model_fn_no_eval_metric_ops(features, labels, mode, params):
      del features, labels, params
      return model_fn_lib.EstimatorSpec(
          mode=mode, loss=constant_op.constant(_EXPECTED_LOSS))

    features, labels = make_dummy_features_labels()
    estimator_spec = eval_model_fn_no_eval_metric_ops(
        features=features, labels=labels, mode=_EVAL, params={})
    with self.test_session() as sess:
      self.assertEqual(sess.run(estimator_spec.loss), _EXPECTED_LOSS)

  def test_eval_no_eval_metrics(self):

    @xla.estimator_model_fn
    def eval_model_fn_no_eval_metrics(features, labels, mode, params):
      del features, labels, params
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, loss=constant_op.constant(_EXPECTED_LOSS))

    features, labels = make_dummy_features_labels()
    estimator_spec = eval_model_fn_no_eval_metrics(
        features=features, labels=labels, mode=_EVAL, params={})

    self.assertEqual(estimator_spec.eval_metric_ops, {})
    with self.test_session() as sess:
      self.assertEqual(sess.run(estimator_spec.loss), _EXPECTED_LOSS)

  def test_eval_fn_missing_input_tensor(self):

    @xla.estimator_model_fn
    def eval_model_fn(features, labels, mode, params):
      del params
      dummy_eval_metric_fn_tensors_dict = {
          'eval_tensor_1': features + labels,
      }
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=constant_op.constant(_EXPECTED_LOSS),
          eval_metrics=(_test_eval_metric_fn,
                        dummy_eval_metric_fn_tensors_dict))

    with self.assertRaisesRegexp(
        ValueError,
        re.escape("Arguments ['eval_tensor_2'] are needed by metric_fn (first "
                  'element of TPUEstimatorSpec.eval_metrics) but they are not '
                  'provided by evaluation tensors (second element of '
                  'TPUEstimatorSpec.eval_metrics).')):
      self._verify_evaluation_result(eval_model_fn)

  def test_eval_fn_extraneous_input_tensor(self):

    @xla.estimator_model_fn
    def eval_model_fn(features, labels, mode, params):
      del params
      dummy_eval_metric_fn_tensors_dict = {
          'eval_tensor_1': features + labels,
          'eval_tensor_2': features - labels,
          'extra_tensor': features * 2 - labels,
      }
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=constant_op.constant(_EXPECTED_LOSS),
          eval_metrics=(_test_eval_metric_fn,
                        dummy_eval_metric_fn_tensors_dict))

    with self.assertRaisesRegexp(
        ValueError,
        re.escape("Arguments ['extra_tensor'] are provided by evaluation "
                  'tensors (second element of TPUEstimatorSpec.eval_metrics) '
                  'but they are not needed by metric_fn (first element of '
                  'TPUEstimatorSpec.eval_metrics).')):
      self._verify_evaluation_result(eval_model_fn)

  def test_eval_tensors_as_list(self):

    @xla.estimator_model_fn
    def eval_model_fn(features, labels, mode, params):
      del params
      dummy_eval_metric_fn_tensors = [features + labels, features - labels]
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=constant_op.constant(_EXPECTED_LOSS),
          eval_metrics=(_test_eval_metric_fn, dummy_eval_metric_fn_tensors))

    self._verify_evaluation_result(eval_model_fn)

  def test_eval_tensors_as_dict(self):

    @xla.estimator_model_fn
    def eval_model_fn(features, labels, mode, params):
      del params
      dummy_eval_metric_fn_tensors_dict = {
          'eval_tensor_1': features + labels,
          'eval_tensor_2': features - labels,
      }
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=constant_op.constant(_EXPECTED_LOSS),
          eval_metrics=(_test_eval_metric_fn,
                        dummy_eval_metric_fn_tensors_dict))

    self._verify_evaluation_result(eval_model_fn)

  def test_model_with_summary(self):
    """Tests that summary ops are disabled."""

    @xla.estimator_model_fn
    def model_fn_with_summary(features, labels, mode, params):
      del features, labels, params
      loss = constant_op.constant(_EXPECTED_LOSS)
      summary.scalar('loss_scalar_summary', loss)
      summary.histogram('loss_histogram_summary', loss)
      summary.image('loss_image_summary', loss)
      return tpu_estimator.TPUEstimatorSpec(mode=mode, loss=loss)

    features, labels = make_dummy_features_labels()
    estimator_spec = model_fn_with_summary(
        features=features, labels=labels, mode=_EVAL, params={})

    with self.test_session() as sess:
      self.assertEqual(sess.run(estimator_spec.loss), _EXPECTED_LOSS)


class XlaDecoratorScaffoldTest(test.TestCase, parameterized.TestCase):

  def _make_scaffold_fn(self, mode):

    def _scaffold_fn_on_cpu():
      scaffold = training.Scaffold()
      self.assertNotIn(mode, self.is_scaffold_fn_called)
      self.is_scaffold_fn_called[mode] = True
      return scaffold

    return _scaffold_fn_on_cpu

  def test_scaffold_fn_return_none(self):

    @xla.estimator_model_fn
    def model_fn(features, labels, mode, params):
      del features, labels, params
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=constant_op.constant(_EXPECTED_LOSS),
          train_op=control_flow_ops.no_op(),
          scaffold_fn=lambda: None)

    features, labels = make_dummy_features_labels()
    with self.assertRaisesRegexp(
        ValueError,
        'TPUEstimatorSpec.scaffold_fn returns None, which is not allowed'):
      model_fn(features=features, labels=labels, mode=_TRAIN, params={})

  @parameterized.named_parameters(
      ('train_mode', _TRAIN),
      ('eval_mode', _EVAL),
      # TODO(ycao): Add predict_mode test after PREDICT mode is implemented.
  )
  def test_scaffold_fn_in_mode(self, mode):

    @xla.estimator_model_fn
    def model_fn(features, labels, mode, params):
      del features, labels, params
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=constant_op.constant(_EXPECTED_LOSS),
          train_op=control_flow_ops.no_op(),
          scaffold_fn=self._make_scaffold_fn(mode))

    features, labels = make_dummy_features_labels()

    self.is_scaffold_fn_called = {}
    model_fn(features=features, labels=labels, mode=mode, params={})
    self.assertTrue(self.is_scaffold_fn_called[mode])


if __name__ == '__main__':
  test.main()
