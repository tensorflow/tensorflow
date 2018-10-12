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
# ==============================================================================
"""Tests for SavedModelEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

from tensorflow.contrib.estimator.python.estimator import export as contrib_export
from tensorflow.contrib.estimator.python.estimator import saved_model_estimator
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training


def dummy_input_fn():
  return dataset_ops.Dataset.from_tensors((
      {'x': constant_op.constant([[1], [-2]], dtype=dtypes.int64)},
      constant_op.constant([[4], [-3]], dtype=dtypes.float32))).repeat()


def dummy_input_fn_features_only():
  return dataset_ops.Dataset.from_tensors(
      {'x': constant_op.constant([[5], [6]], dtype=dtypes.int64)}).repeat()


def dummy_supervised_receiver_fn():
  feature_spec = {
      'x': array_ops.placeholder(
          dtype=dtypes.int64, shape=(2, 1), name='feature_x'),
      }
  label_spec = array_ops.placeholder(
      dtype=dtypes.float32, shape=[2, 1], name='truth')
  return export.build_raw_supervised_input_receiver_fn(
      feature_spec, label_spec)


def dummy_serving_receiver_fn():
  feature_spec = {'x': array_ops.placeholder(
      dtype=dtypes.int64, shape=(2, 1), name='feature_x'),}
  return export.build_raw_serving_input_receiver_fn(feature_spec)


def model_fn_diff_modes(features, labels, mode):
  _, _ = features, labels
  v = variables.Variable(21, name='some_var')
  train_op = None
  loss = constant_op.constant(104)
  if mode == model_fn_lib.ModeKeys.TRAIN:
    loss = constant_op.constant(105)
    predictions = constant_op.constant([501])
    train_op = control_flow_ops.group(
        state_ops.assign_add(training.get_global_step(), 1),
        state_ops.assign_add(v, 3))
  elif mode == model_fn_lib.ModeKeys.EVAL:
    loss = constant_op.constant(106)
    predictions = constant_op.constant([502])
  else:
    loss = constant_op.constant(107)
    predictions = constant_op.constant([503])
  return model_fn_lib.EstimatorSpec(
      mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          'abs_err': metrics_lib.mean_absolute_error(
              constant_op.constant(0), predictions)},
      predictions=predictions)


class SavedModelEstimatorTest(test.TestCase):

  def setUp(self):
    self.tmpdirs = []

  def tearDown(self):
    for tmpdir in self.tmpdirs:
      # gfile.DeleteRecursively fails in the windows cmake test, so use shutil.
      shutil.rmtree(tmpdir, ignore_errors=True)
    self.tmpdirs = []

  def _get_tmp_dir(self):
    tmpdir = tempfile.mkdtemp()
    self.tmpdirs.append(tmpdir)
    return tmpdir

  def _export_estimator(self, train=True, evaluate=True, predict=True,
                        model_fn=model_fn_diff_modes):
    est = estimator.Estimator(model_fn, self._get_tmp_dir())
    est.train(input_fn=dummy_input_fn, steps=10)

    input_receiver_fn_map = {}
    if train:
      input_receiver_fn_map[model_fn_lib.ModeKeys.TRAIN] = (
          dummy_supervised_receiver_fn())
    if evaluate:
      input_receiver_fn_map[model_fn_lib.ModeKeys.EVAL] = (
          dummy_supervised_receiver_fn())
    if predict:
      input_receiver_fn_map[model_fn_lib.ModeKeys.PREDICT] = (
          dummy_serving_receiver_fn())

    export_base_path = self._get_tmp_dir()
    export_dir = contrib_export.export_all_saved_models(
        est, export_base_path, input_receiver_fn_map)
    return export_dir

  def test_load_all_modes(self):
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(), self._get_tmp_dir())
    sme.train(input_fn=dummy_input_fn, steps=1)
    sme.train(input_fn=dummy_input_fn, steps=2)
    self.assertEqual(13, sme.get_variable_value('global_step'))
    self.assertEqual(60, sme.get_variable_value('some_var'))

    eval_results = sme.evaluate(dummy_input_fn, steps=5)

    self.assertEqual(13, eval_results['global_step'])
    self.assertEqual(106, eval_results['loss'])
    self.assertEqual(502, eval_results['metrics/abs_err'])

    predictions = next(sme.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'output': 503}, predictions)

  def test_load_all_modes_no_train(self):
    """Ensure that all functions can be used without requiring a ckpt."""
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(), self._get_tmp_dir())
    eval_results = sme.evaluate(dummy_input_fn, steps=5)
    self.assertEqual(10, eval_results['global_step'])
    self.assertEqual(106, eval_results['loss'])
    self.assertEqual(502, eval_results['metrics/abs_err'])

    predictions = next(sme.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'output': 503}, predictions)

  def test_partial_exported_estimator(self):
    sme1 = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(train=False, predict=False), self._get_tmp_dir())
    sme1.evaluate(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(RuntimeError, 'train mode is not available'):
      sme1.train(input_fn=dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(RuntimeError, 'infer mode is not available'):
      next(sme1.predict(dummy_input_fn_features_only))

    sme2 = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(evaluate=False), self._get_tmp_dir())
    sme2.train(input_fn=dummy_input_fn, steps=1)
    next(sme2.predict(dummy_input_fn_features_only))
    with self.assertRaisesRegexp(RuntimeError, 'eval mode is not available'):
      sme2.evaluate(dummy_input_fn, steps=5)

  def test_with_incorrect_input(self):
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(), self._get_tmp_dir())

    def bad_shape_input_fn():
      return dataset_ops.Dataset.from_tensors((
          {'x': constant_op.constant([1, 2], dtype=dtypes.int64)},
          constant_op.constant([1, 2], dtype=dtypes.float32)))

    with self.assertRaisesRegexp(ValueError, 'Expected shape'):
      sme.train(bad_shape_input_fn, steps=1)

    def bad_dtype_input_fn():
      return dataset_ops.Dataset.from_tensors((
          {'x': constant_op.constant([[1], [1]], dtype=dtypes.int32)},
          constant_op.constant([[1], [1]], dtype=dtypes.int64)))

    with self.assertRaisesRegexp(ValueError, 'Expected dtype'):
      sme.train(bad_dtype_input_fn, steps=1)

  def test_input_fn_with_global_step(self):
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(), self._get_tmp_dir())

    def bad_input_fn():
      training.get_or_create_global_step()
      return dataset_ops.Dataset.from_tensors((
          {'x': constant_op.constant([[1], [1]], dtype=dtypes.int64)},
          constant_op.constant([[1], [1]], dtype=dtypes.float32)))

    with self.assertRaisesRegexp(RuntimeError,
                                 'Graph must not contain a global step tensor'):
      sme.train(bad_input_fn, steps=1)

  def test_re_export_saved_model_serving_only(self):
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(), self._get_tmp_dir())
    sme.train(dummy_input_fn, steps=3)
    self.assertEqual(13, sme.get_variable_value('global_step'))
    self.assertEqual(60, sme.get_variable_value('some_var'))

    predictions = next(sme.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'output': 503}, predictions)

    # Export SavedModel, and test that the variable and prediction values are
    # the same.
    sme_export_dir = sme.export_savedmodel(
        self._get_tmp_dir(), dummy_serving_receiver_fn())

    sme2 = saved_model_estimator.SavedModelEstimator(
        sme_export_dir, self._get_tmp_dir())
    self.assertEqual(60, sme.get_variable_value('some_var'))
    self.assertEqual(13, sme.get_variable_value('global_step'))

    predictions = next(sme2.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'output': 503}, predictions)

  def test_re_export_saved_model(self):
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(), self._get_tmp_dir())
    self.assertDictEqual(
        {'loss': 106, 'metrics/abs_err': 502, 'global_step': 10},
        sme.evaluate(dummy_input_fn, steps=1))

    sme.train(dummy_input_fn, steps=3)
    self.assertDictEqual(
        {'loss': 106, 'metrics/abs_err': 502, 'global_step': 13},
        sme.evaluate(dummy_input_fn, steps=1))
    self.assertEqual(60, sme.get_variable_value('some_var'))

    predictions = next(sme.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'output': 503}, predictions)

    # Export SavedModel for all modes
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: dummy_supervised_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: dummy_supervised_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: dummy_serving_receiver_fn()}
    sme_export_dir = contrib_export.export_all_saved_models(
        sme, self._get_tmp_dir(), input_receiver_fn_map)

    sme2 = saved_model_estimator.SavedModelEstimator(
        sme_export_dir, self._get_tmp_dir())
    self.assertDictEqual(
        {'loss': 106, 'metrics/abs_err': 502, 'global_step': 13},
        sme.evaluate(dummy_input_fn, steps=1))
    self.assertEqual(60, sme.get_variable_value('some_var'))

    sme.train(dummy_input_fn, steps=7)
    self.assertEqual(20, sme.get_variable_value('global_step'))

    predictions = next(sme2.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'output': 503}, predictions)

  def test_load_saved_model_from_serving_only(self):
    def model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant([103]),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([502]),
          export_outputs={'test': export_output.ClassificationOutput(
              constant_op.constant([[32.]]))})

    est = estimator.Estimator(model_fn, self._get_tmp_dir())
    est.train(input_fn=dummy_input_fn, steps=10)

    def serving_input_receiver_fn():
      return export.ServingInputReceiver(
          {'test-features': constant_op.constant([[1], [1]])},
          array_ops.placeholder(dtype=dtypes.string))

    export_dir = est.export_savedmodel(
        self._get_tmp_dir(), serving_input_receiver_fn)

    sme = saved_model_estimator.SavedModelEstimator(
        export_dir, self._get_tmp_dir())

    def input_fn():
      return {'inputs': constant_op.constant('someinputstr')}

    prediction = next(sme.predict(input_fn))
    self.assertDictEqual({'scores': 32}, prediction)

  def test_with_local_init_op(self):
    def model_fn(features, labels, mode):
      _, _ = features, labels
      v = variables.Variable(21, name='some_var')
      scaffold = monitored_session.Scaffold(
          local_init_op=state_ops.assign_add(v, -3).op
      )
      return model_fn_lib.EstimatorSpec(
          mode,
          scaffold=scaffold,
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          loss=array_ops.identity(v))
    export_dir = self._export_estimator(predict=False, model_fn=model_fn)
    sme = saved_model_estimator.SavedModelEstimator(
        export_dir, self._get_tmp_dir())

    eval_results1 = sme.evaluate(dummy_input_fn, steps=2)
    self.assertEqual(15, eval_results1['loss'])

    sme.train(dummy_input_fn, steps=1)
    self.assertEqual(15, sme.get_variable_value('some_var'))

    eval_results2 = sme.evaluate(dummy_input_fn, steps=5)
    self.assertEqual(12, eval_results2['loss'])

  def test_with_working_input_fn(self):
    def model_fn(features, labels, mode):
      loss = None
      if labels is not None:
        loss = labels[0][0] + labels[1][0]
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=loss,
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions={'features_0': array_ops.identity([features['x'][0][0]]),
                       'features_1': array_ops.identity([features['x'][1][0]])})

    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(model_fn=model_fn), self._get_tmp_dir())
    eval_results = sme.evaluate(dummy_input_fn, steps=1)
    self.assertEqual(1, eval_results['loss'])

    predictions = next(sme.predict(dummy_input_fn_features_only))
    self.assertDictEqual({'features_0': 5, 'features_1': 6}, predictions)

  def test_control_dependency(self):
    # Control dependencies are saved with "^" appended to the start of the input
    # name. The input map must include control dependencies as well.
    def model_fn(features, labels, mode):
      _ = labels
      with ops.control_dependencies([features['x']]):
        loss = features['x'][1][0]
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=loss,
          train_op=state_ops.assign_add(training.get_global_step(), 1))
    sme = saved_model_estimator.SavedModelEstimator(
        self._export_estimator(train=False, predict=False, model_fn=model_fn),
        self._get_tmp_dir())
    sme.evaluate(dummy_input_fn, steps=1)  # Should run without error


if __name__ == '__main__':
  test.main()
