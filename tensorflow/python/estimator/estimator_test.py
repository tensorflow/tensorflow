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
"""Tests for Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import glob
import os
import tempfile

import numpy as np
import six

from google.protobuf import text_format

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.export import export_output
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.layers import layers
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.summary import summary
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_state_pb2
from tensorflow.python.training import saver
from tensorflow.python.training import saver_test_utils
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils

_TMP_DIR = '/tmp'
_ANOTHER_TMP_DIR = '/another_tmp'


def dummy_model_fn(features, labels, params):
  _, _, _ = features, labels, params


def summaries_with_matching_keyword(keyword, dir_):
  """Yields summary protos matching given keyword from event file."""

  writer_cache.FileWriterCache.clear()

  event_paths = glob.glob(os.path.join(dir_, 'events*'))
  for event in summary_iterator.summary_iterator(event_paths[-1]):
    if event.summary is not None:
      for value in event.summary.value:
        if keyword in value.tag:
          yield event.summary


def check_eventfile_for_keyword(keyword, dir_):
  """Checks event files for the keyword."""
  return any(summaries_with_matching_keyword(keyword, dir_))


def get_mock_saver():
  real_saver = saver.Saver()
  return test.mock.Mock(wraps=real_saver, saver_def=real_saver.saver_def)


class EstimatorInheritanceConstraintTest(test.TestCase):
  """Tests that sub classes cannot override methods of Estimator."""

  def test_override_a_method(self):
    class _Estimator(estimator.Estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def predict(self, input_fn, predict_keys=None, hooks=None):
        pass

    with self.assertRaisesRegexp(
        ValueError, 'cannot override members of Estimator.*predict'):
      _Estimator()

  def test_override_a_method_with_tricks(self):
    class _Estimator(estimator.Estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def _assert_members_are_not_overridden(self):
        pass  # HAHA! I tricked you!

      def predict(self, input_fn, predict_keys=None, hooks=None):
        pass

    with self.assertRaisesRegexp(
        ValueError, 'cannot override members of Estimator.*predict'):
      _Estimator()

  def test_extension_of_api_is_ok(self):
    class _Estimator(estimator.Estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def predict_proba(self, input_fn, predict_keys=None, hooks=None):
        pass

    _Estimator()

  def test_override_allowed_method(self):
    class _Estimator(estimator.Estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def _call_input_fn(self, input_fn, mode):
        return input_fn()

      def _create_global_step(self, graph):
        pass

      def _convert_train_steps_to_hooks(self, steps, max_steps):
        pass

      def _convert_eval_steps_to_hooks(self, steps):
        pass

    _Estimator()


class EstimatorConstructorTest(test.TestCase):

  def test_config_must_be_a_run_config(self):
    with self.assertRaisesRegexp(ValueError, 'an instance of RunConfig'):
      estimator.Estimator(model_fn=None, config='NotARunConfig')

  def test_model_fn_must_be_provided(self):
    with self.assertRaisesRegexp(ValueError, 'model_fn.* must be'):
      estimator.Estimator(model_fn=None)

  def test_property_accessors(self):

    def model_fn(features, labels, params):
      _, _, _ = features, labels, params

    class FakeConfig(run_config.RunConfig):
      pass

    params = {'hidden_layers': [3, 4]}
    est = estimator.Estimator(
        model_fn=model_fn, model_dir='bla', config=FakeConfig(), params=params)
    self.assertTrue(isinstance(est.config, FakeConfig))
    self.assertEqual(params, est.params)
    self.assertEqual('bla', est.model_dir)

  def test_default_config(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(model_fn=model_fn)
    self.assertTrue(isinstance(est.config, run_config.RunConfig))

  def test_default_model_dir(self):

    def model_fn(features, labels):
      _, _ = features, labels

    with test.mock.patch.object(tempfile, 'mkdtemp', return_value=_TMP_DIR):
      est = estimator.Estimator(model_fn=model_fn)
      self.assertEqual(_TMP_DIR, est.config.model_dir)
      self.assertEqual(_TMP_DIR, est.model_dir)

  def test_model_dir_in_constructor(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(model_fn=model_fn, model_dir=_TMP_DIR)
    self.assertEqual(_TMP_DIR, est.config.model_dir)
    self.assertEqual(_TMP_DIR, est.model_dir)

  def test_model_dir_in_run_config(self):

    class FakeConfig(run_config.RunConfig):

      @property
      def model_dir(self):
        return _TMP_DIR

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(model_fn=model_fn, config=FakeConfig())
    self.assertEqual(_TMP_DIR, est.config.model_dir)
    self.assertEqual(_TMP_DIR, est.model_dir)

  def test_same_model_dir_in_constructor_and_run_config(self):

    class FakeConfig(run_config.RunConfig):

      @property
      def model_dir(self):
        return _TMP_DIR

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(
        model_fn=model_fn, config=FakeConfig(), model_dir=_TMP_DIR)
    self.assertEqual(_TMP_DIR, est.config.model_dir)
    self.assertEqual(_TMP_DIR, est.model_dir)

  def test_different_model_dir_in_constructor_and_run_config(self):

    class FakeConfig(run_config.RunConfig):

      @property
      def model_dir(self):
        return _TMP_DIR

    def model_fn(features, labels):
      _, _ = features, labels

    with self.assertRaisesRegexp(
        ValueError,
        'model_dir are set both in constructor and RunConfig, but '
        'with different values'):
      estimator.Estimator(
          model_fn=model_fn, config=FakeConfig(), model_dir=_ANOTHER_TMP_DIR)

  def test_model_fn_args_must_include_features(self):

    def model_fn(x, labels):
      _, _ = x, labels

    with self.assertRaisesRegexp(ValueError, 'features'):
      estimator.Estimator(model_fn=model_fn)

  def test_model_fn_args_labels_is_optional(self):

    def model_fn(features):
      _ = features

    estimator.Estimator(model_fn=model_fn)

  def test_if_params_provided_then_model_fn_should_accept_it(self):

    def model_fn(features, labels):
      _, _ = features, labels

    estimator.Estimator(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'params'):
      estimator.Estimator(model_fn=model_fn, params={'hidden_layers': 4})

  def test_internal_params_is_a_deepcopy(self):

    def model_fn(features, labels, params):
      _, _, _ = features, labels, params

    params = {'hidden_layers': 4}
    est = estimator.Estimator(model_fn=model_fn, params=params)

    params['hidden_layers'] = 5
    self.assertEqual(4, est.params['hidden_layers'])

  def test_not_known_model_fn_args(self):

    def model_fn(features, labels, something):
      _, _, _ = features, labels, something

    with self.assertRaisesRegexp(ValueError, 'something'):
      estimator.Estimator(model_fn=model_fn)

  def test_not_known_model_fn_args_handled_by_lambda(self):
    def model_fn(features, labels, something):
      _, _, _ = features, labels, something

    new_model_fn = lambda features, labels: model_fn(  # pylint: disable=g-long-lambda
        features, labels, 'something')
    estimator.Estimator(model_fn=new_model_fn)

  def test_if_model_fn_is_a_member_function_of_a_class(self):

    class ModelFnClass(object):

      def __init__(self):
        estimator.Estimator(model_fn=self.model_fn)

      def model_fn(self, features, labels, mode):
        _, _, _ = features, labels, mode

    ModelFnClass()

  def test_model_fn_property_binds_params(self):

    def model_fn(features, labels, mode, config, params):
      _, _, _, _, _ = features, labels, mode, config, params

    est = estimator.Estimator(model_fn=model_fn)
    model_fn_args = function_utils.fn_args(est.model_fn)
    self.assertEqual(
        set(['features', 'labels', 'mode', 'config']), set(model_fn_args))

  def test_model_fn_property_returns_fixed_signature(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.Estimator(model_fn=model_fn)
    model_fn_args = function_utils.fn_args(est.model_fn)
    self.assertEqual(
        set(['features', 'labels', 'mode', 'config']), set(model_fn_args))


def dummy_input_fn():
  return ({'x': constant_op.constant([[1], [1]])},
          constant_op.constant([[1], [1]]))


def model_fn_global_step_incrementer(features, labels, mode):
  _, _ = features, labels
  global_step = training.get_global_step()
  return model_fn_lib.EstimatorSpec(
      mode,
      loss=constant_op.constant(1.),
      train_op=state_ops.assign_add(global_step, 1))


def assert_features_op(expected_features, actual_features):
  return [
      check_ops.assert_equal(
          expected_features[k], actual_features[k], name='assert_%s' % k)
      for k in expected_features
  ]


def _estimator_spec(
    expected_features, expected_labels, actual_features, actual_labels, mode):
  assert_ops = tuple(
      assert_features_op(expected_features, actual_features) + [
          check_ops.assert_equal(
              expected_labels, actual_labels, name='assert_labels')
      ])
  global_step = training.get_global_step()
  with ops.control_dependencies(assert_ops):
    return model_fn_lib.EstimatorSpec(
        mode=mode,
        predictions=constant_op.constant(0.),
        loss=constant_op.constant(0.),
        train_op=state_ops.assign_add(global_step, 1))


def _make_input_fn(features, labels):
  def _input_fn():
    return {
        k: constant_op.constant(v)
        for k, v in six.iteritems(features)
    }, constant_op.constant(labels)
  return _input_fn


class EstimatorTrainTest(test.TestCase):

  def test_callable_model_fn(self):
    expected_features = {'x': 42., 'y': 43.}
    expected_labels = 44.

    model_fn_call_count = [0]

    test_self = self

    class ModelFn(object):

      def __call__(self, features, labels):
        model_fn_call_count[0] += 1
        test_self.assertItemsEqual(expected_features.keys(), features.keys())
        return _estimator_spec(
            expected_features, expected_labels, features, labels,
            model_fn_lib.ModeKeys.TRAIN)

    with self.assertRaisesRegexp(ValueError, 'does not include params'):
      estimator.Estimator(model_fn=ModelFn(), params={'a': 'b'})
    est = estimator.Estimator(model_fn=ModelFn(), config=run_config.RunConfig())
    self.assertEqual(0, model_fn_call_count[0])
    est.train(
        input_fn=_make_input_fn(expected_features, expected_labels), steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_callable_input_fn(self):
    expected_mode = model_fn_lib.ModeKeys.TRAIN
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    test_self = self

    class InputFn(object):

      def __call__(self, mode, params, config):
        input_fn_call_count[0] += 1
        test_self.assertEqual(expected_mode, mode)
        test_self.assertEqual(expected_params, params)
        test_self.assertEqual(4321, config.tf_random_seed)
        return dummy_input_fn()

    est = estimator.Estimator(model_fn=_model_fn,
                              params=expected_params,
                              config=expected_config)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(InputFn(), steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_input_fn_args(self):
    expected_mode = model_fn_lib.ModeKeys.TRAIN
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    def _input_fn(mode, params, config):
      input_fn_call_count[0] += 1
      self.assertEqual(expected_mode, mode)
      self.assertEqual(expected_params, params)
      self.assertEqual(4321, config.tf_random_seed)
      return dummy_input_fn()

    est = estimator.Estimator(model_fn=_model_fn,
                              params=expected_params,
                              config=expected_config)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_minimal_model_fn_args(self):
    expected_features = {'x': 4, 'y': 5}

    def _input_fn():
      return expected_features

    model_fn_call_count = [0]
    def _model_fn(features):
      model_fn_call_count[0] += 1
      self.assertItemsEqual(expected_features.keys(), features.keys())
      with ops.control_dependencies(
          assert_features_op(expected_features, features)):
        return model_fn_lib.EstimatorSpec(
            mode=None,
            predictions=constant_op.constant(0.),
            loss=constant_op.constant(0.),
            train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    self.assertEqual(0, model_fn_call_count[0])
    est.train(input_fn=_input_fn, steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_labels_should_be_none_if_model_fn_does_not_use_labels(self):

    def _input_fn_with_labels():
      return {'x': 4, 'y': 5}, [4]

    def _model_fn(features):
      _ = features
      return model_fn_lib.EstimatorSpec(
          mode=None,
          predictions=constant_op.constant(0.),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    with self.assertRaisesRegexp(ValueError, 'model_fn does not take labels'):
      est.train(input_fn=_input_fn_with_labels, steps=1)

  def test_input_fn_len_should_be_2_if_tuple_or_list(self):

    def _input_fn():
      return 4, 5, 6

    def _model_fn(features):
      _ = features

    est = estimator.Estimator(model_fn=_model_fn)
    with self.assertRaisesRegexp(ValueError, 'len 2 tuple'):
      est.train(input_fn=_input_fn, steps=1)

  def test_all_model_fn_args(self):
    expected_features = {'x': 42., 'y': 43.}
    expected_labels = 44.
    expected_params = {'some_param': 'some_value'}
    expected_config = run_config.RunConfig()
    expected_config.i_am_test = True

    # TODO(ptucker): We have to roll our own mock since Estimator._get_arguments
    # doesn't work with mock fns.
    model_fn_call_count = [0]

    # Note that args are all passed by keyword, so can be in any order.
    def _model_fn(mode, params, features, labels, config):
      model_fn_call_count[0] += 1
      self.assertItemsEqual(expected_features.keys(), features.keys())
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertEqual(expected_params, params)
      self.assertTrue(config.i_am_test)
      return _estimator_spec(
          expected_features, expected_labels, features, labels, mode)

    est = estimator.Estimator(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    self.assertEqual(0, model_fn_call_count[0])
    est.train(
        input_fn=_make_input_fn(expected_features, expected_labels), steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_partial_model_fn_args(self):
    expected_features = {'x': 42., 'y': 43.}
    expected_labels = 44.
    expected_params = {'some_param': 'some_value'}
    expected_config = run_config.RunConfig()
    expected_config.i_am_test = True
    expected_foo = 45.
    expected_bar = 46.

    # TODO(ptucker): We have to roll our own mock since Estimator._get_arguments
    # doesn't work with mock fns.
    model_fn_call_count = [0]

    def _model_fn(features, labels, foo, mode, params, config, bar):
      model_fn_call_count[0] += 1
      self.assertEqual(expected_foo, foo)
      self.assertEqual(expected_bar, bar)
      self.assertItemsEqual(expected_features.keys(), features.keys())
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertEqual(expected_params, params)
      self.assertTrue(config.i_am_test)
      return _estimator_spec(
          expected_features, expected_labels, features, labels, mode)
    partial_model_fn = functools.partial(
        _model_fn, foo=expected_foo, bar=expected_bar)

    est = estimator.Estimator(
        model_fn=partial_model_fn, params=expected_params,
        config=expected_config)
    self.assertEqual(0, model_fn_call_count[0])
    est.train(
        input_fn=_make_input_fn(expected_features, expected_labels), steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_model_fn_must_return_estimator_spec(self):

    def model_fn(features, labels):
      _, _ = features, labels
      return 'NotGoodNotGood'

    est = estimator.Estimator(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'EstimatorSpec'):
      est.train(dummy_input_fn, steps=1)

  def test_run_train_op_and_saves_at_the_end(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_loss_summary(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer,
                              config=run_config.RunConfig(save_summary_steps=1))
    est.train(dummy_input_fn, steps=1)

    # Make sure nothing is stuck in limbo.
    writer_cache.FileWriterCache.clear()

    if check_eventfile_for_keyword('loss', est.model_dir):
      return
    self.fail('{} should be part of reported summaries.'.format('loss'))

  def test_latest_checkpoint(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    self.assertIsNone(est.latest_checkpoint())
    est.train(dummy_input_fn, steps=5)
    self.assertIsNotNone(est.latest_checkpoint())
    self.assertTrue(est.latest_checkpoint().startswith(est.model_dir))

  def test_steps_and_saves_reloads(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))
    est.train(dummy_input_fn, steps=5)
    self.assertEqual(
        10, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_warm_starts(self):
    def _make_model_fn(x):
      def _variable_creating_model_fn(features, labels, mode):
        _, _ = features, labels
        variable_scope.get_variable('x', initializer=x)
        global_step = training.get_global_step()
        return model_fn_lib.EstimatorSpec(
            mode,
            loss=constant_op.constant(1.),
            train_op=state_ops.assign_add(global_step, 1))
      return _variable_creating_model_fn

    est = estimator.Estimator(model_fn=_make_model_fn(42.))
    est.train(dummy_input_fn, steps=10)

    warm_started_est = estimator.Estimator(
        model_fn=_make_model_fn(36.),
        warm_start_from=est.model_dir)
    warm_started_est.train(dummy_input_fn, steps=5)
    # warm_start is called after the model_fn, so x should have the value
    # from the checkpoint.
    self.assertEqual(42., warm_started_est.get_variable_value('x'))
    # global_step should not be warm-started.
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(
            warm_started_est.model_dir))

  def test_warm_starts_from_savedmodel(self):
    def _make_model_fn(x):
      def _variable_creating_and_export_model_fn(features, labels, mode):
        _, _ = features, labels
        variable_scope.get_variable('x', initializer=x)
        global_step = training.get_global_step()
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions={'y': constant_op.constant(1.0)},
            loss=constant_op.constant(1.),
            train_op=state_ops.assign_add(global_step, 1),
            export_outputs={'test': export_output.ClassificationOutput(
                constant_op.constant([4.2]), constant_op.constant(['label']))})
      return _variable_creating_and_export_model_fn

    est = estimator.Estimator(model_fn=_make_model_fn(42.))
    est.train(dummy_input_fn, steps=10)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(
        export_dir_base, serving_input_receiver_fn)

    warm_started_est = estimator.Estimator(
        model_fn=_make_model_fn(36.),
        warm_start_from=export_dir)
    warm_started_est.train(dummy_input_fn, steps=5)
    # warm_start is called after the model_fn, so x should have the value
    # from the SavedModel.
    self.assertEqual(42., warm_started_est.get_variable_value('x'))

  def test_max_step(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, max_steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))
    est.train(dummy_input_fn, max_steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_checkpoint_contains_relative_paths(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(
        model_dir=tmpdir,
        model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=5)

    checkpoint_file_content = file_io.read_file_to_string(
        os.path.join(tmpdir, 'checkpoint'))
    ckpt = checkpoint_state_pb2.CheckpointState()
    text_format.Merge(checkpoint_file_content, ckpt)
    self.assertEqual(ckpt.model_checkpoint_path, 'model.ckpt-5')
    # TODO(b/78461127): Please modify tests to not directly rely on names of
    # checkpoints.
    self.assertAllEqual(
        ['model.ckpt-0', 'model.ckpt-5'], ckpt.all_model_checkpoint_paths)

  def test_train_save_copy_reload(self):
    tmpdir = tempfile.mkdtemp()
    model_dir1 = os.path.join(tmpdir, 'model_dir1')
    est1 = estimator.Estimator(
        model_dir=model_dir1,
        model_fn=model_fn_global_step_incrementer)
    est1.train(dummy_input_fn, steps=5)

    # We have to clear the cache before we can rename the directory,
    # otherwise open file handles will prevent the delete on Windows.
    writer_cache.FileWriterCache.clear()
    model_dir2 = os.path.join(tmpdir, 'model_dir2')
    os.renames(model_dir1, model_dir2)

    est2 = estimator.Estimator(
        model_dir=model_dir2,
        model_fn=model_fn_global_step_incrementer)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est2.model_dir))
    est2.train(dummy_input_fn, steps=5)
    self.assertEqual(
        10, estimator._load_global_step_from_checkpoint_dir(est2.model_dir))

  def test_steps0_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.train(dummy_input_fn, steps=0)

  def test_steps_negative_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.train(dummy_input_fn, steps=-1)

  def test_max_steps0_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify max_steps > 0'):
      est.train(dummy_input_fn, max_steps=0)

  def test_max_steps_negative_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify max_steps > 0'):
      est.train(dummy_input_fn, max_steps=-1)

  def test_scaffold_is_used(self):
    self.is_init_fn_called = False

    def _init_fn(scaffold, sess):
      _, _ = scaffold, sess
      self.is_init_fn_called = True

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=training.Scaffold(init_fn=_init_fn))

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    self.assertTrue(self.is_init_fn_called)

  def test_hooks_should_be_session_run_hook(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    with self.assertRaisesRegexp(TypeError, 'must be a SessionRunHook'):
      est.train(dummy_input_fn, steps=1, hooks=['NotAHook'])

  def test_training_hooks_are_used(self):
    chief_hook = test.mock.MagicMock(
        wraps=training.SessionRunHook(), spec=training.SessionRunHook)
    hook = test.mock.MagicMock(
        wraps=training.SessionRunHook(), spec=training.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          training_chief_hooks=[chief_hook],
          training_hooks=[hook])

    est = estimator.Estimator(model_fn=_model_fn_hooks)
    self.assertFalse(chief_hook.begin.called)
    self.assertFalse(hook.begin.called)
    est.train(dummy_input_fn, steps=1)
    self.assertTrue(chief_hook.begin.called)
    self.assertTrue(hook.begin.called)

  def test_saving_listeners_are_used(self):
    listener = test.mock.Mock(spec=training.CheckpointSaverListener)
    listener.after_save.return_value = None
    est = estimator.Estimator(
        model_fn=model_fn_global_step_incrementer,
        config=run_config.RunConfig(save_checkpoints_steps=10))
    est.train(dummy_input_fn, steps=26, saving_listeners=[listener])
    self.assertEqual(4, listener.before_save.call_count)
    self.assertEqual(4, listener.after_save.call_count)

  def test_saver_hook_should_exist_to_use_saving_listeners(self):
    listener = test.mock.Mock(spec=training.CheckpointSaverListener)
    est = estimator.Estimator(
        model_fn=model_fn_global_step_incrementer,
        config=run_config.RunConfig(save_checkpoints_steps=None,
                                    save_checkpoints_secs=None))
    with self.assertRaisesRegexp(
        ValueError, 'CheckpointSaverHook to use saving_listeners'):
      est.train(dummy_input_fn, steps=1, saving_listeners=[listener])

  def test_listeners_should_be_listeners(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    with self.assertRaisesRegexp(
        TypeError, 'must be a list of CheckpointSaverListener'):
      est.train(dummy_input_fn, steps=1, saving_listeners=['not-a-listener'])

  def test_chief_only_hook_should_not_be_called_on_non_chief(self):
    chief_hook = test.mock.MagicMock(
        wraps=training.SessionRunHook(), spec=training.SessionRunHook)
    hook = test.mock.MagicMock(
        wraps=training.SessionRunHook(), spec=training.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          training_chief_hooks=[chief_hook],
          training_hooks=[hook])

    class NonChiefRunConfig(run_config.RunConfig):
      @property
      def is_chief(self):  # pylint: disable=g-wrong-blank-lines
        return False

    # Mocking the SessionManager.wait_for_session, so that worker doesn't wait
    # for chief.
    def get_initialized_session(*args, **kwargs):
      # Session doesn't take 'max_wait_secs' argument.
      kwargs.pop('max_wait_secs', None)
      scaffold = training.Scaffold().finalize()
      sess = session.Session(*args, **kwargs)
      sess.run(scaffold.init_op)
      return sess

    with test.mock.patch.object(
        training.SessionManager,
        'wait_for_session',
        side_effect=get_initialized_session):
      est = estimator.Estimator(
          model_fn=_model_fn_hooks, config=NonChiefRunConfig())
      self.assertFalse(chief_hook.begin.called)
      self.assertFalse(hook.begin.called)
      est.train(dummy_input_fn, steps=1)
      self.assertFalse(chief_hook.begin.called)
      self.assertTrue(hook.begin.called)

  def test_features_labels_mode(self):
    given_features = {'test-features': [[1], [1]]}
    given_labels = {'test-labels': [[1], [1]]}

    def _input_fn():
      return given_features, given_labels

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    self.assertEqual(given_features, self.features)
    self.assertEqual(given_labels, self.labels)
    self.assertEqual(model_fn_lib.ModeKeys.TRAIN, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed
    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(training.get_global_step())
      self.assertEqual(expected_random_seed, ops.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)


def _model_fn_with_eval_metric_ops(features, labels, mode, params):
  _, _ = features, labels
  metric_name = params.get('metric_name') or 'metric'
  metric_value = params.get('metric_value') or 2.
  global_step = training.get_global_step()
  loss = constant_op.constant(1.)
  metric_update_op = loss.op
  metric_tensor = control_flow_ops.with_dependencies(
      [metric_update_op], constant_op.constant(metric_value))
  return model_fn_lib.EstimatorSpec(
      mode,
      loss=loss,
      predictions={'predictions': constant_op.constant(1.)},
      train_op=state_ops.assign_add(global_step, 1),
      eval_metric_ops={metric_name: (metric_tensor, metric_update_op)})


class _StepCounterHook(session_run_hook.SessionRunHook):
  """Hooks that counts the number of times it is called."""

  def __init__(self):
    self._steps = 0

  def before_run(self, run_context):
    del run_context
    self._steps += 1

  @property
  def steps(self):
    return self._steps


class EstimatorGetVariablesTest(test.TestCase):

  def test_model_should_be_trained(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='one')
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    with self.assertRaisesRegexp(ValueError, 'not find trained model'):
      est.get_variable_names()
    with self.assertRaisesRegexp(ValueError, 'not find trained model'):
      est.get_variable_value('one')

  def test_get_variable_utils(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='one')
      variables.Variable(3., name='three')
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(input_fn=dummy_input_fn, steps=1)
    self.assertEqual(
        set(['one', 'three', 'global_step']), set(est.get_variable_names()))
    self.assertEqual(1., est.get_variable_value('one'))
    self.assertEqual(3., est.get_variable_value('three'))


class EstimatorDatasetIntegrationTest(test.TestCase):
  """Tests dataset integration."""

  def test_returned_by_input_fn(self):

    def _input_fn():
      return dataset_ops.Dataset.from_tensors(([1.], [2.]))

    def _model_fn(features, labels, mode):
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=features + labels,  # 1 + 2
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    scores = est.evaluate(_input_fn, steps=1)
    self.assertEqual(3., scores[model_fn_lib.LOSS_METRIC_KEY])

  def test_with_none_labels(self):

    def _input_fn():
      return dataset_ops.Dataset.from_tensors([7.])

    def _model_fn(features, labels, mode):
      self.assertIsNone(labels)
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=features,  # 7
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    scores = est.evaluate(_input_fn, steps=1)
    self.assertEqual(7., scores[model_fn_lib.LOSS_METRIC_KEY])

  def test_with_predict(self):

    def _input_fn():
      return dataset_ops.Dataset.from_tensors([10.])

    def _model_fn(features, labels, mode):
      _ = labels
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=features,  # 10
          loss=features,  # 10
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    self.assertEqual([10.], next(est.predict(input_fn=_input_fn)))

  def test_batching(self):

    def _input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([[1.], [2.]],
                                                     [[10.], [20.]])).batch(1)

    def _model_fn(features, labels, mode):
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=features,
          loss=features + (0 if labels is None else labels),  # 11, 22
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn)
    scores = est.evaluate(_input_fn)
    # (11 + 22)/2 = 16.5
    self.assertEqual(16.5, scores[model_fn_lib.LOSS_METRIC_KEY])
    self.assertEqual([1., 2.], list(est.predict(_input_fn)))


class EstimatorEvaluateTest(test.TestCase):

  def test_eval_dir(self):
    est = estimator.Estimator(
        model_fn=model_fn_global_step_incrementer,
        model_dir='some_path')
    expected_eval_dir = os.path.join('some_path', 'eval')
    self.assertEqual(expected_eval_dir, est.eval_dir())
    expected_eval_dir_name = os.path.join('some_path', 'eval_a_name')
    self.assertEqual(expected_eval_dir_name, est.eval_dir('a_name'))

  def test_input_fn_args(self):
    expected_mode = model_fn_lib.ModeKeys.EVAL
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    def _input_fn(mode, params, config):
      input_fn_call_count[0] += 1
      self.assertEqual(expected_mode, mode)
      self.assertEqual(expected_params, params)
      self.assertEqual(4321, config.tf_random_seed)
      return dummy_input_fn()

    est = estimator.Estimator(model_fn=_model_fn,
                              params=expected_params,
                              config=expected_config)
    est.train(dummy_input_fn, steps=1)
    self.assertEqual(0, input_fn_call_count[0])
    est.evaluate(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_model_fn_must_return_estimator_spec(self):
    def _model_fn(features, labels, mode):
      _, _ = features, labels
      if mode == model_fn_lib.ModeKeys.EVAL:
        return 'NotGoodNotGood'
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(1.),
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(
        ValueError, 'model_fn should return an EstimatorSpec'):
      est.evaluate(dummy_input_fn, steps=1)

  def test_no_checkpoint_uses_init(self):
    def _model_fn(features, labels, mode, params):
      del features, labels, params
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(1.),
          eval_metric_ops={'metric': metrics_lib.mean(
              variables.Variable(2.) + 1)})
    est = estimator.Estimator(model_fn=_model_fn)
    metrics = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is newly
    # initialized (since there is no checkpoint).
    self.assertEqual(3., metrics['metric'])

  def test_no_checkpoint_uses_init_with_warm_starting(self):
    def _make_model_fn(x):
      def _variable_creating_and_export_model_fn(features, labels, mode):
        _, _ = features, labels
        x_var = variable_scope.get_variable('x', initializer=x)
        global_step = training.get_global_step()
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions={'y': constant_op.constant(1.0)},
            loss=constant_op.constant(1.),
            eval_metric_ops={'metric': metrics_lib.mean(x_var + 1)},
            train_op=state_ops.assign_add(global_step, 1),
            export_outputs={'test': export_output.ClassificationOutput(
                constant_op.constant([4.2]), constant_op.constant(['label']))})
      return _variable_creating_and_export_model_fn

    first_est = estimator.Estimator(model_fn=_make_model_fn(42.))
    first_est.train(dummy_input_fn, steps=10)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    exported_path = first_est.export_savedmodel(export_dir_base,
                                                serving_input_receiver_fn)

    # Test that we can pass either warm_start_from as an external checkpoint
    # or an exported SavedModel.
    est = estimator.Estimator(model_fn=_make_model_fn(52.),
                              warm_start_from=exported_path)
    metrics = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is
    # warm-started from the SavedModel of the first model (42.), as opposed to
    # the initialization in the new model_fn (52.).
    self.assertEqual(43., metrics['metric'])

    est = estimator.Estimator(model_fn=_make_model_fn(62.),
                              warm_start_from=first_est.model_dir)
    metrics = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is
    # warm-started from a checkpoint of the first model (42.), as opposed to
    # the initialization in the new model_fn (52.).
    self.assertEqual(43., metrics['metric'])

  def test_scores(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops,
        params={
            'metric_name': 'metric',
            'metric_value': 2.})
    est.train(dummy_input_fn, steps=5)
    scores = est.evaluate(dummy_input_fn, steps=1)
    self.assertIn('metric', scores)
    self.assertAlmostEqual(2., scores['metric'])

  def test_tuple_metrics(self):
    def _model_fn(features, labels, mode):
      del features  # unused
      del labels
      return model_fn_lib.EstimatorSpec(
          mode,
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          loss=constant_op.constant(1.),
          eval_metric_ops={
              'nested_metric': (
                  ((constant_op.constant(2.), constant_op.constant(1)),
                   constant_op.constant(3., dtype=dtypes.float64)),
                  control_flow_ops.no_op())})
    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    evaluation = est.evaluate(dummy_input_fn, steps=1)
    ((two_float, one_integer), three_double) = evaluation['nested_metric']
    self.assertAlmostEqual(2., two_float)
    self.assertEqual(1, one_integer)
    self.assertAlmostEqual(3., three_double)

  def test_steps0_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops)
    est.train(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.evaluate(dummy_input_fn, steps=0)

  def test_steps_negative_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops)
    est.train(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.evaluate(dummy_input_fn, steps=-1)

  def test_global_step_metric_raises_error(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops,
        params={
            'metric_name': 'global_step',
            'metric_value': 2.})
    est.train(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(
        ValueError, 'Metric with name `global_step` is not allowed'):
      est.evaluate(dummy_input_fn, steps=1)

  def test_global_step_is_reported(self):
    est = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops,
        params={'metric_name': 'metric',
                'metric_value': 2.})
    est.train(dummy_input_fn, steps=5)
    scores = est.evaluate(dummy_input_fn, steps=1)
    self.assertIn('global_step', scores)
    self.assertEqual(5, scores['global_step'])

  def test_loss_metric_is_reported(self):

    def _model_fn_with_incremental_loss(features, labels, mode):
      _, _ = features, labels
      local_weight = variables.Variable(
          0., name='local_weight', collections=[ops.GraphKeys.LOCAL_VARIABLES])
      # Loss will be 2, 4, 6, ...
      loss = 2 * state_ops.assign_add(local_weight, 1.)
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=loss,
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    est = estimator.Estimator(model_fn=_model_fn_with_incremental_loss)
    est.train(dummy_input_fn, steps=1)
    scores = est.evaluate(dummy_input_fn, steps=5)
    self.assertIn(model_fn_lib.LOSS_METRIC_KEY, scores)
    # Average loss will be (2 + 4 + 6 + 8 + 10)/5=6
    self.assertAlmostEqual(6., scores[model_fn_lib.LOSS_METRIC_KEY])

  def test_hooks_should_be_session_run_hook(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(TypeError, 'must be a SessionRunHook'):
      est.evaluate(dummy_input_fn, steps=5, hooks=['NotAHook'])

  def test_hooks_are_used(self):
    step_counter_hook = _StepCounterHook()

    est = estimator.Estimator(model_fn=_model_fn_with_eval_metric_ops)
    est.train(dummy_input_fn, steps=1)
    est.evaluate(dummy_input_fn, steps=5, hooks=[step_counter_hook])
    self.assertEqual(5, step_counter_hook.steps)

  def test_evaluate_from_checkpoint(self):
    params = {
        'metric_name': 'metric',
        'metric_value': 2.}
    est1 = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops,
        params=params)
    est1.train(dummy_input_fn, steps=5)
    est2 = estimator.Estimator(
        model_fn=_model_fn_with_eval_metric_ops,
        params=params)
    scores = est2.evaluate(
        dummy_input_fn, steps=1, checkpoint_path=est1.latest_checkpoint())
    self.assertEqual(5, scores['global_step'])

  def test_wrong_shape_throws_reasonable_error(self):
    """Make sure we are helpful when model_fns change. See b/110263146."""
    def _get_model_fn(val=1):
      def _model_fn(features, labels, mode):
        del features, labels  # unused
        variables.Variable(val, name='weight')
        return model_fn_lib.EstimatorSpec(
            mode=mode,
            predictions=constant_op.constant([[1.]]),
            loss=constant_op.constant(0.),
            train_op=state_ops.assign_add(training.get_global_step(), 1))
      return _model_fn

    model_fn_1 = _get_model_fn()
    model_fn_2 = _get_model_fn(val=[1])

    est1 = estimator.Estimator(model_fn=model_fn_1)
    est1.train(dummy_input_fn, steps=5)
    est2 = estimator.Estimator(
        model_fn=model_fn_2, model_dir=est1.model_dir)

    expected_msg = 'Restoring from checkpoint failed.*a mismatch between'
    with self.assertRaisesRegexp(errors.InvalidArgumentError, expected_msg):
      est2.train(dummy_input_fn, steps=1,)

  def test_scaffold_is_used(self):

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='weight')
      self.mock_saver = get_mock_saver()
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=constant_op.constant([[1.]]),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=training.Scaffold(saver=self.mock_saver))

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    est.evaluate(dummy_input_fn, steps=1)
    self.assertTrue(self.mock_saver.restore.called)

  def test_features_labels_mode(self):
    given_features = {'test-features': [[1], [1]]}
    given_labels = {'test-labels': [[1], [1]]}

    def _input_fn():
      return given_features, given_labels

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    est.evaluate(_input_fn, steps=1)
    self.assertEqual(given_features, self.features)
    self.assertEqual(given_labels, self.labels)
    self.assertEqual(model_fn_lib.ModeKeys.EVAL, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed
    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(training.get_global_step())
      self.assertEqual(expected_random_seed, ops.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    est.evaluate(dummy_input_fn, steps=1)

  def test_evaluation_hooks_are_used(self):
    hook = test.mock.MagicMock(
        wraps=training.SessionRunHook(), spec=training.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          evaluation_hooks=[hook])

    est = estimator.Estimator(model_fn=_model_fn_hooks)
    est.train(dummy_input_fn, steps=1)
    self.assertFalse(hook.begin.called)
    est.evaluate(dummy_input_fn, steps=1)
    self.assertTrue(hook.begin.called)

  def test_summary_writing_with_summary_proto(self):

    def model_fn_global_step_incrementer_image(features, labels, mode):
      _, _ = features, labels
      global_step = training.get_global_step()

      image = array_ops.zeros([5, 3, 3, 1])
      eval_metric_ops = {
          'foo': (summary.image('image', image, max_outputs=3),
                  constant_op.constant(1))
      }
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(1.),
          train_op=state_ops.assign_add(global_step, 1),
          eval_metric_ops=eval_metric_ops)

    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer_image,
                              config=run_config.RunConfig(save_summary_steps=1))
    est.train(dummy_input_fn, steps=200)
    est.evaluate(
        input_fn=dummy_input_fn,
        steps=200,
    )

    # Make sure nothing is stuck in limbo.
    writer_cache.FileWriterCache.clear()

    # Get last evaluation Event written.
    for key in ['foo/0', 'foo/1', 'foo/2']:
      self.assertTrue(
          check_eventfile_for_keyword(key, est.eval_dir()),
          '{} should be part of reported summaries.'.format(key))

    # Verify that evaluated checkpoint path is written to event file.
    checkpoint_path_tag = 'checkpoint_path'
    self.assertTrue(
        check_eventfile_for_keyword(checkpoint_path_tag, est.eval_dir()),
        '{} should be part of reported summaries.'.format(checkpoint_path_tag))

    expected_tensor_proto = tensor_util.make_tensor_proto(
        est.latest_checkpoint(), dtype=dtypes.string)
    summaries = summaries_with_matching_keyword(checkpoint_path_tag,
                                                est.eval_dir())
    self.assertProtoEquals(expected_tensor_proto,
                           next(summaries).value[0].tensor)


class EstimatorPredictTest(test.TestCase):

  def test_input_fn_args(self):
    expected_mode = model_fn_lib.ModeKeys.PREDICT
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del features, labels, params, config
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    def _input_fn(mode, params, config):
      input_fn_call_count[0] += 1
      self.assertEqual(expected_mode, mode)
      self.assertEqual(expected_params, params)
      self.assertEqual(4321, config.tf_random_seed)
      return dummy_input_fn()

    est = estimator.Estimator(model_fn=_model_fn,
                              params=expected_params,
                              config=expected_config)
    est.train(dummy_input_fn, steps=1)
    self.assertEqual(0, input_fn_call_count[0])
    next(est.predict(_input_fn))
    self.assertEqual(1, input_fn_call_count[0])

  def test_no_checkpoint_uses_init(self):
    def _model_fn(features, labels, mode, params, config):
      del features, labels, params, config
      x = variables.Variable([[3.]], name='x')
      return model_fn_lib.EstimatorSpec(mode, predictions=math_ops.add(x, 1.))
    est = estimator.Estimator(model_fn=_model_fn)
    # Expected prediction value is 1 + the value of the Variable that is newly
    # initialized (since there is no checkpoint).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

  def test_no_checkpoint_uses_init_with_warm_starting(self):
    def _make_model_fn(x):
      def _variable_creating_and_export_model_fn(features, labels, mode):
        _, _ = features, labels
        x_var = variables.Variable([[x]], name='x')
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions=math_ops.add(x_var, 1.),
            loss=constant_op.constant(1.),
            train_op=state_ops.assign_add(training.get_global_step(), 1),
            export_outputs={'test': export_output.ClassificationOutput(
                constant_op.constant([4.2]),
                constant_op.constant(['label']))})
      return _variable_creating_and_export_model_fn

    first_est = estimator.Estimator(model_fn=_make_model_fn(3.))
    first_est.train(dummy_input_fn, steps=10)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    exported_path = first_est.export_savedmodel(export_dir_base,
                                                serving_input_receiver_fn)

    # Test that we can pass either warm_start_from as an external checkpoint
    # or an exported SavedModel.
    est = estimator.Estimator(model_fn=_make_model_fn(30.),
                              warm_start_from=exported_path)
    # Prediction here is set to 1 + the value of the Variable that is
    # warm-started from the SavedModel of the first model (3.), as opposed to
    # the initialization in the new model_fn (30.).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

    est = estimator.Estimator(model_fn=_make_model_fn(40.),
                              warm_start_from=first_est.model_dir)
    # Prediction here is set to 1 + the value of the Variable that is
    # warm-started from a checkpoint of the first model (3.), as opposed to
    # the initialization in the new model_fn (40.).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

  def test_no_trained_model_invalid_checkpoint_path(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    with self.assertRaises(ValueError):
      next(
          est.predict(
              dummy_input_fn,
              checkpoint_path=saver.latest_checkpoint('fakedir')))

  def test_tensor_predictions(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    self.assertEqual(10., next(est.predict(dummy_input_fn)))

  def test_predictionhooks_are_used(self):
    hook = test.mock.MagicMock(
        wraps=training.SessionRunHook(), spec=training.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]),
          prediction_hooks=[hook])

    est = estimator.Estimator(model_fn=_model_fn_hooks)
    est.train(dummy_input_fn, steps=1)
    self.assertFalse(hook.begin.called)
    next(est.predict(dummy_input_fn))
    self.assertTrue(hook.begin.called)

  def test_warn_if_no_queue_runner(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with test.mock.patch.object(logging, 'warning') as mock_log:
      next(est.predict(dummy_input_fn))
      self.assertRegexpMatches(
          str(mock_log.call_args),
          'Input graph does not.*contain a QueueRunner.')

  def test_skip_warn_if_dataset_returns_features(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    def _input_fn():
      it = dataset_ops.Dataset.from_tensors([1]).make_one_shot_iterator()
      return it.get_next()

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with test.mock.patch.object(logging, 'warning') as mock_log:
      next(est.predict(_input_fn))
      # The warning should not have keyword QueueRunner.
      self.assertRegexpMatches(str(mock_log.call_args), '^((?!QueueRunner).)*$')

  def test_skip_warn_if_dataset_returns_features_dict(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    def _input_fn():
      it = dataset_ops.Dataset.from_tensors([1]).make_one_shot_iterator()
      features = {'age': it.get_next()}
      return features

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with test.mock.patch.object(logging, 'warning') as mock_log:
      next(est.predict(_input_fn))
      # The warning should not have keyword QueueRunner.
      self.assertRegexpMatches(str(mock_log.call_args), '^((?!QueueRunner).)*$')

  def test_input_fn_can_return_just_features(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)

    def _only_features():
      return {'x': constant_op.constant([[0.]])}

    self.assertEqual([10.], next(est.predict(_only_features)))

  def test_batch_size_mismatch(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions={
              'y1': constant_op.constant([[10.]]),
              'y2': constant_op.constant([[12.], [13]])
          })

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError,
                                 'Batch length of predictions should be same'):
      next(est.predict(dummy_input_fn))

  def test_iterate_batches(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions={
              # First dim is different but the prediction should still work
              'y1': array_ops.zeros(shape=[3]),
              'y2': array_ops.zeros(shape=[5, 3])
          })

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)

    predictions = next(est.predict(dummy_input_fn, yield_single_examples=False))
    self.assertAllEqual(predictions['y1'].shape, [3])
    self.assertAllEqual(predictions['y2'].shape, [5, 3])

  def test_predict_keys_defined_for_tensor(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(
        ValueError,
        'predict_keys argument is not valid in case of non-dict predictions'):
      next(est.predict(dummy_input_fn, predict_keys=['y']))

  def test_predict_keys_does_not_exists(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions={
              'y1': constant_op.constant([[10.]]),
              'y2': constant_op.constant([[12.]])
          })

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError,
                                 'Expected to run at least one output from'):
      next(est.predict(dummy_input_fn, predict_keys=['y3']))

  def test_return_given_predict_keys(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions={
              'y1': constant_op.constant([[10.]]),
              'y2': constant_op.constant([[12.]])
          })

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = next(est.predict(dummy_input_fn, predict_keys=['y1']))
    self.assertIn('y1', results)
    self.assertNotIn('y2', results)

  def test_yield_rows_of_tensor(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.], [12.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = est.predict(dummy_input_fn)
    self.assertEqual([10.], next(results))
    self.assertEqual([12.], next(results))

  def test_yield_rows_of_dict(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions={
              'y1': constant_op.constant([[10.], [12]]),
              'y2': constant_op.constant([[0.], [2.]])
          })

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = est.predict(dummy_input_fn)
    self.assertDictEqual({'y1': [10.], 'y2': [0.]}, next(results))
    self.assertDictEqual({'y1': [12.], 'y2': [2.]}, next(results))

  def test_hooks_should_be_session_run_hook(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(TypeError, 'must be a SessionRunHook'):
      next(est.predict(dummy_input_fn, hooks=['NotAHook']))

  def test_hooks_are_used(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[10.], [12.]]))

    step_counter_hook = _StepCounterHook()
    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = est.predict(dummy_input_fn, hooks=[step_counter_hook])
    self.assertEqual(0, step_counter_hook.steps)  # not called yet
    next(results)
    self.assertEqual(1, step_counter_hook.steps)  # first call
    next(results)
    self.assertEqual(1, step_counter_hook.steps)  # it's in same batch
    next(results)
    self.assertEqual(2, step_counter_hook.steps)  # next batch

  def test_predict_from_old_model_dir(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      v = variables.Variable([[16.]], name='weight')
      prediction = v * 2
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=prediction)

    est1 = estimator.Estimator(model_fn=_model_fn)
    est1.train(dummy_input_fn, steps=1)
    est2 = estimator.Estimator(model_fn=_model_fn, model_dir=est1.model_dir)
    self.assertEqual([32.], next(est2.predict(dummy_input_fn)))

  def test_predict_from_checkpoint_path(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      v = variables.Variable([[16.]], name='weight')
      prediction = v * 2
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=prediction)

    est1 = estimator.Estimator(model_fn=_model_fn)
    est1.train(dummy_input_fn, steps=1)
    est2 = estimator.Estimator(model_fn=_model_fn, model_dir=est1.model_dir)
    self.assertEqual([32.],
                     next(
                         est2.predict(
                             dummy_input_fn,
                             checkpoint_path=est2.latest_checkpoint())))

  def test_scaffold_is_used(self):

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='weight')
      self.mock_saver = get_mock_saver()
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=constant_op.constant([[1.]]),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=training.Scaffold(saver=self.mock_saver))

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    next(est.predict(dummy_input_fn))
    self.assertTrue(self.mock_saver.restore.called)

  def test_features_labels_mode(self):
    given_features = {'test-features': [[1], [1]]}
    given_labels = {'test-labels': [[1], [1]]}

    def _input_fn():
      return given_features, given_labels

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    next(est.predict(_input_fn))
    self.assertEqual(given_features, self.features)
    self.assertIsNone(self.labels)
    self.assertEqual(model_fn_lib.ModeKeys.PREDICT, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed
    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(training.get_global_step())
      self.assertEqual(expected_random_seed, ops.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    next(est.predict(dummy_input_fn))


def _model_fn_for_export_tests(features, labels, mode):
  _, _ = features, labels
  variables.Variable(1., name='weight')
  scores = constant_op.constant([3.])
  classes = constant_op.constant(['wumpus'])
  update_global_step = state_ops.assign_add(training.get_global_step(), 1)
  with ops.control_dependencies([update_global_step]):
    train_op = constant_op.constant(2.)
  return model_fn_lib.EstimatorSpec(
      mode,
      predictions=constant_op.constant(10.),
      loss=constant_op.constant(1.),
      train_op=train_op,
      export_outputs={
          'test': export_output.ClassificationOutput(scores, classes)})


def _x_y_input_fn():
  return ({'x': constant_op.constant([[1], [1]]),
           'y': constant_op.constant([[2], [2]])},
          constant_op.constant([[1], [1]]))


def _model_fn_with_x_y(features, labels, mode):
  _ = labels
  variables.Variable(1., name='weight')
  scores = constant_op.constant([3.])
  classes = constant_op.constant(['wumpus'])
  if mode == model_fn_lib.ModeKeys.PREDICT:
    variables.Variable(36., name='name_collision')
    return model_fn_lib.EstimatorSpec(
        mode,
        predictions=constant_op.constant(10.),
        export_outputs={
            'test': export_output.ClassificationOutput(scores, classes)})
  else:
    prefix = 'eval_' if mode == model_fn_lib.ModeKeys.EVAL else ''

    multiplied = math_ops.multiply(
        features['x'], features['y'], name='{}multiplied'.format(prefix))
    metrics = {'mean': metrics_lib.mean(features['x'] - features['y'],
                                        name='{}mean'.format(prefix))}
    variables.Variable(1., name='later_var')
    variables.Variable(3., name='name_collision')
    return model_fn_lib.EstimatorSpec(
        mode,
        predictions=multiplied,
        loss=constant_op.constant(1.),
        train_op=state_ops.assign_add(training.get_global_step(), 1),
        eval_metric_ops=metrics)


def _model_fn_with_saveables_for_export_tests(features, labels, mode):
  _, _ = features, labels
  table = saver_test_utils.CheckpointedOp(name='v2')
  update_global_step = state_ops.assign_add(training.get_global_step(), 1)
  with ops.control_dependencies([update_global_step]):
    train_op = table.insert('k1', 30.0)
  prediction = table.lookup('k1', 0.0)
  return model_fn_lib.EstimatorSpec(
      mode,
      predictions=prediction,
      loss=constant_op.constant(1.),
      train_op=train_op,
      export_outputs={
          'test': export_output.PredictOutput({'prediction': prediction})})


def _get_serving_input_receiver_fn():
  feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                  'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
  return export.build_parsing_serving_input_receiver_fn(feature_spec)


def _get_supervised_input_receiver_fn():
  feature_spec = {
      'x': array_ops.placeholder(
          dtype=dtypes.int64, shape=(2, 1), name='feature_x'),
      'y': array_ops.placeholder(
          dtype=dtypes.int64, shape=(2, 1), name='feature_y')
      }
  label_spec = array_ops.placeholder(
      dtype=dtypes.float32, shape=[1], name='truth')

  return export.build_raw_supervised_input_receiver_fn(feature_spec, label_spec)


_VOCAB_FILE_CONTENT = 'emerson\nlake\npalmer\n'
_EXTRA_FILE_CONTENT = 'kermit\npiggy\nralph\n'


class EstimatorExportTest(test.TestCase):

  def test_export_savedmodel_proto_roundtrip_raw_receiver(self):
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(
        export_dir_base, serving_input_receiver_fn)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExample' in graph_ops)
        self.assertTrue('weight' in graph_ops)

  def test_export_saved_model_train(self):
    self._test_export_saved_model_for_mode(
        _get_supervised_input_receiver_fn(), model_fn_lib.ModeKeys.TRAIN)

  def test_export_saved_model_eval(self):
    self._test_export_saved_model_for_mode(
        _get_supervised_input_receiver_fn(), model_fn_lib.ModeKeys.EVAL)

  def test_export_saved_model_predict(self):
    self._test_export_saved_model_for_mode(
        _get_serving_input_receiver_fn(), model_fn_lib.ModeKeys.PREDICT)

  def _test_export_saved_model_for_mode(self, input_receiver_fn, mode):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est._export_saved_model_for_mode(
        export_dir_base, input_receiver_fn, mode=mode)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    tag_set = model_fn_lib.EXPORT_TAG_MAP[mode]
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, tag_set, export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertFalse('name_collision_1' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_receiver_map(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExample' in graph_ops)
        self.assertFalse('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_train_only(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('multiplied' in graph_ops)
        self.assertTrue('mean/update_op' in graph_ops)
        self.assertFalse('eval_multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_eval_only(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.EVAL], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('eval_multiplied' in graph_ops)
        self.assertTrue('eval_mean/value' in graph_ops)
        self.assertFalse('multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_no_serving(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('multiplied' in graph_ops)
        self.assertFalse('eval_multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.EVAL], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('eval_multiplied' in graph_ops)
        self.assertFalse('multiplied' in graph_ops)
        # TODO(karmel): is this the desired behavior when names are shared?
        self.assertTrue('feature_x_1' in graph_ops)
        self.assertTrue('feature_y_1' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_three_defs(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    # Restore, to validate that the export was well-formed.
    for tag_set in model_fn_lib.EXPORT_TAG_MAP.values():
      with ops.Graph().as_default() as graph:
        with session.Session(graph=graph) as sess:
          loader.load(sess, tag_set, export_dir)
          graph_ops = [x.name for x in graph.get_operations()]
          self.assertTrue('global_step/Assign' in graph_ops)
          self.assertTrue('global_step/Initializer/zeros' in graph_ops)
          self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_all_vars(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('later_var' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertFalse('later_var' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_name_collision(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('name_collision' in graph_ops)
        self.assertFalse('name_collision_1' in graph_ops)
        collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        self.assertEqual(3, collection_vars[-1].eval())

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('name_collision' in graph_ops)
        self.assertFalse('name_collision_1' in graph_ops)
        collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        # This is a non-obvious detail: when we load the estimator spec
        # for predict, name_collision gets set to 36. However, we then restore
        # from checkpoint, which should overwrite that var and make it the 3
        # from training. In practice, this would not be a good way to write
        # a model_fn, but leaving this check in for now to ensure consistency
        # with what would happen given our current order of spec, then
        # checkpoint.
        self.assertEqual(3, collection_vars[-1].eval())

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def _test_export_all_saved_models(self, input_receiver_fn_map):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_with_x_y)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est._export_all_saved_models(
        export_dir_base, input_receiver_fn_map)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))

    self._validate_exported_files(export_dir)

    return export_dir, tmpdir

  def _validate_exported_files(self, export_dir):
    self.assertTrue(gfile.Exists(export_dir))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('saved_model.pb'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.index'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.data-00000-of-00001'))))

  def test_export_all_saved_models_var_not_found(self):
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }

    def _model_fn_with_predict_only_vars(features, labels, mode):
      _, _ = features, labels
      if mode == model_fn_lib.ModeKeys.PREDICT:
        variables.Variable(1., name='only_in_predict')
      else:
        variables.Variable(1., name='otherwise')

      prediction = constant_op.constant(1.)
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=constant_op.constant(1.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          export_outputs={
              'test': export_output.PredictOutput({'prediction': prediction})
          })

    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_with_predict_only_vars)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))

    err_regex = r'Could not load all requested variables[\w\W]*infer'
    with self.assertRaisesRegexp(ValueError, err_regex):
      est._export_all_saved_models(export_dir_base, input_receiver_fn_map)

  def test_export_all_saved_models_metric_operation(self):
    """Ensures metrics ops.Operations can be expoerted (b/109740581)."""

    def _model_fn(features, labels, mode):
      del features, labels  # Unused
      metrics = {'metrics': (constant_op.constant([0]),
                             control_flow_ops.no_op())}
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=constant_op.constant(10.),
          loss=constant_op.constant(1.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          eval_metric_ops=metrics)

    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn)
    est.train(input_fn=dummy_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('metric_operation_export'))

    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn()}

    export_dir = est._export_all_saved_models(
        export_dir_base, input_receiver_fn_map)

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        meta_graph = loader.load(sess, [tag_constants.EVAL], export_dir)
        sig_outputs = meta_graph.signature_def[
            model_fn_lib.ModeKeys.EVAL].outputs
        self.assertEqual(
            sig_outputs['metrics/update_op'].name, 'metric_op_wrapper:0')

  def test_export_savedmodel_with_saveables_proto_roundtrip(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(
        model_fn=_model_fn_with_saveables_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(
        export_dir_base, serving_input_receiver_fn)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self.assertTrue(gfile.Exists(export_dir))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('saved_model.pb'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.index'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.data-00000-of-00001'))))

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExample' in graph_ops)
        # The original saver is used to restore variables
        self.assertTrue('save/LookupTableImportV2' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_savedmodel_assets(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # Create a fake asset.
    vocab_file_name = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('my_vocab_file'))
    vocab_file = gfile.GFile(vocab_file_name, mode='w')
    vocab_file.write(_VOCAB_FILE_CONTENT)
    vocab_file.close()

    # hack in an op that uses the asset, in order to test asset export.
    # this is not actually valid, of course.
    def serving_input_receiver_with_asset_fn():
      features, receiver_tensor, _ = serving_input_receiver_fn()
      filename = ops.convert_to_tensor(vocab_file_name,
                                       dtypes.string,
                                       name='asset_filepath')
      ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
      features['bogus_filename'] = filename

      return export.ServingInputReceiver(features, receiver_tensor)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(
        export_dir_base, serving_input_receiver_with_asset_fn)

    # Check that the asset files are in the right places.
    expected_vocab_file_name = os.path.join(
        compat.as_bytes(export_dir), compat.as_bytes('assets/my_vocab_file'))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir), compat.as_bytes('assets'))))
    self.assertTrue(gfile.Exists(expected_vocab_file_name))
    self.assertEqual(
        compat.as_bytes(_VOCAB_FILE_CONTENT),
        compat.as_bytes(gfile.GFile(expected_vocab_file_name).read()))

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        assets = [
            x.eval()
            for x in graph.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
        ]
        self.assertItemsEqual([vocab_file_name], assets)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExample' in graph_ops)
        self.assertTrue('asset_filepath' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # cleanup
    gfile.DeleteRecursively(tmpdir)

  def test_export_savedmodel_extra_assets(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # Create a fake asset.
    extra_file_name = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('my_extra_file'))
    extra_file = gfile.GFile(extra_file_name, mode='w')
    extra_file.write(_EXTRA_FILE_CONTENT)
    extra_file.close()

    # Perform the export.
    assets_extra = {'some/sub/directory/my_extra_file': extra_file_name}
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(export_dir_base,
                                       serving_input_receiver_fn,
                                       assets_extra=assets_extra)

    # Check that the asset files are in the right places.
    expected_extra_path = os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('assets.extra/some/sub/directory/my_extra_file'))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir), compat.as_bytes('assets.extra'))))
    self.assertTrue(gfile.Exists(expected_extra_path))
    self.assertEqual(
        compat.as_bytes(_EXTRA_FILE_CONTENT),
        compat.as_bytes(gfile.GFile(expected_extra_path).read()))

    # cleanup
    gfile.DeleteRecursively(tmpdir)

  def test_export_savedmodel_tensor_features(self):
    """Test that models accepting a single raw Tensor can be exported.

    See https://github.com/tensorflow/tensorflow/issues/11674

    If the model_fn and receiver_fn accept raw tensors rather than dictionaries
    as input, export_savedmodel should be okay with that, too.

    """

    tmpdir = tempfile.mkdtemp()

    def _input_fn_tensor_features():
      t = array_ops.constant([1, 2, 3], dtype=dtypes.float32, shape=[1, 3])
      return (t, None)

    def _model_fn_tensor_features(features, labels, mode):
      _ = labels
      prediction = math_ops.matmul(features, features, transpose_b=True)

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=constant_op.constant(1.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          export_outputs={
              'test': export_output.PredictOutput({'prediction': prediction})
          })

    def _serving_input_receiver_fn():
      feat = array_ops.placeholder(dtype=dtypes.float32)
      return export.TensorServingInputReceiver(
          features=feat, receiver_tensors=feat)

    est = estimator.Estimator(model_fn=_model_fn_tensor_features)
    est.train(input_fn=_input_fn_tensor_features, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(
        export_dir_base, _serving_input_receiver_fn)

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name.lower() for x in graph.get_operations()]
        self.assertTrue('const' in graph_ops)
        self.assertTrue('matmul' in graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_scaffold_is_used_for_saver(self):
    tmpdir = tempfile.mkdtemp()

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='weight')
      self.mock_saver = get_mock_saver()
      scores = constant_op.constant([3.])
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=constant_op.constant([[1.]]),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=training.Scaffold(saver=self.mock_saver),
          export_outputs={'test': export_output.ClassificationOutput(scores)})

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    est.export_savedmodel(export_dir_base, serving_input_receiver_fn)

    self.assertTrue(self.mock_saver.restore.called)
    self.assertTrue(self.mock_saver.export_meta_graph.called)
    self.assertTrue(self.mock_saver.save.called)

  def test_scaffold_is_used_for_saver_multiple_modes(self):
    tmpdir = tempfile.mkdtemp()
    savers = {'predict_saver': None, 'train_saver': None}

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='weight')

      scores = constant_op.constant([3.])
      if mode == model_fn_lib.ModeKeys.PREDICT:
        savers['predict_saver'] = get_mock_saver()
        scaffold = training.Scaffold(saver=savers['predict_saver'])
      elif mode == model_fn_lib.ModeKeys.TRAIN:
        savers['train_saver'] = get_mock_saver()
        scaffold = training.Scaffold(saver=savers['train_saver'])
      else:
        scaffold = training.Scaffold()
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=constant_op.constant([[1.]]),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=scaffold,
          export_outputs={'test': export_output.ClassificationOutput(scores)})

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    est._export_all_saved_models(export_dir_base, input_receiver_fn_map)

    self.assertTrue(savers['train_saver'].restore.called)
    self.assertEqual(savers['train_saver'].export_meta_graph.call_count, 1)
    self.assertEqual(savers['train_saver'].save.call_count, 1)

    self.assertTrue(savers['predict_saver'].restore.called)
    self.assertEqual(savers['predict_saver'].export_meta_graph.call_count, 1)
    self.assertEqual(savers['predict_saver'].save.call_count, 0)

  def test_scaffold_is_used_for_local_init(self):
    tmpdir = tempfile.mkdtemp()

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      my_int = variables.Variable(1, name='my_int',
                                  collections=[ops.GraphKeys.LOCAL_VARIABLES])
      scores = constant_op.constant([3.])
      with ops.control_dependencies([
          variables.local_variables_initializer(),
          lookup_ops.tables_initializer()
      ]):
        assign_op = state_ops.assign(my_int, 12345)

      # local_initSop must be an Operation, not a Tensor.
      custom_local_init_op = control_flow_ops.group(assign_op)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=constant_op.constant([[1.]]),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=training.Scaffold(local_init_op=custom_local_init_op),
          export_outputs={'test': export_output.ClassificationOutput(scores)})

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_savedmodel(export_dir_base,
                                       serving_input_receiver_fn)

    # Restore, to validate that the custom local_init_op runs.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        my_int = graph.get_tensor_by_name('my_int:0')
        my_int_value = sess.run(my_int)
        self.assertEqual(12345, my_int_value)

  def test_scaffold_is_used_for_local_init_multiple_modes(self):
    tmpdir = tempfile.mkdtemp()

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      my_int = variables.Variable(1, name='my_int',
                                  collections=[ops.GraphKeys.LOCAL_VARIABLES])
      scores = constant_op.constant([3.])
      with ops.control_dependencies([
          variables.local_variables_initializer(),
          lookup_ops.tables_initializer()
      ]):
        assign_op = state_ops.assign(my_int, 12345)

      custom_local_init_op = None
      if mode == model_fn_lib.ModeKeys.PREDICT:
        # local_initSop must be an Operation, not a Tensor.
        custom_local_init_op = control_flow_ops.group(assign_op)

      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=constant_op.constant([[1.]]),
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          scaffold=training.Scaffold(local_init_op=custom_local_init_op),
          export_outputs={'test': export_output.ClassificationOutput(scores)})

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    input_receiver_fn_map = {
        model_fn_lib.ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        model_fn_lib.ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est._export_all_saved_models(
        export_dir_base, input_receiver_fn_map)

    # Restore, to validate that the custom local_init_op runs.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.SERVING], export_dir)
        my_int = graph.get_tensor_by_name('my_int:0')
        my_int_value = sess.run(my_int)
        self.assertEqual(12345, my_int_value)
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, [tag_constants.TRAINING], export_dir)
        my_int = graph.get_tensor_by_name('my_int:0')
        my_int_value = sess.run(my_int)
        self.assertEqual(1, my_int_value)

  def test_features_labels_mode(self):
    given_features = {'test-features': constant_op.constant([[1], [1]])}

    def serving_input_receiver_fn():
      return export.ServingInputReceiver(
          given_features, array_ops.placeholder(dtype=dtypes.string))

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]),
          export_outputs={
              'test': export_output.ClassificationOutput(
                  constant_op.constant([[0.]]))
          })

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    est.export_savedmodel(tempfile.mkdtemp(), serving_input_receiver_fn)
    self.assertEqual(given_features, self.features)
    self.assertIsNone(self.labels)
    self.assertEqual(model_fn_lib.ModeKeys.PREDICT, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed
    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(training.get_global_step())
      self.assertEqual(expected_random_seed, ops.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=constant_op.constant(0.),
          train_op=state_ops.assign_add(training.get_global_step(), 1),
          predictions=constant_op.constant([[0.]]),
          export_outputs={
              'test': export_output.ClassificationOutput(
                  constant_op.constant([[0.]]))
          })

    def serving_input_receiver_fn():
      return export.ServingInputReceiver(
          {'test-features': constant_op.constant([[1], [1]])},
          array_ops.placeholder(dtype=dtypes.string))

    est = estimator.Estimator(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    est.export_savedmodel(tempfile.mkdtemp(), serving_input_receiver_fn)

  def test_export_savedmodel_respects_soft_placement(self):
    def model_fn_with_a_gpu_op_but_no_kernel(features, labels, mode):
      _, _ = features, labels
      table = saver_test_utils.CheckpointedOp(name='v2')

      update_global_step = state_ops.assign_add(training.get_global_step(), 1)
      with ops.control_dependencies([update_global_step]):
        train_op = table.insert('k1', 30.0)

      #  In this test, there are no GPUs available.  The goal is to verify that
      #  export_savedmodel executes nevertheless.
      with ops.device('/gpu:0'):
        string_op = string_ops.as_string(update_global_step)

      with ops.control_dependencies([string_op]):
        prediction = table.lookup('k1', 0.0)

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=constant_op.constant(1.),
          train_op=train_op,
          export_outputs={
              'test': export_output.PredictOutput({
                  'prediction': prediction
              })
          })

    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(
        model_fn=model_fn_with_a_gpu_op_but_no_kernel)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))

    export_dir = est.export_savedmodel(
        export_dir_base, serving_input_receiver_fn)

    # At this point, if export_savedmodel executed with
    # allow_soft_placement=True, then the GPU-assigned operation was silently
    # placed on the CPU.  Otherwise, an exception would have been raised
    # related to the fact that the requested GPU device isn't available.

    # Expectations below assume that export_savedmodel has completed normally.
    self.assertTrue(gfile.Exists(export_dir_base))
    self.assertTrue(gfile.Exists(export_dir))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('saved_model.pb'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.index'))))
    self.assertTrue(gfile.Exists(os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes('variables/variables.data-00000-of-00001'))))

    gfile.DeleteRecursively(tmpdir)

  def test_export_savedmodel_proto_strip_default_attrs(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {'x': parsing_ops.VarLenFeature(dtype=dtypes.int64),
                    'y': parsing_ops.VarLenFeature(dtype=dtypes.int64)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir_stripped = est.export_savedmodel(
        export_dir_base, serving_input_receiver_fn, strip_default_attrs=True)
    export_dir_not_stripped = est.export_savedmodel(
        export_dir_base, serving_input_receiver_fn, strip_default_attrs=False)

    # Load the SavedModel from disk as-is to verify default attrs
    # are stripped. Reimporting the SavedModel via the loader causes the
    # default attrs to be populated in the NodeDefs.

    # pylint: disable=protected-access
    saved_model_stripped_pb = loader_impl._parse_saved_model(
        export_dir_stripped)
    saved_model_not_stripped_pb = loader_impl._parse_saved_model(
        export_dir_not_stripped)
    self.assertIsNotNone(saved_model_stripped_pb)
    self.assertIsNotNone(saved_model_not_stripped_pb)
    # pylint: enable=protected-access

    meta_graph_def_stripped = [
        x for x in saved_model_stripped_pb.meta_graphs
        if x.meta_info_def.tags == [tag_constants.SERVING]][0]
    meta_graph_def_not_stripped = [
        x for x in saved_model_not_stripped_pb.meta_graphs
        if x.meta_info_def.tags == [tag_constants.SERVING]][0]

    # "weight" node in graph is a "Variable" Op with 2 default valued attrs.
    #   o "container"    : "".
    #   o "shared_name"  : "".

    # saved_model_stripped_pb was exported with strip_default_attrs set to True.
    # "weight" node shouldn't have attributes "container" and "shared_name".
    node_def = test_util.get_node_def_from_graph(
        'weight', meta_graph_def_stripped.graph_def)
    self.assertNotIn('container', node_def.attr)
    self.assertNotIn('shared_name', node_def.attr)

    # saved_model_not_stripped_pb was exported with strip_default_attrs
    # disabled. "weight" node should have attributes "container" and
    # "shared_name".
    node_def = test_util.get_node_def_from_graph(
        'weight', meta_graph_def_not_stripped.graph_def)
    self.assertIn('container', node_def.attr)
    self.assertIn('shared_name', node_def.attr)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_savedmodel_no_export_outputs(self):
    """Ensure that an EstimatorSpec without outputs defined can be exported."""

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      variables.Variable(1., name='weight')
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=constant_op.constant(10.),
          loss=constant_op.constant(1.),
          train_op=state_ops.assign_add(training.get_global_step(), 1))

    tmpdir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=_model_fn)
    est.train(input_fn=dummy_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('no_export_outputs'))
    export_dir = est.export_savedmodel(
        export_dir_base, _get_serving_input_receiver_fn())

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        meta_graph = loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('weight' in graph_ops)

        sig_def = meta_graph.signature_def
        self.assertEqual(len(sig_def), 1)
        sig_outputs = sig_def[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs
        self.assertEqual(sig_outputs['output'].name, 'Const:0')


class EstimatorHookOrderingTest(test.TestCase):

  def testCustomHooksAreCalledBeforeNanTensorHook(self):

    def nan_making_model_fn(mode, features, labels):
      """A graph that generates NaN's for testing."""
      del features, labels

      global_step = variables.Variable(
          0, dtype=dtypes.int64, name='global_step')
      inc_global_step = state_ops.assign_add(global_step, 1)
      nan_const = constant_op.constant(np.nan, dtype=dtypes.float32)
      loss = control_flow_ops.cond(
          inc_global_step > 1, lambda: nan_const, lambda: 1.0)

      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=global_step.read_value(),
          loss=loss,
          train_op=inc_global_step)

    def empty_input_fn():
      return dict(), None

    class AfterRunCountingHook(session_run_hook.SessionRunHook):
      """Hooks that counts the number of times after_run() is called."""

      def __init__(self):
        self.after_run_count = 0

      def after_run(self, run_context, run_values):
        del run_context, run_values
        self.after_run_count += 1

    test_hook = AfterRunCountingHook()
    est = estimator.Estimator(model_fn=nan_making_model_fn)
    with self.assertRaises(basic_session_run_hooks.NanLossDuringTrainingError):
      est.train(input_fn=empty_input_fn, steps=2, hooks=[test_hook])
    self.assertEqual(2, test_hook.after_run_count)


class EstimatorIntegrationTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_complete_flow_with_a_simple_linear_model(self):

    def _model_fn(features, labels, mode):
      predictions = layers.dense(
          features['x'], 1, kernel_initializer=init_ops.zeros_initializer())
      export_outputs = {
          'predictions': export_output.RegressionOutput(predictions)
      }

      if mode == model_fn_lib.ModeKeys.PREDICT:
        return model_fn_lib.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

      loss = losses.mean_squared_error(labels, predictions)
      train_op = training.GradientDescentOptimizer(learning_rate=0.5).minimize(
          loss, training.get_global_step())
      eval_metric_ops = {
          'absolute_error': metrics_lib.mean_absolute_error(
              labels, predictions)
      }

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          export_outputs=export_outputs)

    est = estimator.Estimator(model_fn=_model_fn)
    data = np.linspace(0., 1., 100, dtype=np.float32).reshape(-1, 1)

    # TRAIN
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=50, num_epochs=None, shuffle=True)
    est.train(train_input_fn, steps=200)

    # EVALUTE
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=50, num_epochs=1, shuffle=True)
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores['global_step'])
    self.assertGreater(0.1, scores['absolute_error'])

    # PREDICT
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=None, batch_size=10, num_epochs=1, shuffle=False)
    predictions = list(est.predict(predict_input_fn))
    self.assertAllClose(data, predictions, atol=0.01)

    # EXPORT
    feature_spec = {'x': parsing_ops.FixedLenFeature([1], dtypes.float32)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))


if __name__ == '__main__':
  test.main()
