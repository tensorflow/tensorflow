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

from tensorflow.python.client import session
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test
from tensorflow.python.training import training


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

    class FakeConfig(run_config.RunConfig):  # pylint: disable=g-wrong-blank-lines
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

    est = estimator.Estimator(model_fn=model_fn)
    self.assertTrue(est.model_dir is not None)

  def test_model_fn_args_must_include_features(self):

    def model_fn(x, labels):
      _, _ = x, labels

    with self.assertRaisesRegexp(ValueError, 'features'):
      estimator.Estimator(model_fn=model_fn)

  def test_model_fn_args_must_include_labels(self):

    def model_fn(features, y):
      _, _ = features, y

    with self.assertRaisesRegexp(ValueError, 'labels'):
      estimator.Estimator(model_fn=model_fn)

  def test_if_params_provided_then_model_fn_should_accept_it(self):

    def model_fn(features, labels):
      _, _ = features, labels

    estimator.Estimator(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'params'):
      estimator.Estimator(model_fn=model_fn, params={'hidden_layers': 4})

  def test_not_known_model_fn_args_without_default(self):

    def model_fn(features, labels, something):
      _, _, _ = features, labels, something

    with self.assertRaisesRegexp(ValueError, 'something'):
      estimator.Estimator(model_fn=model_fn)


def dummy_input_fn():
  return {'x': [[1], [1]]}, [[1], [1]]


def model_fn_global_step_incrementer(features, labels, mode):
  _, _ = features, labels
  global_step = training.get_global_step()
  return model_fn_lib.EstimatorSpec(
      mode,
      loss=constant_op.constant(1.),
      train_op=state_ops.assign_add(global_step, 1))


class EstimatorFitTest(test.TestCase):

  def test_model_fn_must_return_estimator_spec(self):

    def model_fn(features, labels):
      _, _ = features, labels
      return 'NotGoodNotGood'

    est = estimator.Estimator(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'EstimatorSpec'):
      est.fit(dummy_input_fn, steps=1)

  def test_run_train_op_and_saves_at_the_end(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.fit(dummy_input_fn, steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_steps_and_saves_reloads(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.fit(dummy_input_fn, steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))
    est.fit(dummy_input_fn, steps=5)
    self.assertEqual(
        10, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_max_step(self):
    est = estimator.Estimator(model_fn=model_fn_global_step_incrementer)
    est.fit(dummy_input_fn, max_steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))
    est.fit(dummy_input_fn, max_steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

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
          train_op=constant_op.constant(0.),
          scaffold=training.Scaffold(init_fn=_init_fn))

    est = estimator.Estimator(model_fn=_model_fn_scaffold)
    est.fit(dummy_input_fn, steps=1)
    self.assertTrue(self.is_init_fn_called)

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
          train_op=constant_op.constant(0.),
          training_chief_hooks=[chief_hook],
          training_hooks=[hook])

    est = estimator.Estimator(model_fn=_model_fn_hooks)
    self.assertFalse(chief_hook.begin.called)
    self.assertFalse(hook.begin.called)
    est.fit(dummy_input_fn, steps=1)
    self.assertTrue(chief_hook.begin.called)
    self.assertTrue(hook.begin.called)

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
          train_op=constant_op.constant(0.),
          training_chief_hooks=[chief_hook],
          training_hooks=[hook])

    class NonChiefRunConfig(run_config.RunConfig):  # pylint: disable=g-wrong-blank-lines
      @property
      def is_chief(self):  # pylint: disable=g-wrong-blank-lines
        return False

    # Mocking the SessionManager.wait_for_session, so that worker doesn't wait
    # for chief.
    def get_initialized_session(*args, **kwargs):
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
      est.fit(dummy_input_fn, steps=1)
      self.assertFalse(chief_hook.begin.called)
      self.assertTrue(hook.begin.called)


if __name__ == '__main__':
  test.main()
