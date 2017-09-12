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

"""Tests for training.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import training
from tensorflow.python.platform import test
from tensorflow.python.training import session_run_hook

_DEFAULT_EVAL_STEPS = 100
_DEFAULT_EVAL_DELAY_SECS = 120
_DEFAULT_EVAL_THROTTLE_SECS = 600
_INVALID_INPUT_FN_MSG = '`input_fn` must be callable'
_INVALID_HOOK_MSG = 'All hooks must be `SessionRunHook` instances'
_INVALID_MAX_STEPS_MSG = 'Must specify max_steps > 0'
_INVALID_STEPS_MSG = 'Must specify steps > 0'
_INVALID_NAME_MSG = '`name` must be string'
_INVALID_EVAL_DELAY_SECS_MSG = 'Must specify delay_secs >= 0'
_INVALID_EVAL_THROTTLE_SECS_MSG = 'Must specify throttle_secs >= 0'
_INVALID_ESTIMATOR_MSG = '`estimator` must have type `tf.estimator.Estimator`'
_INVALID_TRAIN_SPEC_MSG = '`train_spec` must have type `tf.estimator.TrainSpec`'
_INVALID_EVAL_SPEC_MSG = '`eval_spec` must have type `tf.estimator.EvalSpec`'


class _FakeHook(session_run_hook.SessionRunHook):
  """Fake implementation of `SessionRunHook`."""


class _InvalidHook(object):
  """Invalid hook (not a subclass of `SessionRunHook`)."""


class TrainSpecTest(test.TestCase):
  """Tests TrainSpec."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    spec = training.TrainSpec(input_fn=lambda: 1)
    self.assertEqual(1, spec.input_fn())
    self.assertIsNone(spec.max_steps)
    self.assertEqual(0, len(spec.hooks))

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    hooks = [_FakeHook()]
    spec = training.TrainSpec(input_fn=lambda: 1, max_steps=2, hooks=hooks)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(2, spec.max_steps)
    self.assertEqual(tuple(hooks), spec.hooks)

  def testInvalidInputFn(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_INPUT_FN_MSG):
      training.TrainSpec(input_fn='invalid')

  def testInvalidMaxStep(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_MAX_STEPS_MSG):
      training.TrainSpec(input_fn=lambda: 1, max_steps=0)

  def testInvalidHook(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_HOOK_MSG):
      training.TrainSpec(input_fn=lambda: 1, hooks=[_InvalidHook()])


class EvalSpecTest(test.TestCase):
  """Tests EvalSpec."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    spec = training.EvalSpec(input_fn=lambda: 1)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(_DEFAULT_EVAL_STEPS, spec.steps)
    self.assertIsNone(spec.name)
    self.assertEqual(0, len(spec.hooks))
    self.assertEqual(0, len(spec.export_strategies))
    self.assertEqual(_DEFAULT_EVAL_DELAY_SECS, spec.delay_secs)
    self.assertEqual(_DEFAULT_EVAL_THROTTLE_SECS, spec.throttle_secs)

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    hooks = [_FakeHook()]

    # TODO(b/65169058): Replace the export_strategies with valid instances.
    spec = training.EvalSpec(input_fn=lambda: 1, steps=2, name='name',
                             hooks=hooks, export_strategies=hooks,
                             delay_secs=3, throttle_secs=4)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(2, spec.steps)
    self.assertEqual('name', spec.name)
    self.assertEqual(tuple(hooks), spec.hooks)
    self.assertEqual(tuple(hooks), spec.export_strategies)
    self.assertEqual(3, spec.delay_secs)
    self.assertEqual(4, spec.throttle_secs)

  def testInvalidInputFn(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_INPUT_FN_MSG):
      training.EvalSpec(input_fn='invalid')

  def testInvalidMaxStep(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_STEPS_MSG):
      training.EvalSpec(input_fn=lambda: 1, steps=0)

  def testInvalidName(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_NAME_MSG):
      training.EvalSpec(input_fn=lambda: 1, name=123)

  def testInvalidHook(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_HOOK_MSG):
      training.EvalSpec(input_fn=lambda: 1, hooks=[_InvalidHook()])

  def testInvalidDelaySecs(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_EVAL_DELAY_SECS_MSG):
      training.EvalSpec(input_fn=lambda: 1, delay_secs=-1)

  def testInvalidThrottleSecs(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_EVAL_THROTTLE_SECS_MSG):
      training.EvalSpec(input_fn=lambda: 1, throttle_secs=-1)


class TrainingExecutorTest(test.TestCase):
  """Tests _TrainingExecutor."""

  def testRequiredArgumentsSet(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1)

    executor = training._TrainingExecutor(estimator, train_spec, eval_spec)
    self.assertEqual(estimator, executor.estimator)

  def test_invalid_estimator(self):
    invalid_estimator = object()
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1)

    with self.assertRaisesRegexp(TypeError, _INVALID_ESTIMATOR_MSG):
      training._TrainingExecutor(invalid_estimator, train_spec, eval_spec)

  def test_invalid_train_spec(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    invalid_train_spec = object()
    eval_spec = training.EvalSpec(input_fn=lambda: 1)

    with self.assertRaisesRegexp(TypeError, _INVALID_TRAIN_SPEC_MSG):
      training._TrainingExecutor(estimator, invalid_train_spec, eval_spec)

  def test_invalid_eval_spec(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    invalid_eval_spec = object()

    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_SPEC_MSG):
      training._TrainingExecutor(estimator, train_spec, invalid_eval_spec)

if __name__ == '__main__':
  test.main()
