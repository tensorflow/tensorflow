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
"""Tests for early_stopping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import parameterized
from tensorflow.contrib.estimator.python.estimator import early_stopping
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util


class _FakeRunConfig(run_config.RunConfig):

  def __init__(self, is_chief):
    super(_FakeRunConfig, self).__init__()
    self._is_chief = is_chief

  @property
  def is_chief(self):
    return self._is_chief


def _dummy_model_fn(features, labels, params):
  _, _, _ = features, labels, params


class _FakeEstimator(estimator.Estimator):
  """Fake estimator for testing."""

  def __init__(self, config):
    super(_FakeEstimator, self).__init__(
        model_fn=_dummy_model_fn, config=config)


def _write_events(eval_dir, params):
  """Test helper to write events to summary files."""
  for steps, loss, accuracy in params:
    estimator._write_dict_to_summary(eval_dir, {
        'loss': loss,
        'accuracy': accuracy,
    }, steps)


class ReadEvalMetricsTest(test.TestCase):

  def test_read_eval_metrics(self):
    eval_dir = tempfile.mkdtemp()
    _write_events(
        eval_dir,
        [
            # steps, loss, accuracy
            (1000, 1, 2),
            (2000, 3, 4),
            (3000, 5, 6),
        ])
    self.assertEqual({
        1000: {
            'loss': 1,
            'accuracy': 2
        },
        2000: {
            'loss': 3,
            'accuracy': 4
        },
        3000: {
            'loss': 5,
            'accuracy': 6
        },
    }, early_stopping.read_eval_metrics(eval_dir))


class EarlyStoppingHooksTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    config = _FakeRunConfig(is_chief=True)
    self._estimator = _FakeEstimator(config=config)
    eval_dir = self._estimator.eval_dir()
    os.makedirs(eval_dir)
    _write_events(
        eval_dir,
        [
            # steps, loss, accuracy
            (1000, 0.8, 0.5),
            (2000, 0.7, 0.6),
            (3000, 0.4, 0.7),
            (3500, 0.41, 0.68),
        ])

  def run_session(self, hooks, should_stop):
    hooks = hooks if isinstance(hooks, list) else [hooks]
    with ops.Graph().as_default():
      training_util.create_global_step()
      no_op = control_flow_ops.no_op()
      with monitored_session.SingularMonitoredSession(hooks=hooks) as mon_sess:
        mon_sess.run(no_op)
        self.assertEqual(mon_sess.should_stop(), should_stop)

  @parameterized.parameters((0.8, 0, False), (0.6, 4000, False), (0.6, 0, True))
  def test_stop_if_higher_hook(self, threshold, min_steps, should_stop):
    self.run_session(
        early_stopping.stop_if_higher_hook(
            self._estimator,
            metric_name='accuracy',
            threshold=threshold,
            min_steps=min_steps), should_stop)

  @parameterized.parameters((0.3, 0, False), (0.5, 4000, False), (0.5, 0, True))
  def test_stop_if_lower_hook(self, threshold, min_steps, should_stop):
    self.run_session(
        early_stopping.stop_if_lower_hook(
            self._estimator,
            metric_name='loss',
            threshold=threshold,
            min_steps=min_steps), should_stop)

  @parameterized.parameters((1500, 0, False), (500, 4000, False),
                            (500, 0, True))
  def test_stop_if_no_increase_hook(self, max_steps, min_steps, should_stop):
    self.run_session(
        early_stopping.stop_if_no_increase_hook(
            self._estimator,
            metric_name='accuracy',
            max_steps_without_increase=max_steps,
            min_steps=min_steps), should_stop)

  @parameterized.parameters((1500, 0, False), (500, 4000, False),
                            (500, 0, True))
  def test_stop_if_no_decrease_hook(self, max_steps, min_steps, should_stop):
    self.run_session(
        early_stopping.stop_if_no_decrease_hook(
            self._estimator,
            metric_name='loss',
            max_steps_without_decrease=max_steps,
            min_steps=min_steps), should_stop)

  @parameterized.parameters((1500, 0.3, False), (1500, 0.5, True),
                            (500, 0.3, True))
  def test_multiple_hooks(self, max_steps, loss_threshold, should_stop):
    self.run_session([
        early_stopping.stop_if_no_decrease_hook(
            self._estimator,
            metric_name='loss',
            max_steps_without_decrease=max_steps),
        early_stopping.stop_if_lower_hook(
            self._estimator, metric_name='loss', threshold=loss_threshold)
    ], should_stop)

  @parameterized.parameters(False, True)
  def test_make_early_stopping_hook(self, should_stop):
    self.run_session([
        early_stopping.make_early_stopping_hook(
            self._estimator, should_stop_fn=lambda: should_stop)
    ], should_stop)

  def test_make_early_stopping_hook_typeerror(self):
    with self.assertRaises(TypeError):
      early_stopping.make_early_stopping_hook(
          estimator=object(), should_stop_fn=lambda: True)

  def test_make_early_stopping_hook_valueerror(self):
    with self.assertRaises(ValueError):
      early_stopping.make_early_stopping_hook(
          self._estimator,
          should_stop_fn=lambda: True,
          run_every_secs=60,
          run_every_steps=100)


class StopOnPredicateHookTest(test.TestCase):

  def test_stop(self):
    hook = early_stopping._StopOnPredicateHook(
        should_stop_fn=lambda: False, run_every_secs=0)
    with ops.Graph().as_default():
      training_util.create_global_step()
      no_op = control_flow_ops.no_op()
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as mon_sess:
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        self.assertFalse(mon_sess.raw_session().run(hook._stop_var))

    hook = early_stopping._StopOnPredicateHook(
        should_stop_fn=lambda: True, run_every_secs=0)
    with ops.Graph().as_default():
      training_util.create_global_step()
      no_op = control_flow_ops.no_op()
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as mon_sess:
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        self.assertTrue(mon_sess.raw_session().run(hook._stop_var))


class CheckForStoppingHookTest(test.TestCase):

  def test_stop(self):
    hook = early_stopping._CheckForStoppingHook()
    with ops.Graph().as_default():
      no_op = control_flow_ops.no_op()
      assign_op = state_ops.assign(early_stopping._get_or_create_stop_var(),
                                   True)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as mon_sess:
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        mon_sess.run(assign_op)
        self.assertTrue(mon_sess.should_stop())


if __name__ == '__main__':
  test.main()
