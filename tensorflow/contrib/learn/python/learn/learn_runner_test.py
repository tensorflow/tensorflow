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
"""learn_main tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from tensorflow.contrib.learn.python.learn import evaluable  # pylint: disable=g-import-not-at-top
from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn import trainable

from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.contrib.training.python.training import hparam as hparam_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

patch = test.mock.patch

_MODIR_DIR = "/tmp"
_HPARAMS = hparam_lib.HParams(learning_rate=0.01)
_MUST_SPECIFY_OUTPUT_DIR_MSG = "Must specify an output directory"
_MISSING_MODEL_DIR_ERR_MSG = "Must specify a model directory in `run_config`."
_EXP_NOT_CALLABLE_MSG = "Experiment builder .* is not callable"
_INVALID_HPARAMS_ERR_MSG = "`hparams` must be `HParams` instance"
_NOT_EXP_TYPE_MSG = "Experiment builder did not return an Experiment"
_NON_EXIST_TASK_MSG = "Schedule references non-existent task"
_NON_CALLABLE_MSG = "Schedule references non-callable member"
_MUST_SPECIFY_OUTPUT_DIR_OR_CONFIG_MSG = (
    "Must set value for `output_dir` or `run_config`")
_HPARAMS_CANNOT_BE_SET_FOR_OUTPUT_DIR_MSG = (
    "Must set `hparams` as None for `experiment_fn` with `output_dir`.")
_CANNOT_SET_BOTH_OUTPUT_DIR_AND_CONFIG_MSG = (
    "Cannot provide both `output_dir` and `run_config`")
_INVALID_RUN_CONFIG_TYPE_MSG = "`run_config` must be `RunConfig` instance"
_RUN_CONFIG_UID_CHECK_ERR_MSG = (
    "`RunConfig` instance is expected to be used by the `Estimator`")


class TestExperiment(experiment.Experiment):

  def __init__(self, default=None, config=None, model_dir=None):
    self.default = default
    self.config = config
    internal_model_dir = model_dir or config.model_dir
    self._model_dir = internal_model_dir

    class Estimator(evaluable.Evaluable, trainable.Trainable):
      config = self.config

      @property
      def model_dir(self):
        return internal_model_dir

      def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
              monitors=None, max_steps=None):
        raise NotImplementedError

      def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
                   batch_size=None, steps=None, metrics=None, name=None,
                   checkpoint_path=None, hooks=None):
        raise NotImplementedError

    super(TestExperiment, self).__init__(Estimator(), None, None)

  def local_run(self):
    return "local_run-{}".format(self._model_dir)

  def train(self):
    return "train-{}".format(self._model_dir)

  def run_std_server(self):
    return "run_std_server-{}".format(self._model_dir)

  def train_and_evaluate(self):
    return "train_and_evaluate-{}".format(self._model_dir)

  def simple_task(self):
    return "simple_task, default=%s." % self.default


# pylint: disable=unused-argument
def build_experiment(output_dir):
  tf_logging.info("In default build_experiment.")
  return TestExperiment(model_dir=output_dir)


def build_experiment_fn_for_output_dir(run_config=None):
  def _build_experiment(output_dir):
    tf_logging.info("In default build_experiment.")
    return TestExperiment(config=run_config, model_dir=output_dir)
  return _build_experiment


def build_experiment_for_run_config(run_config, hparams):
  if hparams is not None and hparams != _HPARAMS:
    raise ValueError("hparams is not set correctly")
  return TestExperiment(config=run_config)


def build_non_experiment(output_dir):
  return "Ceci n'est pas un Experiment."


# pylint: enable=unused-argument


def build_distributed_cluster_spec():
  return {
      run_config_lib.TaskType.PS: ["localhost:1234", "localhost:1235"],
      run_config_lib.TaskType.WORKER: ["localhost:1236", "localhost:1237"],
      run_config_lib.TaskType.MASTER: ["localhost:1238"],
      "foo_has_no_default_schedule": ["localhost:1239"]
  }


def build_non_distributed_cluster_spec():
  return {"foo": ["localhost:1234"]}


class LearnRunnerRunWithOutputDirTest(test.TestCase):

  def setUp(self):
    # Ensure the TF_CONFIG environment variable is unset for all tests.
    os.environ.pop("TF_CONFIG", None)

  def test_run_with_custom_schedule(self):
    self.assertEqual(
        "simple_task, default=None.",
        learn_runner.run(build_experiment,
                         output_dir=_MODIR_DIR,
                         schedule="simple_task"))

  def test_run_with_explicit_local_run(self):
    self.assertEqual(
        "local_run-" + _MODIR_DIR,
        learn_runner.run(build_experiment,
                         output_dir=_MODIR_DIR,
                         schedule="local_run"))

  def test_fail_output_dir_and_run_config_are_both_set(self):
    with self.assertRaisesRegexp(
        ValueError, _CANNOT_SET_BOTH_OUTPUT_DIR_AND_CONFIG_MSG):
      learn_runner.run(build_experiment,
                       output_dir=_MODIR_DIR,
                       schedule="simple_task",
                       run_config=run_config_lib.RunConfig())

  def test_fail_empty_output_dir(self):
    with self.assertRaisesRegexp(ValueError, _MUST_SPECIFY_OUTPUT_DIR_MSG):
      learn_runner.run(build_experiment, output_dir="", schedule="simple_task")

  def test_fail_no_output_dir(self):
    with self.assertRaisesRegexp(
        ValueError, _MUST_SPECIFY_OUTPUT_DIR_OR_CONFIG_MSG):
      learn_runner.run(build_experiment, None, "simple_task")

  def test_fail_hparams_are_set(self):
    hparams = _HPARAMS
    with self.assertRaisesRegexp(
        ValueError, _HPARAMS_CANNOT_BE_SET_FOR_OUTPUT_DIR_MSG):
      learn_runner.run(
          build_experiment, _MODIR_DIR, schedule="simple_task", hparams=hparams)

  def test_fail_non_callable(self):
    with self.assertRaisesRegexp(TypeError, _EXP_NOT_CALLABLE_MSG):
      learn_runner.run("not callable", _MODIR_DIR, "simple_test")

  def test_fail_not_experiment(self):
    with self.assertRaisesRegexp(TypeError, _NOT_EXP_TYPE_MSG):
      learn_runner.run(build_non_experiment, _MODIR_DIR, "simple_test")

  def test_fail_non_existent_task(self):
    with self.assertRaisesRegexp(ValueError, _NON_EXIST_TASK_MSG):
      learn_runner.run(build_experiment, _MODIR_DIR, "mirage")

  def test_fail_non_callable_task(self):
    with self.assertRaisesRegexp(TypeError, _NON_CALLABLE_MSG):
      learn_runner.run(build_experiment, _MODIR_DIR, "default")


class LearnRunnerRunWithRunConfigTest(test.TestCase):

  def setUp(self):
    # Ensure the TF_CONFIG environment variable is unset for all tests.
    os.environ.pop("TF_CONFIG", None)

  def test_run_with_custom_schedule(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    self.assertEqual(
        "simple_task, default=None.",
        learn_runner.run(build_experiment_for_run_config,
                         run_config=run_config,
                         schedule="simple_task"))

  def test_run_with_hparams(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    self.assertEqual(
        "simple_task, default=None.",
        learn_runner.run(build_experiment_for_run_config,
                         run_config=run_config,
                         schedule="simple_task",
                         hparams=_HPARAMS))

  def test_run_with_explicit_local_run(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    self.assertEqual(
        "local_run-" + _MODIR_DIR,
        learn_runner.run(build_experiment_for_run_config,
                         run_config=run_config,
                         schedule="local_run"))

  def test_fail_empty_output_dir(self):
    run_config = run_config_lib.RunConfig(model_dir="")
    with self.assertRaisesRegexp(ValueError, _MISSING_MODEL_DIR_ERR_MSG):
      learn_runner.run(build_experiment_for_run_config,
                       run_config=run_config,
                       schedule="local_run")

  def test_fail_no_output_dir(self):
    run_config = run_config_lib.RunConfig()
    with self.assertRaisesRegexp(ValueError, _MISSING_MODEL_DIR_ERR_MSG):
      learn_runner.run(build_experiment_for_run_config,
                       run_config=run_config,
                       schedule="local_run")

  def test_fail_invalid_run_config_type(self):
    run_config = "invalid_run_config"
    with self.assertRaisesRegexp(ValueError, _INVALID_RUN_CONFIG_TYPE_MSG):
      learn_runner.run(build_experiment_for_run_config,
                       run_config=run_config,
                       schedule="local_run")

  def test_fail_invalid_hparams_type(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    with self.assertRaisesRegexp(ValueError, _INVALID_HPARAMS_ERR_MSG):
      learn_runner.run(build_experiment_for_run_config,
                       run_config=run_config,
                       schedule="local_run",
                       hparams=["hparams"])

  def test_fail_non_callable(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    with self.assertRaisesRegexp(TypeError, _EXP_NOT_CALLABLE_MSG):
      learn_runner.run("not callable",
                       run_config=run_config,
                       schedule="simple_task")

  def test_fail_not_experiment(self):
    def _experiment_fn(run_config, hparams):
      del run_config, hparams  # unused.
      return "not experiment"

    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    with self.assertRaisesRegexp(TypeError, _NOT_EXP_TYPE_MSG):
      learn_runner.run(_experiment_fn,
                       run_config=run_config,
                       schedule="simple_task")

  def test_fail_non_existent_task(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    with self.assertRaisesRegexp(ValueError, _NON_EXIST_TASK_MSG):
      learn_runner.run(build_experiment_for_run_config,
                       run_config=run_config,
                       schedule="mirage")

  def test_fail_non_callable_task(self):
    run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)
    with self.assertRaisesRegexp(TypeError, _NON_CALLABLE_MSG):
      learn_runner.run(build_experiment_for_run_config,
                       run_config=run_config,
                       schedule="default")

  def test_basic_run_config_uid_check(self):
    expected_run_config = run_config_lib.RunConfig(model_dir=_MODIR_DIR)

    def _experiment_fn(run_config, hparams):
      del run_config, hparams  # unused.
      # Explicitly use a new run_config.
      new_config = run_config_lib.RunConfig(
          model_dir=_MODIR_DIR, save_checkpoints_steps=123)

      return TestExperiment(config=new_config)

    with self.assertRaisesRegexp(RuntimeError, _RUN_CONFIG_UID_CHECK_ERR_MSG):
      learn_runner.run(experiment_fn=_experiment_fn,
                       run_config=expected_run_config)


class LearnRunnerDefaultScheduleTest(test.TestCase):

  def setUp(self):
    # Ensure the TF_CONFIG environment variable is unset for all tests.
    os.environ.pop("TF_CONFIG", None)

  def test_schedule_from_tf_config_runs_train_on_worker(self):
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": run_config_lib.TaskType.WORKER
        }
    })
    # RunConfig constructor will set job_name from TF_CONFIG.
    config = run_config_lib.RunConfig()
    self.assertEqual(
        "train-" + _MODIR_DIR,
        learn_runner.run(
            build_experiment_fn_for_output_dir(config),
            output_dir=_MODIR_DIR))

  def test_schedule_from_tf_config_runs_train_and_evaluate_on_master(self):
    tf_config = {
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": run_config_lib.TaskType.MASTER
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      self.assertEqual(
          "train_and_evaluate-" + _MODIR_DIR,
          learn_runner.run(
              build_experiment_fn_for_output_dir(config),
              output_dir=_MODIR_DIR))

  def test_schedule_from_tf_config_runs_serve_on_ps(self):
    tf_config = {
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": run_config_lib.TaskType.PS
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      self.assertEqual(
          "run_std_server-" + _MODIR_DIR,
          learn_runner.run(
              build_experiment_fn_for_output_dir(config),
              output_dir=_MODIR_DIR))

  def test_no_schedule_and_no_config_runs_train_and_evaluate(self):
    self.assertEqual(
        "train_and_evaluate-" + _MODIR_DIR,
        learn_runner.run(build_experiment, output_dir=_MODIR_DIR))

  def test_no_schedule_and_non_distributed_runs_train_and_evaluate(self):
    tf_config = {"cluster": build_non_distributed_cluster_spec()}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      self.assertEqual(
          "train_and_evaluate-" + _MODIR_DIR,
          learn_runner.run(
              build_experiment_fn_for_output_dir(config),
              output_dir=_MODIR_DIR))

  def test_fail_task_type_with_no_default_schedule(self):
    tf_config = {
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": "foo_has_no_default_schedule"
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      create_experiment_fn = lambda output_dir: TestExperiment(config=config)
      self.assertRaisesRegexp(ValueError,
                              "No default schedule",
                              learn_runner.run,
                              create_experiment_fn,
                              _MODIR_DIR)

  def test_fail_schedule_from_config_with_no_task_type(self):
    tf_config = {"cluster": build_distributed_cluster_spec()}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      self.assertRaisesRegexp(
          ValueError,
          "Must specify a schedule",
          learn_runner.run,
          lambda output_dir: TestExperiment(config=config),
          output_dir=_MODIR_DIR)


if __name__ == "__main__":
  test.main()
