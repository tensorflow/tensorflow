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
from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

patch = test.mock.patch


class TestExperiment(experiment.Experiment):

  def __init__(self, default=None, config=None):
    self.default = default
    self.config = config

    class Estimator(evaluable.Evaluable, trainable.Trainable):
      config = self.config

      def model_dir(self):
        raise NotImplementedError

      def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
              monitors=None, max_steps=None):
        raise NotImplementedError

      def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
                   batch_size=None, steps=None, metrics=None, name=None,
                   checkpoint_path=None, hooks=None):
        raise NotImplementedError

    super(TestExperiment, self).__init__(Estimator(), None, None)

  def local_run(self):
    return "local_run"

  def train(self):
    return "train"

  def run_std_server(self):
    return "run_std_server"

  def train_and_evaluate(self):
    return "train_and_evaluate"

  def simple_task(self):
    return "simple_task, default=%s." % self.default


# pylint: disable=unused-argument
def build_experiment(output_dir):
  tf_logging.info("In default build_experiment.")
  return TestExperiment()


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


class MainTest(test.TestCase):

  def setUp(self):
    # Ensure the TF_CONFIG environment variable is unset for all tests.
    os.environ.pop("TF_CONFIG", None)

  def test_run_with_custom_schedule(self):
    self.assertEqual(
        "simple_task, default=None.",
        learn_runner.run(build_experiment,
                         output_dir="/tmp",
                         schedule="simple_task"))

  def test_run_with_explicit_local_run(self):
    self.assertEqual(
        "local_run",
        learn_runner.run(build_experiment,
                         output_dir="/tmp",
                         schedule="local_run"))

  def test_schedule_from_tf_config_runs_train_on_worker(self):
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": run_config_lib.TaskType.WORKER
        }
    })
    # RunConfig constructor will set job_name from TF_CONFIG.
    config = run_config.RunConfig()
    self.assertEqual(
        "train",
        learn_runner.run(lambda output_dir: TestExperiment(config=config),
                         output_dir="/tmp"))

  def test_schedule_from_tf_config_runs_train_and_evaluate_on_master(self):
    tf_config = {
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": run_config_lib.TaskType.MASTER
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      self.assertEqual(
          "train_and_evaluate",
          learn_runner.run(lambda output_dir: TestExperiment(config=config),
                           output_dir="/tmp"))

  def test_schedule_from_tf_config_runs_serve_on_ps(self):
    tf_config = {
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": run_config_lib.TaskType.PS
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      self.assertEqual(
          "run_std_server",
          learn_runner.run(lambda output_dir: TestExperiment(config=config),
                           output_dir="/tmp"))

  def test_fail_no_output_dir(self):
    self.assertRaisesRegexp(ValueError, "Must specify an output directory",
                            learn_runner.run, build_experiment, "",
                            "simple_task")

  def test_no_schedule_and_no_config_runs_train_and_evaluate(self):
    self.assertEqual(
        "train_and_evaluate",
        learn_runner.run(build_experiment, output_dir="/tmp"))

  def test_no_schedule_and_non_distributed_runs_train_and_evaluate(self):
    tf_config = {"cluster": build_non_distributed_cluster_spec()}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      self.assertEqual(
          "train_and_evaluate",
          learn_runner.run(lambda output_dir: TestExperiment(config=config),
                           output_dir="/tmp"))

  def test_fail_task_type_with_no_default_schedule(self):
    tf_config = {
        "cluster": build_distributed_cluster_spec(),
        "task": {
            "type": "foo_has_no_default_schedule"
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      create_experiment_fn = lambda output_dir: TestExperiment(config=config)
      self.assertRaisesRegexp(ValueError, "No default schedule",
                              learn_runner.run, create_experiment_fn, "/tmp")

  def test_fail_non_callable(self):
    self.assertRaisesRegexp(TypeError, "Experiment builder .* is not callable",
                            learn_runner.run, "not callable", "/tmp",
                            "simple_test")

  def test_fail_not_experiment(self):
    self.assertRaisesRegexp(TypeError,
                            "Experiment builder did not return an Experiment",
                            learn_runner.run, build_non_experiment, "/tmp",
                            "simple_test")

  def test_fail_non_existent_task(self):
    self.assertRaisesRegexp(ValueError, "Schedule references non-existent task",
                            learn_runner.run, build_experiment, "/tmp",
                            "mirage")

  def test_fail_non_callable_task(self):
    self.assertRaisesRegexp(TypeError,
                            "Schedule references non-callable member",
                            learn_runner.run, build_experiment, "/tmp",
                            "default")

  def test_fail_schedule_from_config_with_no_task_type(self):
    tf_config = {"cluster": build_distributed_cluster_spec()}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      self.assertRaisesRegexp(
          ValueError,
          "Must specify a schedule",
          learn_runner.run,
          lambda output_dir: TestExperiment(config=config),
          output_dir="/tmp")


if __name__ == "__main__":
  test.main()
