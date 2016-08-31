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

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn import run_config


class TestExperiment(tf.contrib.learn.Experiment):

  def __init__(self, default=None, config=None):
    self.default = default
    self.config = config

  @property
  def estimator(self):
    class Estimator(object):
      config = self.config
    return Estimator()

  def local_run(self):
    return "train_and_evaluate"

  def train(self):
    return "train"

  def serve(self):
    return "serve"

  def simple_task(self):
    return "simple_task, default=%s." % self.default


# pylint: disable=unused-argument
def build_experiment(output_dir):
  tf.logging.info("In default build_experiment.")
  return TestExperiment()


def build_non_experiment(output_dir):
  return "Ceci n'est pas un Experiment."
# pylint: enable=unused-argument


class MainTest(tf.test.TestCase):

  def setUp(self):
    # Ensure the TF_CONFIG environment variable is unset for all tests.
    os.environ.pop("TF_CONFIG", None)

  def test_run_with_explicit_args(self):
    self.assertEqual(
        "simple_task, default=None.",
        learn_runner.run(build_experiment,
                         output_dir="/tmp",
                         schedule="simple_task"))

  def test_schedule_from_tf_config(self):
    os.environ["TF_CONFIG"] = json.dumps({"task": {"type": "worker"}})
    # RunConfig constructuor will set job_name from TF_CONFIG.
    config = run_config.RunConfig()
    self.assertEqual(
        "train",
        learn_runner.run(lambda output_dir: TestExperiment(config=config),
                         output_dir="/tmp"))

  def test_schedule_from_manually_specified_job_name(self):
    config = run_config.RunConfig(job_name="worker")
    self.assertEqual(
        "train",
        learn_runner.run(lambda output_dir: TestExperiment(config=config),
                         output_dir="/tmp"))

  def test_schedule_from_config_runs_train_and_evaluate_on_master(self):
    config = run_config.RunConfig(job_name="master")
    self.assertEqual(
        "train_and_evaluate",
        learn_runner.run(lambda output_dir: TestExperiment(config=config),
                         output_dir="/tmp"))

  def test_schedule_from_config_runs_serve_on_ps(self):
    config = run_config.RunConfig(job_name="ps")
    self.assertEqual(
        "serve",
        learn_runner.run(lambda output_dir: TestExperiment(config=config),
                         output_dir="/tmp"))

  def test_schedule_from_config_runs_train_on_worker(self):
    config = run_config.RunConfig(job_name="worker")
    self.assertEqual(
        "train",
        learn_runner.run(lambda output_dir: TestExperiment(config=config),
                         output_dir="/tmp"))

  def test_fail_no_output_dir(self):
    self.assertRaisesRegexp(ValueError, "Must specify an output directory",
                            learn_runner.run, build_experiment, "",
                            "simple_task")

  def test_fail_no_schedule_and_no_config(self):
    self.assertRaisesRegexp(ValueError, "Must specify a schedule",
                            learn_runner.run, build_experiment, "/tmp")

  def test_fail_job_name_with_no_default_schedule(self):
    self.assertRaisesRegexp(ValueError, "Must specify a schedule",
                            learn_runner.run, build_experiment, "/tmp")

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

  def test_fail_schedule_from_config_with_no_job_name(self):
    config = run_config.RunConfig(job_name=None)
    self.assertRaisesRegexp(
        ValueError,
        "Must specify a schedule",
        learn_runner.run,
        lambda output_dir: TestExperiment(config=config),
        output_dir="/tmp")


if __name__ == "__main__":
  tf.test.main()
