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

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner


class TestExperiment(tf.contrib.learn.Experiment):

  def __init__(self, default=None):
    self.default = default

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

  def test_run(self):
    self.assertEqual(
        "simple_task, default=None.",
        learn_runner.run(
            build_experiment, output_dir="/tmp", schedule="simple_task"))

  def test_fail_no_output_dir(self):
    self.assertRaisesRegexp(ValueError, "Must specify an output directory",
                            learn_runner.run, build_experiment, "",
                            "simple_task")

  def test_fail_no_schedule(self):
    self.assertRaisesRegexp(ValueError, "Must specify a schedule",
                            learn_runner.run, build_experiment, "/tmp", "")

  def test_fail_non_callable(self):
    self.assertRaisesRegexp(TypeError, "Experiment builder .* is not callable",
                            learn_runner.run, "not callable", "/tmp",
                            "simple_test")

  def test_fail_not_experiment(self):
    self.assertRaisesRegexp(
        TypeError, "Experiment builder did not return an Experiment",
        learn_runner.run, build_non_experiment, "/tmp", "simple_test")

  def test_fail_non_existent_task(self):
    self.assertRaisesRegexp(
        ValueError, "Schedule references non-existent task",
        learn_runner.run, build_experiment, "/tmp", "mirage")

  def test_fail_non_callable_task(self):
    self.assertRaisesRegexp(
        TypeError, "Schedule references non-callable member",
        learn_runner.run, build_experiment, "/tmp", "default")


if __name__ == "__main__":
  tf.test.main()
