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


FLAGS = learn_runner.FLAGS


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

  def setUp(self):
    # Make sure the flags exist. It's unclear why this is necessary.
    if not hasattr(FLAGS, "output_dir"):
      learn_runner.flags.DEFINE_string("output_dir", "/tmp", "Fake")
    if not hasattr(FLAGS, "schedule"):
      learn_runner.flags.DEFINE_string("schedule", "simple_task", "Fake")

  def test_run(self):
    FLAGS.output_dir = "/tmp"
    FLAGS.schedule = "simple_task"
    self.assertEqual("simple_task, default=None.",
                     learn_runner.run(build_experiment))

  def test_fail_no_output_dir(self):
    FLAGS.output_dir = ""
    FLAGS.schedule = "simple_test"
    self.assertRaisesRegexp(RuntimeError,
                            "Must specify an output directory",
                            learn_runner.run, build_experiment)

  def test_fail_no_schedule(self):
    FLAGS.output_dir = "/tmp"
    FLAGS.schedule = ""
    self.assertRaisesRegexp(RuntimeError, "Must specify a schedule",
                            learn_runner.run, build_experiment)

  def test_fail_non_callable(self):
    FLAGS.output_dir = "/tmp"
    FLAGS.schedule = "simple_test"
    self.assertRaisesRegexp(TypeError,
                            "Experiment builder .* is not callable",
                            learn_runner.run, "not callable")

  def test_fail_not_experiment(self):
    FLAGS.output_dir = "/tmp"
    FLAGS.schedule = "simple_test"
    self.assertRaisesRegexp(
        TypeError, "Experiment builder did not return an Experiment",
        learn_runner.run, build_non_experiment)

  def test_fail_non_existent_task(self):
    FLAGS.output_dir = "/tmp"
    FLAGS.schedule = "mirage"
    self.assertRaisesRegexp(
        ValueError, "Schedule references non-existent task",
        learn_runner.run, build_experiment)

  def test_fail_non_callable_task(self):
    FLAGS.output_dir = "/tmp"
    FLAGS.schedule = "default"
    self.assertRaisesRegexp(
        TypeError, "Schedule references non-callable member",
        learn_runner.run, build_experiment)


if __name__ == "__main__":
  tf.test.main()
