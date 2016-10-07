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
"""run_config.py tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import run_config

patch = tf.test.mock.patch


class RunConfigTest(tf.test.TestCase):

  def test_defaults_with_no_tf_config(self):
    config = run_config.RunConfig()
    self.assertEquals(config.master, "")
    self.assertEquals(config.task, 0)
    self.assertEquals(config.num_ps_replicas, 0)
    self.assertIsNone(config.cluster_spec)
    self.assertIsNone(config.job_name)
    self.assertTrue(config.is_chief)
    self.assertEquals(config.evaluation_master, "")

  def test_values_from_tf_config(self):
    tf_config = {"cluster": {"ps": ["host1:1", "host2:2"],
                             "worker": ["host3:3", "host4:4", "host5:5"]},
                 "task": {"type": "worker",
                          "index": 1}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()

    self.assertEquals(config.master, "grpc://host4:4")
    self.assertEquals(config.task, 1)
    self.assertEquals(config.num_ps_replicas, 2)
    self.assertEquals(config.cluster_spec.as_dict(), tf_config["cluster"])
    self.assertEquals(config.job_name, "worker")
    self.assertFalse(config.is_chief)
    self.assertEquals(config.evaluation_master, "")

  def test_explicitly_specified_values(self):
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["localhost:9990"],
        "my_job_name": ["localhost:9991", "localhost:9992", "localhost:0"]
    })
    config = run_config.RunConfig(
        master="localhost:0",
        task=2,
        job_name="my_job_name",
        cluster_spec=cluster_spec,
        evaluation_master="localhost:9991"
    )

    self.assertEquals(config.master, "localhost:0")
    self.assertEquals(config.task, 2)
    self.assertEquals(config.num_ps_replicas, 1)
    self.assertEquals(config.cluster_spec, cluster_spec)
    self.assertEquals(config.job_name, "my_job_name")
    self.assertFalse(config.is_chief)
    self.assertEquals(config.evaluation_master, "localhost:9991")

  def test_tf_config_with_overrides(self):
    # Purpose: to test the case where TF_CONFIG is set, but then
    # values are overridden by manually passing them to the constructor.

    # Setup the TF_CONFIG environment variable
    tf_config = {"cluster": {"ps": ["host1:1", "host2:2"],
                             "worker": ["host3:3", "host4:4", "host5:5"]},
                 "task": {"type": "worker",
                          "index": 1}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      # Run, but override all of the values that would otherwise have been
      # set by TF_CONFIG.
      cluster_spec_override = tf.train.ClusterSpec({
          "ps": ["my_host1:314"],
          "my_job_name": ["my_host2:314", "my_host4:314", "my_host5:314"],
          "master": ["my_host6:313"]
      })
      config = run_config.RunConfig(
          master="my_master",
          task=0,
          job_name="master",
          cluster_spec=cluster_spec_override,
          is_chief=True)

    # To protect against changes to the test itself (either
    # the TF_CONFIG variable or the manual overrides), we will assert
    # that the overrides are in fact different than TF_CONFIG.
    self.assertNotEquals(tf_config["cluster"], cluster_spec_override)
    self.assertNotIn("my_job_name", tf_config["cluster"])

    # Now we assert that the correct values were indeed returned.
    self.assertEquals(config.master, "my_master")
    self.assertEquals(config.task, 0)
    self.assertEquals(config.num_ps_replicas, 1)
    self.assertEquals(config.cluster_spec, cluster_spec_override)
    self.assertEquals(config.job_name, "master")
    self.assertTrue(config.is_chief)

  def test_explicitly_setting_task_to_0_overrides_tf_config(self):
    # Setup the TF_CONFIG environment variable
    tf_config = {"task": {"index": 1}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig(task=0)
    self.assertEquals(config.task, 0)

  def test_num_ps_replicas_and_cluster_spec_are_mutually_exclusive(self):
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ["host1:1", "host2:2"],
         "worker": ["host3:3", "host4:4", "host5:5"]})
    expected_msg_regexp = "Cannot specify both num_ps_replicas and cluster_spec"
    with self.assertRaisesRegexp(ValueError, expected_msg_regexp):
      run_config.RunConfig(
          num_ps_replicas=2,
          cluster_spec=cluster_spec,)

  def test_num_ps_replicas_from_tf_config(self):
    tf_config = {"cluster": {"ps": ["host1:1", "host2:2"],
                             "worker": ["host3:3", "host4:4", "host5:5"]},
                 "task": {"type": "worker",
                          "index": 1}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      expected_msg_regexp = ("Cannot specify both num_ps_replicas and "
                             "cluster_spec.*cluster_spec may have been set in "
                             "the TF_CONFIG environment variable")
      with self.assertRaisesRegexp(ValueError, expected_msg_regexp):
        run_config.RunConfig(num_ps_replicas=2)

  def test_no_cluster_spec_results_in_empty_master(self):
    config = run_config.RunConfig()
    self.assertEquals(config.master, "")

  def test_single_node_in_cluster_spec_produces_empty_master(self):
    cluster_spec = tf.train.ClusterSpec({"worker": ["host1:1"]})
    config = run_config.RunConfig(cluster_spec=cluster_spec)
    self.assertEquals(config.master, "")

  def test_no_job_name_produces_empty_master(self):
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ["host1:1", "host2:2"],
         "worker": ["host3:3", "host4:4", "host5:5"]})
    # NB: omitted job_name; better to omit than explictly set to None
    # as this better mimics client behavior.
    config = run_config.RunConfig(cluster_spec=cluster_spec)
    self.assertEquals(config.master, "")

  def test_invalid_job_name_raises(self):
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ["host1:1", "host2:2"],
         "worker": ["host3:3", "host4:4", "host5:5"]})
    expected_msg_regexp = "not_in_cluster_spec is not a valid task"
    with self.assertRaisesRegexp(ValueError, expected_msg_regexp):
      run_config.RunConfig(
          cluster_spec=cluster_spec, job_name="not_in_cluster_spec")

  def test_illegal_task_index_raises(self):
    cluster_spec = tf.train.ClusterSpec(
        {"ps": ["host1:1", "host2:2"],
         "worker": ["host3:3", "host4:4", "host5:5"]})
    expected_msg_regexp = "3 is not a valid task index"
    with self.assertRaisesRegexp(ValueError, expected_msg_regexp):
      run_config.RunConfig(
          cluster_spec=cluster_spec, job_name="worker", task=3)

  def test_empty_cluster_spec(self):
    config = run_config.RunConfig(cluster_spec=tf.train.ClusterSpec({}))
    self.assertEquals(config.cluster_spec.as_dict(), {})

  def test_num_ps_replicas_can_be_set_if_cluster_spec_is_empty(self):
    config = run_config.RunConfig(
        num_ps_replicas=2,
        cluster_spec=tf.train.ClusterSpec({}))
    # Basically, just make sure no exception is being raised.
    self.assertEquals(config.num_ps_replicas, 2)

  def test_is_chief_from_tf_config(self):
    # is_chief should be true when ["task"]["type"] == "master" and
    # index == 0. Note that test_values_from_tf_config covers the
    # non-master case.
    tf_config = {"cluster": {"ps": ["host1:1", "host2:2"],
                             "master": ["host3:3"],
                             "worker": ["host4:4", "host5:5", "host6:6"]},
                 "task": {"type": "master",
                          "index": 0}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()

    self.assertTrue(config.is_chief)

  def test_default_is_chief_from_tf_config_without_job_name(self):
    tf_config = {"cluster": {},
                 "task": {}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()

    self.assertTrue(config.is_chief)

  def test_default_is_chief_without_tf_config_or_job_name(self):
    # When is_chief is omitted, there is no TF_CONFIG and no job_name
    # (legacy behavior), then is_chief should be True iff task == 0.
    config = run_config.RunConfig(task=0)
    self.assertTrue(config.is_chief)

    config = run_config.RunConfig(task=1)
    self.assertFalse(config.is_chief)

  def test_default_is_chief_without_tf_config_but_has_job_name(self):
    # When is_chief is omitted, there is no TF_CONFIG but there is a job_name,
    # then is_chief is True iff job_name is "worker" and task == 0.
    config = run_config.RunConfig(job_name="worker", task=0)
    self.assertTrue(config.is_chief)

    config = run_config.RunConfig(
        job_name="worker", task=1)
    self.assertFalse(config.is_chief)

    config = run_config.RunConfig(job_name="ps", task=0)
    self.assertFalse(config.is_chief)

    config = run_config.RunConfig(job_name="ps", task=1)
    self.assertFalse(config.is_chief)

  def test_bad_is_chief_combinations_raise(self):
    msg = "Task is 1, but only task 0 may be chief"
    with self.assertRaisesRegexp(ValueError, msg):
      run_config.RunConfig(is_chief=True, task=1)

    msg = "job_name is \'ps\', but only masters or workers may be chiefs"
    with self.assertRaisesRegexp(ValueError, msg):
      run_config.RunConfig(is_chief=True, task=0, job_name="ps")

    with self.assertRaisesRegexp(ValueError, "Master task 0 must be chief"):
      run_config.RunConfig(is_chief=False, task=0, job_name="master")


if __name__ == "__main__":
  tf.test.main()
