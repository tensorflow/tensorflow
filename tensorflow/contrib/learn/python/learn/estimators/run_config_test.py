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

import copy
import json

from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import run_config as core_run_config
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

TEST_DIR = "test_dir"
ANOTHER_TEST_DIR = "another_test_dir"
MASTER = "master_"
RANDOM_SEED = 123

patch = test.mock.patch


def _create_run_config_with_cluster_spec(tf_config_str):
  with patch.dict("os.environ", {"TF_CONFIG": tf_config_str}):
    return run_config_lib.RunConfig(
        tf_random_seed=RANDOM_SEED, model_dir=TEST_DIR)


class RunConfigTest(test.TestCase):

  def test_instance_of_core_run_config(self):
    config = run_config_lib.RunConfig()
    self.assertTrue(isinstance(config, core_run_config.RunConfig))

  def test_defaults_with_no_tf_config(self):
    config = run_config_lib.RunConfig()
    self.assertEqual(config.master, "")
    self.assertEqual(config.task_id, 0)
    self.assertEqual(config.num_ps_replicas, 0)
    self.assertEqual(config.cluster_spec, {})
    self.assertIsNone(config.task_type)
    self.assertTrue(config.is_chief)
    self.assertEqual(config.evaluation_master, "")

  def test_values_from_tf_config(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
        "task": {
            "type": run_config_lib.TaskType.WORKER,
            "index": 1
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()

    self.assertEqual(config.master, "grpc://host4:4")
    self.assertEqual(config.task_id, 1)
    self.assertEqual(config.num_ps_replicas, 2)
    self.assertEqual(config.num_worker_replicas, 3)
    self.assertEqual(config.cluster_spec.as_dict(), tf_config["cluster"])
    self.assertEqual(config.task_type, run_config_lib.TaskType.WORKER)
    self.assertFalse(config.is_chief)
    self.assertEqual(config.evaluation_master, "")

  def test_explicitly_specified_values(self):
    cluster_spec = {
        run_config_lib.TaskType.PS: ["localhost:9990"],
        "my_job_name": ["localhost:9991", "localhost:9992", "localhost:0"]
    }
    tf_config = {
        "cluster": cluster_spec,
        "task": {
            "type": run_config_lib.TaskType.WORKER,
            "index": 2
        }
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig(
          master="localhost:0", evaluation_master="localhost:9991")

    self.assertEqual(config.master, "localhost:0")
    self.assertEqual(config.task_id, 2)
    self.assertEqual(config.num_ps_replicas, 1)
    self.assertEqual(config.num_worker_replicas, 0)
    self.assertEqual(config.cluster_spec, server_lib.ClusterSpec(cluster_spec))
    self.assertEqual(config.task_type, run_config_lib.TaskType.WORKER)
    self.assertFalse(config.is_chief)
    self.assertEqual(config.evaluation_master, "localhost:9991")

  def test_single_node_in_cluster_spec_produces_empty_master(self):
    tf_config = {"cluster": {run_config_lib.TaskType.WORKER: ["host1:1"]}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      self.assertEqual(config.master, "")

  def test_no_task_type_produces_empty_master(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
        # Omits "task": {"type": "worker}
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()
      self.assertEqual(config.master, "")

  def test_invalid_job_name_raises(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
        "task": {
            "type": "not_in_cluster_spec"
        }
    }
    expected_msg_regexp = "not_in_cluster_spec is not a valid task"
    with patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), self.assertRaisesRegexp(
            ValueError, expected_msg_regexp):
      run_config_lib.RunConfig()

  def test_illegal_task_index_raises(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
        "task": {
            "type": run_config_lib.TaskType.WORKER,
            "index": 3
        }
    }
    expected_msg_regexp = "3 is not a valid task_id"
    with patch.dict(
        "os.environ",
        {"TF_CONFIG": json.dumps(tf_config)}), self.assertRaisesRegexp(
            ValueError, expected_msg_regexp):
      run_config_lib.RunConfig()

  def test_is_chief_from_cloud_tf_config(self):
    # is_chief should be true when ["task"]["type"] == "master" and
    # index == 0 and ["task"]["environment"] == "cloud". Note that
    # test_values_from_tf_config covers the non-master case.
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.MASTER: ["host3:3"],
            run_config_lib.TaskType.WORKER: ["host4:4", "host5:5", "host6:6"]
        },
        "task": {
            "type": run_config_lib.TaskType.MASTER,
            "index": 0
        },
        "environment": "cloud"
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()

    self.assertTrue(config.is_chief)

  def test_is_chief_from_noncloud_tf_config(self):
    # is_chief should be true when ["task"]["type"] == "worker" and
    # index == 0 if ["task"]["environment"] != "cloud".
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.MASTER: ["host3:3"],
            run_config_lib.TaskType.WORKER: ["host4:4", "host5:5", "host6:6"]
        },
        "task": {
            "type": run_config_lib.TaskType.WORKER,
            "index": 0
        },
        "environment": "random"
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()

    self.assertTrue(config.is_chief)

    # But task 0 for a job named "master" should not be.
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.MASTER: ["host3:3"],
            run_config_lib.TaskType.WORKER: ["host4:4", "host5:5", "host6:6"]
        },
        "task": {
            "type": run_config_lib.TaskType.MASTER,
            "index": 0
        },
        "environment": "random"
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()

    self.assertFalse(config.is_chief)

  def test_default_is_chief_from_tf_config_without_job_name(self):
    tf_config = {"cluster": {}, "task": {}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config_lib.RunConfig()

    self.assertTrue(config.is_chief)

  def test_model_dir(self):
    empty_config = run_config_lib.RunConfig()
    self.assertIsNone(empty_config.model_dir)

    config = run_config_lib.RunConfig(model_dir=TEST_DIR)
    self.assertEqual(TEST_DIR, config.model_dir)

  def test_model_dir_in_tf_config(self):
    tf_config = {"model_dir": TEST_DIR}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      run_config = run_config_lib.RunConfig()
    self.assertEqual(TEST_DIR, run_config.model_dir)

  def test_model_dir_both_in_tf_config_and_constructor(self):
    tf_config = {"model_dir": TEST_DIR}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      run_config = run_config_lib.RunConfig(model_dir=TEST_DIR)
    self.assertEqual(TEST_DIR, run_config.model_dir)

  def test_model_dir_fail_if_constructor_value_mismatch_tf_config(self):
    tf_config = {"model_dir": TEST_DIR}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      with self.assertRaisesRegexp(
          ValueError,
          "`model_dir` provided in RunConfig .* must have "
          "the same value .* in TF_CONFIG"):
        run_config_lib.RunConfig(model_dir=TEST_DIR + "/sub_dir")

  def test_replace(self):
    config = run_config_lib.RunConfig(
        tf_random_seed=RANDOM_SEED, model_dir=TEST_DIR)
    self.assertEqual(TEST_DIR, config.model_dir)
    self.assertEqual(RANDOM_SEED, config.tf_random_seed)

    new_config = config.replace(model_dir=ANOTHER_TEST_DIR)
    self.assertEqual(ANOTHER_TEST_DIR, new_config.model_dir)
    self.assertEqual(RANDOM_SEED, new_config.tf_random_seed)
    self.assertEqual(RANDOM_SEED, config.tf_random_seed)

  def test_uid_for_different_configs(self):
    config = run_config_lib.RunConfig(
        tf_random_seed=RANDOM_SEED, model_dir=TEST_DIR)

    expected_uid = config.uid()
    # Check for 10 times, which should prove something.
    for _ in range(10):
      self.assertEqual(expected_uid, config.uid())

    new_config = config.replace(model_dir=ANOTHER_TEST_DIR)
    self.assertEqual(TEST_DIR, config.model_dir)
    self.assertNotEqual(expected_uid, new_config.uid())
    self.assertEqual(ANOTHER_TEST_DIR, new_config.model_dir)

  def test_uid_for_whitelist(self):
    whitelist = ["model_dir"]
    config = run_config_lib.RunConfig(
        tf_random_seed=RANDOM_SEED, model_dir=TEST_DIR)

    expected_uid = config.uid(whitelist)
    self.assertEqual(expected_uid, config.uid(whitelist))

    new_config = config.replace(model_dir=ANOTHER_TEST_DIR)
    self.assertEqual(TEST_DIR, config.model_dir)
    self.assertEqual(expected_uid, new_config.uid(whitelist))
    self.assertEqual(ANOTHER_TEST_DIR, new_config.model_dir)

  def test_uid_for_default_whitelist(self):
    config = run_config_lib.RunConfig(
        tf_random_seed=11,
        save_summary_steps=12,
        save_checkpoints_steps=13,
        save_checkpoints_secs=14,
        session_config=config_pb2.ConfigProto(allow_soft_placement=True),
        keep_checkpoint_max=16,
        keep_checkpoint_every_n_hours=17)
    self.assertEqual(11, config.tf_random_seed)
    self.assertEqual(12, config.save_summary_steps)
    self.assertEqual(13, config.save_checkpoints_steps)
    self.assertEqual(14, config.save_checkpoints_secs)
    self.assertEqual(config_pb2.ConfigProto(allow_soft_placement=True),
                     config.session_config)
    self.assertEqual(16, config.keep_checkpoint_max)
    self.assertEqual(17, config.keep_checkpoint_every_n_hours)

    new_config = run_config_lib.RunConfig(
        tf_random_seed=21,
        save_summary_steps=22,
        save_checkpoints_steps=23,
        save_checkpoints_secs=24,
        session_config=config_pb2.ConfigProto(allow_soft_placement=False),
        keep_checkpoint_max=26,
        keep_checkpoint_every_n_hours=27)
    self.assertEqual(config.uid(), new_config.uid())
    # model_dir is not on the default whitelist.
    self.assertNotEqual(config.uid(whitelist=[]),
                        new_config.uid(whitelist=[]))
    new_config = new_config.replace(model_dir=ANOTHER_TEST_DIR)
    self.assertNotEqual(config.uid(), new_config.uid())

  def test_uid_for_deepcopy(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
        "task": {
            "type": run_config_lib.TaskType.WORKER,
            "index": 1
        }
    }

    config = _create_run_config_with_cluster_spec(json.dumps(tf_config))
    expected_uid = config.uid()
    self.assertEqual(tf_config["cluster"], config.cluster_spec.as_dict())

    new_config = copy.deepcopy(config)
    self.assertEqual(tf_config["cluster"], new_config.cluster_spec.as_dict())
    self.assertEqual(expected_uid, new_config.uid())

  def test_uid_for_different_cluster_spec_order(self):
    tf_config_1_str = (
        "{\"cluster\": {\"ps\": [\"host1:1\", \"host2:2\"], "
        "\"worker\": [\"host3:3\", \"host4:4\", \"host5:5\"]}}")

    tf_config_2_str = (
        "{\"cluster\": {\"worker\": [\"host3:3\", \"host4:4\", \"host5:5\"],"
        "\"ps\": [\"host1:1\", \"host2:2\"]}}")

    # Wraps in a loop to check flakiness.
    for _ in range(100):
      uid_1 = _create_run_config_with_cluster_spec(tf_config_1_str).uid()
      uid_2 = _create_run_config_with_cluster_spec(tf_config_2_str).uid()
      self.assertEqual(uid_1, uid_2)

  def test_uid_for_different_cluster_specs(self):
    tf_config_1 = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
    }

    tf_config_2 = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
    }

    uid_1 = _create_run_config_with_cluster_spec(json.dumps(tf_config_1)).uid()
    uid_2 = _create_run_config_with_cluster_spec(json.dumps(tf_config_2)).uid()
    self.assertNotEqual(uid_1, uid_2)

  def test_num_worker_replicas_counts_in_master_too(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.MASTER: ["host6:6"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"],
        },
        "task": {
            "type": run_config_lib.TaskType.WORKER,
            "index": 1
        }
    }

    config = _create_run_config_with_cluster_spec(json.dumps(tf_config))
    self.assertEqual(config.num_worker_replicas, 4)


if __name__ == "__main__":
  test.main()
