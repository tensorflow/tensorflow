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
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

patch = test.mock.patch


class RunConfigTest(test.TestCase):

  def test_defaults_with_no_tf_config(self):
    config = run_config.RunConfig()
    self.assertEquals(config.master, "")
    self.assertEquals(config.task_id, 0)
    self.assertEquals(config.num_ps_replicas, 0)
    self.assertEquals(config.cluster_spec, {})
    self.assertIsNone(config.task_type)
    self.assertTrue(config.is_chief)
    self.assertEquals(config.evaluation_master, "")

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
      config = run_config.RunConfig()

    self.assertEquals(config.master, "grpc://host4:4")
    self.assertEquals(config.task_id, 1)
    self.assertEquals(config.num_ps_replicas, 2)
    self.assertEquals(config.cluster_spec.as_dict(), tf_config["cluster"])
    self.assertEquals(config.task_type, run_config_lib.TaskType.WORKER)
    self.assertFalse(config.is_chief)
    self.assertEquals(config.evaluation_master, "")

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
      config = run_config.RunConfig(
          master="localhost:0", evaluation_master="localhost:9991")

    self.assertEquals(config.master, "localhost:0")
    self.assertEquals(config.task_id, 2)
    self.assertEquals(config.num_ps_replicas, 1)
    self.assertEquals(config.cluster_spec, server_lib.ClusterSpec(cluster_spec))
    self.assertEquals(config.task_type, run_config_lib.TaskType.WORKER)
    self.assertFalse(config.is_chief)
    self.assertEquals(config.evaluation_master, "localhost:9991")

  def test_single_node_in_cluster_spec_produces_empty_master(self):
    tf_config = {"cluster": {run_config_lib.TaskType.WORKER: ["host1:1"]}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      self.assertEquals(config.master, "")

  def test_no_task_type_produces_empty_master(self):
    tf_config = {
        "cluster": {
            run_config_lib.TaskType.PS: ["host1:1", "host2:2"],
            run_config_lib.TaskType.WORKER: ["host3:3", "host4:4", "host5:5"]
        },
        # Omits "task": {"type": "worker}
    }
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()
      self.assertEquals(config.master, "")

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
      run_config.RunConfig()

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
      run_config.RunConfig()

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
      config = run_config.RunConfig()

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
      config = run_config.RunConfig()

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
      config = run_config.RunConfig()

    self.assertFalse(config.is_chief)

  def test_default_is_chief_from_tf_config_without_job_name(self):
    tf_config = {"cluster": {}, "task": {}}
    with patch.dict("os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
      config = run_config.RunConfig()

    self.assertTrue(config.is_chief)


if __name__ == "__main__":
  test.main()
