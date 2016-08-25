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

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import run_config


class RunConfigTest(tf.test.TestCase):

  def test_defaults(self):
    config = run_config.RunConfig()
    self.assertEquals(config.master, "")
    self.assertEquals(config.task, 0)
    self.assertEquals(config.num_ps_replicas, 0)
    self.assertEquals(config.cluster_spec, None)
    self.assertEquals(config.job_name, None)

  def test_explicitly_specified_values(self):
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["localhost:9990"],
        "my_job_name": ["localhost:9991", "localhost:9992", "localhost:0"]
    })
    config = run_config.RunConfig(
        master="localhost:0",
        task=2,
        job_name="my_job_name",
        cluster_spec=cluster_spec,)

    self.assertEquals(config.master, "localhost:0")
    self.assertEquals(config.task, 2)
    self.assertEquals(config.cluster_spec, cluster_spec)
    self.assertEquals(config.job_name, "my_job_name")


if __name__ == "__main__":
  tf.test.main()
