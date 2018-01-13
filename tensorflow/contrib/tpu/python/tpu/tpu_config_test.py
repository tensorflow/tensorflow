# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TPU RunConfig tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.contrib.tpu.python.tpu import tpu_config as tpu_config_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.platform import test


def _set_tf_config_env_variable(tf_config):
  return test.mock.patch.dict('os.environ', {
      'TF_CONFIG': json.dumps(tf_config)
  })


class TPURunConfigTest(test.TestCase):

  def test_fail_with_invalid_num_shards(self):
    with self.assertRaisesRegexp(ValueError, 'must be positive'):
      tpu_config_lib.RunConfig(
          tpu_config=tpu_config_lib.TPUConfig(num_shards=0))

  def test_fail_with_iterations_per_loop(self):
    with self.assertRaisesRegexp(ValueError, 'must be positive'):
      tpu_config_lib.RunConfig(
          tpu_config=tpu_config_lib.TPUConfig(iterations_per_loop=0))


class TPURunConfigMasterTest(test.TestCase):

  def test_default_values(self):
    run_config = tpu_config_lib.RunConfig()
    self.assertEqual('', run_config.master)
    self.assertEqual('', run_config.evaluation_master)

  def test_user_provided_master_and_evaluation_master(self):
    run_config = tpu_config_lib.RunConfig(
        master='_master_123', evaluation_master='_eval_master_123')
    self.assertEqual('_master_123', run_config.master)
    self.assertEqual('_eval_master_123', run_config.evaluation_master)

  def test_evaluation_master_defaults_to_master(self):
    run_config = tpu_config_lib.RunConfig(master='_master_123')
    self.assertEqual('_master_123', run_config.master)
    self.assertEqual('_master_123', run_config.evaluation_master)

  def test_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
        'eval_session_master': '_eval_master_123'
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig()
      self.assertEqual('_master_123', run_config.master)
      self.assertEqual('_eval_master_123', run_config.evaluation_master)

  def test_evaluation_master_defaults_to_master_in_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig()
      self.assertEqual('_master_123', run_config.master)
      self.assertEqual('_master_123', run_config.evaluation_master)

  def test_respect_evaluation_master_in_tf_config(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
        },
        'task': {
            'type': run_config_lib.TaskType.EVALUATOR,
            'index': 0
        },
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig(master='_something')
      self.assertEqual('', run_config.evaluation_master)

  def test_user_overwrites_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
        'eval_session_master': '_eval_master_123'
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig(
          master='_new_master_123', evaluation_master='_new_eval_master_123')
      self.assertEqual('_new_master_123', run_config.master)
      self.assertEqual('_new_eval_master_123', run_config.evaluation_master)

  def test_user_overwrites_master_in_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
        'eval_session_master': '_eval_master_123'
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig(master='_new_master_123')
      self.assertEqual('_new_master_123', run_config.master)
      self.assertEqual('_eval_master_123', run_config.evaluation_master)


class TPUJobNameTest(test.TestCase):

  def test_default_name(self):
    config = tpu_config_lib.RunConfig()
    self.assertIsNone(config.tpu_config.tpu_job_name)

  def test_with_tf_config(self):
    tf_config = {'service': {'tpu_worker_job_name': '_my_new_name',}}
    with _set_tf_config_env_variable(tf_config):
      config = tpu_config_lib.RunConfig()
      self.assertEqual('_my_new_name', config.tpu_config.tpu_job_name)


if __name__ == '__main__':
  test.main()
