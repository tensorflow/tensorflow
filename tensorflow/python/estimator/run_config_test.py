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
"""RunConfig tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.platform import test

_TEST_DIR = 'test_dir'
_MASTER = 'master_'
_NOT_SUPPORTED_REPLACE_PROPERTY_MSG = 'Replacing .*is not supported'
_SAVE_CKPT_ERR = (
    '`save_checkpoints_steps` and `save_checkpoints_secs` cannot be both set.'
)
_MODEL_DIR_ERR = 'model_dir should be non-empty'
_SAVE_SUMMARY_STEPS_ERR = 'save_summary_steps should be >= 0'
_SAVE_CKPT_STEPS_ERR = 'save_checkpoints_steps should be >= 0'
_SAVE_CKPT_SECS_ERR = 'save_checkpoints_secs should be >= 0'
_SESSION_CONFIG_ERR = 'session_config must be instance of ConfigProto'
_KEEP_CKPT_MAX_ERR = 'keep_checkpoint_max should be >= 0'
_KEEP_CKPT_HOURS_ERR = 'keep_checkpoint_every_n_hours should be > 0'
_TF_RANDOM_SEED_ERR = 'tf_random_seed must be integer'


class RunConfigTest(test.TestCase):

  def test_default_property_values(self):
    config = run_config_lib.RunConfig()
    self.assertIsNone(config.model_dir)
    self.assertIsNone(config.session_config)
    self.assertEqual(1, config.tf_random_seed)
    self.assertEqual(100, config.save_summary_steps)
    self.assertEqual(600, config.save_checkpoints_secs)
    self.assertIsNone(config.save_checkpoints_steps)
    self.assertEqual(5, config.keep_checkpoint_max)
    self.assertEqual(10000, config.keep_checkpoint_every_n_hours)

  def test_model_dir(self):
    empty_config = run_config_lib.RunConfig()
    self.assertIsNone(empty_config.model_dir)

    new_config = empty_config.replace(model_dir=_TEST_DIR)
    self.assertEqual(_TEST_DIR, new_config.model_dir)

  def test_replace_with_allowed_properties(self):
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)

    config = run_config_lib.RunConfig().replace(
        tf_random_seed=11,
        save_summary_steps=12,
        save_checkpoints_secs=14,
        session_config=session_config,
        keep_checkpoint_max=16,
        keep_checkpoint_every_n_hours=17)
    self.assertEqual(11, config.tf_random_seed)
    self.assertEqual(12, config.save_summary_steps)
    self.assertEqual(14, config.save_checkpoints_secs)
    self.assertEqual(session_config, config.session_config)
    self.assertEqual(16, config.keep_checkpoint_max)
    self.assertEqual(17, config.keep_checkpoint_every_n_hours)

  def test_replace_none_value(self):
    config = run_config_lib.RunConfig().replace(
        tf_random_seed=None,
        model_dir=None,
        save_summary_steps=None,
        save_checkpoints_secs=None,
        save_checkpoints_steps=None,
        session_config=None,
        keep_checkpoint_max=None,
        keep_checkpoint_every_n_hours=None)
    self.assertIsNone(config.tf_random_seed)
    self.assertIsNone(config.model_dir)
    self.assertIsNone(config.save_summary_steps)
    self.assertIsNone(config.save_checkpoints_secs)
    self.assertIsNone(config.save_checkpoints_steps)
    self.assertIsNone(config.session_config)
    self.assertIsNone(config.keep_checkpoint_max)
    self.assertIsNone(config.keep_checkpoint_every_n_hours)

  def test_replace_with_disallowallowed_properties(self):
    config = run_config_lib.RunConfig()
    with self.assertRaises(ValueError):
      # tf_random_seed is not allowed to be replaced.
      config.replace(master='_master')
    with self.assertRaises(ValueError):
      config.replace(some_undefined_property=123)

  def test_replace(self):
    config = run_config_lib.RunConfig()

    with self.assertRaisesRegexp(
        ValueError, _NOT_SUPPORTED_REPLACE_PROPERTY_MSG):
      # master is not allowed to be replaced.
      config.replace(master=_MASTER)

    with self.assertRaisesRegexp(
        ValueError, _NOT_SUPPORTED_REPLACE_PROPERTY_MSG):
      config.replace(some_undefined_property=_MASTER)

  def test_replace_invalid_values(self):
    config = run_config_lib.RunConfig()

    with self.assertRaisesRegexp(ValueError, _MODEL_DIR_ERR):
      config.replace(model_dir='')
    with self.assertRaisesRegexp(ValueError, _SAVE_SUMMARY_STEPS_ERR):
      config.replace(save_summary_steps=-1)
    with self.assertRaisesRegexp(ValueError, _SAVE_CKPT_STEPS_ERR):
      config.replace(save_checkpoints_steps=-1)
    with self.assertRaisesRegexp(ValueError, _SAVE_CKPT_SECS_ERR):
      config.replace(save_checkpoints_secs=-1)
    with self.assertRaisesRegexp(ValueError, _SESSION_CONFIG_ERR):
      config.replace(session_config={})
    with self.assertRaisesRegexp(ValueError, _KEEP_CKPT_MAX_ERR):
      config.replace(keep_checkpoint_max=-1)
    with self.assertRaisesRegexp(ValueError, _KEEP_CKPT_HOURS_ERR):
      config.replace(keep_checkpoint_every_n_hours=0)
    with self.assertRaisesRegexp(ValueError, _TF_RANDOM_SEED_ERR):
      config.replace(tf_random_seed=1.0)

  def test_init_with_allowed_properties(self):
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)

    config = run_config_lib.RunConfig(
        tf_random_seed=11,
        save_summary_steps=12,
        save_checkpoints_secs=14,
        session_config=session_config,
        keep_checkpoint_max=16,
        keep_checkpoint_every_n_hours=17)
    self.assertEqual(11, config.tf_random_seed)
    self.assertEqual(12, config.save_summary_steps)
    self.assertEqual(14, config.save_checkpoints_secs)
    self.assertEqual(session_config, config.session_config)
    self.assertEqual(16, config.keep_checkpoint_max)
    self.assertEqual(17, config.keep_checkpoint_every_n_hours)

  def test_init_none_value(self):
    config = run_config_lib.RunConfig(
        tf_random_seed=None,
        model_dir=None,
        save_summary_steps=None,
        save_checkpoints_secs=None,
        save_checkpoints_steps=None,
        session_config=None,
        keep_checkpoint_max=None,
        keep_checkpoint_every_n_hours=None)
    self.assertIsNone(config.tf_random_seed)
    self.assertIsNone(config.model_dir)
    self.assertIsNone(config.save_summary_steps)
    self.assertIsNone(config.save_checkpoints_secs)
    self.assertIsNone(config.save_checkpoints_steps)
    self.assertIsNone(config.session_config)
    self.assertIsNone(config.keep_checkpoint_max)
    self.assertIsNone(config.keep_checkpoint_every_n_hours)

  def test_init_invalid_values(self):
    with self.assertRaisesRegexp(ValueError, _MODEL_DIR_ERR):
      run_config_lib.RunConfig(model_dir='')
    with self.assertRaisesRegexp(ValueError, _SAVE_SUMMARY_STEPS_ERR):
      run_config_lib.RunConfig(save_summary_steps=-1)
    with self.assertRaisesRegexp(ValueError, _SAVE_CKPT_STEPS_ERR):
      run_config_lib.RunConfig(save_checkpoints_steps=-1)
    with self.assertRaisesRegexp(ValueError, _SAVE_CKPT_SECS_ERR):
      run_config_lib.RunConfig(save_checkpoints_secs=-1)
    with self.assertRaisesRegexp(ValueError, _SESSION_CONFIG_ERR):
      run_config_lib.RunConfig(session_config={})
    with self.assertRaisesRegexp(ValueError, _KEEP_CKPT_MAX_ERR):
      run_config_lib.RunConfig(keep_checkpoint_max=-1)
    with self.assertRaisesRegexp(ValueError, _KEEP_CKPT_HOURS_ERR):
      run_config_lib.RunConfig(keep_checkpoint_every_n_hours=0)
    with self.assertRaisesRegexp(ValueError, _TF_RANDOM_SEED_ERR):
      run_config_lib.RunConfig(tf_random_seed=1.0)


class RunConfigSaveCheckpointsTest(test.TestCase):

  def test_save_checkpoint(self):
    empty_config = run_config_lib.RunConfig()
    self.assertEqual(600, empty_config.save_checkpoints_secs)
    self.assertIsNone(empty_config.save_checkpoints_steps)

    config_with_steps = empty_config.replace(save_checkpoints_steps=100)
    del empty_config
    self.assertEqual(100, config_with_steps.save_checkpoints_steps)
    self.assertIsNone(config_with_steps.save_checkpoints_secs)

    config_with_secs = config_with_steps.replace(save_checkpoints_secs=200)
    del config_with_steps
    self.assertEqual(200, config_with_secs.save_checkpoints_secs)
    self.assertIsNone(config_with_secs.save_checkpoints_steps)

  def test_save_checkpoint_both_steps_and_secs_are_not_none(self):
    empty_config = run_config_lib.RunConfig()
    with self.assertRaisesRegexp(ValueError, _SAVE_CKPT_ERR):
      empty_config.replace(save_checkpoints_steps=100,
                           save_checkpoints_secs=200)

    with self.assertRaisesRegexp(ValueError, _SAVE_CKPT_ERR):
      run_config_lib.RunConfig(save_checkpoints_steps=100,
                               save_checkpoints_secs=200)

  def test_save_checkpoint_both_steps_and_secs_are_none(self):
    config_with_secs = run_config_lib.RunConfig()
    config_without_ckpt = config_with_secs.replace(
        save_checkpoints_steps=None, save_checkpoints_secs=None)
    self.assertIsNone(config_without_ckpt.save_checkpoints_steps)
    self.assertIsNone(config_without_ckpt.save_checkpoints_secs)

  def test_save_checkpoint_flip_secs_to_none(self):
    config_with_secs = run_config_lib.RunConfig()
    config_without_ckpt = config_with_secs.replace(save_checkpoints_secs=None)
    self.assertIsNone(config_without_ckpt.save_checkpoints_steps)
    self.assertIsNone(config_without_ckpt.save_checkpoints_secs)

  def test_save_checkpoint_flip_steps_to_none(self):
    config_with_steps = run_config_lib.RunConfig().replace(
        save_checkpoints_steps=100)
    config_without_ckpt = config_with_steps.replace(save_checkpoints_steps=None)
    self.assertIsNone(config_without_ckpt.save_checkpoints_steps)
    self.assertIsNone(config_without_ckpt.save_checkpoints_secs)


if __name__ == '__main__':
  test.main()
