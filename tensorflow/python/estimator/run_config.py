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
"""Environment configuration object for Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six

from tensorflow.core.protobuf import config_pb2


# A list of the property names in RunConfig user allows to change.
_DEFAULT_REPLACEABLE_LIST = [
    'model_dir',
    'tf_random_seed',
    'save_summary_steps',
    'save_checkpoints_steps',
    'save_checkpoints_secs',
    'session_config',
    'keep_checkpoint_max',
    'keep_checkpoint_every_n_hours',
    'log_step_count_steps'
]

_SAVE_CKPT_ERR = (
    '`save_checkpoints_steps` and `save_checkpoints_secs` cannot be both set.'
)


def _validate_save_ckpt_with_replaced_keys(new_copy, replaced_keys):
  """Validates the save ckpt properties."""
  # Ensure one (and only one) of save_steps and save_secs is not None.
  # Also, if user sets one save ckpt property, say steps, the other one (secs)
  # should be set as None to improve usability.

  save_steps = new_copy.save_checkpoints_steps
  save_secs = new_copy.save_checkpoints_secs

  if ('save_checkpoints_steps' in replaced_keys and
      'save_checkpoints_secs' in replaced_keys):
    # If user sets both properties explicitly, we need to error out if both
    # are set or neither of them are set.
    if save_steps is not None and save_secs is not None:
      raise ValueError(_SAVE_CKPT_ERR)
  elif 'save_checkpoints_steps' in replaced_keys and save_steps is not None:
    new_copy._save_checkpoints_secs = None  # pylint: disable=protected-access
  elif 'save_checkpoints_secs' in replaced_keys and save_secs is not None:
    new_copy._save_checkpoints_steps = None  # pylint: disable=protected-access


def _validate_properties(run_config):
  """Validates the properties."""
  def _validate(property_name, cond, message):
    property_value = getattr(run_config, property_name)
    if property_value is not None and not cond(property_value):
      raise ValueError(message)

  _validate('model_dir', lambda dir: dir,
            message='model_dir should be non-empty')

  _validate('save_summary_steps', lambda steps: steps >= 0,
            message='save_summary_steps should be >= 0')

  _validate('save_checkpoints_steps', lambda steps: steps >= 0,
            message='save_checkpoints_steps should be >= 0')
  _validate('save_checkpoints_secs', lambda secs: secs >= 0,
            message='save_checkpoints_secs should be >= 0')

  _validate('session_config',
            lambda sc: isinstance(sc, config_pb2.ConfigProto),
            message='session_config must be instance of ConfigProto')

  _validate('keep_checkpoint_max', lambda keep_max: keep_max >= 0,
            message='keep_checkpoint_max should be >= 0')
  _validate('keep_checkpoint_every_n_hours', lambda keep_hours: keep_hours > 0,
            message='keep_checkpoint_every_n_hours should be > 0')
  _validate('log_step_count_steps', lambda num_steps: num_steps > 0,
            message='log_step_count_steps should be > 0')

  _validate('tf_random_seed', lambda seed: isinstance(seed, six.integer_types),
            message='tf_random_seed must be integer.')


class TaskType(object):
  MASTER = 'master'
  PS = 'ps'
  WORKER = 'worker'


class RunConfig(object):
  """This class specifies the configurations for an `Estimator` run."""

  def __init__(self):
    self._model_dir = None
    self._tf_random_seed = 1
    self._save_summary_steps = 100
    self._save_checkpoints_secs = 600
    self._save_checkpoints_steps = None
    self._session_config = None
    self._keep_checkpoint_max = 5
    self._keep_checkpoint_every_n_hours = 10000
    self._log_step_count_steps = 100
    _validate_properties(self)

  @property
  def cluster_spec(self):
    return None

  @property
  def evaluation_master(self):
    return ''

  @property
  def is_chief(self):
    return True

  @property
  def master(self):
    return ''

  @property
  def num_ps_replicas(self):
    return 0

  @property
  def num_worker_replicas(self):
    return 1

  @property
  def task_id(self):
    return 0

  @property
  def task_type(self):
    return TaskType.WORKER

  @property
  def tf_random_seed(self):
    return self._tf_random_seed

  @property
  def save_summary_steps(self):
    return self._save_summary_steps

  @property
  def save_checkpoints_secs(self):
    return self._save_checkpoints_secs

  @property
  def session_config(self):
    return self._session_config

  @property
  def save_checkpoints_steps(self):
    return self._save_checkpoints_steps

  @property
  def keep_checkpoint_max(self):
    return self._keep_checkpoint_max

  @property
  def keep_checkpoint_every_n_hours(self):
    return self._keep_checkpoint_every_n_hours

  @property
  def log_step_count_steps(self):
    return self._log_step_count_steps

  @property
  def model_dir(self):
    return self._model_dir

  def replace(self, **kwargs):
    """Returns a new instance of `RunConfig` replacing specified properties.

    Only the properties in the following list are allowed to be replaced:
      - `model_dir`.
      - `tf_random_seed`,
      - `save_summary_steps`,
      - `save_checkpoints_steps`,
      - `save_checkpoints_secs`,
      - `session_config`,
      - `keep_checkpoint_max`,
      - `keep_checkpoint_every_n_hours`,
      - `log_step_count_steps`,

    In addition, either `save_checkpoints_steps` or `save_checkpoints_secs`
    can be set (should not be both).

    Args:
      **kwargs: keyword named properties with new values.

    Raises:
      ValueError: If any property name in `kwargs` does not exist or is not
        allowed to be replaced, or both `save_checkpoints_steps` and
        `save_checkpoints_secs` are set.

    Returns:
      a new instance of `RunConfig`.
    """
    return self._replace(
        allowed_properties_list=_DEFAULT_REPLACEABLE_LIST, **kwargs)

  def _replace(self, allowed_properties_list=None, **kwargs):
    """See `replace`.

    N.B.: This implementation assumes that for key named "foo", the underlying
    property the RunConfig holds is "_foo" (with one leading underscore).

    Args:
      allowed_properties_list: The property name list allowed to be replaced.
      **kwargs: keyword named properties with new values.

    Raises:
      ValueError: If any property name in `kwargs` does not exist or is not
        allowed to be replaced, or both `save_checkpoints_steps` and
        `save_checkpoints_secs` are set.

    Returns:
      a new instance of `RunConfig`.
    """

    new_copy = copy.deepcopy(self)

    allowed_properties_list = allowed_properties_list or []

    for key, new_value in six.iteritems(kwargs):
      if key in allowed_properties_list:
        setattr(new_copy, '_' + key, new_value)
        continue

      raise ValueError(
          'Replacing {} is not supported. Allowed properties are {}.'.format(
              key, allowed_properties_list))

    _validate_save_ckpt_with_replaced_keys(new_copy, kwargs.keys())
    _validate_properties(new_copy)
    return new_copy
