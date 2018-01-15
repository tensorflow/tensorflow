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
# ===================================================================

"""A RunConfig subclass with TPU support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from tensorflow.contrib.tpu.python.tpu import util as util_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.platform import tf_logging as logging

# pylint: disable=protected-access
_TF_CONFIG_ENV = run_config_lib._TF_CONFIG_ENV
_SERVICE_KEY = run_config_lib._SERVICE_KEY
_TPU_WORKER_JOB_NAME = 'tpu_worker_job_name'

# pylint: enable=protected-access


class TPUConfig(
    collections.namedtuple('TPUConfig', [
        'iterations_per_loop',
        'num_shards',
        'per_host_input_for_training',
        'tpu_job_name',
        'initial_infeed_sleep_secs',
    ])):
  """TPU related configuration required by `TPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps runnining in TPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
    num_shards: The number of TPU shards in the system.
    per_host_input_for_training: If `True`, `input_fn` is invoked Per-Host
      rather than Per-Core. With Per-Host input pipeline deployment, `input_fn`
      is invoked once on each host. To be precise, with a global batch size
      `train_batch_size` in `TPUEstimator` constructor, the batch size for each
      shard is `train_batch_size` // #hosts. With Per-Core input pipeline
      deployment, the shard batch size is `train_batch_size` // #cores.
    tpu_job_name: The name of the TPU job. Typically, this name is auto-inferred
      within TPUEstimator, however when using ClusterSpec propagation in more
      esoteric cluster configurations, you may need to specify the job name as a
      string.
    initial_infeed_sleep_secs: The number of seconds the infeed thread should
      wait before enqueueing the first batch. This helps avoid timeouts for
      models that require a long compilation time.
  """

  def __new__(cls,
              iterations_per_loop=2,
              num_shards=2,
              per_host_input_for_training=True,
              tpu_job_name=None,
              initial_infeed_sleep_secs=None):

    # Check iterations_per_loop.
    util_lib.check_positive_integer(iterations_per_loop,
                                    'TPUConfig iterations_per_loop')

    # Check num_shards.
    util_lib.check_positive_integer(num_shards, 'TPUConfig num_shards')

    # Check initial_infeed_sleep_secs.
    if initial_infeed_sleep_secs:
      util_lib.check_positive_integer(initial_infeed_sleep_secs,
                                      'TPUConfig initial_infeed_sleep_secs')

    tpu_job_name = tpu_job_name or _get_tpu_job_name_from_tf_config()

    return super(TPUConfig, cls).__new__(
        cls,
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        per_host_input_for_training=per_host_input_for_training,
        tpu_job_name=tpu_job_name,
        initial_infeed_sleep_secs=initial_infeed_sleep_secs)


class RunConfig(run_config_lib.RunConfig):
  """RunConfig with TPU support."""

  def __init__(self,
               tpu_config=None,
               evaluation_master=None,
               master=None,
               **kwargs):
    """Constructs a RunConfig.

    Args:
      tpu_config: the TPUConfig that specifies TPU-specific configuration.
      evaluation_master: a string. The address of the master to use for eval.
        Defaults to master if not set.
      master: a string. The address of the master to use for training.
      tf_random_seed: an int. Sets the TensorFlow random seed. Defaults to None,
        which initializes it randomly based on the environment.
    """
    super(RunConfig, self).__init__(**kwargs)
    self._tpu_config = tpu_config or TPUConfig()

    # If user sets master and/or evaluation_master explicilty, including empty
    # string '', take it. Otherwise, take the values set by parent class.
    if master is not None:
      self._master = master

    if evaluation_master is not None:
      self._evaluation_master = evaluation_master
    elif (not self._evaluation_master and
          self.task_type != run_config_lib.TaskType.EVALUATOR):
      # If the task type is EVALUATOR, it means some cluster manager sets the
      # TF_CONFIG. In that case, we respect the configuration in TF_CONFIG.
      #
      # Otherwise, it means user executes the code without external cluster
      # manager. For that, we optimize the user experience by setting
      # evaluation_master to master, unless user overwrites it.
      self._evaluation_master = self._master

  @property
  def evaluation_master(self):
    return self._evaluation_master

  @property
  def master(self):
    return self._master

  @property
  def tpu_config(self):
    return self._tpu_config

  def replace(self, **kwargs):
    if 'tpu_config' not in kwargs:
      return super(RunConfig, self).replace(**kwargs)

    tpu_config = kwargs.pop('tpu_config')
    new_instance = super(RunConfig, self).replace(**kwargs)
    new_instance._tpu_config = tpu_config  # pylint: disable=protected-access
    return new_instance


def _get_tpu_job_name_from_tf_config():
  """Extracts the TPU job name from TF_CONFIG env variable."""
  # TODO(xiejw): Extends this to support both TF_CONFIG env variable and cluster
  # spec propagation.
  tf_config = json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))
  tpu_job_name = tf_config.get(_SERVICE_KEY, {}).get(_TPU_WORKER_JOB_NAME)
  if tpu_job_name:
    logging.info('Load TPU job name from TF_CONFIG: %s', tpu_job_name)
  return tpu_job_name
