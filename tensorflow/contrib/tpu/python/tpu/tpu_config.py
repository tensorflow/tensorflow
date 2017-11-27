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

from tensorflow.contrib.tpu.python.tpu import util as util_lib
from tensorflow.python.estimator import run_config as run_config_lib


class TPUConfig(
    collections.namedtuple('TPUConfig', [
        'iterations_per_loop',
        'num_shards',
        'per_host_input_for_training',
        'tpu_job_name',
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
  """

  def __new__(cls,
              iterations_per_loop=2,
              num_shards=2,
              per_host_input_for_training=True,
              tpu_job_name=None):

    # Check iterations_per_loop.
    util_lib.check_positive_integer(iterations_per_loop,
                                    'TPUConfig iterations_per_loop')

    # Check num_shards.
    util_lib.check_positive_integer(num_shards, 'TPUConfig num_shards')
    return super(TPUConfig, cls).__new__(
        cls,
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        per_host_input_for_training=per_host_input_for_training,
        tpu_job_name=tpu_job_name)


class RunConfig(run_config_lib.RunConfig):
  """RunConfig with TPU support."""

  def __init__(self, tpu_config=None, evaluation_master=None, master='',
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
    if evaluation_master is None:
      self._evaluation_master = master
    else:
      self._evaluation_master = evaluation_master
    self._master = master

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
