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

import numpy as np

from tensorflow.contrib.tpu.python.tpu import util as util_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.platform import tf_logging as logging

# pylint: disable=protected-access
_TF_CONFIG_ENV = run_config_lib._TF_CONFIG_ENV
_SERVICE_KEY = run_config_lib._SERVICE_KEY
_TPU_WORKER_JOB_NAME = 'tpu_worker_job_name'
_NUM_CORES_PER_HOST = 8
# pylint: enable=protected-access


class InputPipelineConfig(object):
  r"""Please see the definition of these values in TPUConfig."""
  PER_SHARD_V1 = 1
  PER_HOST_V1 = 2
  PER_HOST_V2 = 3


# TODO(b/72511246) Provide a simplified api to configure model parallelism.
class TPUConfig(
    collections.namedtuple('TPUConfig', [
        'iterations_per_loop',
        'num_shards',
        'computation_shape',
        'per_host_input_for_training',
        'tpu_job_name',
        'initial_infeed_sleep_secs',
    ])):
  r"""TPU related configuration required by `TPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps running in TPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
    num_shards: (Deprecated, ignored by TPUEstimator).
      The number of model replicas in the system. For non-model-parallelism
      case, this number equals the total number of TPU cores. For
      model-parallelism, the total number of TPU cores equals
      product(computation_shape) * num_shards.
    computation_shape: Defaults to `None`, which disables model parallelism. A
      list of size 3 which describes the shape of a model replica's block of
      cores. This is required by model-parallelism which enables partitioning
      the model to multiple cores. For example, [2, 2, 1] means the model is
      partitioned across 4 cores which span two cores in both x and y
      coordinates.  Please refer to @{tf.contrib.tpu.Topology} for the
      geometry of a TPU mesh.
    per_host_input_for_training: If `True`, `PER_HOST_V1`, or `PER_HOST_V2`,
      `input_fn` is invoked per-host rather than per-core. With per-host input
      pipeline configuration, `input_fn` is invoked once on each host. With the
      per-core input pipeline configuration, it is invoked once for each core.
      With a global batch size `train_batch_size` in `TPUEstimator` constructor,
      the batch size for each shard is `train_batch_size` // #hosts in the
      `True` or `PER_HOST_V1` mode. In `PER_HOST_V2` mode, it is
      `train_batch_size` // #cores. With the per-core input pipeline
      configuration, the shard batch size is also `train_batch_size` // #cores.
      Note: per_host_input_for_training==PER_SHARD_V1 only supports mode.TRAIN.
    tpu_job_name: The name of the TPU job. Typically, this name is auto-inferred
      within TPUEstimator, however when using ClusterSpec propagation in more
      esoteric cluster configurations, you may need to specify the job name as a
      string.
    initial_infeed_sleep_secs: The number of seconds the infeed thread should
      wait before enqueueing the first batch. This helps avoid timeouts for
      models that require a long compilation time.

    Raises:
      ValueError: If `computation_shape` or `computation_shape` are invalid.
  """

  def __new__(cls,
              iterations_per_loop=2,
              num_shards=None,
              computation_shape=None,
              per_host_input_for_training=True,
              tpu_job_name=None,
              initial_infeed_sleep_secs=None):

    # Check iterations_per_loop.
    util_lib.check_positive_integer(iterations_per_loop,
                                    'TPUConfig iterations_per_loop')

    # Check num_shards.
    if num_shards is not None:
      util_lib.check_positive_integer(num_shards, 'TPUConfig num_shards')

    # Check computation_shape
    if computation_shape is not None and len(computation_shape) != 3:
      raise ValueError(
          'computation_shape must be a list with length 3 or None; got {}'.
          format(str(computation_shape)))

    if computation_shape is not None:
      computation_shape_array = np.asarray(computation_shape, dtype=np.int32)
      # This prevents any computation being replicated across multiple hosts, so
      # that each host feeds the same number of computations.
      if any(computation_shape_array < 1) or any(computation_shape_array > 2):
        raise ValueError('computation_shape elements can only be 1 or 2; got '
                         'computation_shape={}'.format(computation_shape))

    # per_host_input_for_training may be True, False, or integer in [1..3].
    # Map legacy values (True, False) to numeric values.
    if per_host_input_for_training is False:
      per_host_input_for_training = InputPipelineConfig.PER_SHARD_V1
    elif per_host_input_for_training is True:
      per_host_input_for_training = InputPipelineConfig.PER_HOST_V1

    # Check initial_infeed_sleep_secs.
    if initial_infeed_sleep_secs:
      util_lib.check_positive_integer(initial_infeed_sleep_secs,
                                      'TPUConfig initial_infeed_sleep_secs')

    tpu_job_name = tpu_job_name or _get_tpu_job_name_from_tf_config()

    return super(TPUConfig, cls).__new__(
        cls,
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        computation_shape=computation_shape,
        per_host_input_for_training=per_host_input_for_training,
        tpu_job_name=tpu_job_name,
        initial_infeed_sleep_secs=initial_infeed_sleep_secs)


class RunConfig(run_config_lib.RunConfig):
  """RunConfig with TPU support."""

  def __init__(self,
               tpu_config=None,
               evaluation_master=None,
               master=None,
               cluster=None,
               **kwargs):
    """Constructs a RunConfig.

    Args:
      tpu_config: the TPUConfig that specifies TPU-specific configuration.
      evaluation_master: a string. The address of the master to use for eval.
        Defaults to master if not set.
      master: a string. The address of the master to use for training.
      cluster: a ClusterResolver
      **kwargs: keyword config parameters.

    Raises:
      ValueError: if cluster is not None and the provided session_config has a
        cluster_def already.
    """
    super(RunConfig, self).__init__(**kwargs)
    self._tpu_config = tpu_config or TPUConfig()
    self._cluster = cluster

    # If user sets master and/or evaluation_master explicitly, including empty
    # string '', take it. Otherwise, take the values set by parent class.
    if master is not None:
      if cluster is not None:
        raise ValueError('Both master and cluster are set.')
      self._master = master
    else:
      if cluster:
        self._master = cluster.master()

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

    # Set the ClusterSpec to use
    if cluster:
      self._cluster_spec = cluster.cluster_spec()

      # Merge the cluster_def into the ConfigProto.
      if self._session_config is None:  # pylint: disable=access-member-before-definition
        self._session_config = config_pb2.ConfigProto(allow_soft_placement=True)
      if self._session_config.HasField('cluster_def'):
        raise ValueError(
            'You cannot provide a ClusterResolver and '
            'session_config.cluster_def.')
      self._session_config.cluster_def.CopyFrom(
          self._cluster_spec.as_cluster_def())

  @property
  def evaluation_master(self):
    return self._evaluation_master

  @property
  def master(self):
    return self._master

  @property
  def tpu_config(self):
    return self._tpu_config

  @property
  def cluster(self):
    return self._cluster

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
