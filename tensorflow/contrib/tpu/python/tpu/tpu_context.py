# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TPU system metdata and associated tooling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import copy

import numpy as np

from tensorflow.contrib.tpu.python.tpu import device_assignment  as tpu_device_assignment
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.platform import tf_logging as logging


_DEFAULT_JOB_NAME = 'tpu_worker'
_DEFAULT_COORDINATOR_JOB_NAME = 'coordinator'
_LOCAL_MASTERS = ('', 'local')


class _TPUContext(object):
  """A context holds immutable states of TPU computation.

  This immutable object holds TPUEstimator config, train/eval batch size, and
  `TPUEstimator.use_tpu`, which is expected to be passed around. It also
  provides utility functions, based on the current state, to determine other
  information commonly required by TPU computation, such as TPU device names,
  TPU hosts, shard batch size, etc.

  N.B. As `mode` is not immutable state in Estimator, but essential to
  distinguish between TPU training and evaluation, a common usage for
  _TPUContext with `mode` is as follows:
  ```
  with _ctx.with_mode(mode) as ctx:
    if ctx.is_running_on_cpu():
       ...
  ```
  """

  def __init__(self, config, train_batch_size, eval_batch_size,
               predict_batch_size, use_tpu):
    self._config = config
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._predict_batch_size = predict_batch_size
    self._use_tpu = use_tpu
    self._model_parallelism_enabled = (
        use_tpu and config.tpu_config.computation_shape)
    self._mode = None

    self._lazy_tpu_system_metadata_dict = {}  # key by master address
    self._lazy_device_assignment_dict = {}  # key by master address
    self._lazy_validation_dict = {}  # key by ModeKeys

  def _assert_mode(self):
    if self._mode is None:
      raise RuntimeError(
          '`mode` needs to be set via contextmanager `with_mode`.')
    return self._mode

  @contextmanager
  def with_mode(self, mode):
    # NOTE(xiejw): Shallow copy is enough. It will share he lazy dictionaries,
    # such as _lazy_tpu_system_metadata_dict between new copy and the original
    # one. Note that all lazy states stored in properties _lazy_foo are sort of
    # immutable as they should be same for the process lifetime.
    new_ctx = copy.copy(self)
    new_ctx._mode = mode  # pylint: disable=protected-access
    yield new_ctx

  @property
  def mode(self):
    return self._assert_mode()

  def _get_master_address(self):
    mode = self._assert_mode()
    config = self._config
    master = (
        config.master
        if mode != model_fn_lib.ModeKeys.EVAL else config.evaluation_master)
    return master

  def _get_tpu_system_metadata(self):
    """Gets the (maybe cached) TPU system metadata."""
    master = self._get_master_address()
    tpu_system_metadata = self._lazy_tpu_system_metadata_dict.get(master)
    if tpu_system_metadata is not None:
      return tpu_system_metadata

    # pylint: disable=protected-access
    tpu_system_metadata = (
        tpu_system_metadata_lib._query_tpu_system_metadata(
            master,
            run_config=self._config,
            query_topology=self.model_parallelism_enabled))

    self._lazy_tpu_system_metadata_dict[master] = tpu_system_metadata
    return tpu_system_metadata

  def _get_device_assignment(self):
    """Gets the (maybe cached) TPU device assignment."""
    master = self._get_master_address()
    device_assignment = self._lazy_device_assignment_dict.get(master)
    if device_assignment is not None:
      return device_assignment

    tpu_system_metadata = self._get_tpu_system_metadata()

    device_assignment = tpu_device_assignment.device_assignment(
        tpu_system_metadata.topology,
        computation_shape=self._config.tpu_config.computation_shape,
        num_replicas=self.num_replicas)

    logging.info('computation_shape: %s',
                 str(self._config.tpu_config.computation_shape))
    logging.info('num_replicas: %d', self.num_replicas)
    logging.info('device_assignment.topology.device_coordinates: %s',
                 str(device_assignment.topology.device_coordinates))
    logging.info('device_assignment.core_assignment: %s',
                 str(device_assignment.core_assignment))

    self._lazy_device_assignment_dict[master] = device_assignment
    return device_assignment

  @property
  def model_parallelism_enabled(self):
    return self._model_parallelism_enabled

  @property
  def device_assignment(self):
    return (self._get_device_assignment()
            if self._model_parallelism_enabled else None)

  @property
  def num_of_cores_per_host(self):
    metadata = self._get_tpu_system_metadata()
    return metadata.num_of_cores_per_host

  @property
  def num_cores(self):
    metadata = self._get_tpu_system_metadata()
    return metadata.num_cores

  @property
  def num_of_replicas_per_host(self):
    if self.model_parallelism_enabled:
      return self.num_replicas // self.num_hosts
    else:
      return self.num_of_cores_per_host

  @property
  def num_replicas(self):
    num_cores_in_system = self.num_cores

    if self.model_parallelism_enabled:
      computation_shape_array = np.asarray(
          self._config.tpu_config.computation_shape, dtype=np.int32)
      num_cores_per_replica = np.prod(computation_shape_array)
      if num_cores_per_replica > num_cores_in_system:
        raise ValueError(
            'The num of cores required by the model parallelism, specified by '
            'TPUConfig.computation_shape, is larger than the total num of '
            'TPU cores in the system. computation_shape: {}, num cores '
            'in the system: {}'.format(
                self._config.tpu_config.computation_shape,
                num_cores_in_system))

      if num_cores_in_system % num_cores_per_replica != 0:
        raise RuntimeError(
            'The num of cores in the system ({}) is not divisible by the num '
            'of cores ({}) required by the model parallelism, specified by '
            'TPUConfig.computation_shape. This should never happen!'.format(
                num_cores_in_system, num_cores_per_replica))

      return num_cores_in_system // num_cores_per_replica
    else:
      return num_cores_in_system

  @property
  def num_hosts(self):
    metadata = self._get_tpu_system_metadata()
    return metadata.num_hosts

  @property
  def config(self):
    return self._config

  def is_input_sharded_per_core(self):
    """Return true if input_fn is invoked per-core (other than per-host)."""
    mode = self._assert_mode()
    return (mode == model_fn_lib.ModeKeys.TRAIN and
            (self._config.tpu_config.per_host_input_for_training is
             tpu_config.InputPipelineConfig.PER_SHARD_V1))

  def is_input_per_host_with_iterators(self):
    """Return true if input_fn should be run in the per-host v2 config."""
    return (self._config.tpu_config.per_host_input_for_training is
            tpu_config.InputPipelineConfig.PER_HOST_V2)

  def is_running_on_cpu(self, is_export_mode=False):
    """Determines whether the input_fn and model_fn should be invoked on CPU.

    This API also validates user provided configuration, such as batch size,
    according the lazy initialized TPU system metadata.

    Args:
      is_export_mode: Indicates whether the current mode is for exporting the
        model, when mode == PREDICT. Only with this bool, we could
        tell whether user is calling the Estimator.predict or
        Estimator.export_savedmodel, which are running on TPU and CPU
        respectively. Parent class Estimator does not distinguish these two.

    Returns:
      bool, whether current input_fn or model_fn should be running on CPU.

    Raises:
      ValueError: any configuration is invalid.
    """

    is_running_on_cpu = self._is_running_on_cpu(is_export_mode)
    if not is_running_on_cpu:
      self._validate_tpu_configuration()
    return is_running_on_cpu

  def _is_running_on_cpu(self, is_export_mode):
    """Determines whether the input_fn and model_fn should be invoked on CPU."""
    mode = self._assert_mode()

    if not self._use_tpu:
      return True

    if mode != model_fn_lib.ModeKeys.PREDICT:
      return False

    # There are actually 2 use cases when running with mode.PREDICT: prediction
    # and saving the model.  We run actual predictions on the TPU, but
    # model export is run on the CPU.
    if is_export_mode:
      return True

    return False

  @property
  def global_batch_size(self):
    mode = self._assert_mode()
    if mode == model_fn_lib.ModeKeys.TRAIN:
      return self._train_batch_size
    elif mode == model_fn_lib.ModeKeys.EVAL:
      return self._eval_batch_size
    elif mode == model_fn_lib.ModeKeys.PREDICT:
      return self._predict_batch_size
    else:
      return None

  @property
  def batch_size_for_input_fn(self):
    """Returns the shard batch size for `input_fn`."""
    global_batch_size = self.global_batch_size

    if self.is_running_on_cpu():
      return global_batch_size

    # On TPU
    if self.is_input_sharded_per_core() or (
        self.is_input_per_host_with_iterators()):
      # We prohibit per core input sharding for the model parallelism case,
      # therefore it is safe to use num_cores here.
      return global_batch_size // self.num_cores
    else:
      return global_batch_size // self.num_hosts

  @property
  def batch_size_for_model_fn(self):
    """Returns the shard batch size for `model_fn`."""
    global_batch_size = self.global_batch_size

    if self.is_running_on_cpu():
      return global_batch_size

    # On TPU. always sharded per shard.
    return global_batch_size // self.num_replicas

  @property
  def master_job(self):
    """Returns the job name to use to place TPU computations on.

    Returns:
      A string containing the job name, or None if no job should be specified.

    Raises:
      ValueError: If the user needs to specify a tpu_job_name, because we are
        unable to infer the job name automatically, or if the user-specified job
        names are inappropriate.
    """
    run_config = self._config
    # If the user specifies the tpu_job_name, use that.
    if run_config.tpu_config.tpu_job_name:
      return run_config.tpu_config.tpu_job_name

    # The tpu job is determined by the run_config. Right now, this method is
    # required as tpu_config is not part of the RunConfig.
    mode = self._assert_mode()
    master = (
        run_config.evaluation_master
        if mode == model_fn_lib.ModeKeys.EVAL else run_config.master)
    if master in _LOCAL_MASTERS:
      return None

    if (not run_config.session_config or
        not run_config.session_config.cluster_def.job):
      return _DEFAULT_JOB_NAME
    cluster_def = run_config.session_config.cluster_def
    job_names = set([job.name for job in cluster_def.job])
    if _DEFAULT_JOB_NAME in job_names:
      # b/37868888 tracks allowing ClusterSpec propagation to reuse job names.
      raise ValueError('Currently, tpu_worker is not an allowed job name.')
    if len(job_names) == 1:
      return cluster_def.job[0].name
    if len(job_names) == 2:
      if _DEFAULT_COORDINATOR_JOB_NAME in job_names:
        job_names.remove(_DEFAULT_COORDINATOR_JOB_NAME)
        return job_names.pop()
      # TODO(b/67716447): Include more sophisticated heuristics.
    raise ValueError(
        'Could not infer TPU job name. Please specify a tpu_job_name as part '
        'of your TPUConfig.')

  @property
  def tpu_host_placement_function(self):
    """Returns the TPU host place function."""
    master = self.master_job

    def _placement_function(_sentinal=None, core_id=None, host_id=None):  # pylint: disable=invalid-name
      assert _sentinal is None
      if core_id is not None and host_id is not None:
        raise RuntimeError(
            'core_id and host_id can have only one non-None value.')

      if master is None:
        return '/replica:0/task:0/device:CPU:0'
      else:
        if core_id is not None:
          host_id = core_id / self.num_of_cores_per_host
        return '/job:%s/task:%d/device:CPU:0' % (master, host_id)

    return _placement_function

  @property
  def tpu_device_placement_function(self):
    """Returns a TPU device placement Fn."""
    master = self.master_job
    job_device = '' if master is None else ('/job:%s' % master)

    def _placement_function(i):
      if self.model_parallelism_enabled:
        return self.device_assignment.tpu_device(replica=i, job=master)
      else:
        num_of_cores_per_host = self.num_of_cores_per_host
        host_id = i / num_of_cores_per_host
        ordinal_id = i % num_of_cores_per_host
        return '%s/task:%d/device:TPU:%d' % (job_device, host_id, ordinal_id)

    return _placement_function

  @property
  def tpu_ordinal_function(self):
    """Returns the TPU ordinal fn."""

    def _tpu_ordinal_function(index):
      """Return the TPU ordinal associated with a shard.

      Required because the enqueue ops are placed on CPU.

      Args:
        index: the shard index

      Returns:
        The ordinal of the TPU device the shard's infeed should be placed on.
      """
      if self.model_parallelism_enabled:
        return self.device_assignment.tpu_ordinal(replica=index)
      else:
        return index % self.num_of_cores_per_host

    return _tpu_ordinal_function

  def _validate_tpu_configuration(self):
    """Validates the configuration based on the TPU system metadata."""
    mode = self._assert_mode()
    if self._lazy_validation_dict.get(mode):
      return

    # All following information is obtained from TPU system metadata.
    num_cores = self.num_cores
    num_replicas = self.num_replicas
    num_hosts = self.num_hosts

    if not num_cores:
      tpu_system_metadata = self._get_tpu_system_metadata()
      raise RuntimeError(
          'Cannot find any TPU cores in the system. Please double check '
          'Tensorflow master address and TPU worker(s). Available devices '
          'are {}.'.format(tpu_system_metadata.devices))

    if self._config.tpu_config.num_shards:
      user_provided_num_replicas = self._config.tpu_config.num_shards
      if user_provided_num_replicas != num_replicas:
        message = (
            'TPUConfig.num_shards is not set correctly. According to TPU '
            'system metadata for Tensorflow master ({}): num_replicas should '
            'be ({}), got ({}). For non-model-parallelism, num_replicas should '
            'be the total num of TPU cores in the system. For '
            'model-parallelism, the total number of TPU cores should be '
            'product(computation_shape) * num_replicas. Please set it '
            'accordingly or leave it as `None`'.format(
                self._get_master_address(), num_replicas,
                user_provided_num_replicas))

        raise ValueError(message)

    if mode == model_fn_lib.ModeKeys.TRAIN:
      if self._train_batch_size % num_replicas != 0:
        raise ValueError(
            'train batch size {} must be divisible by number of replicas {}'
            .format(self._train_batch_size, num_replicas))

    elif mode == model_fn_lib.ModeKeys.EVAL:
      if self._eval_batch_size is None:
        raise ValueError(
            'eval_batch_size in TPUEstimator constructor cannot be `None`'
            'if .evaluate is running on TPU.')
      if self._eval_batch_size % num_replicas != 0:
        raise ValueError(
            'eval batch size {} must be divisible by number of replicas {}'
            .format(self._eval_batch_size, num_replicas))
      if num_hosts > 1:
        raise ValueError(
            'TPUEstimator.evaluate should be running on single TPU worker. '
            'got {}.'.format(num_hosts))
    else:
      assert mode == model_fn_lib.ModeKeys.PREDICT
      if self._predict_batch_size is None:
        raise ValueError(
            'predict_batch_size in TPUEstimator constructor should not be '
            '`None` if .predict is running on TPU.')
      if self._predict_batch_size % num_replicas != 0:
        raise ValueError(
            'predict batch size {} must be divisible by number of replicas {}'
            .format(self._predict_batch_size, num_replicas))
      if num_hosts > 1:
        raise ValueError(
            'TPUEstimator.predict should be running on single TPU worker. '
            'got {}.'.format(num_hosts))

    # Record the state "validated" into lazy dictionary.
    self._lazy_validation_dict[mode] = True


class _OneCoreTPUContext(_TPUContext):
  """Special _TPUContext for one core usage."""

  def __init__(self, config, train_batch_size, eval_batch_size,
               predict_batch_size, use_tpu):

    super(_OneCoreTPUContext, self).__init__(
        config, train_batch_size, eval_batch_size,
        predict_batch_size, use_tpu)

  def _get_tpu_system_metadata(self):
    """Gets the (maybe cached) TPU system metadata."""
    master = self._get_master_address()
    tpu_system_metadata = self._lazy_tpu_system_metadata_dict.get(master)
    if tpu_system_metadata is not None:
      return tpu_system_metadata

    tpu_system_metadata = (
        tpu_system_metadata_lib._TPUSystemMetadata(  # pylint: disable=protected-access
            num_cores=1,
            num_hosts=1,
            num_of_cores_per_host=1,
            topology=None,
            devices=[]))

    self._lazy_tpu_system_metadata_dict[master] = tpu_system_metadata
    return tpu_system_metadata


def _get_tpu_context(config, train_batch_size, eval_batch_size,
                     predict_batch_size, use_tpu):
  """Returns an instance of `_TPUContext`."""

  if (config.tpu_config.num_shards == 1 and
      config.tpu_config.computation_shape is None):
    logging.warning(
        'Setting TPUConfig.num_shards==1 is an unsupported behavior. '
        'Please fix as soon as possible (leaving num_shards as None.')
    return _OneCoreTPUContext(config, train_batch_size, eval_batch_size,
                              predict_batch_size, use_tpu)

  return _TPUContext(config, train_batch_size, eval_batch_size,
                     predict_batch_size, use_tpu)
