# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""TPU specific APIs to be used in conjunction with TPU Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver import TPUClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


_INITIALIZED_TPU_SYSTEMS = {}
_LOCAL_MASTERS = ("", "local")


@tf_export("tpu.experimental.initialize_tpu_system")
def initialize_tpu_system(cluster_resolver=None):
  """Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster. If called
    inside tf.function, it returns the serialized topology object instead.

  Raises:
    RuntimeError: If running inside a tf.function.
    NotFoundError: If no TPU devices found in eager mode.
  """
  job = None
  if cluster_resolver is None:
    # If no cluster resolver is specified, and running eagerly, execute the init
    # ops in the current device scope.
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)

    cluster_resolver = TPUClusterResolver("")
  assert isinstance(cluster_resolver, TPUClusterResolver)

  tpu_name = compat.as_text(cluster_resolver._tpu)  # pylint: disable=protected-access
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    logging.warning(
        "TPU system %s has already been initialized. "
        "Reinitializing the TPU can cause previously created "
        "variables on TPU to be lost.", tpu_name)

  logging.info("Initializing the TPU system: %s", tpu_name)

  # This function looks as it is for the following non-intuitive reasons.
  # tpu.initialize_system creates a dummy op whose sole purpose is to trigger
  # DistributedTPURewritePass. This pass actually adds real ops that
  # initialize the TPU system. Thus, we can't simply run tpu.initialize_system
  # eagerly. We need to wrap it in defun and trigger the rewrite passes on it.
  if tpu_name not in _LOCAL_MASTERS:
    # Explicitly place the tpu.initialize_system in the first worker to
    # avoid the output node match multiple devices error.
    job = "{}/replica:0/task:0".format(cluster_resolver.get_job_name())

  if context.executing_eagerly():
    @function.defun
    def _tpu_init_fn():
      # In TF1, we usually close chips when compilation fails to clear the data
      # in infeed. In TF2, we don't need to do this because infeed is no longer
      # used, so user can recover from TPU compilation failures more smoothly.
      return tpu.initialize_system(
          job=job, compilation_failure_closes_chips=False)

    # The TPU_SYSTEM device must match the device used in tpu.initialize_system
    # exactly, otherwise you can get errors if there are multiple TPU_SYSTEM
    # devices available.
    try:
      with ops.device(tpu._tpu_system_device_name(job)):  # pylint: disable=protected-access
        output = _tpu_init_fn()
      context.async_wait()
    except errors.InvalidArgumentError as e:
      raise errors.NotFoundError(
          None, None,
          "TPUs not found in the cluster. Failed in initialization: "
          + str(e))

    # Clear out the eager context caches since the memory is invalid now.
    logging.info("Clearing out eager caches")
    context.context()._clear_caches()  # pylint: disable=protected-access

    serialized_topology = output.numpy()
  elif not ops.executing_eagerly_outside_functions():
    master = cluster_resolver.master()
    cluster_spec = cluster_resolver.cluster_spec()

    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    if cluster_spec:
      session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

    with ops.Graph().as_default():
      with session_lib.Session(config=session_config, target=master) as sess:
        serialized_topology = sess.run(tpu.initialize_system())
  else:
    with ops.device(tpu._tpu_system_device_name(job)):  # pylint: disable=protected-access
      serialized_topology = tpu.initialize_system(
          job=job, compilation_failure_closes_chips=False)
      # If initialize_tpu_system is called inside tf.function, we only return
      # the serialized topology object as the tf.tpu.Topology object has to be
      # constructed in eager mode.
      return serialized_topology

  logging.info("Finished initializing TPU system.")
  tpu_topology = topology.Topology(serialized=serialized_topology)
  _INITIALIZED_TPU_SYSTEMS[tpu_name] = tpu_topology

  return tpu_topology


@tf_export("tpu.experimental.shutdown_tpu_system")
def shutdown_tpu_system(cluster_resolver=None):
  """Shuts down the TPU devices.

  This will clear all caches, even those that are maintained through sequential
  calls to tf.tpu.experimental.initialize_tpu_system, such as the compilation
  cache.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.

  Raises:
    RuntimeError: If no TPU devices found for eager execution or if run in a
        tf.function.
  """
  job = None
  if cluster_resolver is None:
    # If no cluster resolver is specified, and running eagerly, execute the init
    # ops in the current device scope.
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)

    cluster_resolver = TPUClusterResolver("")
  assert isinstance(cluster_resolver, TPUClusterResolver)

  tpu_name = compat.as_text(cluster_resolver._tpu)  # pylint: disable=protected-access
  if tpu_name not in _INITIALIZED_TPU_SYSTEMS:
    logging.warning("You are shutting down a TPU system %s that has not been "
                    "initialized." % tpu_name)

  logging.info("Shutting down the TPU system: %s", tpu_name)

  if context.executing_eagerly():
    # This function looks as it is for the following non-intuitive reasons.
    # tpu.shutdown_system creates a dummy op whose sole purpose is to trigger
    # DistributedTPURewritePass. This pass actually adds real ops that
    # shutdown the TPU system. Thus, we can't simply run tpu.shutdown_system
    # eagerly. We need to wrap it in defun and trigger the rewrite passes on it.
    if tpu_name not in _LOCAL_MASTERS:
      # Explicitly place the tpu.shutdown_system in the first worker to
      # avoid the output node match multiple devices error.
      job = "{}/replica:0/task:0".format(cluster_resolver.get_job_name())

    @function.defun
    def _tpu_shutdown_fn():
      tpu.shutdown_system(job=job)

    # The TPU_SYSTEM device must match the device used in tpu.shutdown_system
    # exactly, otherwise you can get errors if there are multiple TPU_SYSTEM
    # devices available.
    with ops.device(tpu._tpu_system_device_name(job)):  # pylint: disable=protected-access
      _tpu_shutdown_fn()

    # Clear out the eager context caches since the memory is invalid now.
    logging.info("Clearing out eager caches")
    context.context()._clear_caches()  # pylint: disable=protected-access
  elif not ops.executing_eagerly_outside_functions():
    master = cluster_resolver.master()
    cluster_spec = cluster_resolver.cluster_spec()

    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    if cluster_spec:
      session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

    with ops.Graph().as_default():
      with session_lib.Session(config=session_config, target=master) as sess:
        sess.run(tpu.shutdown_system())
  else:
    raise RuntimeError("initialize_tpu_system is not supported within "
                       "tf.functions.")

  logging.info("Finished shutting down TPU system.")
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    del _INITIALIZED_TPU_SYSTEMS[tpu_name]
