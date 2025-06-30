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

import gc
import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat


_INITIALIZED_TPU_SYSTEMS = {}
_LOCAL_MASTERS = ("", "local")


def _is_colab_v2_environment():
  """Check if running in Google Colab V2 environment."""
  try:
    import google.colab  # pylint: disable=import-outside-toplevel,unused-import
    colab_env = os.environ.get('COLAB_TPU_ADDR', '')
    runtime_name = os.environ.get('COLAB_RUNTIME_NAME', '')
    return 'v2-8' in colab_env or 'v2' in runtime_name.lower()
  except ImportError:
    return False


def _ensure_libtpu_available():
  """Ensure libtpu is available and provide helpful error messages."""
  if _is_colab_v2_environment():
    try:
      # Try to import TPU-related modules to check if libtpu is available
      from tensorflow.python.tpu import tpu_strategy_util  # pylint: disable=import-outside-toplevel,unused-import
      return True
    except ImportError:
      logging.error(
          "TPU libraries not found in Google Colab V2. Please install with:\n"
          "!pip install -U \"https://storage.googleapis.com/libtpu-releases/libtpu-nightly.tar.gz\"\n"
          "Then restart the runtime and import tensorflow again."
      )
      return False
  return True


def _apply_colab_v2_workarounds():
  """Apply workarounds for Google Colab V2 TPU issues."""
  if not _is_colab_v2_environment():
    return
    
  # Set environment variables that help with TPU detection in Colab V2
  if 'TPU_LIBRARY_PATH' not in os.environ:
    # Try common libtpu locations
    possible_paths = [
        '/usr/local/lib/python*/site-packages/libtpu/libtpu.so',
        '/opt/conda/lib/python*/site-packages/libtpu/libtpu.so',
        'libtpu.so'
    ]
    
    import glob
    for path_pattern in possible_paths:
      matches = glob.glob(path_pattern)
      if matches:
        os.environ['TPU_LIBRARY_PATH'] = matches[0]
        logging.info(f"Set TPU_LIBRARY_PATH to {matches[0]} for Colab V2 compatibility")
        break


_tpu_worker_address = monitoring.StringGauge(
    "/tensorflow/tpu/worker_address",
    "The worker address that the coordinator/client connects to.", "address")


def initialize_tpu_system_impl(cluster_resolver, tpu_cluster_resolver_cls):
  """Implementation for tpu.experimental.initialize_tpu_system.

  Kept separate to avoid tpu_oss code duplication.

  Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
    tpu_cluster_resolver_cls: a reference to
        tf.distribute.cluster_resolver.TPUClusterResolver so that an instance
        of it can be initialized if cluster_resolver is None.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster. If called
    inside tf.function, it returns the serialized topology object instead.

  Raises:
    RuntimeError: If running inside a tf.function.
    NotFoundError: If no TPU devices found in eager mode.
    TypeError: If tpu_cluster_resolver_cls is
        not tf.distribute.cluster_resolver.TPUClusterResolver.
  """
  # check that tpu_cluster_resolver_cls is a
  # tf.distribute.cluster_resolver.TPUClusterResolver
  if tpu_cluster_resolver_cls is None or not issubclass(
      tpu_cluster_resolver_cls, cluster_resolver_lib.ClusterResolver
  ) or not hasattr(tpu_cluster_resolver_cls, "tpu_hardware_feature"):
    raise TypeError(
        "tpu_cluster_resolver_cls is not"
        " tf.distribute.cluster_resolver.TPUClusterResolver.")
  # Deallocate all TPU buffers by clearing out eager context caches and
  # triggering garbage collection to avoid keeping invalid tpu buffer around
  # after reinitialized tpu system.
  logging.info("Deallocate tpu buffers before initializing tpu system.")
  context.context()._clear_caches()  # pylint: disable=protected-access
  context.context().clear_kernel_cache()
  gc.collect()

  job = None
  if cluster_resolver is None:
    # If no cluster resolver is specified, and running eagerly, execute the init
    # ops in the current device scope.
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)

    cluster_resolver = tpu_cluster_resolver_cls("")
  assert isinstance(cluster_resolver, tpu_cluster_resolver_cls)

  tpu_name = compat.as_text(cluster_resolver._tpu)  # pylint: disable=protected-access
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    logging.warning(
        "TPU system %s has already been initialized. "
        "Reinitializing the TPU can cause previously created "
        "variables on TPU to be lost.", tpu_name)

  logging.info("Initializing the TPU system: %s", tpu_name)

  # Apply Colab V2 specific workarounds
  _apply_colab_v2_workarounds()
  
  # Check if libtpu is available, especially for Colab V2
  if not _ensure_libtpu_available():
    raise errors.NotFoundError(
        None, None,
        "TPU libraries not available. In Google Colab V2, please install with: "
        "!pip install -U \"https://storage.googleapis.com/libtpu-releases/libtpu-nightly.tar.gz\""
    )

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
    @def_function.function(autograph=False)
    def _tpu_init_fn():
      # In TF1, we usually close chips when compilation fails to clear the data
      # in infeed. In TF2, we don't need to do this because infeed is no longer
      # used, so user can recover from TPU compilation failures more smoothly.
      # Same for the cancellation of a TPU excution.
      return tpu.initialize_system(
          job=job,
          compilation_failure_closes_chips=False,
          tpu_cancellation_closes_chips=False)

    # The TPU_SYSTEM device must match the device used in tpu.initialize_system
    # exactly, otherwise you can get errors if there are multiple TPU_SYSTEM
    # devices available.
    run_eagerly = def_function.functions_run_eagerly()
    if run_eagerly:
      logging.warning(
          "It looks like tf.function behavior was disabled, perhaps using"
          " tf.config.run_functions_eagerly."
          " tf.tpu.experimental.initialize_tpu_system requires tf.function to"
          " work. This primitive will override the disable."
      )
      def_function.run_functions_eagerly(False)
    try:
      with ops.device(tpu._tpu_system_device_name(job)):  # pylint: disable=protected-access
        output = _tpu_init_fn()
      context.async_wait()
    except errors.InvalidArgumentError as e:
      # Enhanced error handling for Colab V2
      error_msg = str(e)
      if _is_colab_v2_environment() and "No OpKernel was registered" in error_msg:
        raise errors.NotFoundError(
            None, None,
            "TPU initialization failed in Google Colab V2. This is a known issue. "
            "Please try the following steps:\n"
            "1. Install the correct TPU libraries: "
            "!pip install -U \"https://storage.googleapis.com/libtpu-releases/libtpu-nightly.tar.gz\"\n"
            "2. Restart the runtime\n"
            "3. Import tensorflow after installing libtpu\n"
            f"Original error: {error_msg}")
      else:
        raise errors.NotFoundError(
            None, None,
            "TPUs not found in the cluster. Failed in initialization: "
            + str(e))
    finally:
      if run_eagerly is not None:
        def_function.run_functions_eagerly(run_eagerly)
    # Clear out the eager context caches since the memory is invalid now.
    context.context()._initialize_logical_devices()  # pylint: disable=protected-access

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
  cluster_resolver.set_tpu_topology(serialized_topology)
  _INITIALIZED_TPU_SYSTEMS[tpu_name] = tpu_topology

  # Record the address of the TPU worker-0 that the coordinator connects to.
  # This can be used to associate the TPU worker with the right coordinator when
  # aggregating the metrics for the application. An example of the address:
  # /bns/mb/borg/mb/bns/chienchunh/chienchunh_group_49640234.1.tfm_train_tpu_worker/0
  _tpu_worker_address.get_cell("address").set(cluster_resolver.get_master())

  return tpu_topology


def get_initialized_tpu_systems():
  """Returns all currently initialized tpu systems.

  Returns:
     A dictionary, with tpu name as the key and the tpu topology as the value.
  """
  return _INITIALIZED_TPU_SYSTEMS.copy()


def shutdown_tpu_system_impl(cluster_resolver, tpu_cluster_resolver_cls):
  """Implementation for tpu.experimental.shutdown_tpu_system.

  Kept separate to avoid tpu_oss code duplication.

  Shuts down the TPU devices.

  This will clear all caches, even those that are maintained through sequential
  calls to tf.tpu.experimental.initialize_tpu_system, such as the compilation
  cache.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
    tpu_cluster_resolver_cls: a reference to
        tf.distribute.cluster_resolver.TPUClusterResolver so that an instance
        of it can be initialized if cluster_resolver is None.

  Raises:
    RuntimeError: If no TPU devices found for eager execution or if run in a
        tf.function.
    TypeError: If tpu_cluster_resolver_cls is
        not tf.distribute.cluster_resolver.TPUClusterResolver.
  """
  # check that tpu_cluster_resolver_cls is a
  # tf.distribute.cluster_resolver.TPUClusterResolver
  if tpu_cluster_resolver_cls is None or not issubclass(
      tpu_cluster_resolver_cls, cluster_resolver_lib.ClusterResolver
  ) or not hasattr(tpu_cluster_resolver_cls, "tpu_hardware_feature"):
    raise TypeError(
        "tpu_cluster_resolver_cls is not"
        " tf.distribute.cluster_resolver.TPUClusterResolver.")

  job = None
  if cluster_resolver is None:
    # If no cluster resolver is specified, and running eagerly, execute the init
    # ops in the current device scope.
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)

    cluster_resolver = tpu_cluster_resolver_cls("")
  assert isinstance(cluster_resolver, tpu_cluster_resolver_cls)

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

    @def_function.function(autograph=False)
    def _tpu_shutdown_fn():
      tpu.shutdown_system(job=job)

    # The TPU_SYSTEM device must match the device used in tpu.shutdown_system
    # exactly, otherwise you can get errors if there are multiple TPU_SYSTEM
    # devices available.
    run_eagerly = def_function.functions_run_eagerly()
    if run_eagerly:
      logging.warning(
          "It looks like tf.function behavior was disabled, perhaps using"
          " tf.config.run_functions_eagerly."
          " tf.tpu.experimental.shutdown_tpu_system requires tf.function to"
          " work. This primitive will override the disable."
      )
      def_function.run_functions_eagerly(False)
    try:
      with ops.device(tpu._tpu_system_device_name(job)):  # pylint: disable=protected-access
        _tpu_shutdown_fn()
    finally:
      if run_eagerly is not None:
        def_function.run_functions_eagerly(run_eagerly)

    # Clear out the eager context caches since the memory is invalid now.
    logging.info("Clearing out eager caches")
    context.context()._clear_caches()  # pylint: disable=protected-access
    context.context().clear_kernel_cache()
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
    raise RuntimeError(
        "initialize_tpu_system is not supported within "
        "tf.functions.  You should call initialize_tpu_system outside of your tf.function. "
    )

  logging.info("Finished shutting down TPU system.")
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    del _INITIALIZED_TPU_SYSTEMS[tpu_name]
