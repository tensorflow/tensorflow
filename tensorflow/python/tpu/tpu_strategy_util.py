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
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional_ops
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


_INITIALIZED_TPU_SYSTEMS = {}


@tf_export("tpu.experimental.initialize_tpu_system")
def initialize_tpu_system(cluster_resolver=None):
  """Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster.

  Raises:
    RuntimeError: If no TPU devices found for eager execution.
  """
  if cluster_resolver is None:
    cluster_resolver = TPUClusterResolver("")
  assert isinstance(cluster_resolver, TPUClusterResolver)

  tpu_name = compat.as_text(cluster_resolver._tpu)  # pylint: disable=protected-access
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    logging.warning("TPU system %s has already been initialized. "
                    "Reinitializing the TPU can cause previously created "
                    "variables on TPU to be lost.")

  logging.info("Initializing the TPU system.")

  if context.executing_eagerly():
    # This function looks as it is for the following non-intuitive reasons.
    # tpu.initialize_system creates a dummy op whose sole purpose is to trigger
    # DistributedTPURewritePass. This pass actually adds real ops that
    # initialize the TPU system. Thus, we can't simply run tpu.initialize_system
    # eagerly. We need to wrap it in defun and trigger the rewrite passes on it.
    # The easiest way to trigger a rewrite is to run the function with
    # TPUPartitionedCallOp.
    @function.defun
    def _tpu_init_fn():
      return tpu.initialize_system()

    # We can't call _tpu_init_fn normally (because it contains just a dummy op,
    # see above) but need to define it to get it added to eager context
    # and get its assigned name.
    # pylint: disable=protected-access
    graph_func = _tpu_init_fn._get_concrete_function_internal()
    func_name = compat.as_str(graph_func._inference_function.name)
    # pylint: enable=protected-access

    tpu_devices = sorted(
        [x for x in context.list_devices() if "device:TPU:" in x])

    if not tpu_devices:
      raise RuntimeError("Could not find any TPU devices")

    with ops.device(device_util.get_host_for_device(tpu_devices[0])):
      output = tpu_functional_ops.TPUPartitionedCall(
          args=[], device_ordinal=0, Tout=[dtypes.string], f=func_name)
    serialized_topology = output[0].numpy()
  else:
    master = cluster_resolver.master()
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    with ops.Graph().as_default():
      with session_lib.Session(config=session_config, target=master) as sess:
        serialized_topology = sess.run(tpu.initialize_system())

  logging.info("Finished initializing TPU system.")
  tpu_topology = topology.Topology(serialized=serialized_topology)
  _INITIALIZED_TPU_SYSTEMS[tpu_name] = tpu_topology

  return tpu_topology
