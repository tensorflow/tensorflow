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
from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional_ops
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


def get_first_tpu_host_device(cluster_resolver):
  """Get the device spec for the first TPU host."""
  if context.executing_eagerly():
    tpu_devices = sorted(
        [x for x in context.list_devices() if "device:TPU:" in x])
    if not tpu_devices:
      raise RuntimeError("Could not find any TPU devices")
    spec = tf_device.DeviceSpec.from_string(tpu_devices[0])
    task_id = spec.task
  else:
    # Session master needs to be configured and the coordinator is not part
    # of the cluster.
    task_id = 0
  if cluster_resolver.get_master() in ("", "local"):
    return "/replica:0/task:0/device:CPU:0"
  job_name = cluster_resolver.get_job_name() or "tpu_worker"
  return "/job:%s/task:%d/device:CPU:0" % (job_name, task_id)


@tf_export("tpu.experimental.initialize_tpu_system")
def initialize_tpu_system(cluster_resolver=None):
  """Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster.
  """
  if cluster_resolver is None:
    cluster_resolver = TPUClusterResolver("")

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

    with ops.device(get_first_tpu_host_device(cluster_resolver)):
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
  return topology.Topology(serialized=serialized_topology)
