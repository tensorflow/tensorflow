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
"""XLA backend that runs XRT operators via TensorFlow remote eager.

This module implements the Python XLA client's `Backend` abstraction using XRT,
which embeds XLA's compiler/runtime operations as TensorFlow
operations. The module uses TensorFlow's remote eager RPC API to invoke XRT
operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.xla.python import xla_extension as _xla
# pylint: enable=g-direct-tensorflow-import


def get_tf_context(target, worker):
  """Returns a TensorFlow RPC client object.

  Args:
    target: string; a host:port pair (e.g., '10.0.101.1:8470') naming an XRT
      server.
    worker: string; the task name of the remote TensorFlow worker.
  """
  client = _xla.xrt.GetTfClient(target, worker)
  options = _xla.xrt.XrtTfContextOptions()
  options.max_queue_size = 10000
  return _xla.xrt.XrtTfContext.Create(options, client, worker, 0)


class XrtBackend(xla_client.Backend):
  """XLA backend using XRT.

  Args:
    tf_context: an XrtTfContext object.
    tf_device_type: the type of TensorFlow device to use for XRT (e.g. `"TPU"`).
  """

  def __init__(self, tf_context, tf_device_type, platform="tpu"):
    super(XrtBackend, self).__init__(platform)
    self.tf_device_type = tf_device_type

    self.context = _xla.xrt.XrtContext.Create(tf_context, tf_device_type)

  def device_count(self):
    return self.context.DeviceCount()

  def local_device_count(self):
    raise NotImplementedError()

  def devices(self):
    raise NotImplementedError()

  def host_id(self):
    raise NotImplementedError()

  def buffer_from_pyval(self, pyval, device=0):
    return _xla.xrt.XrtBuffer.from_literal(self.context, device, pyval)

  def make_tuple(self, buffers, device_ordinal):
    return _xla.xrt.XrtBuffer.make_tuple(self.context, buffers, device_ordinal)

  def compile(self, computation, compile_options):
    # pylint: disable=protected-access
    program_shape = computation.GetProgramShape()
    # pylint: enable=protected-access
    proto = computation.GetSerializedProto()
    # TODO(phawkins): use the layouts in compile_options.
    arg_shapes = [
        shape.with_major_to_minor_layout_if_absent()
        for shape in program_shape.parameter_shapes()
    ]
    result_shape = (
        program_shape.result_shape().with_major_to_minor_layout_if_absent())
    device_assignment = _xla.xrt.AssignDevices(compile_options.num_replicas, 1)
    return _xla.xrt.XrtExecutable.Compile(self.context, proto, arg_shapes,
                                          result_shape, device_assignment)
