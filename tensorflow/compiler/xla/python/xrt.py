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

This module implements the Python XLA client's `Backend` abstraction using XRT
, which embeds XLA's compiler/runtime operations as TensorFlow
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


def _make_xla_shape(shape):
  if shape.is_tuple():
    return _xla.Shape.Tuple([_make_xla_shape(s) for s in shape.tuple_shapes()])
  return _xla.Shape.Array(shape.xla_element_type(), shape.dimensions(),
                          shape.minor_to_major())


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

  def __init__(self, tf_context, tf_device_type):
    self.tf_device_type = tf_device_type

    self.context = _xla.xrt.XrtContext.Create(tf_context, tf_device_type)

  def device_count(self):
    return self.context.DeviceCount()

  def buffer_from_pyval(self, pyval, device=0):
    return _xla.xrt.XrtBuffer.FromLiteral(self.context, device, pyval)

  def delete_buffer(self, c_buffer):
    c_buffer.Delete()

  def destructure_tuple(self, c_buffer):
    return c_buffer.DestructureTuple()

  def compile(self, computation, arg_shapes, result_shape, compile_options):
    del arg_shapes
    del result_shape
    # pylint: disable=protected-access
    program_shape = xla_client._wrap_program_shape(
        computation.GetProgramShape())
    # pylint: enable=protected-access
    proto = computation.GetSerializedProto()
    arg_shapes = [
        _make_xla_shape(shape.with_major_to_minor_layout_if_absent())
        for shape in program_shape.parameter_shapes
    ]
    result_shape = _make_xla_shape(
        program_shape.result_shape.with_major_to_minor_layout_if_absent())
    device_assignment = _xla.xrt.AssignDevices(compile_options.num_replicas, 1)
    return _xla.xrt.XrtExecutable.Compile(self.context, proto, arg_shapes,
                                          result_shape, device_assignment)

  def delete_executable(self, executable):
    executable.Delete()

  def execute(self, executable, args):
    return executable.Execute(args)

  def execute_replicated(self, executable, per_replica_args):
    # The extra list packing and unpacking is to handle multiple
    # computations per replica, which we don't support yet.
    return executable.ExecuteReplicated([per_replica_args])[0]
