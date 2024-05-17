# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=unidiomatic-typecheck
"""A shim layer for working with functions exported/restored from saved models.

This functionality should ultimately be moved into a first-class core API.
"""

import numpy

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable


@registration.register_tf_serializable()
class TrackableConstant(trackable.Trackable):
  """Trackable class for captured constants."""
  __slots__ = ("capture", "function", "_exported_tensor")

  def __init__(self, capture, function):
    self.capture = capture
    self.function = function
    self._exported_tensor = None

  def _export_to_saved_model_graph(self, tensor_map, **unused_kwargs):
    capture_constant_value = tensor_util.constant_value(self.capture)
    if capture_constant_value is None:
      raise ValueError(
          f"Unable to save function {self.function.name} because it "
          f"captures graph tensor {self.capture} from a parent function which "
          "cannot be converted to a constant with `tf.get_static_value`.")

    if numpy.prod(self.capture.shape.as_list()) > 1 and numpy.all(
        capture_constant_value == capture_constant_value.flat[0]):
      # For the common case of a constant array filled with the same
      # value, rebuild the constant op specifically with the shape arg,
      # since otherwise the whole array is written into the node def,
      # causing performance and graph proto size issues (protos cannot be
      # bigger than 2GB).
      copied_tensor = constant_op.constant(
          capture_constant_value.flat[0],
          dtype=self.capture.dtype,
          shape=self.capture.shape)
    else:
      copied_tensor = constant_op.constant(capture_constant_value)

    tensor_map[self.capture] = copied_tensor
    self._exported_tensor = copied_tensor
    return [self.capture]

  def _serialize_to_proto(self, object_proto=None, **kwargs):
    object_proto.constant.operation = self._exported_tensor.op.name

  @classmethod
  def _deserialize_from_proto(cls, object_proto, operation_attributes,
                              **kwargs):
    tensor_proto = (
        operation_attributes[object_proto.constant.operation]["value"].tensor)
    ndarray = tensor_util.MakeNdarray(tensor_proto)
    if dtypes.as_dtype(tensor_proto.dtype) == dtypes.string:
      with ops.device("CPU"):
        # String operations should be done on the CPU.
        imported_constant = constant_op.constant(ndarray)
    else:
      imported_constant = constant_op.constant(ndarray)
    return imported_constant
