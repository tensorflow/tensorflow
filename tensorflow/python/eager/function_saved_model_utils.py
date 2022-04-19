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
import warnings

import numpy

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import registration
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import tracking


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


def get_tensor_from_node(node):
  """Resolves a saved model graph node into a tensor to be captured.

  Args:
    node: a tensor, variable, or resource to be resolved into a capturable
      tensor

  Returns:
    A list of tensors.
  Raises:
    ValueError: if the node cannot be converted into a tensor.
  """
  with ops.init_scope():
    # TODO(b/210144904): Use __tf_tensor__ instead of `is_[...]` checks
    if getattr(node, "is_distributed_variable", False):
      return node
    elif getattr(node, "is_distributed_table", False):
      return node
    elif getattr(node, "is_sharded_variable", False):
      return node
    elif resource_variable_ops.is_resource_variable(node):
      return node.handle
    elif isinstance(node, tracking.Asset):
      return node.asset_path
    elif tensor_util.is_tf_type(node):
      return node
    elif isinstance(node, tracking.CapturableResource):
      # Note: this executes restored functions in the CapturableResource.
      return node.resource_handle
    raise ValueError(f"Cannot convert node {node} to tensor.")


def restore_captures(concrete_function, inputs):
  """Restore captures for the concrete function.

  Used at deserialization time.  For functions that are being deserialized,
  saved model restores objects that tensors were captured from, but functions
  only know about their tensors -- object information is destroyed by tracing.
  This additional logic extracts the tensors which the function originally
  captured.

  Args:
    concrete_function: the concrete function for which to restore captures
    inputs: a list tensors or other Python objects (such as variables) which
      contain tensors that were originally captured by the function
  """
  bound_inputs = [get_tensor_from_node(obj) for obj in inputs]
  bound_variables = [
      obj for obj in inputs
      if isinstance(obj, (variables_lib.Variable,
                          resource_variable_ops.BaseResourceVariable))
  ]
  # TODO(b/205010575): This is only injecting the captured inputs into the
  # concrete function, note that we did not modify the FuncGraph
  # itself.
  captured_inputs_list = []
  concrete_function.set_variables(bound_variables)
  if bound_inputs:
    for bound_input, internal_capture in zip(
        bound_inputs, concrete_function.inputs[-len(bound_inputs):]):
      # Distributed inputs have special logic for capturing, so we call their
      # custom restoration methods
      if hasattr(bound_input, "__tf_experimental_restore_capture__"):
        captured_inputs_list.append(
            bound_input.__tf_experimental_restore_capture__(
                concrete_function, internal_capture))
      else:
        captured_inputs_list.append(bound_input)
        concrete_function.graph.replace_capture(bound_input, internal_capture)
        if internal_capture.dtype == dtypes.resource:
          if resource_variable_ops.is_resource_variable(bound_input):
            try:
              handle = bound_input.handle
            except ValueError:
              # For mirrored variables we'll copy handle data for components
              # as they get captured.
              pass
            else:
              handle_data_util.copy_handle_data(handle, internal_capture)
          else:
            # TODO(b/213451747): Remove need to call copy_handle_data
            handle_data_util.copy_handle_data(bound_input, internal_capture)
        # Setting "captures" first means "capture" won't create a new
        # placeholder for this input.
        concrete_function.graph.capture(bound_input)

  if any([inp is None for inp in captured_inputs_list]):
    warnings.warn("Trying to load ShardedVariables using tf.saved_model.load. "
                  "This won't work if using a tf.distribute.Strategy, and may "
                  "use excess memory if not using a Strategy. Ignore this "
                  "warning if using tf.keras.models.load_model.")
  concrete_function.set_external_captures(captured_inputs_list)
