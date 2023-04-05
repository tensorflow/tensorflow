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
# ==============================================================================
"""Functions used by multiple converter files."""

import copy
import datetime
import sys

from absl import logging

import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph

# The field name of conversion metadata in the flatbuffer file.
CONVERSION_METADATA_FIELD_NAME = "CONVERSION_METADATA"

# Keras functions used by TFLite
model_input_signature = _tflite_keras_util.model_input_signature
trace_model_call = _tflite_keras_util.trace_model_call

# Jax functions used by TFLite
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
try:
  from jax import xla_computation as _xla_computation
except ImportError:
  _xla_computation = None
# pylint: enable=g-import-not-at-top
# pylint: enable=unused-import

# Defined as per TFLite schema
_MAP_TFLITE_ENUM_TO_TF_TYPES = {
    0: dtypes.float32,
    1: dtypes.float16,
    2: dtypes.int32,
    3: dtypes.uint8,
    4: dtypes.int64,
    5: dtypes.string,
    6: dtypes.bool,
    7: dtypes.int16,
    8: dtypes.complex64,
    9: dtypes.int8,
    10: dtypes.float64,
    11: dtypes.complex128,
    16: dtypes.uint32,
}

_TFLITE_FILE_IDENTIFIER = b"TFL3"

_MAP_QUANT_TO_IO_TYPES = {
    dtypes.int8: {dtypes.int8, dtypes.uint8},
    dtypes.int16: {dtypes.int16},
}


def _convert_tflite_enum_type_to_tf_type(tflite_enum_type):
  """Converts tflite enum type (eg: 0) to tf type (eg: tf.float32).

  Args:
    tflite_enum_type: tflite enum type (eg: 0, that corresponds to float32)

  Raises:
    ValueError: If an invalid tflite enum type is provided.

  Returns:
    tf type (eg: tf.float32)
  """
  tf_type = _MAP_TFLITE_ENUM_TO_TF_TYPES.get(tflite_enum_type)
  if tf_type is None:
    raise ValueError(
        "Unsupported enum {}. The valid map of enum to tf types is : {}"
        .format(tflite_enum_type, _MAP_TFLITE_ENUM_TO_TF_TYPES))
  return tf_type


def get_tf_type_name(tf_type):
  """Converts tf.dtype (eg: tf.float32) to str (eg: "tf.float32")."""
  return "tf." + tf_type.name if tf_type else None


def get_tensor_name(tensor):
  """Returns name of the input tensor.

  Args:
    tensor: tf.Tensor

  Returns:
    str
  """
  parts = tensor.name.split(":")
  if len(parts) > 2:
    raise ValueError("Tensor name invalid. Expect 0 or 1 colon, got {0}".format(
        len(parts) - 1))

  # To be consistent with the tensor naming scheme in tensorflow, we need
  # drop the ':0' suffix for the first tensor.
  if len(parts) > 1 and parts[1] != "0":
    return tensor.name
  return parts[0]


def get_tensors_from_tensor_names(graph, tensor_names):
  """Gets the Tensors associated with the `tensor_names` in the provided graph.

  Args:
    graph: TensorFlow Graph.
    tensor_names: List of strings that represent names of tensors in the graph.

  Returns:
    A list of Tensor objects in the same order the names are provided.

  Raises:
    ValueError:
      tensor_names contains an invalid tensor name.
  """
  # Get the list of all of the tensors.
  tensor_name_to_tensor = {}
  for op in graph.get_operations():
    for tensor in op.values():
      tensor_name_to_tensor[get_tensor_name(tensor)] = tensor

  # Get the tensors associated with tensor_names.
  tensors = []
  invalid_tensors = []
  for name in tensor_names:
    if not isinstance(name, str):
      raise ValueError("Invalid type for a tensor name in the provided graph. "
                       "Expected type for a tensor name is 'str', instead got "
                       "type '{}' for tensor name '{}'".format(
                           type(name), name))

    tensor = tensor_name_to_tensor.get(name)
    if tensor is None:
      invalid_tensors.append(name)
    else:
      tensors.append(tensor)

  # Throw ValueError if any user input names are not valid tensors.
  if invalid_tensors:
    raise ValueError("Invalid tensors '{}' were found.".format(
        ",".join(invalid_tensors)))
  return tensors


def set_tensor_shapes(tensors, shapes):
  """Sets Tensor shape for each tensor if the shape is defined.

  Args:
    tensors: TensorFlow ops.Tensor.
    shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).

  Raises:
    ValueError:
      `shapes` contains an invalid tensor.
      `shapes` contains an invalid shape for a valid tensor.
  """
  if shapes:
    tensor_names_to_tensor = {
        get_tensor_name(tensor): tensor for tensor in tensors
    }
    for name, shape in shapes.items():
      if name not in tensor_names_to_tensor:
        raise ValueError("Invalid tensor \'{}\' found in tensor shapes "
                         "map.".format(name))
      if shape is not None:
        tensor = tensor_names_to_tensor[name]
        try:
          tensor.set_shape(shape)
        except ValueError as error:
          message = ("The shape of tensor '{0}' cannot be changed from {1} to "
                     "{2}. {3}".format(name, tensor.shape, shape, str(error)))
          raise ValueError(message)


def get_grappler_config(optimizers_list):
  """Creates a tf.compat.v1.ConfigProto for configuring Grappler.

  Args:
    optimizers_list: List of strings that represents the list of optimizers.

  Returns:
    tf.ConfigProto.
  """
  config = _config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  for optimizer in optimizers_list:
    rewrite_options.optimizers.append(optimizer)
  return config


def run_graph_optimizations(graph_def,
                            input_arrays,
                            output_arrays,
                            config,
                            graph=None):
  """Apply standard TensorFlow optimizations to the graph_def.

  Args:
    graph_def: Frozen GraphDef to be optimized.
    input_arrays: List of arrays that are considered inputs of the graph.
    output_arrays: List of arrays that are considered outputs of the graph.
    config: tf.ConfigProto.
    graph: TensorFlow Graph. Required when Eager mode is enabled. (default None)

  Returns:
    A new, optimized GraphDef.
  """
  meta_graph = _export_meta_graph(graph_def=graph_def, graph=graph)

  signature = _meta_graph_pb2.SignatureDef()
  for array in input_arrays:
    signature.inputs[array.name].name = array.name
    signature.inputs[array.name].dtype = array.dtype.as_datatype_enum
    signature.inputs[array.name].tensor_shape.CopyFrom(array.shape.as_proto())

  for array in output_arrays:
    signature.outputs[array.name].name = array.name
    signature.outputs[array.name].dtype = array.dtype.as_datatype_enum
    signature.outputs[array.name].tensor_shape.CopyFrom(array.shape.as_proto())

  meta_graph.signature_def["not_used_key"].CopyFrom(signature)

  # We need to add a collection called 'train_op' so that grappler
  # knows what the outputs are.
  fetch_collection = _meta_graph_pb2.CollectionDef()
  for array in input_arrays + output_arrays:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  return tf_optimizer.OptimizeGraph(config, meta_graph)


def _convert_op_hints_if_present(sess, graph_def, output_tensors,
                                 hinted_outputs_nodes):
  if is_frozen_graph(sess):
    raise ValueError("Try to convert op hints, needs unfrozen graph.")
  output_arrays = [get_tensor_name(tensor) for tensor in output_tensors]
  graph_def = _convert_to_constants.convert_variables_to_constants(
      sess, graph_def, output_arrays + hinted_outputs_nodes)
  graph_def = convert_op_hints_to_stubs(graph_def=graph_def)
  return graph_def


def freeze_graph(sess, input_tensors, output_tensors):
  """Returns a frozen GraphDef.

  Runs a Grappler pass and freezes a graph with Variables in it. Otherwise the
  existing GraphDef is returned. The Grappler pass is only run on models that
  are frozen in order to inline the functions in the graph.
  If OpHints is present, it will try to convert the OpHint graph.

  Args:
    sess: TensorFlow Session.
    input_tensors: List of input tensors.
    output_tensors: List of output tensors (only .name is used from this).

  Returns:
    Frozen GraphDef.
  """
  # Runs a Grappler pass in order to inline any functions in the graph.
  # Asides from inlining any simple function, Grappler will also try to lower
  # while loop into switch merge representation which is undesired for Ophints,
  # so we simply remove those attributes to prevent Grappler from doing so.
  graph_def = _convert_to_constants.disable_lower_using_switch_merge(
      sess.graph_def)
  config = get_grappler_config(["function"])
  graph_def = run_graph_optimizations(
      graph_def, input_tensors, output_tensors, config, graph=sess.graph)

  # If ophints are present, just convert them.
  hinted_outputs_nodes = find_all_hinted_output_nodes(sess)
  if hinted_outputs_nodes:
    return _convert_op_hints_if_present(sess, graph_def, output_tensors,
                                        hinted_outputs_nodes)

  if not is_frozen_graph(sess):
    output_node_names = [tensor.name.split(":")[0] for tensor in output_tensors]
    return _convert_to_constants.convert_variables_to_constants(
        sess, graph_def, output_node_names
    )
  else:
    return sess.graph_def


def is_frozen_graph(sess):
  """Determines if the graph is frozen.

  Determines if a graph has previously been frozen by checking for any
  operations of type Variable*. If variables are found, the graph is not frozen.

  Args:
    sess: TensorFlow Session.

  Returns:
    Bool.
  """
  for op in sess.graph.get_operations():
    if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
      return False
  return True


def build_debug_info_func(original_graph):
  """Returns a method to retrieve the `GraphDebugInfo` from the original graph.

  Args:
    original_graph: The original `Graph` containing all the op stack traces.

  Returns:
    A function which retrieves the stack traces from the original graph and
    converts them to a `GraphDebugInfo` for a given set of nodes.
  """

  def f(original_nodes):
    """Function to create `GraphDebugInfo` for the given `original_nodes`."""
    if not original_graph:
      return None
    # For the given nodes, gets all the op definitions in the original graph.
    useful_ops = []
    for func, name in original_nodes:
      try:
        if not func:
          useful_ops.append((func, original_graph.get_operation_by_name(name)))
        else:
          sub_func = original_graph._get_function(func)  # pylint: disable=protected-access
          if isinstance(sub_func, function.AtomicFunction):  # pylint: disable=protected-access
            useful_ops.append(
                (func, sub_func.graph.get_operation_by_name(name)))
          else:
            sys.stderr.write(
                "Use '@tf.function' or '@defun' to decorate the function.\n")
            continue
      except KeyError:
        # New node created by graph optimizer. No stack trace from source code.
        continue
    # Convert all the op definitions to stack traces in terms of GraphDebugInfo.
    return _error_interpolation.create_graph_debug_info_def(useful_ops)

  return f


def convert_debug_info_func(saved_debug_info):
  """Returns a method to retrieve the `GraphDebugInfo` from the original graph.

  Args:
    saved_debug_info: The `GraphDebugInfo` containing all the debug info.

  Returns:
    A function which retrieves the stack traces from the original graph and
    converts them to a `GraphDebugInfo` for a given set of nodes.
  """

  def f(original_nodes):
    """Function to create `GraphDebugInfo` for the given `original_nodes`."""
    if not saved_debug_info:
      return None

    output_debug_info = graph_debug_info_pb2.GraphDebugInfo()
    # All the files are copied over, so the index wouldn't be changed.
    output_debug_info.files[:] = saved_debug_info.files
    # We only copy over the debug info for the input nodes
    for func, node in original_nodes:
      debug_key = node + "@" + func
      output_debug_info.traces[debug_key].CopyFrom(
          saved_debug_info.traces[debug_key])
    return output_debug_info

  return f


def get_debug_info(nodes_to_debug_info_func, converted_graph):
  """Returns the debug info for the original nodes in the `converted_graph`.

  Args:
    nodes_to_debug_info_func: The method to collect the op debug info for the
      nodes.
    converted_graph: A `GraphDef` after optimization and transformation.

  Returns:
    `GraphDebugInfo` for all the original nodes in `converted_graph`.
  """
  if not nodes_to_debug_info_func:
    return None

  # Collect all the debug info nodes from the converted_graph
  original_nodes = set()
  for node in converted_graph.node:
    debug_nodes = node.experimental_debug_info.original_node_names
    debug_funcs = node.experimental_debug_info.original_func_names
    # If the `original_node_names` are empty, uses the node name directly.
    if not debug_nodes:
      original_nodes.add(("", node.name))
    else:
      for i in range(len(debug_nodes)):
        debug_func = "" if i >= len(debug_funcs) else debug_funcs[i]
        original_nodes.add((debug_func, debug_nodes[i]))

  # Convert the nodes to the debug info proto object.
  return nodes_to_debug_info_func(original_nodes)


def convert_bytes_to_c_source(data,
                              array_name,
                              max_line_width=80,
                              include_guard=None,
                              include_path=None,
                              use_tensorflow_license=False):
  """Returns strings representing a C constant array containing `data`.

  Args:
    data: Byte array that will be converted into a C constant.
    array_name: String to use as the variable name for the constant array.
    max_line_width: The longest line length, for formatting purposes.
    include_guard: Name to use for the include guard macro definition.
    include_path: Optional path to include in the source file.
    use_tensorflow_license: Whether to include the standard TensorFlow Apache2
      license in the generated files.

  Returns:
    Text that can be compiled as a C source file to link in the data as a
    literal array of values.
    Text that can be used as a C header file to reference the literal array.
  """

  starting_pad = "   "
  array_lines = []
  array_line = starting_pad
  for value in bytearray(data):
    if (len(array_line) + 4) > max_line_width:
      array_lines.append(array_line + "\n")
      array_line = starting_pad
    array_line += " 0x%02x," % (value,)
  if len(array_line) > len(starting_pad):
    array_lines.append(array_line + "\n")
  array_values = "".join(array_lines)

  if include_guard is None:
    include_guard = "TENSORFLOW_LITE_UTIL_" + array_name.upper() + "_DATA_H_"

  if include_path is not None:
    include_line = "#include \"{include_path}\"\n".format(
        include_path=include_path)
  else:
    include_line = ""

  if use_tensorflow_license:
    license_text = """
/* Copyright {year} The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
""".format(year=datetime.date.today().year)
  else:
    license_text = ""

  source_template = """{license_text}
// This is a TensorFlow Lite model file that has been converted into a C data
// array using the tensorflow.lite.util.convert_bytes_to_c_source() function.
// This form is useful for compiling into a binary for devices that don't have a
// file system.

{include_line}
// We need to keep the data array aligned on some architectures.
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

const unsigned char {array_name}[] DATA_ALIGN_ATTRIBUTE = {{
{array_values}}};
const int {array_name}_len = {array_length};
"""

  source_text = source_template.format(
      array_name=array_name,
      array_length=len(data),
      array_values=array_values,
      license_text=license_text,
      include_line=include_line)

  header_template = """
{license_text}

// This is a TensorFlow Lite model file that has been converted into a C data
// array using the tensorflow.lite.util.convert_bytes_to_c_source() function.
// This form is useful for compiling into a binary for devices that don't have a
// file system.

#ifndef {include_guard}
#define {include_guard}

extern const unsigned char {array_name}[];
extern const int {array_name}_len;

#endif  // {include_guard}
"""

  header_text = header_template.format(
      array_name=array_name,
      include_guard=include_guard,
      license_text=license_text)

  return source_text, header_text


def _convert_model_from_bytearray_to_object(model_bytearray):
  """Converts a tflite model from a bytearray into a parsable object."""
  model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
  model_object = schema_fb.ModelT.InitFromObj(model_object)
  model_object = copy.deepcopy(model_object)
  return model_object


def _convert_model_from_object_to_bytearray(model_object):
  """Converts a tflite model from a parsable object into a bytearray."""
  # Initial size of the buffer, which will grow automatically if needed
  builder = flatbuffers.Builder(1024)
  model_offset = model_object.Pack(builder)
  builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
  return bytes(builder.Output())


def get_quantize_opcode_idx(model):
  """Returns the quantize op idx."""
  quant_opcode_idxs = []
  for idx, opcode in enumerate(model.operatorCodes):
    builtin_code = schema_util.get_builtin_code_from_operator_code(opcode)
    if builtin_code == schema_fb.BuiltinOperator.QUANTIZE:
      quant_opcode_idxs.append(idx)
  return quant_opcode_idxs


def get_dequantize_opcode_idx(model):
  """Returns the quantize op idx."""
  quant_opcode_idxs = []
  for idx, opcode in enumerate(model.operatorCodes):
    builtin_code = schema_util.get_builtin_code_from_operator_code(opcode)
    if builtin_code == schema_fb.BuiltinOperator.DEQUANTIZE:
      quant_opcode_idxs.append(idx)
  return quant_opcode_idxs


def _update_signature_def_tensors(tensor_maps, map_old_to_new_tensors):
  """Update the tensors in the SignatureDef's TensorMaps."""
  for i in range(len(tensor_maps)):
    if tensor_maps[i].tensorIndex in map_old_to_new_tensors:
      tensor_maps[i].tensorIndex = (
          map_old_to_new_tensors[tensor_maps[i].tensorIndex])


def _remove_tensors_from_model(model, remove_tensors_idxs):
  """Remove tensors from model."""
  if not remove_tensors_idxs:
    return
  if len(model.subgraphs) > 1:
    logging.info("Skipping the removal of dangled tensors since the model has "
                 "multiple subgraphs and tensors can be used in the different "
                 "subgraph(s)")
    return
  subgraph = model.subgraphs[0]
  tensors = subgraph.tensors
  operators = subgraph.operators

  logging.debug("Removing tensors at indices : %s", remove_tensors_idxs)
  # An optimized check to validate if "remove_tensors_idxs" (eg: [4,5,6]) is an
  # exact subset, with ordering, of "tensors" indices (eg: [0,1,2,3,4,5,6]).
  if min(remove_tensors_idxs) == len(tensors) - len(remove_tensors_idxs):
    logging.debug("Removing tensors only at the end of the tensor list")
    del tensors[min(remove_tensors_idxs):]
  else:
    logging.debug("Removing tensors requires updating the model")
    # Map the old tensor indices to new tensor indices
    d_old_to_new_tensors = {}
    left_shift_by = 0
    for idx in range(len(tensors)):
      if idx in remove_tensors_idxs:
        left_shift_by += 1
      else:
        d_old_to_new_tensors[idx] = idx - left_shift_by
    logging.debug("Old to new tensors map: %s", d_old_to_new_tensors.__str__())
    # Update tensor indices referenced throughout the model
    def update_tensors(tensor_idxs):
      for i, ti in enumerate(tensor_idxs):
        tensor_idxs[i] = d_old_to_new_tensors.get(ti, -1)
    update_tensors(subgraph.inputs)
    update_tensors(subgraph.outputs)
    for op in operators:
      update_tensors(op.inputs)
      update_tensors(op.outputs)
    if model.signatureDefs:
      signature_def = model.signatureDefs[0]
      _update_signature_def_tensors(signature_def.inputs, d_old_to_new_tensors)
      _update_signature_def_tensors(signature_def.outputs, d_old_to_new_tensors)
    # Delete the tensors
    for idx in sorted(remove_tensors_idxs, reverse=True):
      tensors.pop(idx)
    logging.debug("Removed tensors marked for deletion")


def _modify_model_input_type(model, inference_input_type=dtypes.float32):
  """Modify model input type."""
  if inference_input_type == dtypes.float32:
    return

  if not model.signatureDefs:
    _modify_model_input_type_per_subgraph(model, 0, -1, inference_input_type)
    return

  for signature_index, signature_def in enumerate(model.signatureDefs):
    _modify_model_input_type_per_subgraph(model, signature_def.subgraphIndex,
                                          signature_index, inference_input_type)


def _modify_model_input_type_per_subgraph(model, subgraph_index,
                                          signature_index,
                                          inference_input_type):
  """Modify model input type per subgraph."""
  subgraph = model.subgraphs[subgraph_index]
  tensors = subgraph.tensors
  operators = subgraph.operators

  # Find all quantize operators
  quant_opcode_idxs = get_quantize_opcode_idx(model)
  if operators and not quant_opcode_idxs:
    for input_idx in subgraph.inputs:
      input_type = _convert_tflite_enum_type_to_tf_type(tensors[input_idx].type)
      if input_type == dtypes.float32:
        raise ValueError("Model input is not dequantized.")
    # None of the inputs have float32, then they must be int16, int8, or bool
    return

  # Validate that the model input is quantized
  input_quant_ops = []
  for op in operators:
    # Find operators that quantize model input
    if op.opcodeIndex in quant_opcode_idxs and op.inputs[0] in subgraph.inputs:
      float_tensor, quant_tensor = tensors[op.inputs[0]], tensors[op.outputs[0]]
      # If found, validate that the operator's input type is float
      float_type = _convert_tflite_enum_type_to_tf_type(float_tensor.type)
      if float_type != dtypes.float32:
        if float_type == inference_input_type:
          continue
        else:
          raise ValueError(
              "Initial model input type must be tf.float32. Expected type for "
              "tensor with name '{}' is tf.float32, instead type is {}".format(
                  float_tensor.name, get_tf_type_name(float_type)))
      # If found, validate that the operator output is quantized and compatible
      # with the final model input type
      quant_type = _convert_tflite_enum_type_to_tf_type(quant_tensor.type)
      if quant_type not in _MAP_QUANT_TO_IO_TYPES:
        raise ValueError(
            "Initial model input is not quantized. Expected type for "
            "tensor with name '{}' should be in {}, instead type is {}".format(
                quant_tensor.name,
                tuple(get_tf_type_name(t) for t in
                      _MAP_QUANT_TO_IO_TYPES.keys()),
                get_tf_type_name(quant_type)))
      else:
        inference_io_types = _MAP_QUANT_TO_IO_TYPES[quant_type]
        if inference_input_type not in inference_io_types:
          raise ValueError(
              "Unsupported `inference_input_type` value. Expected to be in "
              "{}, instead got {}.".format(
                  tuple(get_tf_type_name(t) for t in inference_io_types),
                  get_tf_type_name(inference_input_type)))
      input_quant_ops.append(op)

  if len(subgraph.inputs) != len(input_quant_ops):
    logging.warning(
        "For model inputs containing unsupported operations which cannot be "
        "quantized, the `inference_input_type` attribute will default to the "
        "original type."
        )

  # Modify model input type
  if inference_input_type == dtypes.uint8:
    # Change quant op (float to int8) to quant op (uint8 to int8)
    for op in input_quant_ops:
      int8_quantization = tensors[op.outputs[0]].quantization
      uint8_quantization = schema_fb.QuantizationParametersT()
      uint8_quantization.scale = [int8_quantization.scale[0]]
      uint8_quantization.zeroPoint = [int8_quantization.zeroPoint[0] + 128]
      tensors[op.inputs[0]].quantization = uint8_quantization
      tensors[op.inputs[0]].type = schema_fb.TensorType.UINT8
  elif inference_input_type in _MAP_QUANT_TO_IO_TYPES:
    # Remove the inputs and the quant operator
    remove_tensors_idxs = set()
    for op in input_quant_ops:
      subgraph.inputs[subgraph.inputs == op.inputs[0]] = op.outputs[0]
      if signature_index >= 0:
        signature_def = model.signatureDefs[signature_index]
        for i in range(len(signature_def.inputs)):
          if signature_def.inputs[i].tensorIndex == op.inputs[0]:
            signature_def.inputs[i].tensorIndex = op.outputs[0]
      remove_tensors_idxs.add(op.inputs[0])
      operators.remove(op)
    # Remove tensors marked for deletion.
    _remove_tensors_from_model(model, remove_tensors_idxs)
  else:
    raise ValueError(
        "Unsupported `inference_input_type` value {}.".format(
            get_tf_type_name(inference_input_type)))


def _modify_model_output_type(model, inference_output_type=dtypes.float32):
  """Modify model output type."""
  if inference_output_type == dtypes.float32:
    return

  if not model.signatureDefs:
    _modify_model_output_type_per_subgraph(model, 0, -1, inference_output_type)
    return

  for signature_index, signature_def in enumerate(model.signatureDefs):
    _modify_model_output_type_per_subgraph(model, signature_def.subgraphIndex,
                                           signature_index,
                                           inference_output_type)


def _modify_model_output_type_per_subgraph(model, subgraph_index,
                                           signature_index,
                                           inference_output_type):
  """Modify model output type per subgraph."""
  subgraph = model.subgraphs[subgraph_index]
  tensors = subgraph.tensors
  operators = subgraph.operators

  # Find all dequantize operators
  dequant_opcode_idxs = get_dequantize_opcode_idx(model)
  if operators and not dequant_opcode_idxs:
    for output in subgraph.outputs:
      output_type = _convert_tflite_enum_type_to_tf_type(tensors[output].type)
      if output_type == dtypes.float32:
        raise ValueError("Model output is not dequantized.")
    # None of the outputs have float32, then they must be int16, int8, or bool
    return

  # Validate that the model output is dequantized
  output_dequant_ops = []
  for op in operators:
    # Find operators that dequantize model output
    if (op.opcodeIndex in dequant_opcode_idxs and
        op.outputs[0] in subgraph.outputs):
      # If found, validate that the operator's output type is float
      quant_tensor, float_tensor = tensors[op.inputs[0]], tensors[op.outputs[0]]
      float_type = _convert_tflite_enum_type_to_tf_type(float_tensor.type)
      if float_type != dtypes.float32:
        if float_type == inference_output_type:
          continue
        else:
          raise ValueError(
              "Initial model output type must be tf.float32. Expected type for "
              "tensor with name '{}' is tf.float32, instead type is {}".format(
                  float_tensor.name, get_tf_type_name(float_type)))
      # If found, validate that the operator input is quantized and compatible
      # with the final model output type
      quant_type = _convert_tflite_enum_type_to_tf_type(quant_tensor.type)
      if quant_type not in _MAP_QUANT_TO_IO_TYPES:
        raise ValueError(
            "Initial model output is not dequantized. Expected type for "
            "tensor with name '{}' should be in {}, instead type is {}".format(
                quant_tensor.name,
                tuple(get_tf_type_name(t) for t in
                      _MAP_QUANT_TO_IO_TYPES.keys()),
                get_tf_type_name(quant_type)))
      else:
        inference_io_types = _MAP_QUANT_TO_IO_TYPES[quant_type]
        if inference_output_type not in inference_io_types:
          raise ValueError(
              "Unsupported `inference_output_type` value. Expected to be in "
              "{}, instead got {}.".format(
                  tuple(get_tf_type_name(t) for t in inference_io_types),
                  get_tf_type_name(inference_output_type)))
      output_dequant_ops.append(op)

  if len(subgraph.outputs) != len(output_dequant_ops):
    logging.warning(
        "For model outputs containing unsupported operations which cannot be "
        "quantized, the `inference_output_type` attribute will default to the "
        "original type."
        )

  # Modify model output type
  if inference_output_type == dtypes.uint8:
    # Find a quantize operator
    quant_opcode_idx = -1
    for idx, opcode in enumerate(model.operatorCodes):
      builtin_code = schema_util.get_builtin_code_from_operator_code(opcode)
      if builtin_code == schema_fb.BuiltinOperator.QUANTIZE:
        quant_opcode_idx = idx
        break
    # Create a quantize operator, if none exist
    if quant_opcode_idx == -1:
      quant_op = schema_fb.OperatorCodeT()
      quant_op.builtinCode = schema_fb.BuiltinOperator.QUANTIZE
      quant_op.deprecatedBuiltinCode = schema_fb.BuiltinOperator.QUANTIZE
      model.operatorCodes.append(quant_op)
      quant_opcode_idx = len(model.operatorCodes) - 1
    # Change dequant op (int8 to float) to quant op (int8 to uint8)
    for op in output_dequant_ops:
      op.opcodeIndex = quant_opcode_idx
      int8_quantization = tensors[op.inputs[0]].quantization
      uint8_quantization = schema_fb.QuantizationParametersT()
      uint8_quantization.scale = [int8_quantization.scale[0]]
      uint8_quantization.zeroPoint = [int8_quantization.zeroPoint[0] + 128]
      tensors[op.outputs[0]].quantization = uint8_quantization
      tensors[op.outputs[0]].type = schema_fb.TensorType.UINT8
  elif inference_output_type in _MAP_QUANT_TO_IO_TYPES:
    # Remove the outputs and the dequant operator
    remove_tensors_idxs = set()
    for op in output_dequant_ops:
      subgraph.outputs[subgraph.outputs == op.outputs[0]] = op.inputs[0]
      if signature_index >= 0:
        signature_def = model.signatureDefs[signature_index]
        for i in range(len(signature_def.outputs)):
          if signature_def.outputs[i].tensorIndex == op.outputs[0]:
            signature_def.outputs[i].tensorIndex = op.inputs[0]
      remove_tensors_idxs.add(op.outputs[0])
      operators.remove(op)
    # Remove tensors marked for deletion.
    _remove_tensors_from_model(model, remove_tensors_idxs)
  else:
    raise ValueError(
        "Unsupported `inference_output_type` value {}.".format(
            get_tf_type_name(inference_output_type)))


def _remove_redundant_quantize_ops(model):
  """Finds back to back quantize ops and remove the first quantize op."""
  if not model.signatureDefs:
    _remove_redundant_quantize_ops_per_subgraph(model, 0, -1)
    return

  for signature_index, signature_def in enumerate(model.signatureDefs):
    _remove_redundant_quantize_ops_per_subgraph(model,
                                                signature_def.subgraphIndex,
                                                signature_index)


def _remove_redundant_quantize_ops_per_subgraph(model, subgraph_index,
                                                signature_index):
  """Remove redundant quantize ops per subgraph."""
  subgraph = model.subgraphs[subgraph_index]
  tensors = subgraph.tensors
  operators = subgraph.operators

  # Find all quantize operators.
  quant_opcode_idxs = get_quantize_opcode_idx(model)
  dequant_opcode_idxs = get_dequantize_opcode_idx(model)

  # Find all redundant quant tensors.
  all_quant_ops = []
  redundant_quant_tensors = {}
  output_dequant_tensors = {}
  for op in operators:
    if op.opcodeIndex in quant_opcode_idxs:
      all_quant_ops.append(op)
      input_tensor = tensors[op.inputs[0]]
      output_tensor = tensors[op.outputs[0]]
      input_type = _convert_tflite_enum_type_to_tf_type(input_tensor.type)
      output_type = _convert_tflite_enum_type_to_tf_type(output_tensor.type)
      # This is a requantize op, so write down its input tensor index.
      if input_type != dtypes.float32 and output_type != dtypes.float32:
        redundant_quant_tensors[op.inputs[0]] = op
    if (op.opcodeIndex in dequant_opcode_idxs and
        op.outputs[0] in subgraph.outputs):
      output_dequant_tensors[op.inputs[0]] = op

  # Remove all the quant ops which produce the redundant quant tensors.
  for op in all_quant_ops:
    output_tensor_idx = op.outputs[0]
    if output_tensor_idx in redundant_quant_tensors:
      requantize_op = redundant_quant_tensors[output_tensor_idx]
      if model.signatureDefs:
        signature_def = model.signatureDefs[0]
        for output in signature_def.outputs:
          if output.tensorIndex == op.outputs[0]:
            output.tensorIndex = op.inputs[0]
      # Reset the input of the requantize op to the float input
      requantize_op.inputs[0] = op.inputs[0]
      operators.remove(op)

  # Remove all the quant ops which connect to the output dequant op.
  for op in all_quant_ops:
    output_tensor_idx = op.outputs[0]
    if output_tensor_idx in output_dequant_tensors:
      dequant_op = output_dequant_tensors[output_tensor_idx]
      subgraph.outputs[subgraph.outputs == dequant_op.outputs[0]] = op.inputs[0]
      if signature_index >= 0:
        signature_def = model.signatureDefs[signature_index]
        for output in signature_def.outputs:
          if output.tensorIndex == dequant_op.outputs[0]:
            output.tensorIndex = op.inputs[0]
      operators.remove(op)
      operators.remove(dequant_op)


def modify_model_io_type(
    model, inference_input_type=dtypes.float32,
    inference_output_type=dtypes.float32):
  """Modify the input/output type of a tflite model.

  Args:
    model: A tflite model.
    inference_input_type: tf.DType representing modified input type.
      (default tf.float32. If model input is int8 quantized, it must be in
      {tf.float32, tf.int8,tf.uint8}, else if model input is int16 quantized,
      it must be in {tf.float32, tf.int16}, else it must be tf.float32)
    inference_output_type: tf.DType representing modified output type.
      (default tf.float32. If model output is int8 dequantized, it must be in
      {tf.float32, tf.int8,tf.uint8}, else if model output is int16 dequantized,
      it must be in {tf.float32, tf.int16}, else it must be tf.float32)
  Returns:
    A tflite model with modified input/output type.

  Raises:
    ValueError: If `inference_input_type`/`inference_output_type` is unsupported
      or a supported integer type is specified for a model whose input/output is
      not quantized/dequantized.
    RuntimeError: If the modification was unsuccessful.

  """
  if (inference_input_type == dtypes.float32 and
      inference_output_type == dtypes.float32):
    return model

  model_object = _convert_model_from_bytearray_to_object(model)

  _modify_model_input_type(model_object, inference_input_type)

  _modify_model_output_type(model_object, inference_output_type)

  _remove_redundant_quantize_ops(model_object)

  return _convert_model_from_object_to_bytearray(model_object)


def get_sparsity_modes(model_object):
  """Get sparsity modes used in a tflite model.

  The sparsity modes are listed in conversion_metadata.fbs file.

  Args:
    model_object: A tflite model in object form.

  Returns:
    The list of sparsity modes used in the model.
  """
  if not model_object or not model_object.metadata:
    return []

  result = set()
  for subgraph in model_object.subgraphs:
    for tensor in subgraph.tensors:
      if not tensor.sparsity:
        continue

      # Block map is the list if indexes where the block size is larger than 1.
      # So empty block map means it is random sparsity.
      if not tensor.sparsity.blockMap:
        result.add(
            conversion_metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY)
      else:
        result.add(
            conversion_metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY)

  return list(result)


def populate_conversion_metadata(model_object, metadata):
  """Add or update conversion metadata to a tflite model.

  Args:
    model_object: A tflite model in object form.
    metadata: The conversion metadata.

  Returns:
    A tflite model object with embedded conversion metadata.
  """
  try:
    metadata_builder = flatbuffers.Builder(0)
    metadata_builder.Finish(metadata.Pack(metadata_builder))
    buffer_field = schema_fb.BufferT()
    buffer_field.data = metadata_builder.Output()

    if not model_object.metadata:
      model_object.metadata = []
    else:
      # Check if metadata has already been populated.
      for meta in model_object.metadata:
        if meta.name.decode("utf-8") == CONVERSION_METADATA_FIELD_NAME:
          model_object.buffers[meta.buffer] = buffer_field
          return model_object

    if not model_object.buffers:
      model_object.buffers = []
    model_object.buffers.append(buffer_field)
    # Creates a new metadata field.
    metadata_field = schema_fb.MetadataT()
    metadata_field.name = CONVERSION_METADATA_FIELD_NAME
    metadata_field.buffer = len(model_object.buffers) - 1
    model_object.metadata.append(metadata_field)

    return model_object
  except Exception:  # pylint: disable=broad-except
    return model_object


def get_conversion_metadata(model_buffer):
  """Read conversion metadata from a tflite model.

  Args:
    model_buffer: A tflite model.

  Returns:
    The conversion metadata or None if it is not populated.
  """
  model_object = flatbuffer_utils.convert_bytearray_to_object(model_buffer)
  if not model_object or not model_object.metadata:
    return None

  for meta in model_object.metadata:
    if meta.name.decode("utf-8") == CONVERSION_METADATA_FIELD_NAME:
      metadata_buf = model_object.buffers[meta.buffer].data.tobytes()
      return conversion_metadata_fb.ConversionMetadataT.InitFromObj(
          conversion_metadata_fb.ConversionMetadata.GetRootAsConversionMetadata(
              metadata_buf, 0))

  return None
