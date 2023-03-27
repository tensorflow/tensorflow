# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for FlatBuffers.

All functions that are commonly used to work with FlatBuffers.

Refer to the tensorflow lite flatbuffer schema here:
tensorflow/lite/schema/schema.fbs

"""

import copy
import random
import re
import struct
import sys

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile

_TFLITE_FILE_IDENTIFIER = b'TFL3'


def convert_bytearray_to_object(model_bytearray):
  """Converts a tflite model from a bytearray to an object for parsing."""
  model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
  return schema_fb.ModelT.InitFromObj(model_object)


def read_model(input_tflite_file):
  """Reads a tflite model as a python object.

  Args:
    input_tflite_file: Full path name to the input tflite file

  Raises:
    RuntimeError: If input_tflite_file path is invalid.
    IOError: If input_tflite_file cannot be opened.

  Returns:
    A python object corresponding to the input tflite file.
  """
  if not gfile.Exists(input_tflite_file):
    raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
  with gfile.GFile(input_tflite_file, 'rb') as input_file_handle:
    model_bytearray = bytearray(input_file_handle.read())
  model = convert_bytearray_to_object(model_bytearray)
  if sys.byteorder == 'big':
    byte_swap_tflite_model_obj(model, 'little', 'big')
  return model


def read_model_with_mutable_tensors(input_tflite_file):
  """Reads a tflite model as a python object with mutable tensors.

  Similar to read_model() with the addition that the returned object has
  mutable tensors (read_model() returns an object with immutable tensors).

  Args:
    input_tflite_file: Full path name to the input tflite file

  Raises:
    RuntimeError: If input_tflite_file path is invalid.
    IOError: If input_tflite_file cannot be opened.

  Returns:
    A mutable python object corresponding to the input tflite file.
  """
  return copy.deepcopy(read_model(input_tflite_file))


def convert_object_to_bytearray(model_object):
  """Converts a tflite model from an object to a immutable bytearray."""
  # Initial size of the buffer, which will grow automatically if needed
  builder = flatbuffers.Builder(1024)
  model_offset = model_object.Pack(builder)
  builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
  model_bytearray = bytes(builder.Output())
  return model_bytearray


def write_model(model_object, output_tflite_file):
  """Writes the tflite model, a python object, into the output file.

  Args:
    model_object: A tflite model as a python object
    output_tflite_file: Full path name to the output tflite file.

  Raises:
    IOError: If output_tflite_file path is invalid or cannot be opened.
  """
  if sys.byteorder == 'big':
    model_object = copy.deepcopy(model_object)
    byte_swap_tflite_model_obj(model_object, 'big', 'little')
  model_bytearray = convert_object_to_bytearray(model_object)
  with gfile.GFile(output_tflite_file, 'wb') as output_file_handle:
    output_file_handle.write(model_bytearray)


def strip_strings(model):
  """Strips all nonessential strings from the model to reduce model size.

  We remove the following strings:
  (find strings by searching ":string" in the tensorflow lite flatbuffer schema)
  1. Model description
  2. SubGraph name
  3. Tensor names
  We retain OperatorCode custom_code and Metadata name.

  Args:
    model: The model from which to remove nonessential strings.
  """

  model.description = None
  for subgraph in model.subgraphs:
    subgraph.name = None
    for tensor in subgraph.tensors:
      tensor.name = None
  # We clear all signature_def structure, since without names it is useless.
  model.signatureDefs = None


def type_to_name(tensor_type):
  """Converts a numerical enum to a readable tensor type."""
  for name, value in schema_fb.TensorType.__dict__.items():
    if value == tensor_type:
      return name
  return None


def randomize_weights(model, random_seed=0, buffers_to_skip=None):
  """Randomize weights in a model.

  Args:
    model: The model in which to randomize weights.
    random_seed: The input to the random number generator (default value is 0).
    buffers_to_skip: The list of buffer indices to skip. The weights in these
                     buffers are left unmodified.
  """

  # The input to the random seed generator. The default value is 0.
  random.seed(random_seed)

  # Parse model buffers which store the model weights
  buffers = model.buffers
  buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None
  if buffers_to_skip is not None:
    buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]

  buffer_types = {}
  for graph in model.subgraphs:
    for op in graph.operators:
      if op.inputs is None:
        break
      for input_idx in op.inputs:
        tensor = graph.tensors[input_idx]
        buffer_types[tensor.buffer] = type_to_name(tensor.type)

  for i in buffer_ids:
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
    if buffer_i_size == 0:
      continue

    # Raw data buffers are of type ubyte (or uint8) whose values lie in the
    # range [0, 255]. Those ubytes (or unint8s) are the underlying
    # representation of each datatype. For example, a bias tensor of type
    # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
    # For floats, we need to generate a valid float and then pack it into
    # the raw bytes in place.
    buffer_type = buffer_types.get(i, 'INT8')
    if buffer_type.startswith('FLOAT'):
      format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
      for offset in range(0, buffer_i_size, struct.calcsize(format_code)):
        value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
        struct.pack_into(format_code, buffer_i_data, offset, value)
    else:
      for j in range(buffer_i_size):
        buffer_i_data[j] = random.randint(0, 255)


def rename_custom_ops(model, map_custom_op_renames):
  """Rename custom ops so they use the same naming style as builtin ops.

  Args:
    model: The input tflite model.
    map_custom_op_renames: A mapping from old to new custom op names.
  """
  for op_code in model.operatorCodes:
    if op_code.customCode:
      op_code_str = op_code.customCode.decode('ascii')
      if op_code_str in map_custom_op_renames:
        op_code.customCode = map_custom_op_renames[op_code_str].encode('ascii')


def opcode_to_name(model, op_code):
  """Converts a TFLite op_code to the human readable name.

  Args:
    model: The input tflite model.
    op_code: The op_code to resolve to a readable name.

  Returns:
    A string containing the human readable op name, or None if not resolvable.
  """
  op = model.operatorCodes[op_code]
  code = max(op.builtinCode, op.deprecatedBuiltinCode)
  for name, value in vars(schema_fb.BuiltinOperator).items():
    if value == code:
      return name
  return None


def xxd_output_to_bytes(input_cc_file):
  """Converts xxd output C++ source file to bytes (immutable).

  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd

  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.

  Returns:
    A bytearray corresponding to the input cc file array.
  """
  # Match hex values in the string with comma as separator
  pattern = re.compile(r'\W*(0x[0-9a-fA-F,x ]+).*')

  model_bytearray = bytearray()

  with open(input_cc_file) as file_handle:
    for line in file_handle:
      values_match = pattern.match(line)

      if values_match is None:
        continue

      # Match in the parentheses (hex array only)
      list_text = values_match.group(1)

      # Extract hex values (text) from the line
      # e.g. 0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c,
      values_text = filter(None, list_text.split(','))

      # Convert to hex
      values = [int(x, base=16) for x in values_text]
      model_bytearray.extend(values)

  return bytes(model_bytearray)


def xxd_output_to_object(input_cc_file):
  """Converts xxd output C++ source file to object.

  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd

  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.

  Returns:
    A python object corresponding to the input tflite file.
  """
  model_bytes = xxd_output_to_bytes(input_cc_file)
  return convert_bytearray_to_object(model_bytes)


def byte_swap_buffer_content(buffer, chunksize, from_endiness, to_endiness):
  """Helper function for byte-swapping the buffers field."""
  to_swap = [
      buffer.data[i : i + chunksize]
      for i in range(0, len(buffer.data), chunksize)
  ]
  buffer.data = b''.join(
      [
          int.from_bytes(byteswap, from_endiness).to_bytes(
              chunksize, to_endiness
          )
          for byteswap in to_swap
      ]
  )


def byte_swap_string_content(buffer, from_endiness, to_endiness):
  """Helper function for byte-swapping the string buffer.

  Args:
    buffer: TFLite string buffer of from_endiness format.
    from_endiness: The original endianness format of the string buffer.
    to_endiness: The destined endianness format of the string buffer.
  """
  num_of_strings = int.from_bytes(buffer.data[0:4], from_endiness)
  string_content = bytearray(buffer.data[4*(num_of_strings+2):])
  prefix_data = b''.join([int.from_bytes(
    buffer.data[i:i+4], from_endiness).to_bytes(
      4, to_endiness) for i in range(
        0, (num_of_strings+1)*4+1, 4)])
  buffer.data = prefix_data + string_content


def byte_swap_tflite_model_obj(model, from_endiness, to_endiness):
  """Byte swaps the buffers field in a TFLite model.

  Args:
    model: TFLite model object of from_endiness format.
    from_endiness: The original endianness format of the buffers in model.
    to_endiness: The destined endianness format of the buffers in model.
  """
  if model is None:
    return
  # Get all the constant buffers, byte swapping them as per their data types
  buffer_swapped = []
  types_of_16_bits = [
      schema_fb.TensorType.FLOAT16,
      schema_fb.TensorType.INT16,
      schema_fb.TensorType.UINT16,
  ]
  types_of_32_bits = [
      schema_fb.TensorType.FLOAT32,
      schema_fb.TensorType.INT32,
      schema_fb.TensorType.COMPLEX64,
      schema_fb.TensorType.UINT32,
  ]
  types_of_64_bits = [
      schema_fb.TensorType.INT64,
      schema_fb.TensorType.FLOAT64,
      schema_fb.TensorType.COMPLEX128,
      schema_fb.TensorType.UINT64,
  ]
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      if (
          tensor.buffer > 0
          and tensor.buffer < len(model.buffers)
          and tensor.buffer not in buffer_swapped
          and model.buffers[tensor.buffer].data is not None
      ):
        if tensor.type == schema_fb.TensorType.STRING:
          byte_swap_string_content(
            model.buffers[tensor.buffer], from_endiness, to_endiness
          )
        elif tensor.type in types_of_16_bits:
          byte_swap_buffer_content(
              model.buffers[tensor.buffer], 2, from_endiness, to_endiness
          )
        elif tensor.type in types_of_32_bits:
          byte_swap_buffer_content(
              model.buffers[tensor.buffer], 4, from_endiness, to_endiness
          )
        elif tensor.type in types_of_64_bits:
          byte_swap_buffer_content(
              model.buffers[tensor.buffer], 8, from_endiness, to_endiness
          )
        else:
          continue
        buffer_swapped.append(tensor.buffer)


def byte_swap_tflite_buffer(tflite_model, from_endiness, to_endiness):
  """Generates a new model byte array after byte swapping its buffers field.

  Args:
    tflite_model: TFLite flatbuffer in a byte array.
    from_endiness: The original endianness format of the buffers in
      tflite_model.
    to_endiness: The destined endianness format of the buffers in tflite_model.

  Returns:
    TFLite flatbuffer in a byte array, after being byte swapped to to_endiness
    format.
  """
  if tflite_model is None:
    return None
  # Load TFLite Flatbuffer byte array into an object.
  model = convert_bytearray_to_object(tflite_model)

  # Byte swapping the constant buffers as per their data types
  byte_swap_tflite_model_obj(model, from_endiness, to_endiness)

  # Return a TFLite flatbuffer as a byte array.
  return convert_object_to_bytearray(model)


def count_resource_variables(model):
  """Calculates the number of unique resource variables in a model.

  Args:
    model: the input tflite model, either as bytearray or object.

  Returns:
    An integer number representing the number of unique resource variables.
  """
  if not isinstance(model, schema_fb.ModelT):
    model = convert_bytearray_to_object(model)
  unique_shared_names = set()
  for subgraph in model.subgraphs:
    if subgraph.operators is None:
      continue
    for op in subgraph.operators:
      builtin_code = schema_util.get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex])
      if builtin_code == schema_fb.BuiltinOperator.VAR_HANDLE:
        unique_shared_names.add(op.builtinOptions.sharedName)
  return len(unique_shared_names)
