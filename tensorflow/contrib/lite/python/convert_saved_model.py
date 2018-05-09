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
"""Functions to convert SavedModel to frozen GraphDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.lite.python import convert
from tensorflow.contrib.lite.python import lite_constants
from tensorflow.contrib.lite.toco import model_flags_pb2
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util as tf_graph_util
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


def _write_and_flush_file(file_path, data_str):
  """Writes data to file path.

  Args:
    file_path: Full path of the file to store data in.
    data_str: Data represented as a string.

  Returns: None.
  """
  with gfile.Open(file_path, "wb") as data_file:
    data_file.write(data_str)
    data_file.flush()


def _log_tensor_details(tensor_info):
  """Log tensor details: name, shape, and type."""
  for key in tensor_info:
    val = tensor_info[key]
    dtype = types_pb2.DataType.Name(val.dtype)
    if val.tensor_shape.unknown_rank:
      shape = "unknown_rank"
    else:
      dims = [str(dim.size) for dim in val.tensor_shape.dim]
      shape = "({})".format(", ".join(dims))

    logging.info("Tensor's key in saved_model's tensor_map: %s", key)
    logging.info(" tensor name: %s, shape: %s, type: %s", val.name, shape,
                 dtype)


def _get_meta_graph_def(saved_model_dir, tag_set):
  """Validate saved_model and extract MetaGraphDef.

  Args:
    saved_model_dir: saved_model path to convert.
    tag_set: Set of tag(s) of the MetaGraphDef to load.

  Returns:
    The meta_graph_def used for tflite conversion.

  Raises:
    ValueError: No valid MetaGraphDef for given tag_set.
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  tag_sets = []
  result_meta_graph_def = None
  for meta_graph_def in saved_model.meta_graphs:
    meta_graph_tag_set = set(meta_graph_def.meta_info_def.tags)
    tag_sets.append(meta_graph_tag_set)
    if meta_graph_tag_set == tag_set:
      result_meta_graph_def = meta_graph_def
  logging.info("The given saved_model contains the following tags: %s",
               tag_sets)
  if result_meta_graph_def is not None:
    return result_meta_graph_def
  else:
    raise ValueError("No valid MetaGraphDef for this tag_set '{}'. Possible "
                     "values are '{}'. ".format(tag_set, tag_sets))


def _get_signature_def(meta_graph, signature_key):
  """Get the signature def from meta_graph with given signature_key.

  Args:
    meta_graph: meta_graph_def.
    signature_key: signature_def in the meta_graph_def.

  Returns:
    The signature_def used for tflite conversion.

  Raises:
    ValueError: Given signature_key is not valid for this meta_graph.
  """
  signature_def_map = meta_graph.signature_def
  signature_def_keys = set(signature_def_map.keys())
  logging.info(
      "The given saved_model MetaGraphDef contains SignatureDefs with the "
      "following keys: %s", signature_def_keys)
  if signature_key not in signature_def_keys:
    raise ValueError("No '{}' in the saved_model\'s SignatureDefs. Possible "
                     "values are '{}'. ".format(signature_key,
                                                signature_def_keys))
  signature_def = signature_def_utils.get_signature_def_by_key(
      meta_graph, signature_key)
  return signature_def


def _get_inputs_outputs(signature_def):
  """Get inputs and outputs from SignatureDef.

  Args:
    signature_def: SignatureDef in the meta_graph_def for conversion.

  Returns:
    The inputs and outputs in the graph for conversion.
  """
  inputs_tensor_info = signature_def.inputs
  outputs_tensor_info = signature_def.outputs
  logging.info("input tensors info: ")
  _log_tensor_details(inputs_tensor_info)
  logging.info("output tensors info: ")
  _log_tensor_details(outputs_tensor_info)

  def gather_names(tensor_info):
    return [tensor_info[key].name for key in tensor_info]

  inputs = gather_names(inputs_tensor_info)
  outputs = gather_names(outputs_tensor_info)
  return inputs, outputs


def _get_tensors(graph, signature_def_tensor_names=None,
                 user_tensor_names=None):
  """Gets the tensors associated with the tensor names.

  Either signature_def_tensor_names or user_tensor_names should be provided. If
  the user provides tensors, the tensors associated with the user provided
  tensor names are provided. Otherwise, the tensors associated with the names in
  the SignatureDef are provided.

  Args:
    graph: GraphDef representing graph.
    signature_def_tensor_names: Tensor names stored in either the inputs or
      outputs of a SignatureDef. (default None)
    user_tensor_names: Tensor names provided by the user. (default None)

  Returns:
    List of tensors.

  Raises:
    ValueError:
      signature_def_tensors and user_tensor_names are undefined or empty.
      user_tensor_names are not valid.
  """
  tensors = []
  if user_tensor_names:
    # Get the list of all of the tensors with and without the tensor index.
    all_tensor_names = [
        tensor.name for op in graph.get_operations() for tensor in op.outputs
    ]
    all_tensor_names_only = [name.split(":")[0] for name in all_tensor_names]

    # Sort the tensor names.
    user_tensor_names = sorted(user_tensor_names)

    # Get the tensors associated with the tensor names.
    tensors = []
    invalid_tensors = []
    for name in user_tensor_names:
      if name not in all_tensor_names_only:
        invalid_tensors.append(name)
      else:
        idx = all_tensor_names_only.index(name)
        tensors.append(graph.get_tensor_by_name(all_tensor_names[idx]))

    # Throw ValueError if any user input names are not valid tensors.
    if invalid_tensors:
      raise ValueError("Invalid tensors '{}' were found.".format(
          ",".join(invalid_tensors)))
  elif signature_def_tensor_names:
    tensors = [
        graph.get_tensor_by_name(name)
        for name in sorted(signature_def_tensor_names)
    ]
  else:
    # Throw ValueError if signature_def_tensors and user_tensor_names are both
    # either undefined or empty.
    raise ValueError(
        "Specify either signature_def_tensor_names or user_tensor_names")

  return tensors


def _freeze_saved_model(saved_model_dir, input_arrays, input_shapes,
                        output_arrays, tag_set, signature_key, batch_size):
  """Converts a SavedModel to a frozen graph.

  Args:
    saved_model_dir: SavedModel directory to convert.
    input_arrays: List of input tensors to freeze graph with. Uses input arrays
      from SignatureDef when none are provided. (default None)
    input_shapes: Map of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
      (default None)
    output_arrays: List of output tensors to freeze graph with. Uses output
      arrays from SignatureDef when none are provided. (default None)
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present. (default "serve")
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    batch_size: Batch size for the model. Replaces the first dimension of an
      input size array if undefined. (default 1)

  Returns:
    frozen_graph_def: Frozen GraphDef.
    in_tensors: List of input tensors for the graph.
    out_tensors: List of output tensors for the graph.

  Raises:
    ValueError:
      SavedModel doesn't contain a MetaGraphDef identified by tag_set.
      signature_key is not in the MetaGraphDef.
      input_shapes does not match the length of input_arrays.
      input_shapes has a None value after the 1st dimension.
      input_arrays or output_arrays are not valid.
      Unable to load Session.
  """
  # Set default values for inputs if they are set to None.
  if signature_key is None:
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  if tag_set is None:
    tag_set = set([tag_constants.SERVING])
  if batch_size is None:
    batch_size = 1

  # Read SignatureDef.
  meta_graph = _get_meta_graph_def(saved_model_dir, tag_set)
  signature_def = _get_signature_def(meta_graph, signature_key)
  inputs, outputs = _get_inputs_outputs(signature_def)

  graph = ops.Graph()
  with session.Session(graph=graph) as sess:
    # TODO(nupurgarg): Throw ValueError if SavedModel has assets/ directory.
    loader.load(sess, meta_graph.meta_info_def.tags, saved_model_dir)

    # Gets input and output tensors.
    # TODO(zhixianyan): Use TFLite supported Op list to filter outputs.
    in_tensors = _get_tensors(graph, inputs, input_arrays)
    out_tensors = _get_tensors(graph, outputs, output_arrays)

    # Gets fully defined tensor shape. An input tensor with None in the first
    # dimension, e.g. (None, 224, 224, 3), is replaced with the batch_size.
    # Shapes with None after the first dimension result in a ValueError.
    # TODO(zhixianyan): Add supports for input tensor with more None in shape.
    for tensor in in_tensors:
      if (input_shapes and tensor.name in input_shapes and
          input_shapes[tensor.name] is not None):
        shape = input_shapes[tensor.name]
      else:
        shape = tensor.get_shape().as_list()

      if None in shape[1:]:
        raise ValueError(
            "None is only supported in the 1st dimension. Tensor '{0}' has "
            "invalid shape '{1}'.".format(tensor.name, shape))
      elif shape[0] is None:
        shape[0] = batch_size
      tensor.set_shape(shape)

    output_names = [node.split(":")[0] for node in outputs]
    frozen_graph_def = tf_graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_names)

    return frozen_graph_def, in_tensors, out_tensors
  raise ValueError("Unable to load Session.")


def saved_model_to_frozen_graphdef(
    saved_model_dir,
    output_file_model,
    output_file_flags,
    input_arrays=None,
    input_shapes=None,
    output_arrays=None,
    tag_set=None,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    batch_size=1):
  """Converts a SavedModel to a frozen graph. Writes graph to tmp directory.

  Stores frozen graph and command line flags in the tmp directory.

  Args:
    saved_model_dir: SavedModel directory to convert.
    output_file_model: Full file path to save frozen graph.
    output_file_flags: Full file path to save ModelFlags.
    input_arrays: List of input tensors to freeze graph with. Uses input arrays
      from SignatureDef when none are provided. (default None)
    input_shapes: Map of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
      (default None)
    output_arrays: List of output tensors to freeze graph with. Uses output
      arrays from SignatureDef when none are provided. (default None)
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present. (default "serve")
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    batch_size: Batch size for the model. Replaces the first dimension of an
      input size array if undefined. (default 1)

  Returns: None.

  Raises:
    ValueError: Unable to convert to frozen graph.
  """
  frozen_graph_def, in_tensors, out_tensors = _freeze_saved_model(
      saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set,
      signature_key, batch_size)

  # Initialize model flags.
  model = model_flags_pb2.ModelFlags()

  for input_tensor in in_tensors:
    input_array = model.input_arrays.add()
    input_array.name = convert.tensor_name(input_tensor)
    input_array.shape.dims.extend(map(int, input_tensor.get_shape()))

  for output_tensor in out_tensors:
    model.output_arrays.append(convert.tensor_name(output_tensor))

  # Write model and ModelFlags to file. ModelFlags contain input array and
  # output array information that is parsed from the SignatureDef and used for
  # analysis by TOCO.
  _write_and_flush_file(output_file_model, frozen_graph_def.SerializeToString())
  _write_and_flush_file(output_file_flags, model.SerializeToString())


def tflite_from_saved_model(
    saved_model_dir,
    output_file=None,
    input_arrays=None,
    input_shapes=None,
    output_arrays=None,
    tag_set=None,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    batch_size=1,
    inference_type=lite_constants.FLOAT,
    input_format=lite_constants.TENSORFLOW_GRAPHDEF,
    output_format=lite_constants.TFLITE,
    quantized_input_stats=None,
    drop_control_dependency=True):
  """Converts a SavedModel to TFLite FlatBuffer.

  Args:
    saved_model_dir: SavedModel directory to convert.
    output_file: File path to write result TFLite FlatBuffer.
    input_arrays: List of input tensors to freeze graph with. Uses input arrays
      from SignatureDef when none are provided. (default None)
    input_shapes: Map of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
      (default None)
    output_arrays: List of output tensors to freeze graph with. Uses output
      arrays from SignatureDef when none are provided. (default None)
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present. (default "serve")
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    batch_size: Batch size for the model. Replaces the first dimension of an
      input size array if undefined. (default 1)
    inference_type: Currently must be `{FLOAT, QUANTIZED_UINT8}`.
    input_format: Type of data to read (currently must be TENSORFLOW_GRAPHDEF).
    output_format: Type of data to write (currently must be TFLITE or
      GRAPHVIZ_DOT)
    quantized_input_stats: For each member of input_tensors the mean and
      std deviation of training data. Only needed if `inference_type` is
      `QUANTIZED_UINT8`.
    drop_control_dependency: Drops control dependencies silently. This is due
      to tf lite not supporting control dependencies.

  Returns:
    The converted data. For example if tflite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    ValueError: Unable to convert to frozen graph.
  """
  frozen_graph_def, in_tensors, out_tensors = _freeze_saved_model(
      saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set,
      signature_key, batch_size)

  result = convert.toco_convert(
      input_data=frozen_graph_def,
      input_tensors=in_tensors,
      output_tensors=out_tensors,
      inference_type=inference_type,
      input_format=input_format,
      output_format=output_format,
      quantized_input_stats=quantized_input_stats,
      drop_control_dependency=drop_control_dependency)

  if output_file is not None:
    with gfile.Open(output_file, "wb") as f:
      f.write(result)
    logging.info("Successfully converted to: %s", output_file)

  return result
