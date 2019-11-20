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

from tensorflow.lite.python import util
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader


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


def get_meta_graph_def(saved_model_dir, tag_set):
  """Validate saved_model and extract MetaGraphDef.

  Args:
    saved_model_dir: saved_model path to convert.
    tag_set: Set of tag(s) of the MetaGraphDef to load.

  Returns:
    The meta_graph_def used for tflite conversion.

  Raises:
    ValueError: No valid MetaGraphDef for given tag_set.
  """
  with session.Session(graph=ops.Graph()) as sess:
    return loader.load(sess, tag_set, saved_model_dir)


def get_signature_def(meta_graph, signature_key):
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
      "The given SavedModel MetaGraphDef contains SignatureDefs with the "
      "following keys: %s", signature_def_keys)
  if signature_key not in signature_def_keys:
    raise ValueError("No '{}' in the SavedModel\'s SignatureDefs. Possible "
                     "values are '{}'.".format(signature_key,
                                               ",".join(signature_def_keys)))
  return signature_def_map[signature_key]


def get_inputs_outputs(signature_def):
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
    # Sort the tensor names.
    user_tensor_names = sorted(user_tensor_names)

    tensors = util.get_tensors_from_tensor_names(graph, user_tensor_names)
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


def freeze_saved_model(saved_model_dir, input_arrays, input_shapes,
                       output_arrays, tag_set, signature_key):
  """Converts a SavedModel to a frozen graph.

  Args:
    saved_model_dir: SavedModel directory to convert.
    input_arrays: List of input tensors to freeze graph with. Uses input arrays
      from SignatureDef when none are provided.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
    output_arrays: List of output tensors to freeze graph with. Uses output
      arrays from SignatureDef when none are provided.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.

  Returns:
    frozen_graph_def: Frozen GraphDef.
    in_tensors: List of input tensors for the graph.
    out_tensors: List of output tensors for the graph.
    graph: `Graph` object.

  Raises:
    ValueError:
      SavedModel doesn't contain a MetaGraphDef identified by tag_set.
      signature_key is not in the MetaGraphDef.
      assets/ directory is in the MetaGraphDef.
      input_shapes does not match the length of input_arrays.
      input_arrays or output_arrays are not valid.
  """
  # Read SignatureDef.
  meta_graph = get_meta_graph_def(saved_model_dir, tag_set)
  signature_def = get_signature_def(meta_graph, signature_key)
  inputs, outputs = get_inputs_outputs(signature_def)

  # Check SavedModel for assets directory.
  collection_def = meta_graph.collection_def
  if constants.ASSETS_KEY in collection_def:
    raise ValueError("SavedModels with assets/ directory are not supported.")

  graph = ops.Graph()
  with session.Session(graph=graph) as sess:
    loader.load(sess, meta_graph.meta_info_def.tags, saved_model_dir)

    # Gets input and output tensors.
    # TODO(zhixianyan): Use TFLite supported Op list to filter outputs.
    in_tensors = _get_tensors(graph, inputs, input_arrays)
    out_tensors = _get_tensors(graph, outputs, output_arrays)
    util.set_tensor_shapes(in_tensors, input_shapes)

    frozen_graph_def = util.freeze_graph(sess, in_tensors, out_tensors)
    return frozen_graph_def, in_tensors, out_tensors, sess.graph
