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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2 as _rewriter_config_pb2
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util as tf_graph_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph

# Map of tf.dtypes to TFLite types_flag_pb2.
_MAP_TF_TO_TFLITE_TYPES = {
    dtypes.float32: _types_pb2.FLOAT,
    dtypes.int32: _types_pb2.INT32,
    dtypes.int64: _types_pb2.INT64,
    dtypes.string: _types_pb2.STRING,
    dtypes.uint8: _types_pb2.QUANTIZED_UINT8,
    dtypes.int8: _types_pb2.INT8,
    dtypes.complex64: _types_pb2.COMPLEX64
}


def convert_dtype_to_tflite_type(tf_dtype):
  """Converts tf.dtype to TFLite proto type.

  Args:
    tf_dtype: tf.dtype

  Raises:
    ValueError: Unsupported tf.dtype.

  Returns:
    types_flag_pb2.
  """
  result = _MAP_TF_TO_TFLITE_TYPES.get(tf_dtype)
  if result is None:
    raise ValueError("Unsupported tf.dtype {0}".format(tf_dtype))
  return result


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


def get_grappler_config(enable_layout_optimizer=False, function_only=False):
  """Creates a tf.ConfigProto for configuring Grappler.

  Args:
    enable_layout_optimizer: Bool indicating whether to run the layout
      optimizer. This turns NHCW to NCHW. This provides performance
      optimizations when Flex mode is enabled. (default False)
    function_only: Bool indiciating whether to only run the function optimizer.
      This inlines functions and is required for freezing models with functions.
      (default False)

  Returns:
    tf.ConfigProto.
  """
  config = _config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  if function_only:
    rewrite_options.optimizers.append("function")
  else:
    if enable_layout_optimizer:
      rewrite_options.layout_optimizer = _rewriter_config_pb2.RewriterConfig.ON
    else:
      rewrite_options.layout_optimizer = _rewriter_config_pb2.RewriterConfig.OFF

    # Avoid remapping as it creates ops like _FusedConv2D, which are not
    # supported by TFLite.
    rewrite_options.remapping = _rewriter_config_pb2.RewriterConfig.OFF
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

  # We need to add a collection called 'train_op' so that grappler
  # knows what the outputs are.
  fetch_collection = _meta_graph_pb2.CollectionDef()
  for array in input_arrays + output_arrays:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  return tf_optimizer.OptimizeGraph(config, meta_graph)


def _convert_op_hints_if_present(sess, output_tensors):
  if is_frozen_graph(sess):
    raise ValueError("Try to convert op hints, needs unfrozen graph.")
  hinted_outputs_nodes = find_all_hinted_output_nodes(sess)
  output_arrays = [get_tensor_name(tensor) for tensor in output_tensors]
  graph_def = tf_graph_util.convert_variables_to_constants(
      sess, sess.graph_def, output_arrays + hinted_outputs_nodes)
  graph_def = convert_op_hints_to_stubs(graph_def=graph_def)
  graph_def = tf_graph_util.remove_training_nodes(graph_def)
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
  # Grappler inline function optimization will break OpHints graph
  # transformation, so if OpHints are present, just convert it.
  hinted_outputs_nodes = find_all_hinted_output_nodes(sess)
  if len(hinted_outputs_nodes) > 0:  #  pylint: disable=g-explicit-length-test
    return _convert_op_hints_if_present(sess, output_tensors)

  # Runs a Grappler pass in order to inline any functions in the graph.
  config = get_grappler_config(function_only=True)
  graph_def = run_graph_optimizations(
      sess.graph_def, input_tensors, output_tensors, config, graph=sess.graph)

  if not is_frozen_graph(sess):
    output_arrays = [get_tensor_name(tensor) for tensor in output_tensors]
    return tf_graph_util.convert_variables_to_constants(sess, graph_def,
                                                        output_arrays)
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
