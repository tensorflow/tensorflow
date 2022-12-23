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
"""Defines utilities involving SavedModel."""

from typing import Collection, Dict, Mapping, Optional, Sequence

from absl import logging

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import constants as saved_model_constants
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import tag_constants

# Mapping of signature def key -> SignatureDef.
_SignatureDefMap = Mapping[str, meta_graph_pb2.SignatureDef]


def get_signatures_from_saved_model(
    saved_model_path: str,
    signature_keys: Optional[Sequence[str]] = None,
    tags: Optional[Collection[str]] = None
) -> Dict[str, meta_graph_pb2.SignatureDef]:
  """Gets a map from signature keys to their SignatureDef.

  Args:
    saved_model_path: Path to the saved model.
    signature_keys: List of keys identifying SignatureDef to retrieve. If None,
      retrieve all except the init signature.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel.

  Returns:
    A map from signature_key to its SignatureDef.
  """
  if tags is None:
    tags = {tag_constants.SERVING}

  loader = saved_model_loader.SavedModelLoader(saved_model_path)
  meta_graphdef = loader.get_meta_graph_def_from_tags(tags)
  signatures = {}
  for key, signature_def in meta_graphdef.signature_def.items():
    if key == saved_model_constants.INIT_OP_SIGNATURE_KEY:
      continue
    if signature_keys is not None and key not in signature_keys:
      continue
    signatures[key] = signature_def

  return signatures


def _restore_output_tensor_names(
    graph_def: graph_pb2.GraphDef) -> graph_pb2.GraphDef:
  """Restores the output tensor names of the converted model.

  During the conversion, the output tensor names of the original model are
  embedded in the `tf_saved_model.index_path` attribute of the RetVal nodes and
  might become the name of Retval nodes as well (with an index suffix if there
  are multiple output tensors from one node). Since Retval nodes are not used in
  SavedModel, this function removes them and restore the names to the actual
  output tensors.

  Args:
    graph_def: the converted GraphDef.

  Returns:
    The GraphDef with Retval nodes removed and output tensor names restored.
  """
  output_renaming_map = {}
  with session.Session(graph=ops.Graph()):
    importer.import_graph_def(graph_def, name='')
    graph = ops.get_default_graph()
    for op in graph.get_operations():
      if op.type == '_Retval':
        expected_node_name = op.name
        if op.get_attr('tf_saved_model.index_path') is not None:
          index_path_name = op.get_attr('tf_saved_model.index_path')[0]
          index_path_name = index_path_name.decode('utf-8').split(':')[0]
          try:
            # Only use the index_path name if it points to a Retval node.
            index_path_node = graph.get_operation_by_name(index_path_name)
            if index_path_node.type == '_Retval':
              expected_node_name = index_path_name
          except KeyError:
            pass
        retval_input_node_name = op.inputs[0].op.name
        output_renaming_map[retval_input_node_name] = expected_node_name

  for node in reversed(graph_def.node):
    if node.name in output_renaming_map:
      node.name = output_renaming_map[node.name]
    elif node.op == '_Retval':
      graph_def.node.remove(node)
    else:
      # Update the inputs referring to the pre-renaming node.
      for idx, input_name in enumerate(node.input):
        if input_name in output_renaming_map:
          node.input[idx] = output_renaming_map[input_name]
      # Update the control inputs referring to the pre-renaming node.
      updating_inputs = []
      for input_name in reversed(node.input):
        if input_name.startswith('^') and input_name[1:] in output_renaming_map:
          updating_inputs.append(input_name[1:])
          node.input.remove(input_name)
      for updating_input in updating_inputs:
        node.input.append('^' + output_renaming_map[updating_input])
  return graph_def


def _create_empty_output_dir(output_directory: str) -> None:
  """Creates the `output_directory`.

  If `output_directory` already exists, it recursively deletes all contents
  inside the directory.

  Also creates the parent & intermediate directories.

  Args:
    output_directory: Output directory.
  """
  if file_io.file_exists_v2(output_directory):
    logging.info('Deleting existing directory for quantized model output: %s .',
                 output_directory)
    file_io.delete_recursively_v2(output_directory)

  file_io.recursive_create_dir_v2(output_directory)


def _validate_signatures(signature_def_map: _SignatureDefMap,
                         exported_graph: ops.Graph) -> _SignatureDefMap:
  """Validates if the tensor names in signatures are consistent with the graph.

  This function checks if the input and output tensor names in the signatures
  exist if the graph. The output tensor names might change during conversion,
  we try to fix that with `_restore_output_tensor_names`. Besides, if there
  are duplicated tensor names, they we will be prefixed with the signature name.
  However, if that doesn't work the signatures can't be used with the converted
  graph.

  Args:
    signature_def_map: the signatures to validate.
    exported_graph: The PTQ-exported GraphDef.

  Returns:
    The signatures with tensor names prefixed with signature name if necessary.

  Raises:
    ValueError: Iff the signatures are not consistent with the graph.
  """
  for signature_key, signature_def in signature_def_map.items():
    for tensor_info in signature_def.inputs.values():
      try:
        exported_graph.get_tensor_by_name(tensor_info.name)
      except KeyError as exc:
        try:
          prefixed_name = signature_key + '_' + tensor_info.name
          exported_graph.get_tensor_by_name(prefixed_name)
          tensor_info.name = prefixed_name
        except KeyError:
          raise ValueError(
              'Cannot find the input tensor with name %s in the graph.' %
              tensor_info.name) from exc

    for tensor_info in signature_def.outputs.values():
      try:
        exported_graph.get_tensor_by_name(tensor_info.name)
      except KeyError as exc:
        try:
          prefixed_name = signature_key + '_' + tensor_info.name
          exported_graph.get_tensor_by_name(prefixed_name)
          tensor_info.name = prefixed_name
        except KeyError:
          raise ValueError(
              'Cannot find the output tensor with name %s in the graph.' %
              tensor_info.name) from exc

  return signature_def_map


def _find_op(graph: ops.Graph,
             op_name: Optional[str]) -> Optional[ops.Operation]:
  """Finds the operation with `op_name`.

  Args:
    graph: The graph to find from.
    op_name: Name of the node.

  Returns:
    The operation that corresponds to `op_name`. Returns None iff op_name is an
    empty string or None.

  Raises:
    ValueError: `op_name` is malformed.
  """
  if not op_name:
    return None

  init_op = graph.get_operation_by_name(op_name)
  logging.debug('Op found in the graph: %s', op_name)

  return init_op


def save_model_v1(graph_def: graph_pb2.GraphDef,
                  output_dir: str,
                  signature_def_map: _SignatureDefMap,
                  tags: Collection[str],
                  init_op_name: Optional[str] = None) -> None:
  """Saves the model.

  Saves the provided graph def as SavedModel.
  Uses TF1 SavedModel semantics (i.e. no object graph).

  Args:
    graph_def: Graph to save.
    output_dir: Output directory for the SavedModel.
    signature_def_map: Mapping of signature def key -> SignatureDef.
    tags: Tags for the meta graph def.
    init_op_name: Name of the node for initialization.

  Raises:
    ValueError iff the graph does not contain a valid signature.
  """
  _create_empty_output_dir(output_dir)
  v1_builder = builder.SavedModelBuilder(output_dir)

  graph_def = _restore_output_tensor_names(graph_def)
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def, name='')

    signature_def_map = _validate_signatures(signature_def_map,
                                             ops.get_default_graph())
    v1_builder.add_meta_graph_and_variables(
        sess,
        tags,
        signature_def_map=signature_def_map,
        main_op=_find_op(sess.graph, op_name=init_op_name))

  v1_builder.save()
