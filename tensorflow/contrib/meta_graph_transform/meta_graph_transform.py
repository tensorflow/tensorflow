# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Apply graph_transforms tool to MetaGraphDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re as _re

from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.python.client import session as _session
from tensorflow.python.framework import graph_util as _graph_util
from tensorflow.python.framework import importer as _importer
from tensorflow.python.framework import ops as _ops
from tensorflow.python.saved_model import constants as _saved_model_constants
from tensorflow.python.training import saver as _saver_lib
from tensorflow.python.util import compat
from tensorflow.tools import graph_transforms as _graph_transforms


_FREEZE_GRAPH_TRANSFORM = 'freeze_graph'
_SPARSIFY_GATHER_TRANSFORM = 'sparsify_gather'


def _op_name(tensor_name):
  """Get the op name from a tensor name."""
  # control dependency inputs start with ^
  if tensor_name[0] == '^':
    tensor_name = tensor_name[1:]
  if ':' in tensor_name:
    op_name, _ = tensor_name.split(':')
    return op_name
  return tensor_name


def _get_shared_init_op(initializer_names):
  """Obtain the shared init op name, if it exists.

  Args:
   initializer_names: Dictionary of the "infrastructural" nodes (initializers,
     save and restore ops, etc.). The keys in this dictionary
     indicate the collection where these nodes were obtained from.

  Returns:
    A string indicating the shared init op name or none if None if none exists.
  """
  return_value = initializer_names.get(_saved_model_constants.MAIN_OP_KEY, None)
  if not return_value:
    return_value = initializer_names.get(
        _saved_model_constants.LEGACY_INIT_OP_KEY, None)
  return str(return_value[0]) if return_value else None


def _gtt_transforms(graph_def, input_names, output_names, initializer_names,
                    transforms):
  """Pass through gtt transforms, applying them to the graph_def.

  Args:
    graph_def: A GraphDef proto to be transformed.
    input_names: Names of input nodes.
    output_names: Names of output nodes.
    initializer_names: Dictionary of the "infrastructural" nodes (initializers,
      save and restore ops, etc.) that should be retained even if they are not
      transitively reachable from output nodes. The keys in this dictionary
      indicate the collection where these nodes were obtained from.
    transforms: A list of strings naming the graph transforms to be applied in
      order.
  Returns:
    The transformed GraphDef.
  """
  if not transforms:
    transformed_graph_def = _graph_pb2.GraphDef()
    transformed_graph_def.CopyFrom(graph_def)
    return transformed_graph_def

  initializer_names_flat = sorted(
      [k for l in initializer_names.values() for k in l])
  all_output_names = output_names + initializer_names_flat
  return _graph_transforms.TransformGraph(graph_def, input_names,
                                          all_output_names, transforms)


def _freeze_transform(graph_def, output_names, initializer_names, saver_def,
                      checkpoint_path):
  """Handle the freeze transform.

  Determine which initializer nodes should be retained by the freeze transform.
  Retain those nodes and return an updated dictionary containing them.

  Args:
    graph_def: A GraphDef proto to be transformed.
    output_names: Names of output nodes.
    initializer_names: Dictionary of the "infrastructural" nodes (initializers,
      save and restore ops, etc.). The keys in this dictionary
      indicate the collection where these nodes were obtained from.
    saver_def: A SaverDef proto used for restoring a checkpoint during freezing,
      if needed (default None).
    checkpoint_path:  A path to a checkpoint to restore during freezing,
      if needed (default None).

  Returns:
    A tuple containing the GraphDef and a Dict of pruned initializer nodes.
  """
  table_initializers = initializer_names.get(_ops.GraphKeys.TABLE_INITIALIZERS,
                                             [])
  shared_init_op = _get_shared_init_op(initializer_names)

  graph_def = _freeze_graph_with_def_protos(graph_def, output_names,
                                            table_initializers, shared_init_op,
                                            saver_def, checkpoint_path)
  pruned_initializer_names = {}
  # Freeze graph prunes all initializers and shared init nodes that are not
  # explicitly maintained. Create new initializer_names dictionary to reflect
  # this.
  if table_initializers:
    pruned_initializer_names[_ops.GraphKeys.TABLE_INITIALIZERS] = (
        table_initializers)
    if _saved_model_constants.LEGACY_INIT_OP_KEY in initializer_names:
      pruned_initializer_names[_saved_model_constants.LEGACY_INIT_OP_KEY] = (
          initializer_names[_saved_model_constants.LEGACY_INIT_OP_KEY])
    if _saved_model_constants.MAIN_OP_KEY in initializer_names:
      pruned_initializer_names[_saved_model_constants.MAIN_OP_KEY] = (
          initializer_names[_saved_model_constants.MAIN_OP_KEY])
  return (graph_def, pruned_initializer_names)


def _clean_save_and_restore(graph_def, op, removed_op_names):
  """Clean the specified save and restore op.

  Updates the dtypes attribute of the save / restore op and the associated name
  and shape tensors to remove entries for variables that have been removed.

  Args:
    graph_def: A GraphDef proto to be transformed.
    op: The save or restore op to update.
    removed_op_names: List of op names that have been removed.
  """
  name = op.name + '/tensor_names'
  shape = op.name + '/shape_and_slices'
  name_op = _find_op(graph_def, name)
  shape_op = _find_op(graph_def, shape)
  name_op_value_tensor = name_op.attr['value'].tensor
  shape_op_value_tensor = shape_op.attr['value'].tensor
  names = []
  shapes = []
  dtypes = []
  for index, value in enumerate(name_op_value_tensor.string_val):
    if not _is_removed(compat.as_str(value), removed_op_names):
      names.append(value)
      shapes.append(shape_op_value_tensor.string_val[index])
      dtypes.append(op.attr['dtypes'].list.type[index])
  name_op_value_tensor.string_val[:] = names
  name_op_value_tensor.tensor_shape.dim[0].size = len(names)
  shape_op_value_tensor.string_val[:] = shapes
  shape_op_value_tensor.tensor_shape.dim[0].size = len(shapes)
  op.attr['dtypes'].list.type[:] = dtypes

  name_op.attr['_output_shapes'].list.shape[0].dim[0].size = len(names)
  shape_op.attr['_output_shapes'].list.shape[0].dim[0].size = len(shapes)


def _sparsify_gather_transform(graph_def, input_names, output_names,
                               initializer_names, checkpoint_path):
  """Handle the sparsify gather transform.

  Provides the transform the checkpoint and keeps track of the newly created
  initializer nodes.

  Args:
    graph_def: A GraphDef proto to be transformed.
    input_names: Names of input nodes.
    output_names: Names of output nodes.
    initializer_names: Dictionary of the "infrastructural" nodes (initializers,
      save and restore ops, etc.). The keys in this dictionary
      indicate the collection where these nodes were obtained from.
    checkpoint_path:  A path to a checkpoint.

  Returns:
    A tuple containing the GraphDef and a Dict of updated initializer nodes.
  Raises:
    ValueError: if the restore_op_name does not have the expected format.
  """
  # Ensure that sparsify_shared_init_op is unique.
  sparsify_shared_init_op = 'sparify_gather_init_op'
  while _find_op(graph_def, sparsify_shared_init_op):
    sparsify_shared_init_op += '_1'

  input_flag = ''
  if checkpoint_path:
    input_flag = 'input_checkpoint="%s", ' % checkpoint_path

  sparsify_cmd = [
      'sparsify_gather(%sgroup_init_node="%s")' % (input_flag,
                                                   sparsify_shared_init_op)
  ]

  starting_op_names = [node.name for node in graph_def.node]

  graph_def = _gtt_transforms(graph_def, input_names, output_names,
                              initializer_names, sparsify_cmd)
  ending_op_names = [node.name for node in graph_def.node]
  removed_op_names = list(set(starting_op_names) - set(ending_op_names))
  removed_op_names.sort()

  for op_index, op_name in enumerate(removed_op_names):
    op_name_parts = op_name.rsplit('/', 1)
    # Remove part to get the checkpoint names used by the saver.
    if len(op_name_parts) == 2 and op_name_parts[1].startswith('part_'):
      removed_op_names[op_index] = op_name_parts[0]
    else:
      removed_op_names[op_index] = op_name

  # Obtain newly created table inits from gtt sparsify transform.
  added_table_inits = []
  for index, node in enumerate(graph_def.node):
    if node.name == sparsify_shared_init_op:
      added_table_inits = [n.lstrip('^') for n in node.input]

      table_initializers = initializer_names.get(
          _ops.GraphKeys.TABLE_INITIALIZERS, [])
      table_initializers.extend(added_table_inits)
      initializer_names[_ops.GraphKeys.TABLE_INITIALIZERS] = table_initializers

      del graph_def.node[index]
      break

  # Add inits to existing shared init op.
  node = _find_op(graph_def, _get_shared_init_op(initializer_names))
  for init in added_table_inits:
    node.input.append('^' + init)

  # Update saver.
  for node in graph_def.node:
    if node.name.endswith('SaveV2'):
      _clean_save_and_restore(graph_def, node, removed_op_names)

  return (graph_def, initializer_names)


def _do_transforms(graph_def,
                   input_names,
                   output_names,
                   initializer_names,
                   transforms,
                   saver_def=None,
                   checkpoint_path=None):
  """Apply requested transforms to a GraphDef, including freezing.

  Args:
    graph_def: A GraphDef proto to be transformed.
    input_names: Names of input nodes.
    output_names: Names of output nodes.
    initializer_names: Dictionary of the "infrastructural" nodes (initializers,
      save and restore ops, etc.) that should be retained even if they are not
      transitively reachable from output nodes. The keys in this dictionary
      indicate the collection where these nodes were obtained from.
    transforms: A list of strings naming the graph transforms to be applied in
      order.  These transform names are exactly those supported by the Graph
      Transform Tool, with the addition of the 'freeze_graph' and
      'sparsify_gather' transforms.
    saver_def: A SaverDef proto used for restoring a checkpoint during freezing,
      if needed (default None).
    checkpoint_path:  A path to a checkpoint to restore during freezing,
      if needed (default None).
  Returns:
     A tuple containing the GraphDef and a Dict of updated initializer nodes.
  """
  transformed_graph_def = _graph_pb2.GraphDef()
  transformed_graph_def.CopyFrom(graph_def)
  transformed_initializer_names = initializer_names.copy()

  if not transforms:
    return transformed_graph_def, transformed_initializer_names

  current_gtt_transforms = []
  for t in transforms:
    if t == _FREEZE_GRAPH_TRANSFORM:
      transformed_graph_def = _gtt_transforms(
          transformed_graph_def, input_names, output_names,
          transformed_initializer_names, current_gtt_transforms)
      output_node_names = [_op_name(x) for x in output_names]
      transformed_graph_def, transformed_initializer_names = _freeze_transform(
          transformed_graph_def, output_node_names,
          transformed_initializer_names, saver_def, checkpoint_path)
      current_gtt_transforms = []
    elif t == _SPARSIFY_GATHER_TRANSFORM:
      transformed_graph_def = _gtt_transforms(
          transformed_graph_def, input_names, output_names,
          transformed_initializer_names, current_gtt_transforms)
      transformed_graph_def, transformed_initializer_names = (
          _sparsify_gather_transform(
              transformed_graph_def, input_names, output_names,
              transformed_initializer_names, checkpoint_path))
      current_gtt_transforms = []
    else:
      current_gtt_transforms.append(t)

  transformed_graph_def = _gtt_transforms(
      transformed_graph_def, input_names, output_names,
      transformed_initializer_names, current_gtt_transforms)
  return transformed_graph_def, transformed_initializer_names


def _connect_to_shared_init_op(graph_def, shared_init_op_name,
                               nodes_to_connect):
  """Creates a new shared init node that is connected to via control deps.

  Args:
    graph_def: The GraphDef proto to add the shared init node to.
    shared_init_op_name: A string specifying the name of the shared init node to
      create.
    nodes_to_connect: A list of strings specifying the names of nodes to connect
      to the shared node via control dependencies.
  """
  if nodes_to_connect:
    init_op = graph_def.node.add()
    init_op.name = shared_init_op_name
    init_op.op = 'NoOp'
    init_op.input.extend(['^' + i for i in nodes_to_connect])


# forked and modified from freeze_graph.py
def _freeze_graph_with_def_protos(input_graph_def, output_node_names,
                                  initializer_names, shared_init_op_name,
                                  input_saver_def, input_checkpoint):
  """Converts all variables in a graph and checkpoint into constants.

  During this process, we need to retain certain initialzer nodes (e.g. table
  initializer nodes). Instead of determining which dependencies
  of the shared initializer node (e.g. group_deps) to keep, we
  reconstruct the connections between the individual initializer nodes and
  the shared node after freezing the graph.

  Args:
    input_graph_def: A GraphDef proto to be frozen.
    output_node_names: Names of output nodes.
    initializer_names: Names of initializer nodes to keep.
    shared_init_op_name: The name of the shared initializer node to connect the
      nodes in initializer names to.
    input_saver_def: A SaverDef proto used for restoring a checkpoint.
    input_checkpoint: A path to a checkpoint to restore.

  Returns:
    A frozen GraphDef.
  """

  with _ops.Graph().as_default():
    _ = _importer.import_graph_def(input_graph_def, name='')

    with _session.Session() as sess:
      saver = _saver_lib.Saver(saver_def=input_saver_def)
      saver.restore(sess, input_checkpoint)
      output_graph_def = _graph_util.convert_variables_to_constants(
          sess, input_graph_def, output_node_names + initializer_names)
      _connect_to_shared_init_op(output_graph_def, shared_init_op_name,
                                 initializer_names)
  return output_graph_def


def _find_all_mandatory_retain_ops(base_meta_graph_def):
  """Identify all infrastructural Ops, to ensure that they are retained.

  We need to retain infrastructural Ops (init and saver stuff), in addition
  to the desired outputs.

  For now we retain *all* save and restore ops, variable initializers,
  table initializers, and main init ops.
  This means that strip_unused_nodes will not remove unused variables.

  Args:
    base_meta_graph_def: a GraphDef proto in which to identify nodes to retain.

  Returns:
    A dictionary corresponding to the nodes associated with each collection
    that are to be retained.
  """
  # TODO(b/63447631): implement variable stripping.

  initializer_names = {}

  # Primary SaverDef and SAVERS collection
  saver_defs = []
  if base_meta_graph_def.HasField('saver_def'):
    saver_defs.append(base_meta_graph_def.saver_def)
  saver_defs.extend(_get_all_protos_from_collection(
      base_meta_graph_def, _ops.GraphKeys.SAVERS))
  for saver_def in saver_defs:
    savers = initializer_names.get(_ops.GraphKeys.SAVERS, [])
    savers.extend([
        saver_def.filename_tensor_name, saver_def.save_tensor_name,
        saver_def.restore_op_name
    ])
    initializer_names[_ops.GraphKeys.SAVERS] = savers

  # Variable initializers
  variable_collections = [
      _ops.GraphKeys.GLOBAL_VARIABLES,
      _ops.GraphKeys.TRAINABLE_VARIABLES,
      _ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
      _ops.GraphKeys.LOCAL_VARIABLES,
      _ops.GraphKeys.MODEL_VARIABLES]
  for var_coll in variable_collections:
    variables = _get_all_protos_from_collection(base_meta_graph_def, var_coll)
    var_init_names = [v.initializer_name for v in variables]
    if var_init_names:
      # Sanity check to ensure we don't overwrite dictionary entries.
      assert var_coll not in initializer_names
      initializer_names[var_coll] = var_init_names

  # Table initializers
  op_names = _get_all_node_names_from_collection(
      base_meta_graph_def, _ops.GraphKeys.TABLE_INITIALIZERS)
  if op_names:
    # Sanity check to ensure we don't overwrite dictionary entries.
    assert _ops.GraphKeys.TABLE_INITIALIZERS not in initializer_names
    table_initializers = [t for t in op_names]
    initializer_names[_ops.GraphKeys.TABLE_INITIALIZERS] = table_initializers

  # Various init ops
  various_init_op_collections = [_saved_model_constants.LEGACY_INIT_OP_KEY,
                                 _saved_model_constants.MAIN_OP_KEY,
                                 _ops.GraphKeys.INIT_OP,
                                 _ops.GraphKeys.LOCAL_INIT_OP,
                                 _ops.GraphKeys.READY_OP,
                                 _ops.GraphKeys.READY_FOR_LOCAL_INIT_OP]
  for op_coll in various_init_op_collections:
    op_name = _get_single_node_name_from_collection(
        base_meta_graph_def, op_coll)
    if op_name:
      # Sanity check to ensure we don't overwrite dictionary entries.
      assert op_coll not in initializer_names
      initializer_names[op_coll] = [op_name]
  return initializer_names


def _add_pruned_collection(base_meta_graph_def, meta_graph_def,
                           collection_name, removed_op_names):
  """Copy collection to the transformed MetaGraphDef, omitting removed items."""

  base_collection = base_meta_graph_def.collection_def[collection_name]
  collection = meta_graph_def.collection_def[collection_name]

  if base_collection.HasField('any_list'):
    for any_value in base_collection.any_list.value:
      # just search the serialized proto as a string
      if not _is_removed_mentioned(any_value.value, removed_op_names):
        copied_any = collection.any_list.value.add()
        copied_any.CopyFrom(any_value)
  elif base_collection.HasField('bytes_list'):
    collection.bytes_list.value[:] = [
        s for s in base_collection.bytes_list.value
        if not _is_removed_mentioned(s, removed_op_names)]
  elif base_collection.HasField('node_list'):
    collection.node_list.value[:] = [
        s for s in base_collection.node_list.value
        if not _is_removed(s, removed_op_names)]
  else:
    collection.CopyFrom(base_collection)


def _add_pruned_saver(base_meta_graph_def, meta_graph_def, removed_op_names):
  """Copy the Saver into the transformed MetaGraphDef, if valid.

  Currently this copies the Saver as is, after verifying that none of the
  referenced Save & Restore ops were removed.  A future version will modify
  the Save and Restore ops themselves as needed to account for removed
  Variables.

  Args:
    base_meta_graph_def: The untransformed MetaGraphDef.
    meta_graph_def: The transformed MetaGraphDef being built.
    removed_op_names: An iterable of names of ops that were removed.
  """

  # Note this does surgery on meta_graph_def.graph_def too, so that should have
  # been copied already.
  if base_meta_graph_def.HasField('saver_def'):
    filename_tensor_name = base_meta_graph_def.saver_def.filename_tensor_name
    save_tensor_name = base_meta_graph_def.saver_def.save_tensor_name
    restore_op_name = base_meta_graph_def.saver_def.restore_op_name

    _check_tensor_not_removed(filename_tensor_name, removed_op_names)
    _check_tensor_not_removed(save_tensor_name, removed_op_names)
    _check_tensor_not_removed(restore_op_name, removed_op_names)

    # TODO(b/63447631): Once we strip unused variables, remove references to
    # them from save and restore ops.  Retain those ops only if they also refer
    # to retained Variables. See if we can use _clean_save_and_restore() for
    # this.

    # saver_name, restore_all = restore_op_name.rsplit('/', 1)
    # if restore_all != 'restore_all':
    #   raise ValueError(
    #       'SaverDef restore_op_name did not have expected form */restore_all')

    # save_tensor_names_op_name = '{}/SaveV2/tensor_names'.format(saver_name)
    # restore_tensor_names_op_name = (
    #     '{}/RestoreV2/tensor_names'.format(saver_name))

    # save_tensor_names_op = _find_op(meta_graph_def.graph_def,
    #                                 save_tensor_names_op_name)
    # save_tensor_names_value_tensor = save_tensor_names_op.attr['value'].tensor
    # save_tensor_names_value_tensor.string_val[:] = [
    #     s for s in save_tensor_names_value_tensor.string_val
    #     if not _is_removed(s, removed_op_names)]

    # restore_tensor_names_op = _find_op(
    #     meta_graph_def.graph_def, restore_tensor_names_op_name)
    # restore_tensor_names_value_tensor = (
    #     restore_tensor_names_op.attr['value'].tensor)
    # restore_tensor_names_value_tensor.string_val[:] = [
    #     s for s in restore_tensor_names_value_tensor.string_val
    #     if not _is_removed(s, removed_op_names)]

    # if (save_tensor_names_value_tensor.string_val
    #     or restore_tensor_names_value_tensor.string_val):
    meta_graph_def.saver_def.CopyFrom(base_meta_graph_def.saver_def)


def _find_op(graph_def, op_name):
  """Fetch a node from a GraphDef proto by name."""
  for node_def in graph_def.node:
    if node_def.name == op_name:
      return node_def
  return None


def _add_pruned_signature(base_meta_graph_def, meta_graph_def,
                          signature_name, removed_op_names):
  """Copy the named signature into the transformed MetaGraphDef, if valid.

  If any input or output mentioned in the signature was removed by the graph
  transform, the signature is silently omitted from the transformed
  MetaGraphDef.

  Args:
    base_meta_graph_def: The untransformed MetaGraphDef.
    meta_graph_def: The transformed MetaGraphDef being built.
    signature_name: The name of the signature to copy.
    removed_op_names: An iterable of names of ops that were removed.
  """
  try:
    base_signature = base_meta_graph_def.signature_def[signature_name]
    for key in base_signature.inputs:
      _check_tensor_not_removed(base_signature.inputs[key].name,
                                removed_op_names)
    for key in base_signature.outputs:
      _check_tensor_not_removed(base_signature.outputs[key].name,
                                removed_op_names)
    meta_graph_def.signature_def[signature_name].CopyFrom(base_signature)
  except ValueError:
    # exclude any signature that mentions a removed node
    pass


def _get_single_node_name_from_collection(meta_graph_def, collection_key):
  """Obtain a node name that is the single element of a collection."""
  if collection_key not in meta_graph_def.collection_def:
    return None
  collection = meta_graph_def.collection_def[collection_key]
  if not collection.node_list.value:
    raise ValueError(
        'Collection {} is present but type is not node_list.'.format(
            collection_key))
  if len(collection.node_list.value) != 1:
    raise ValueError(
        'Collection {} is has {} elements; expected exactly one.'.format(
            collection_key, collection.bytes_list))
  return collection.node_list.value[0]


def _get_all_node_names_from_collection(meta_graph_def, collection_key):
  """Obtain node names from a collection."""
  if collection_key not in meta_graph_def.collection_def:
    return None
  collection = meta_graph_def.collection_def[collection_key]
  if not collection.node_list.value:
    raise ValueError(
        'Collection {} is present but type is not node_list.'.format(
            collection_key))
  return collection.node_list.value


def _get_all_protos_from_collection(meta_graph_def, collection_key):
  """Obtain node names from a collection."""
  if collection_key not in meta_graph_def.collection_def:
    return []
  collection = meta_graph_def.collection_def[collection_key]
  if not collection.bytes_list.value:
    raise ValueError(
        'Collection {} is present but type is not bytes_list.'.format(
            collection_key))
  proto_type = _ops.get_collection_proto_type(collection_key)
  result = []
  for value in collection.bytes_list.value:
    proto = proto_type()
    proto.ParseFromString(value)
    result.append(proto)
  return result


def _is_removed(tensor_name, removed_op_names):
  """Determine whether the named tensor is an output of a removed op."""
  for removed_op_name in removed_op_names:
    if tensor_name.split(':')[0] == removed_op_name:
      return True
  return False


def _is_removed_mentioned(s, removed_op_names):
  """Determine whether any removed op is mentioned in the given object.

  This relies on the string representation of the object.  This is used for
  proto messages that may mention ops by name in nested fields.  The string
  representation of the proto includes those field values, so this string
  search approach is sufficient.

  Args:
    s: an object to search for removed op names.
    removed_op_names: An iterable of names of ops that were removed.

  Returns:
    True if any removed op is mentioned in the given object, False otherwise.
  """
  # A common approach taken by some of the transforms in gtt is to add new nodes
  # that have the same prefix as the node they are removing. For example, if
  # the original node name was /foo, they may remove that node and add in
  # /foo/bar. This regex ensures that we handle these two nodes
  # as separate entities.  It matches on nodes having names in the form of
  # '/foo/bar_x' as well as nodes having names in the form of 'foo.'
  s_names = _re.findall(r'((?:[\/]?[a-zA-Z0-9\_]*)*)', compat.as_str_any(s))
  for removed_op_name in removed_op_names:
    for s_name in s_names:
      if s_name.endswith(removed_op_name):
        return True
  return False


def _check_tensor_not_removed(tensor_name, removed_op_names):
  """Verify that the named tensor was not removed.

  Args:
    tensor_name: the name of a tensor to check.
    removed_op_names: An iterable of names of ops that were removed.

  Raises:
    ValueError: if the tensor was removed.
  """
  if not tensor_name:
    raise ValueError('Tensor name should not be empty')
  if _is_removed(tensor_name, removed_op_names):
    raise ValueError(
        'Expected Tensor, but it was removed: {}'.format(tensor_name))


def _add_new_inits_to_collection(meta_graph_def, updated_initializer_names):
  """Add new inits to collection.

  Args:
    meta_graph_def: The MetaGraphDef protocol buffer to update.
    updated_initializer_names: Dictionary of the updated "infrastructural" nodes
      (initializers, save and restore ops, etc.). The keys in this dictionary
      indicate the collection where these nodes were obtained from.

  Raises:
    ValueError: if the tensor was removed.
  """
  # TODO(dzats): Extend this to support all collections.
  if _ops.GraphKeys.TABLE_INITIALIZERS in updated_initializer_names:
    orig_table_inits = _get_all_node_names_from_collection(
        meta_graph_def, _ops.GraphKeys.TABLE_INITIALIZERS)
    orig_table_inits = orig_table_inits if orig_table_inits else []
    updated_table_inits = updated_initializer_names[
        _ops.GraphKeys.TABLE_INITIALIZERS]
    new_table_inits = list(set(updated_table_inits) - set(orig_table_inits))
    new_table_inits.sort()
    meta_graph_def.collection_def[
        _ops.GraphKeys.TABLE_INITIALIZERS].node_list.value.extend(
            new_table_inits)


def meta_graph_transform(
    base_meta_graph_def, input_names, output_names, transforms, tags,
    checkpoint_path=None):
  """Apply the Graph Transform tool to a MetaGraphDef.

  Args:
    base_meta_graph_def: A MetaGraphDef protocol buffer to transform.
    input_names: Names of input nodes.
    output_names: Names of output nodes.
    transforms: A list of strings naming the graph transforms to be applied in
      order.  These transform names are exactly those supported by the Graph
      Transform Tool, with the addition of the 'freeze_graph' and
      'sparsify_gather' transforms.
    tags: A list of tags with which to annotate the transformed MetaGraphDef.
    checkpoint_path: A path to a checkpoint to restore during freezing,
      if needed (default None).

  Returns:
    A new transformed MetaGraphDef protocol buffer.
  """
  meta_graph_def = _meta_graph_pb2.MetaGraphDef()

  initializer_names = _find_all_mandatory_retain_ops(base_meta_graph_def)

  transformed_graph_def, updated_initializer_names = _do_transforms(
      base_meta_graph_def.graph_def, input_names, output_names,
      initializer_names, transforms, base_meta_graph_def.saver_def,
      checkpoint_path)

  meta_graph_def.graph_def.CopyFrom(transformed_graph_def)
  meta_graph_def.meta_info_def.CopyFrom(base_meta_graph_def.meta_info_def)
  meta_graph_def.meta_info_def.ClearField('tags')
  for tag in tags:
    meta_graph_def.meta_info_def.tags.append(tag)

  base_op_names = [compat.as_str(node.name)
                   for node in base_meta_graph_def.graph_def.node]
  retained_op_names = [compat.as_str(node.name)
                       for node in meta_graph_def.graph_def.node]
  removed_op_names = set(base_op_names) - set(retained_op_names)

  # Copy saver, excluding any pruned nodes if graph was not frozen.
  # TODO(b/63447631): Revisit this once the problem is addressed. Currently
  # _add_pruned_saver assumes that the save and restore nodes have not been
  # removed but freeze_graph (correctly) removes them.
  if _FREEZE_GRAPH_TRANSFORM not in transforms:
    _add_pruned_saver(base_meta_graph_def, meta_graph_def, removed_op_names)

  # Copy collections, excluding any pruned nodes
  for collection_name in base_meta_graph_def.collection_def:
    _add_pruned_collection(
        base_meta_graph_def, meta_graph_def, collection_name,
        removed_op_names)

  # Append newly added initializers to collection.
  _add_new_inits_to_collection(meta_graph_def, updated_initializer_names)

  # Copy signature_defs, excluding any pruned nodes
  for signature_name in base_meta_graph_def.signature_def:
    _add_pruned_signature(
        base_meta_graph_def, meta_graph_def, signature_name,
        removed_op_names)

  return meta_graph_def
