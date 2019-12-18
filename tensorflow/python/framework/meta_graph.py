# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""MetaGraph and related functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from distutils import version as distutils_version  # pylint: disable=g-bad-import-order
import os.path
import re

import six
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


# Prefix to be added to unbound input names so they are easily identifiable.
_UNBOUND_INPUT_PREFIX = "$unbound_inputs_"

# List of collections that didn't register proto functions, as a result in
# a previously exported meta_graph the items are of a different data type.
_COMPAT_COLLECTION_LIST = [ops.GraphKeys.LOCAL_VARIABLES,
                           ops.GraphKeys.MODEL_VARIABLES,
                           ops.GraphKeys.METRIC_VARIABLES]


def _node_def(from_node_def, export_scope, unbound_inputs, clear_devices=False):
  """Create a `NodeDef` proto with export_scope stripped.

  Args:
    from_node_def: A `node_def_pb2.NodeDef` protocol buffer.
    export_scope: A `string` representing the name scope to remove.
    unbound_inputs: An array of unbound input names if they exist.
    clear_devices: Boolean which controls whether to clear device information
      from node_def. Default false.

  Returns:
    A `node_def_pb2.NodeDef` protocol buffer.
  """
  node_def = copy.deepcopy(from_node_def)
  for i, v in enumerate(node_def.input):
    if (export_scope and
        not node_def.input[i].lstrip("^").startswith(export_scope)):
      # Adds "$unbound_inputs_" prefix to the unbound name so they are easily
      # identifiable.
      node_def.input[i] = re.sub(r"([\^]|^)(.*)",
                                 r"\1" + _UNBOUND_INPUT_PREFIX + r"\2",
                                 compat.as_str(v))
      unbound_inputs.append(node_def.input[i])
    else:
      node_def.input[i] = ops.strip_name_scope(v, export_scope)
  node_def.name = compat.as_bytes(
      ops.strip_name_scope(from_node_def.name, export_scope))
  for k, v in six.iteritems(from_node_def.attr):
    if k == "_class":
      new_s = [compat.as_bytes(
          ops.strip_name_scope(s, export_scope)) for s in v.list.s
               if not export_scope or
               compat.as_str(s).split("@")[1].startswith(export_scope)]
      node_def.attr[k].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=new_s)))
    elif node_def.op in ("Enter", "RefEnter") and k == "frame_name":
      if not export_scope or compat.as_str(v.s).startswith(export_scope):
        new_s = compat.as_bytes(ops.strip_name_scope(v.s, export_scope))
      node_def.attr[k].CopyFrom(attr_value_pb2.AttrValue(s=new_s))
    else:
      node_def.attr[k].CopyFrom(v)

  if clear_devices:
    node_def.device = ""

  return node_def


def _read_file(filename):
  """Reads a file containing `GraphDef` and returns the protocol buffer.

  Args:
    filename: `graph_def` filename including the path.

  Returns:
    A `GraphDef` protocol buffer.

  Raises:
    IOError: If the file doesn't exist, or cannot be successfully parsed.
  """
  graph_def = graph_pb2.GraphDef()
  if not file_io.file_exists(filename):
    raise IOError("File %s does not exist." % filename)
  # First try to read it as a binary file.
  file_content = file_io.FileIO(filename, "rb").read()
  try:
    graph_def.ParseFromString(file_content)
    return graph_def
  except Exception:  # pylint: disable=broad-except
    pass

  # Next try to read it as a text file.
  try:
    text_format.Merge(file_content, graph_def)
  except text_format.ParseError as e:
    raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

  return graph_def


def ops_used_by_graph_def(graph_def):
  """Collect the list of ops used by a graph.

  Does not validate that the ops are all registered.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    A list of strings, each naming an op used by the graph.
  """
  # Map function names to definitions
  name_to_function = {}
  for fun in graph_def.library.function:
    name_to_function[fun.signature.name] = fun

  # Collect the list of op names.  Since functions can reference functions, we
  # need a recursive traversal.
  used_ops = set()  # Includes both primitive ops and functions
  functions_to_process = []  # A subset of used_ops

  def mark_op_as_used(op):
    if op not in used_ops and op in name_to_function:
      functions_to_process.append(name_to_function[op])
    used_ops.add(op)

  for node in graph_def.node:
    mark_op_as_used(node.op)
  while functions_to_process:
    fun = functions_to_process.pop()
    for node in fun.node_def:
      mark_op_as_used(node.op)

  return [op for op in used_ops if op not in name_to_function]


def stripped_op_list_for_graph(graph_def):
  """Collect the stripped OpDefs for ops used by a graph.

  This function computes the `stripped_op_list` field of `MetaGraphDef` and
  similar protos.  The result can be communicated from the producer to the
  consumer, which can then use the C++ function
  `RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    An `OpList` of ops used by the graph.
  """
  # This is similar to StrippedOpListForGraph in C++, but unlike its
  # C++ counterpart, this version does not require all ops to be registered.
  # This is done to support Prelu fusion in tfjs.
  used_ops = ops_used_by_graph_def(graph_def)
  op_defs = []
  for op in sorted(used_ops):
    op_def = op_def_registry.get(op)
    if op_def is not None:
      op_defs.append(op_def)
  return op_def_pb2.OpList(op=op_defs)


def _get_kind_name(item):
  """Returns the kind name in CollectionDef.

  Args:
    item: A data item.

  Returns:
    The string representation of the kind in CollectionDef.
  """
  if isinstance(item, (six.string_types, six.binary_type)):
    kind = "bytes_list"
  elif isinstance(item, six.integer_types):
    kind = "int64_list"
  elif isinstance(item, float):
    kind = "float_list"
  elif isinstance(item, Any):
    kind = "any_list"
  else:
    kind = "node_list"
  return kind


SAVE_AND_RESTORE_OPS = ["SaveV2",
                        "Save", "SaveSlice",
                        "LegacySave", "LegacySaveSlice",
                        "RestoreV2",
                        "Restore", "RestoreSlice",
                        "LegacyRestore", "LegacyRestoreSlice"]


def _op_name(tensor_name):
  """Extract the Op name from a Tensor name.

  The Op name is everything before a colon, if present,
  not including any ^ prefix denoting a control dependency.

  Args:
    tensor_name: the full name of a Tensor in the graph.
  Returns:
    The name of the Op of which the given Tensor is an output.
  Raises:
    ValueError: if tensor_name is None or empty.
  """
  if not tensor_name:
    raise ValueError("Tensor name cannot be empty or None.")

  # Control dependency inputs start with ^.
  if tensor_name.startswith("^"):
    tensor_name = tensor_name[1:]
  if ":" in tensor_name:
    op_name, _ = tensor_name.split(":")
    return op_name
  return tensor_name


def _get_scope(node_name):
  """Extract the scope name from a node name.

  The scope name is everything before the final slash,
  not including any ^ prefix denoting a control dependency.

  Args:
    node_name: the full name of an Op or a Tensor in the graph.
  Returns:
    The deepest named scope containing the node.
  Raises:
    ValueError: if tensor_name is None or empty
  """
  if not node_name:
    raise ValueError("Node name cannot be empty or None.")

  # Control dependency inputs start with ^.
  if node_name.startswith("^"):
    node_name = node_name[1:]
  if "/" in node_name:
    scope, _ = node_name.rsplit("/", 1)
    return scope

  return ""


def _find_extraneous_saver_nodes(graph_def, saver_def):
  """Identifies any nodes in the graph_def related to unused Savers.

  This approach assumes that each Saver is cleanly isolated in its own name
  scope, so we need only identify the scopes associated with extraneous Savers
  and return all the nodes in those scopes.

  Args:
    graph_def: a GraphDef proto to evaluate.
    saver_def: a SaverDef proto referencing Save/Restore ops to be retained.
  Returns:
    An iterable of node names that may be safely omitted.
  """
  # TODO(soergel): confirm that the assumption of scope isolation is valid.
  # If not, we need to walk up the graph from any restore_all nodes, and walk
  # down the graph from any Save/Restore nodes.  I drafted that approach too,
  # but it seems unnecessarily complex given the name scope solution.

  # load the graph DAG in minimal form, without initializing a full Graph object
  nodes = {
      node_def.name: (set(_op_name(x) for x in node_def.input), node_def.op)
      for node_def in graph_def.node
  }

  retain_scope_save = None
  retain_scope_restore = None
  # It's possible to have no saver if the graph has no Variables
  if saver_def is not None:
    save_op_name = _op_name(saver_def.save_tensor_name)
    restore_op_name = _op_name(saver_def.restore_op_name)

    # The save and restore scopes should always be the same, but if they differ
    # for some reason, we retain them both to be safe.
    retain_scope_restore = _get_scope(restore_op_name) + "/"
    retain_scope_save = _get_scope(save_op_name) + "/"

  all_saver_node_names = set(
      name for name, (_, op) in nodes.items() if op in SAVE_AND_RESTORE_OPS)

  all_saver_scopes = (
      set(_get_scope(x) for x in all_saver_node_names) - all_saver_node_names)
  all_saver_scopes = set(x + "/" for x in all_saver_scopes)

  extraneous_scopes = all_saver_scopes - set([retain_scope_save,
                                              retain_scope_restore])

  extraneous_node_names = set()
  for name, _ in nodes.items():
    for extraneous_scope in extraneous_scopes:
      if name.startswith(extraneous_scope):
        extraneous_node_names.add(name)
        break

  return extraneous_node_names


def _should_include_node(node_or_node_name, export_scope, exclude_nodes):
  """Returns `True` if a node should be included.

  Args:
    node_or_node_name: A node or `string` node name.
    export_scope: `string`. Name scope under which to extract the subgraph. The
      scope name will be stripped from the node definitions for easy import
      later into new name scopes.
    exclude_nodes: An iterable of nodes or `string` node names to omit from the
      export, or None.  Note no sanity-checking is done, so this list must be
      carefully constructed to avoid producing an invalid graph.

  Returns:
    `True` if the node should be included.
  """
  if not isinstance(node_or_node_name, six.string_types):
    try:
      node_name = node_or_node_name.name
    except AttributeError:
      # Keep the object that we don't know how to process.
      return True
  else:
    node_name = node_or_node_name

  if exclude_nodes and (node_or_node_name in exclude_nodes
                        or node_name in exclude_nodes):
    return False

  return (node_name.startswith(_UNBOUND_INPUT_PREFIX) or
          (not export_scope or node_name.startswith(export_scope)))


def add_collection_def(meta_graph_def, key, graph=None,
                       export_scope=None, exclude_nodes=None,
                       override_contents=None):
  """Adds a collection to MetaGraphDef protocol buffer.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    key: One of the GraphKeys or user-defined string.
    graph: The `Graph` from which to get collections.
    export_scope: Optional `string`. Name scope to remove.
    exclude_nodes: An iterable of nodes or `string` node names to omit from the
      collection, or None.
    override_contents: An iterable of values to place in the collection,
      ignoring the current values (if set).
  """
  if graph and not isinstance(graph, ops.Graph):
    raise TypeError("graph must be of type Graph, not %s", type(graph))

  if not isinstance(key, six.string_types) and not isinstance(key, bytes):
    logging.warning("Only collections with string type keys will be "
                    "serialized. This key has %s", type(key))
    return

  # Sets graph to default graph if it's not passed in.
  graph = graph or ops.get_default_graph()

  if override_contents:
    collection_list = override_contents
  else:
    collection_list = graph.get_collection(key)

  # Remove nodes that should not be exported from the collection list.
  collection_list = [x for x in collection_list if
                     _should_include_node(x, export_scope, exclude_nodes)]
  if not collection_list:
    return

  try:
    col_def = meta_graph_def.collection_def[key]
    to_proto = ops.get_to_proto_function(key)
    proto_type = ops.get_collection_proto_type(key)
    if to_proto:
      kind = "bytes_list"
      for x in collection_list:
        # Additional type check to make sure the returned proto is indeed
        # what we expect.
        proto = to_proto(x, export_scope=export_scope)
        if proto:
          assert isinstance(proto, proto_type)
          getattr(col_def, kind).value.append(proto.SerializeToString())
    else:
      kind = _get_kind_name(collection_list[0])
      if kind == "node_list":
        for x in collection_list:
          if not export_scope or x.name.startswith(export_scope):
            getattr(col_def, kind).value.append(
                ops.strip_name_scope(x.name, export_scope))
      elif kind == "bytes_list":
        # NOTE(opensource): This force conversion is to work around the fact
        # that Python3 distinguishes between bytes and strings.
        getattr(col_def, kind).value.extend(
            [compat.as_bytes(x) for x in collection_list])
      else:
        getattr(col_def, kind).value.extend([x for x in collection_list])
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("Issue encountered when serializing %s.\n"
                    "Type is unsupported, or the types of the items don't "
                    "match field type in CollectionDef. Note this is a warning "
                    "and probably safe to ignore.\n%s", key, str(e))
    if key in meta_graph_def.collection_def:
      del meta_graph_def.collection_def[key]
    return


def _is_default_attr_value(op_def, attr_name, attr_value):
  """Checks if given attribute matches the default value in the op def."""
  for attr_def in op_def.attr:
    if attr_def.name == attr_name:
      if not attr_def.HasField("default_value"):
        return False
      # pywrap_tensorflow.EqualAttrValueWrapper returns an empty string
      # if both arguments represent an equivalent AttrValue instance.
      return not pywrap_tensorflow.EqualAttrValueWrapper(
          attr_value.SerializeToString(),
          attr_def.default_value.SerializeToString())
  return False


def strip_graph_default_valued_attrs(meta_graph_def):
  """Strips default valued attributes for node defs in given MetaGraphDef.

  This method also sets `meta_info_def.stripped_default_attrs` in the given
  `MetaGraphDef` proto to True.

  Args:
    meta_graph_def: `MetaGraphDef` protocol buffer

  Returns:
    None.
  """
  # Map function op names to their function definitions.
  op_name_to_function = {}
  for function_def in meta_graph_def.graph_def.library.function:
    op_name_to_function[function_def.signature.name] = function_def

  def _strip_node_default_valued_attrs(node_def):
    """Removes default valued attributes from a single node def."""
    if node_def.op in op_name_to_function:
      return

    op_def = op_def_registry.get(node_def.op)
    if op_def is None:
      return

    attrs_to_strip = set()
    for attr_name, attr_value in node_def.attr.items():
      if _is_default_attr_value(op_def, attr_name, attr_value):
        attrs_to_strip.add(attr_name)

    for attr in attrs_to_strip:
      del node_def.attr[attr]

  # Process all NodeDef instances in graph_def.
  for node_def in meta_graph_def.graph_def.node:
    _strip_node_default_valued_attrs(node_def)

  # Process all NodeDef instances in graph_def.library.function.
  for function_def in meta_graph_def.graph_def.library.function:
    for function_node_def in function_def.node_def:
      _strip_node_default_valued_attrs(function_node_def)

  # Tell consumers of this graph that default valued attrs have been stripped.
  meta_graph_def.meta_info_def.stripped_default_attrs = True


def create_meta_graph_def(meta_info_def=None,
                          graph_def=None,
                          saver_def=None,
                          collection_list=None,
                          graph=None,
                          export_scope=None,
                          exclude_nodes=None,
                          clear_extraneous_savers=False,
                          strip_default_attrs=False):
  # pylint: disable=line-too-long
  """Construct and returns a `MetaGraphDef` protocol buffer.

  Args:
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    graph: The `Graph` to create `MetaGraphDef` out of.
    export_scope: Optional `string`. Name scope to remove.
    exclude_nodes: An iterable of nodes or `string` node names to omit from all
      collection, or None.
    clear_extraneous_savers: Remove any preexisting SaverDefs from the SAVERS
        collection.  Note this method does not alter the graph, so any
        extraneous Save/Restore ops should have been removed already, as needed.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    MetaGraphDef protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
  # pylint: enable=line-too-long
  # Type check.
  if graph and not isinstance(graph, ops.Graph):
    raise TypeError("graph must be of type Graph, not %s", type(graph))
  if meta_info_def and not isinstance(meta_info_def,
                                      meta_graph_pb2.MetaGraphDef.MetaInfoDef):
    raise TypeError("meta_info_def must be of type MetaInfoDef, not %s",
                    type(meta_info_def))
  if graph_def and not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError("graph_def must be of type GraphDef, not %s",
                    type(graph_def))
  if saver_def and not isinstance(saver_def, saver_pb2.SaverDef):
    raise TypeError("saver_def must be of type SaverDef, not %s",
                    type(saver_def))

  # Sets graph to default graph if it's not passed in.
  graph = graph or ops.get_default_graph()

  # Creates a MetaGraphDef proto.
  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  # Adds meta_info_def.
  if not meta_info_def:
    meta_info_def = meta_graph_pb2.MetaGraphDef.MetaInfoDef()

  # Set the tf version strings to the current tf build.
  meta_info_def.tensorflow_version = versions.__version__
  meta_info_def.tensorflow_git_version = versions.__git_version__
  meta_graph_def.meta_info_def.MergeFrom(meta_info_def)

  # Adds graph_def or the default.
  if not graph_def:
    meta_graph_def.graph_def.MergeFrom(graph.as_graph_def(add_shapes=True))
  else:
    meta_graph_def.graph_def.MergeFrom(graph_def)

  # Fills in meta_info_def.stripped_op_list using the ops from graph_def.
  # pylint: disable=g-explicit-length-test
  if len(meta_graph_def.meta_info_def.stripped_op_list.op) == 0:
    meta_graph_def.meta_info_def.stripped_op_list.MergeFrom(
        stripped_op_list_for_graph(meta_graph_def.graph_def))
  # pylint: enable=g-explicit-length-test

  # Strip default valued attributes in graph_def.
  if strip_default_attrs:
    strip_graph_default_valued_attrs(meta_graph_def)

  # Adds saver_def.
  if saver_def:
    meta_graph_def.saver_def.MergeFrom(saver_def)

  # Adds collection_list.
  if collection_list is not None:
    clist = collection_list
  else:
    clist = graph.get_all_collection_keys()

  for ctype in clist:
    if clear_extraneous_savers and ctype == ops.GraphKeys.SAVERS:
      # Avoid importing Saver here
      from_proto = ops.get_from_proto_function(ctype)
      add_collection_def(meta_graph_def, ctype,
                         graph=graph,
                         export_scope=export_scope,
                         exclude_nodes=exclude_nodes,
                         override_contents=[from_proto(saver_def)])
    else:
      add_collection_def(meta_graph_def, ctype,
                         graph=graph,
                         export_scope=export_scope,
                         exclude_nodes=exclude_nodes)
  return meta_graph_def


def read_meta_graph_file(filename):
  """Reads a file containing `MetaGraphDef` and returns the protocol buffer.

  Args:
    filename: `meta_graph_def` filename including the path.

  Returns:
    A `MetaGraphDef` protocol buffer.

  Raises:
    IOError: If the file doesn't exist, or cannot be successfully parsed.
  """
  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  if not file_io.file_exists(filename):
    raise IOError("File %s does not exist." % filename)
  # First try to read it as a binary file.
  file_content = file_io.FileIO(filename, "rb").read()
  try:
    meta_graph_def.ParseFromString(file_content)
    return meta_graph_def
  except Exception:  # pylint: disable=broad-except
    pass

  # Next try to read it as a text file.
  try:
    text_format.Merge(file_content.decode("utf-8"), meta_graph_def)
  except text_format.ParseError as e:
    raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

  return meta_graph_def


def import_scoped_meta_graph(meta_graph_or_file,
                             clear_devices=False,
                             graph=None,
                             import_scope=None,
                             input_map=None,
                             unbound_inputs_col_name="unbound_inputs",
                             restore_collections_predicate=(lambda key: True)):
  """Recreates a `Graph` saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates the desired collections, and returns a dictionary of
  all the Variables imported into the name scope.

  In combination with `export_scoped_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      from graph_def. Default false.
    graph: The `Graph` to import into. If `None`, use the default graph.
    import_scope: Optional `string`. Name scope into which to import the
      subgraph. If `None`, the graph is imported to the root name scope.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    unbound_inputs_col_name: Collection name for looking up unbound inputs.
    restore_collections_predicate: a predicate on collection names. A collection
      named c (i.e whose key is c) will be restored iff
      1) `restore_collections_predicate(c)` is True, and
      2) `c != unbound_inputs_col_name`.

  Returns:
    A dictionary of all the `Variables` imported into the name scope.

  Raises:
    ValueError: If the graph_def contains unbound inputs.
  """
  return import_scoped_meta_graph_with_return_elements(
      meta_graph_or_file, clear_devices, graph, import_scope, input_map,
      unbound_inputs_col_name, restore_collections_predicate)[0]


def import_scoped_meta_graph_with_return_elements(
    meta_graph_or_file,
    clear_devices=False,
    graph=None,
    import_scope=None,
    input_map=None,
    unbound_inputs_col_name="unbound_inputs",
    restore_collections_predicate=(lambda key: True),
    return_elements=None):
  """Imports graph from `MetaGraphDef` and returns vars and return elements.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates the desired collections, and returns a dictionary of
  all the Variables imported into the name scope.

  In combination with `export_scoped_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      from graph_def. Default false.
    graph: The `Graph` to import into. If `None`, use the default graph.
    import_scope: Optional `string`. Name scope into which to import the
      subgraph. If `None`, the graph is imported to the root name scope.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    unbound_inputs_col_name: Collection name for looking up unbound inputs.
    restore_collections_predicate: a predicate on collection names. A collection
      named c (i.e whose key is c) will be restored iff
      1) `restore_collections_predicate(c)` is True, and
      2) `c != unbound_inputs_col_name`.
    return_elements:  A list of strings containing operation names in the
      `MetaGraphDef` that will be returned as `Operation` objects; and/or
      tensor names in `MetaGraphDef` that will be returned as `Tensor` objects.

  Returns:
    A tuple of (
      dictionary of all the `Variables` imported into the name scope,
      list of `Operation` or `Tensor` objects from the `return_elements` list).

  Raises:
    ValueError: If the graph_def contains unbound inputs.

  """
  if context.executing_eagerly():
    raise ValueError("Exporting/importing meta graphs is not supported when "
                     "eager execution is enabled.")
  if isinstance(meta_graph_or_file, meta_graph_pb2.MetaGraphDef):
    meta_graph_def = meta_graph_or_file
  else:
    meta_graph_def = read_meta_graph_file(meta_graph_or_file)

  if unbound_inputs_col_name:
    for key, col_def in meta_graph_def.collection_def.items():
      if key == unbound_inputs_col_name:
        kind = col_def.WhichOneof("kind")
        field = getattr(col_def, kind)
        if field.value and (
            not input_map or
            sorted([compat.as_str(v) for v in field.value]) !=
            sorted(input_map)):
          raise ValueError("Graph contains unbound inputs: %s. Must "
                           "provide these inputs through input_map." % ",".join(
                               compat.as_str(v)
                               for v in field.value
                               if not input_map or v not in input_map))
        break

  # Sets graph to default graph if it's not passed in.
  graph = graph or ops.get_default_graph()

  # Gathers the list of nodes we are interested in.
  with graph.as_default():
    producer_op_list = None
    if meta_graph_def.meta_info_def.HasField("stripped_op_list"):
      producer_op_list = meta_graph_def.meta_info_def.stripped_op_list
    input_graph_def = meta_graph_def.graph_def
    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
      for node in input_graph_def.node:
        node.device = ""

    scope_to_prepend_to_names = graph.unique_name(
        import_scope or "", mark_as_used=False)

    imported_return_elements = importer.import_graph_def(
        input_graph_def,
        name=(import_scope or scope_to_prepend_to_names),
        input_map=input_map,
        producer_op_list=producer_op_list,
        return_elements=return_elements)

    # TensorFlow versions before 1.9 (not inclusive) exported SavedModels
    # without a VariableDef.trainable field set.
    tf_version = meta_graph_def.meta_info_def.tensorflow_version
    if not tf_version:
      variables_have_trainable = True
    else:
      variables_have_trainable = (
          distutils_version.LooseVersion(tf_version)
          >= distutils_version.LooseVersion("1.9"))

    # Sort collections so we see TRAINABLE_VARIABLES first and can default these
    # variables to trainable if the value is not set in their VariableDef.
    sorted_collections = []
    if ops.GraphKeys.TRAINABLE_VARIABLES in meta_graph_def.collection_def:
      sorted_collections.append(
          (ops.GraphKeys.TRAINABLE_VARIABLES,
           meta_graph_def.collection_def[ops.GraphKeys.TRAINABLE_VARIABLES]))
    for key, value in sorted(meta_graph_def.collection_def.items()):
      if key != ops.GraphKeys.TRAINABLE_VARIABLES:
        sorted_collections.append((key, value))

    # Restores all the other collections.
    variable_objects = {}
    for key, col_def in sorted_collections:
      # Don't add unbound_inputs to the new graph.
      if key == unbound_inputs_col_name:
        continue
      if not restore_collections_predicate(key):
        continue

      kind = col_def.WhichOneof("kind")
      if kind is None:
        logging.error("Cannot identify data type for collection %s. Skipping.",
                      key)
        continue
      from_proto = ops.get_from_proto_function(key)

      # Temporary change to allow the TFMA evaluator to read metric variables
      # saved as a bytes list.
      # TODO(kathywu): Remove this hack once cl/248406059 has been submitted.
      if key == ops.GraphKeys.METRIC_VARIABLES:
        # Metric variables will use the same proto functions as GLOBAL_VARIABLES
        from_proto = ops.get_from_proto_function(ops.GraphKeys.GLOBAL_VARIABLES)
      if from_proto and kind == "bytes_list":
        proto_type = ops.get_collection_proto_type(key)
        if key in ops.GraphKeys._VARIABLE_COLLECTIONS:  # pylint: disable=protected-access
          for value in col_def.bytes_list.value:
            variable = variable_objects.get(value, None)
            if variable is None:
              proto = proto_type()
              proto.ParseFromString(value)
              if not variables_have_trainable:
                # If the VariableDef proto does not contain a "trainable"
                # property because it was exported before that property was
                # added, we default it to whether the variable is in the
                # TRAINABLE_VARIABLES collection. We've sorted
                # TRAINABLE_VARIABLES to be first, so trainable variables will
                # be created from that collection.
                proto.trainable = (key == ops.GraphKeys.TRAINABLE_VARIABLES)
              variable = from_proto(
                  proto, import_scope=scope_to_prepend_to_names)
              variable_objects[value] = variable
            graph.add_to_collection(key, variable)
        else:
          for value in col_def.bytes_list.value:
            proto = proto_type()
            proto.ParseFromString(value)
            graph.add_to_collection(
                key, from_proto(
                    proto, import_scope=scope_to_prepend_to_names))
      else:
        field = getattr(col_def, kind)
        if key in _COMPAT_COLLECTION_LIST:
          logging.warning(
              "The saved meta_graph is possibly from an older release:\n"
              "'%s' collection should be of type 'byte_list', but instead "
              "is of type '%s'.", key, kind)
        if kind == "node_list":
          for value in field.value:
            col_op = graph.as_graph_element(
                ops.prepend_name_scope(value, scope_to_prepend_to_names))
            graph.add_to_collection(key, col_op)
        elif kind == "int64_list":
          # NOTE(opensource): This force conversion is to work around the fact
          # that Python2 distinguishes between int and long, while Python3 has
          # only int.
          for value in field.value:
            graph.add_to_collection(key, int(value))
        else:
          for value in field.value:
            graph.add_to_collection(
                key, ops.prepend_name_scope(value, scope_to_prepend_to_names))

    var_list = {}
    variables = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                     scope=scope_to_prepend_to_names)
    for v in variables:
      var_list[ops.strip_name_scope(v.name, scope_to_prepend_to_names)] = v

  return var_list, imported_return_elements


def export_scoped_meta_graph(filename=None,
                             graph_def=None,
                             graph=None,
                             export_scope=None,
                             as_text=False,
                             unbound_inputs_col_name="unbound_inputs",
                             clear_devices=False,
                             saver_def=None,
                             clear_extraneous_savers=False,
                             strip_default_attrs=False,
                             save_debug_info=False,
                             **kwargs):
  """Returns `MetaGraphDef` proto. Optionally writes it to filename.

  This function exports the graph, saver, and collection objects into
  `MetaGraphDef` protocol buffer with the intention of it being imported
  at a later time or location to restart training, run inference, or be
  a subgraph.

  Args:
    filename: Optional filename including the path for writing the
      generated `MetaGraphDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    graph: The `Graph` to export. If `None`, use the default graph.
    export_scope: Optional `string`. Name scope under which to extract
      the subgraph. The scope name will be stripped from the node definitions
      for easy import later into new name scopes. If `None`, the whole graph
      is exported.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.
    unbound_inputs_col_name: Optional `string`. If provided, a string collection
      with the given name will be added to the returned `MetaGraphDef`,
      containing the names of tensors that must be remapped when importing the
      `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      before exporting the graph.
    saver_def: `SaverDef` protocol buffer.
    clear_extraneous_savers: Remove any Saver-related information from the
        graph (both Save/Restore ops and SaverDefs) that are not associated
        with the provided SaverDef.
    strip_default_attrs: Set to true if default valued attributes must be
      removed while exporting the GraphDef.
    save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
      which in the same directory of filename and with `_debug` added before the
      file extension.
    **kwargs: Optional keyed arguments, including meta_info_def and
        collection_list.

  Returns:
    A `MetaGraphDef` proto and dictionary of `Variables` in the exported
    name scope.

  Raises:
    ValueError: When the `GraphDef` is larger than 2GB.
    ValueError: When executing in Eager mode and either `graph_def` or `graph`
      is undefined.
  """
  if context.executing_eagerly() and not (graph_def is not None and
                                          graph is not None):
    raise ValueError("Exporting/importing meta graphs is not supported when "
                     "Eager Execution is enabled.")
  graph = graph or ops.get_default_graph()

  exclude_nodes = None
  unbound_inputs = []
  if export_scope or clear_extraneous_savers or clear_devices:
    if graph_def:
      new_graph_def = graph_pb2.GraphDef()
      new_graph_def.versions.CopyFrom(graph_def.versions)
      new_graph_def.library.CopyFrom(graph_def.library)

      if clear_extraneous_savers:
        exclude_nodes = _find_extraneous_saver_nodes(graph_def, saver_def)

      for node_def in graph_def.node:
        if _should_include_node(node_def.name, export_scope, exclude_nodes):
          new_node_def = _node_def(node_def, export_scope, unbound_inputs,
                                   clear_devices=clear_devices)
          new_graph_def.node.extend([new_node_def])
      graph_def = new_graph_def
    else:
      # Only do this complicated work if we want to remove a name scope.
      graph_def = graph_pb2.GraphDef()
      # pylint: disable=protected-access
      graph_def.versions.CopyFrom(graph.graph_def_versions)
      bytesize = 0

      if clear_extraneous_savers:
        exclude_nodes = _find_extraneous_saver_nodes(graph.as_graph_def(),
                                                     saver_def)

      for key in sorted(graph._nodes_by_id):
        if _should_include_node(graph._nodes_by_id[key].name,
                                export_scope,
                                exclude_nodes):
          value = graph._nodes_by_id[key]
          # pylint: enable=protected-access
          node_def = _node_def(value.node_def, export_scope, unbound_inputs,
                               clear_devices=clear_devices)
          graph_def.node.extend([node_def])
          if value.outputs:
            assert "_output_shapes" not in graph_def.node[-1].attr
            graph_def.node[-1].attr["_output_shapes"].list.shape.extend([
                output.get_shape().as_proto() for output in value.outputs])
          bytesize += value.node_def.ByteSize()
          if bytesize >= (1 << 31) or bytesize < 0:
            raise ValueError("GraphDef cannot be larger than 2GB.")

      graph._copy_functions_to_graph_def(graph_def, bytesize)  # pylint: disable=protected-access

    # It's possible that not all the inputs are in the export_scope.
    # If we would like such information included in the exported meta_graph,
    # add them to a special unbound_inputs collection.
    if unbound_inputs_col_name:
      # Clears the unbound_inputs collections.
      graph.clear_collection(unbound_inputs_col_name)
      for k in unbound_inputs:
        graph.add_to_collection(unbound_inputs_col_name, k)

  var_list = {}
  variables = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                   scope=export_scope)
  for v in variables:
    if _should_include_node(v, export_scope, exclude_nodes):
      var_list[ops.strip_name_scope(v.name, export_scope)] = v

  scoped_meta_graph_def = create_meta_graph_def(
      graph_def=graph_def,
      graph=graph,
      export_scope=export_scope,
      exclude_nodes=exclude_nodes,
      clear_extraneous_savers=clear_extraneous_savers,
      saver_def=saver_def,
      strip_default_attrs=strip_default_attrs,
      **kwargs)

  if filename:
    graph_io.write_graph(
        scoped_meta_graph_def,
        os.path.dirname(filename),
        os.path.basename(filename),
        as_text=as_text)
    if save_debug_info:
      name, _ = os.path.splitext(filename)
      debug_filename = "{name}{ext}".format(name=name, ext=".debug")

      # Gets the operation from the graph by the name. Exludes variable nodes,
      # so only the nodes in the frozen models are included.
      # TODO(liufengdb): fix this for functions.
      ops_to_export = []
      for node in scoped_meta_graph_def.graph_def.node:
        scoped_op_name = ops.prepend_name_scope(node.name, export_scope)
        ops_to_export.append(("", graph.get_operation_by_name(scoped_op_name)))

      graph_debug_info = error_interpolation.create_graph_debug_info_def(
          ops_to_export)

      graph_io.write_graph(
          graph_debug_info,
          os.path.dirname(debug_filename),
          os.path.basename(debug_filename),
          as_text=as_text)

  return scoped_meta_graph_def, var_list


def copy_scoped_meta_graph(from_scope, to_scope,
                           from_graph=None, to_graph=None):
  """Copies a sub-meta_graph from one scope to another.

  Args:
    from_scope: `String` name scope containing the subgraph to be copied.
    to_scope: `String` name scope under which the copied subgraph will reside.
    from_graph: Optional `Graph` from which to copy the subgraph. If `None`, the
      default graph is use.
    to_graph: Optional `Graph` to which to copy the subgraph. If `None`, the
      default graph is used.

  Returns:
    A dictionary of `Variables` that has been copied into `to_scope`.

  Raises:
    ValueError: If `from_scope` and `to_scope` are the same while
      `from_graph` and `to_graph` are also the same.
  """
  from_graph = from_graph or ops.get_default_graph()
  to_graph = to_graph or ops.get_default_graph()

  if from_graph == to_graph and from_scope == to_scope:
    raise ValueError("'from_scope' and 'to_scope' need to be different "
                     "when performing copy in the same graph.")

  orig_meta_graph, var_list = export_scoped_meta_graph(
      export_scope=from_scope, graph=from_graph)
  var_list = import_scoped_meta_graph(orig_meta_graph,
                                      graph=to_graph,
                                      import_scope=to_scope)
  return var_list
