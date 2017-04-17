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

  Raises:
    ValueError: If an unregistered op is used.
  """
  # This is the Python equivalent of StrippedOpListForGraph in C++.
  # Unfortunately, since the Python op registry can differ from that in C++, we
  # can't remove the duplication using swig (at least naively).
  # TODO(irving): Support taking graphs directly.

  used_ops = ops_used_by_graph_def(graph_def)

  # Verify that all used ops are registered.
  registered_ops = op_def_registry.get_registered_ops()
  # These internal ops used by functions are not registered, so we need to
  # whitelist them.  # TODO(irving): Do something better here.
  op_whitelist = ("_Arg", "_Retval", "_ListToArray", "_ArrayToList")
  for op in used_ops:
    if op not in registered_ops and op not in op_whitelist:
      raise ValueError("Op %s is used by the graph, but is not registered" % op)

  # Build the stripped op list in sorted order
  return op_def_pb2.OpList(op=[registered_ops[op] for op in sorted(used_ops)
                               if op in registered_ops])


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


def _should_include_node(node_or_node_name, export_scope):
  """Returns `True` if a node should be included.

  Args:
    node_or_node_name: A node or `string` node name.
    export_scope: `string`. Name scope under which to extract the subgraph. The
      scope name will be striped from the node definitions for easy import later
      into new name scopes.

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

  return (node_name.startswith(_UNBOUND_INPUT_PREFIX) or
          (not export_scope or node_name.startswith(export_scope)))


def add_collection_def(meta_graph_def, key, graph=None,
                       export_scope=None):
  """Adds a collection to MetaGraphDef protocol buffer.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    key: One of the GraphKeys or user-defined string.
    graph: The `Graph` from which to get collections.
    export_scope: Optional `string`. Name scope to remove.
  """
  if graph and not isinstance(graph, ops.Graph):
    raise TypeError("graph must be of type Graph, not %s", type(graph))

  if not isinstance(key, six.string_types) and not isinstance(key, bytes):
    logging.warning("Only collections with string type keys will be "
                    "serialized. This key has %s", type(key))
    return

  # Sets graph to default graph if it's not passed in.
  graph = graph or ops.get_default_graph()

  collection_list = graph.get_collection(key)
  # Remove nodes that should not be exported from the collection list.
  collection_list = [x for x in collection_list if
                     _should_include_node(x, export_scope)]
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
    logging.warning("Error encountered when serializing %s.\n"
                    "Type is unsupported, or the types of the items don't "
                    "match field type in CollectionDef.\n%s", key, str(e))
    if key in meta_graph_def.collection_def:
      del meta_graph_def.collection_def[key]
    return


def create_meta_graph_def(meta_info_def=None,
                          graph_def=None,
                          saver_def=None,
                          collection_list=None,
                          graph=None,
                          export_scope=None):
  """Construct and returns a `MetaGraphDef` protocol buffer.

  Args:
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    graph: The `Graph` to create `MetaGraphDef` out of.
    export_scope: Optional `string`. Name scope to remove.

  Returns:
    MetaGraphDef protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
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

  # Adds saver_def.
  if saver_def:
    meta_graph_def.saver_def.MergeFrom(saver_def)

  # Adds collection_list.
  if collection_list:
    clist = collection_list
  else:
    clist = graph.get_all_collection_keys()
  for ctype in clist:
    add_collection_def(meta_graph_def, ctype,
                       graph=graph,
                       export_scope=export_scope)
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
                             unbound_inputs_col_name="unbound_inputs"):
  """Recreates a`Graph` saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates all the collections, and returns a saver
  constructed from the `saver_def` field.

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

  Returns:
    A dictionary of all the `Variables` imported into the name scope.

  Raises:
    ValueError: If the graph_def contains unbound inputs.
  """
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
                           "provide these inputs through input_map." %
                           ",".join([compat.as_str(v) for v in field.value
                                     if not input_map or v not in input_map]))
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
    importer.import_graph_def(
        input_graph_def, name=(import_scope or ""), input_map=input_map,
        producer_op_list=producer_op_list)

    # Restores all the other collections.
    for key, col_def in meta_graph_def.collection_def.items():
      # Don't add unbound_inputs to the new graph.
      if key == unbound_inputs_col_name:
        continue

      kind = col_def.WhichOneof("kind")
      if kind is None:
        logging.error("Cannot identify data type for collection %s. Skipping.",
                      key)
        continue
      from_proto = ops.get_from_proto_function(key)
      if from_proto:
        assert kind == "bytes_list"
        proto_type = ops.get_collection_proto_type(key)
        for value in col_def.bytes_list.value:
          proto = proto_type()
          proto.ParseFromString(value)
          graph.add_to_collection(
              key, from_proto(proto, import_scope=import_scope))
      else:
        field = getattr(col_def, kind)
        if kind == "node_list":
          for value in field.value:
            col_op = graph.as_graph_element(
                ops.prepend_name_scope(value, import_scope))
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
                key, ops.prepend_name_scope(value, import_scope))

    var_list = {}
    variables = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                     scope=import_scope)
    for v in variables:
      var_list[ops.strip_name_scope(v.name, import_scope)] = v

  return var_list


def export_scoped_meta_graph(filename=None,
                             graph_def=None,
                             graph=None,
                             export_scope=None,
                             as_text=False,
                             unbound_inputs_col_name="unbound_inputs",
                             clear_devices=False,
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
    graph: The `Graph` to import into. If `None`, use the default graph.
    export_scope: Optional `string`. Name scope under which to extract
      the subgraph. The scope name will be striped from the node definitions
      for easy import later into new name scopes. If `None`, the whole graph
      is exported. graph_def and export_scope cannot both be specified.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.
    unbound_inputs_col_name: Optional `string`. If provided, a string collection
      with the given name will be added to the returned `MetaGraphDef`,
      containing the names of tensors that must be remapped when importing the
      `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      before exporting the graph.
    **kwargs: Optional keyed arguments, including meta_info_def,
      saver_def, collection_list.

  Returns:
    A `MetaGraphDef` proto and dictionary of `Variables` in the exported
    name scope.

  Raises:
    ValueError: When the `GraphDef` is larger than 2GB.
  """
  graph = graph or ops.get_default_graph()
  unbound_inputs = []
  if export_scope or clear_devices:
    if graph_def:
      new_graph_def = graph_pb2.GraphDef()
      new_graph_def.versions.CopyFrom(graph_def.versions)
      for node_def in graph_def.node:
        if _should_include_node(node_def.name, export_scope):
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
      for key in sorted(graph._nodes_by_id):
        if _should_include_node(graph._nodes_by_id[key].name, export_scope):
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
    if _should_include_node(v, export_scope):
      var_list[ops.strip_name_scope(v.name, export_scope)] = v

  scoped_meta_graph_def = create_meta_graph_def(
      graph_def=graph_def,
      graph=graph,
      export_scope=export_scope,
      **kwargs)

  if filename:
    graph_io.write_graph(
        scoped_meta_graph_def,
        os.path.dirname(filename),
        os.path.basename(filename),
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
