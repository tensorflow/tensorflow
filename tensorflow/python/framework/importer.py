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
"""A utility function for importing TensorFlow graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


# TODO(josh11b): SWIG the code from node_def_util instead of duplicating
# the logic here.
def _GetNodeAttr(node_def, attr_name):
  if attr_name not in node_def.attr:
    raise ValueError('Expected one attr with name %r in %s.' % (attr_name,
                                                                str(node_def)))
  return node_def.attr[attr_name]


def _ArgToTypesNoRef(node_def, arg_def):
  if arg_def.number_attr:
    repeats = _GetNodeAttr(node_def, arg_def.number_attr).i
    if arg_def.type_attr:
      dtype = _GetNodeAttr(node_def, arg_def.type_attr).type
    else:
      assert arg_def.type != types_pb2.DT_INVALID
      dtype = arg_def.type
    return [dtype] * repeats
  elif arg_def.type_attr:
    return [_GetNodeAttr(node_def, arg_def.type_attr).type]
  elif arg_def.type_list_attr:
    return _GetNodeAttr(node_def, arg_def.type_list_attr).list.type
  else:
    assert arg_def.type != types_pb2.DT_INVALID
    return [arg_def.type]


def _SingleArgToTypes(node_def, arg_def):
  types = _ArgToTypesNoRef(node_def, arg_def)
  if arg_def.is_ref:
    return [dtypes.as_dtype(dt)._as_ref.as_datatype_enum for dt in types]  # pylint: disable=protected-access
  return types


def _ArgsToTypes(node_def, arg_list):
  types = []
  for arg_def in arg_list:
    types.extend(_SingleArgToTypes(node_def, arg_def))
  return types


def _InputTypes(node_def, op_dict):
  op_def = op_dict[node_def.op]
  return _ArgsToTypes(node_def, op_def.input_arg)


def _OutputTypes(node_def, op_dict):
  op_def = op_dict[node_def.op]
  return _ArgsToTypes(node_def, op_def.output_arg)


def _IsControlInput(input_name):
  # Expected format: '^operation_name' (control input).
  return input_name.startswith('^')


def _ParseTensorName(tensor_name):
  """Parses a tensor name into an operation name and output index.

  This function will canonicalize tensor names as follows:

  * "foo:0"       -> ("foo", 0)
  * "foo:7"       -> ("foo", 7)
  * "foo"         -> ("foo", 0)
  * "foo:bar:baz" -> ValueError

  Args:
    tensor_name: The name of a tensor.

  Returns:
    A tuple containing the operation name, and the output index.

  Raises:
    ValueError: If `tensor_name' cannot be interpreted as the name of a tensor.
  """
  components = tensor_name.split(':')
  if len(components) == 2:
    # Expected format: 'operation_name:output_index'.
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError('Cannot convert %r to a tensor name.' % (tensor_name,))
    return components[0], output_index
  elif len(components) == 1:
    # Expected format: 'operation_name' (implicit 0th output).
    return components[0], 0
  else:
    raise ValueError('Cannot convert %r to a tensor name.' % (tensor_name,))


def _CanonicalInputName(input_name):
  input_name = compat.as_str(input_name)
  if _IsControlInput(input_name):
    return input_name
  input_op_name, output_index = _ParseTensorName(input_name)
  return '%s:%d' % (input_op_name, output_index)


def _InvalidNodeMessage(node, message):
  return 'graph_def is invalid at node %r: %s.' % (node.name, message)


@contextlib.contextmanager
def _MaybeDevice(device):
  """Applies the given device only if device is not None or empty."""
  if device:
    with ops.device(device):
      yield
  else:
    yield


def _ProcessGraphDefParam(graph_def, op_dict):
  """Type-checks and possibly canonicalizes `graph_def`."""
  if not isinstance(graph_def, graph_pb2.GraphDef):
    # `graph_def` could be a dynamically-created message, so try a duck-typed
    # approach
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('graph_def must be a GraphDef proto.')
  else:
    # If we're using the graph_def provided by the caller, modify graph_def
    # in-place to add attr defaults to the NodeDefs (this is visible to the
    # caller).
    # NOTE(skyewm): this is undocumented behavior that at least meta_graph.py
    # depends on. It might make sense to move this to meta_graph.py and have
    # import_graph_def not modify the graph_def argument (we'd have to make sure
    # this doesn't break anything else.)
    for node in graph_def.node:
      if node.op not in op_dict:
        # Assume unrecognized ops are functions for now. TF_ImportGraphDef will
        # report an error if the op is actually missing.
        continue
      op_def = op_dict[node.op]
      _SetDefaultAttrValues(node, op_def)

  return graph_def


def _ProcessInputMapParam(input_map):
  """Type-checks and possibly canonicalizes `input_map`."""
  if input_map is None:
    input_map = {}
  else:
    if not (isinstance(input_map, dict) and all(
        isinstance(k, compat.bytes_or_text_types) for k in input_map.keys())):
      raise TypeError('input_map must be a dictionary mapping strings to '
                      'Tensor objects.')
  return input_map


def _ProcessReturnElementsParam(return_elements):
  """Type-checks and possibly canonicalizes `return_elements`."""
  if return_elements is None:
    return None
  if not all(
      isinstance(x, compat.bytes_or_text_types) for x in return_elements):
    raise TypeError('return_elements must be a list of strings.')
  return tuple(compat.as_str(x) for x in return_elements)


def _FindAttrInOpDef(attr_name, op_def):
  for attr_def in op_def.attr:
    if attr_name == attr_def.name:
      return attr_def
  return None


def _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def):
  """Removes unknown default attrs according to `producer_op_list`.

  Removes any unknown attrs in `graph_def` (i.e. attrs that do not appear in
  the OpDefs in `op_dict`) that have a default value in `producer_op_list`.

  Args:
    op_dict: dict mapping operation name to OpDef.
    producer_op_list: OpList proto.
    graph_def: GraphDef proto
  """
  producer_op_dict = {op.name: op for op in producer_op_list.op}
  for node in graph_def.node:
    # Remove any default attr values that aren't in op_def.
    if node.op in producer_op_dict:
      op_def = op_dict[node.op]
      producer_op_def = producer_op_dict[node.op]
      # We make a copy of node.attr to iterate through since we may modify
      # node.attr inside the loop.
      for key in list(node.attr):
        if _FindAttrInOpDef(key, op_def) is None:
          # No attr_def in consumer, look in producer.
          attr_def = _FindAttrInOpDef(key, producer_op_def)
          if (attr_def and attr_def.HasField('default_value') and
              node.attr[key] == attr_def.default_value):
            # Unknown attr had default value in producer, delete it so it can be
            # understood by consumer.
            del node.attr[key]


def _ConvertInputMapValues(name, input_map):
  """Ensures all input map values are tensors.

  This should be called from inside the import name scope.

  Args:
    name: the `name` argument passed to import_graph_def
    input_map: the `input_map` argument passed to import_graph_def.

  Returns:
    An possibly-updated version of `input_map`.

  Raises:
    ValueError: if input map values cannot be converted due to empty name scope.
  """
  if not all(isinstance(v, ops.Tensor) for v in input_map.values()):
    if name == '':  # pylint: disable=g-explicit-bool-comparison
      raise ValueError(
          'tf.import_graph_def() requires a non-empty `name` if `input_map` '
          'contains non-Tensor values. Try calling tf.convert_to_tensor() on '
          '`input_map` values before calling tf.import_graph_def().')
    with ops.name_scope('_inputs'):
      input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}
  return input_map


def _PopulateTFImportGraphDefOptions(options, prefix, input_map,
                                     return_elements):
  """Populates the TF_ImportGraphDefOptions `options`."""
  c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix)
  c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options, True)

  for input_src, input_dst in input_map.items():
    input_src = compat.as_str(input_src)
    if input_src.startswith('^'):
      src_name = compat.as_bytes(input_src[1:])
      dst_op = input_dst._as_tf_output().oper  # pylint: disable=protected-access
      c_api.TF_ImportGraphDefOptionsRemapControlDependency(
          options, src_name, dst_op)
    else:
      src_name, src_idx = _ParseTensorName(input_src)
      src_name = compat.as_str(src_name)
      dst_output = input_dst._as_tf_output()  # pylint: disable=protected-access
      c_api.TF_ImportGraphDefOptionsAddInputMapping(options, src_name, src_idx,
                                                    dst_output)
  for name in return_elements or []:
    if ':' in name:
      op_name, index = _ParseTensorName(name)
      op_name = compat.as_str(op_name)
      c_api.TF_ImportGraphDefOptionsAddReturnOutput(options, op_name, index)
    else:
      c_api.TF_ImportGraphDefOptionsAddReturnOperation(options,
                                                       compat.as_str(name))


def _ProcessNewOps(graph):
  """Processes the newly-added TF_Operations in `graph`."""
  # Maps from a node to the names of the ops it's colocated with, if colocation
  # is specified in the attributes.
  colocation_pairs = {}

  for new_op in graph._add_new_tf_operations(compute_devices=False):  # pylint: disable=protected-access
    original_device = new_op.device
    new_op._set_device('')  # pylint: disable=protected-access
    colocation_names = _GetColocationNames(new_op)
    if colocation_names:
      colocation_pairs[new_op] = colocation_names
      # Don't set a device for this op, since colocation constraints override
      # device functions and the original device. Note that this op's device may
      # still be set by the loop below.
      # TODO(skyewm): why does it override the original device?
    else:
      with _MaybeDevice(original_device):
        graph._apply_device_functions(new_op)  # pylint: disable=protected-access

  # The following loop populates the device field of ops that are colocated
  # with another op.  This is implied by the colocation attribute, but we
  # propagate the device field for completeness.
  for op, coloc_op_list in colocation_pairs.items():
    coloc_device = None
    # Find any device in the list of colocated ops that have a device, if it
    # exists.  We assume that if multiple ops have devices, they refer to the
    # same device.  Otherwise, a runtime error will occur since the colocation
    # property cannot be guaranteed.
    #
    # One possible improvement is to try to check for compatibility of all
    # devices in this list at import time here, which would require
    # implementing a compatibility function for device specs in python.
    for coloc_op_name in coloc_op_list:
      try:
        coloc_op = graph._get_operation_by_name_unsafe(coloc_op_name)  # pylint: disable=protected-access
      except KeyError:
        raise ValueError('Specified colocation to an op that '
                         'does not exist during import: %s in %s' %
                         (coloc_op_name, op.name))
      if coloc_op.device:
        coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
        break
    if coloc_device:
      op._set_device(coloc_device)  # pylint: disable=protected-access


def _GetColocationNames(op):
  """Returns names of the ops that `op` should be colocated with."""
  colocation_names = []
  try:
    class_values = op.get_attr('_class')
  except ValueError:
    # No _class attr
    return
  for val in class_values:
    val = compat.as_str(val)
    if val.startswith('loc:@'):
      colocation_node_name = val[len('loc:@'):]
      if colocation_node_name != op.name:
        colocation_names.append(colocation_node_name)
  return colocation_names


def _GatherReturnElements(requested_return_elements, graph, results):
  """Returns the requested return elements from results.

  Args:
    requested_return_elements: list of strings of operation and tensor names
    graph: Graph
    results: wrapped TF_ImportGraphDefResults

  Returns:
    list of `Operation` and/or `Tensor` objects
  """
  return_outputs = c_api.TF_ImportGraphDefResultsReturnOutputs(results)
  return_opers = c_api.TF_ImportGraphDefResultsReturnOperations(results)

  combined_return_elements = []
  outputs_idx = 0
  opers_idx = 0
  for name in requested_return_elements:
    if ':' in name:
      combined_return_elements.append(
          graph._get_tensor_by_tf_output(return_outputs[outputs_idx]))  # pylint: disable=protected-access
      outputs_idx += 1
    else:
      combined_return_elements.append(
          graph._get_operation_by_tf_operation(return_opers[opers_idx]))  # pylint: disable=protected-access
      opers_idx += 1
  return combined_return_elements


def _SetDefaultAttrValues(node_def, op_def):
  """Set any default attr values in `node_def` that aren't present."""
  assert node_def.op == op_def.name
  for attr_def in op_def.attr:
    key = attr_def.name
    if attr_def.HasField('default_value'):
      value = node_def.attr[key]
      if value is None or value.WhichOneof('value') is None:
        node_def.attr[key].CopyFrom(attr_def.default_value)


@tf_export('import_graph_def')
@deprecated_args(None, 'Please file an issue at '
                 'https://github.com/tensorflow/tensorflow/issues if you depend'
                 ' on this feature.', 'op_dict')
def import_graph_def(graph_def,
                     input_map=None,
                     return_elements=None,
                     name=None,
                     op_dict=None,
                     producer_op_list=None):
  """Imports the graph from `graph_def` into the current default `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  @{tf.Tensor} and @{tf.Operation} objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  @{tf.Graph.as_graph_def} for a way to create a `GraphDef`
  proto.

  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into
      the default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def`
      to `Tensor` objects. The values of the named input tensors in the
      imported graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in
      `graph_def` that will be returned as `Operation` objects; and/or
      tensor names in `graph_def` that will be returned as `Tensor` objects.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    op_dict: (Optional.) Deprecated, do not use.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  op_dict = op_def_registry.get_registered_ops()

  graph_def = _ProcessGraphDefParam(graph_def, op_dict)
  input_map = _ProcessInputMapParam(input_map)
  return_elements = _ProcessReturnElementsParam(return_elements)

  if producer_op_list is not None:
    # TODO(skyewm): make a copy of graph_def so we're not mutating the argument?
    _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def)

  graph = ops.get_default_graph()

  if graph._c_graph:  # pylint: disable=protected-access
    with ops.name_scope(name, 'import', input_map.values()) as scope:
      # Save unique prefix generated by name_scope
      if scope:
        assert scope.endswith('/')
        prefix = scope[:-1]
      else:
        prefix = ''

      # Generate any input map tensors inside name scope
      input_map = _ConvertInputMapValues(name, input_map)

    scoped_options = c_api_util.ScopedTFImportGraphDefOptions()
    options = scoped_options.options
    _PopulateTFImportGraphDefOptions(options, prefix, input_map,
                                     return_elements)

    # _ProcessNewOps mutates the new operations. _lock ensures a Session.run
    # call cannot occur between creating the TF_Operations in the
    # TF_GraphImportGraphDefWithResults call and mutating the them in
    # _ProcessNewOps.
    with graph._lock:  # pylint: disable=protected-access
      with c_api_util.tf_buffer(graph_def.SerializeToString()) as serialized:
        try:
          results = c_api.TF_GraphImportGraphDefWithResults(
              graph._c_graph, serialized, options)  # pylint: disable=protected-access
          results = c_api_util.ScopedTFImportGraphDefResults(results)
        except errors.InvalidArgumentError as e:
          # Convert to ValueError for backwards compatibility.
          raise ValueError(str(e))

      # Create _DefinedFunctions for any imported functions.
      #
      # We do this by creating _DefinedFunctions directly from `graph_def`, and
      # adding them to `graph`. Adding an existing function to a TF_Graph is a
      # no-op, so this only has the effect of updating the Python state (usually
      # _DefinedFunction.add_to_graph also adds the function to the TF_Graph).
      #
      # TODO(skyewm): fetch the TF_Functions directly from the TF_Graph
      # TODO(skyewm): avoid sending serialized FunctionDefs back to the TF_Graph
      # TODO(b/74620627): move this after _ProcessNewOps outside the lock once
      # _USE_C_SHAPES is removed.
      if graph_def.library and graph_def.library.function:
        # pylint: disable=protected-access
        functions = function._from_library(graph_def.library)
        for f in functions:
          f.add_to_graph(graph)
        # pylint: enable=protected-access

      _ProcessNewOps(graph)

    # Treat input mappings that don't appear in the graph as an error, because
    # they are likely to be due to a typo.
    missing_unused_input_keys = (
        c_api.TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
            results.results))
    if missing_unused_input_keys:
      missing_unused_input_keys = [
          compat.as_str(s) for s in missing_unused_input_keys
      ]
      raise ValueError(
          'Attempted to map inputs that were not found in graph_def: [%s]' %
          ', '.join(missing_unused_input_keys))

    if return_elements is None:
      return None
    else:
      return _GatherReturnElements(return_elements, graph, results.results)

  else:
    g = graph

    # Use a canonical representation for all tensor names.
    input_map = {_CanonicalInputName(k): v for k, v in input_map.items()}
    used_input_keys = set()
    name_to_op = {}

    # Add any functions defined in `graph_def` to `g`
    if graph_def.library and graph_def.library.function:
      # Copy op_dict so we don't clobber the original
      op_dict = copy.copy(op_dict)
      # pylint: disable=protected-access
      # Note that we do not prepend `name` to the function name. The reasoning
      # is that function names are similar to op definition names, which
      # currently do not have a scoped name or namespace scheme.
      functions = function._from_library(graph_def.library)
      for f in functions:
        f.add_to_graph(g)
        op_dict[f.name] = f.definition.signature
      # pylint: enable=protected-access

    # LINT.IfChange
    with ops.name_scope(name, 'import', input_map.values()) as scope:
      # TODO(ashankar): Should this just copy over or should it do some
      # more nuanced merging? For example, the graph may already have some
      # marked "bad versions" and we don't want to lose those because of
      # what's in graph_def.versions? The C++ ImporGraphDef does something
      # more nuanced.
      g.graph_def_versions.CopyFrom(graph_def.versions)

      input_map = _ConvertInputMapValues(name, input_map)

      # NOTE(mrry): We do this in two passes, because there may be a cycle in
      # `graph_def`.

      # 1. Add operations without their inputs.
      for node in graph_def.node:
        # Check to see if this op's name matches a previously seen op
        if node.name in name_to_op:
          raise ValueError('Duplicate name \'%s\' in GraphDef.' % node.name)
        if node.op not in op_dict:
          raise ValueError(
              'No op named %s in defined operations. If the Graph you are '
              'importing uses custom ops or any parts of tf.contrib, you '
              'should explicitly import the libraries defining those ops '
              'before loading the Graph. Note that tf.contrib is lazily loaded '
              'when accessed, so simply referencing (e.g.) '
              '`tf.contrib.resampler` will cause those ops to be made '
              'available.' % node.op)
        op_def = op_dict[node.op]

        output_types = _OutputTypes(node, op_dict)
        name_to_op[node.name] = g.create_op(
            node.op, [], output_types, name=node.name, attrs=node.attr,
            compute_shapes=False, compute_device=False,
            op_def=op_def)

      # Maps from a node to the ops it is colocated with, if colocation
      # is specified in the attributes.
      colocation_pairs = collections.defaultdict(list)

      # 2. Add inputs to the operations.
      for node in graph_def.node:
        op = name_to_op[node.name]
        input_types = _InputTypes(node, op_dict)
        apply_device_function = True

        # Rewrite the colocation attributes in the graph, since the
        # names of new ops may have changed.
        for key, value in op.node_def.attr.items():
          if key == '_class':
            class_values = value.list
            new_class_values = []
            for class_value in class_values.s:
              if class_value.startswith(b'loc:@'):
                op_to_bind_to = class_value[5:].decode()
                # Find the op by its original name.
                if op_to_bind_to not in name_to_op:
                  raise ValueError('Specified colocation to an op that '
                                   'does not exist during import: %s in %s' % (
                                       op_to_bind_to, node.name))
                original_op = name_to_op[op_to_bind_to]
                new_class_values.append(compat.as_bytes(
                    'loc:@' + original_op.name))
                if op_to_bind_to != node.name:
                  # Keep track of this mapping for a later phase.
                  colocation_pairs[op].append(original_op)
                  # Don't apply this op's device function,
                  # the colocation constraint will ensure
                  # the proper device gets assigned at runtime.
                  apply_device_function = False

              else:
                new_class_values.append(class_value)
            value.list.CopyFrom(attr_value_pb2.AttrValue.ListValue(
                s=new_class_values))

        # NOTE(mrry): We cannot use zip here because control inputs do not
        # appear in the list of input_types.
        for i, input_name in enumerate(
            [_CanonicalInputName(x) for x in node.input]):

          if _IsControlInput(input_name):
            # (a) Input is a control input that should be taken from an op
            #     in "graph_def".
            try:
              source_op = name_to_op[input_name[1:]]
            except KeyError:
              raise ValueError(
                  _InvalidNodeMessage(
                      node,
                      'Control input %r not found in graph_def.'
                      % (input_name,)))
            # pylint: disable=protected-access
            op._add_control_input(source_op)
            # pylint: enable=protected-access

          else:
            try:
              input_type = input_types[i]
            except IndexError:
              raise ValueError(_InvalidNodeMessage(
                  node, 'More inputs specified (%r) than the op expects.'
                  % (input_name,)))

            if input_name in input_map:
              # (b) Input should be replaced by a tensor from the caller.
              source_tensor = input_map[input_name]
              used_input_keys.add(input_name)

            else:
              # (c) Input should be taken from an op in `graph_def`.
              operation_name, output_index = _ParseTensorName(input_name)
              try:
                source_op = name_to_op[operation_name]
                source_tensor = list(source_op.values())[output_index]
              except (KeyError, IndexError):
                raise ValueError(
                    _InvalidNodeMessage(
                        node,
                        'Input tensor %r not found in graph_def.'
                        % (input_name,)))

            try:
              # pylint: disable=protected-access
              op._add_input(source_tensor, dtype=input_type)
              # pylint: enable=protected-access
            except TypeError as te:
              raise ValueError(_InvalidNodeMessage(
                  node, 'Input tensor %r %s' % (input_name, te)))

        # pylint: disable=protected-access
        if op._input_types != input_types:
          raise ValueError(
              _InvalidNodeMessage(
                  node,
                  'Input types mismatch (expected %r but got %r)'
                  % (', '.join(dtypes.as_dtype(x).name for x in input_types),
                     ', '.join(x.name for x in op._input_types))))
        # pylint: enable=protected-access

        # Execute shape inference for this op.
        # NOTE(mrry): If the graph contains a cycle, the full shape
        # information may not be available for this op's inputs.
        ops.set_shape_and_handle_data_for_outputs(op)
        # For nodes with _output_shapes set, set the output shapes.
        if '_output_shapes' in op.node_def.attr:
          for i, output in enumerate(op.outputs):
            dims = op.node_def.attr['_output_shapes'].list.shape[i]
            output_shape = tensor_shape.TensorShape(
                None if dims.unknown_rank else
                [dim.size if dim.size >= 0 else None for dim in dims.dim])

            try:
              output.set_shape(output_shape)
            except ValueError as e:
              # If the output shape is incompatible with what is inferred
              # by the graph for a very specific whitelist of ops, then we
              # ignore this output shape.  This can happen if there is a
              # bug in the shape function for some operation, and the
              # serialized graph def has the incorrect shape set when
              # running on a newer binary with the fixed shape function.
              # This is an escape hatch that allows us to correct shape
              # functions that are not critical to correct execution but
              # would cause graphs to fail if imported after correcting.
              #
              # This can be removed after 2017/03/08.
              if op.type in ['RandomShuffleQueue', 'PaddingFIFOQueue',
                             'FIFOQueue', 'PriorityQueue', 'QueueSize',
                             'Stack', 'Barrier', 'BarrierReadySize',
                             'BarrierIncompleteSize', 'HashTable',
                             'MutableHashTable',
                             'MutableHashTableOfTensors', 'Mutex',
                             'CuckooTable', 'IndexTable',
                             'WholeFileReader', 'TextLineReader',
                             'FixedLengthRecordReader',
                             'TFRecordReader', 'IdentityReader',
                             'LMDBReader',
                             'RefSwitch', 'RefEnter', 'RefNextIteration',
                             'RefMerge', 'RefIdentity']:
                pass
              elif op.type in [
                  'ConditionalAccumulator', 'SparseConditionalAccumulator',
                  'Table'
              ]:
                # This can be removed after 2017/04/24.
                pass
              else:
                raise e

          del op.node_def.attr['_output_shapes']

        # NOTE(mrry): We do this after configuring the inputs, because
        # the result of the device functions may depend on the inputs.
        if apply_device_function:
          with _MaybeDevice(node.device):
            g._apply_device_functions(op)  # pylint: disable=protected-access

      # The following loop populates the device field of ops that are
      # colocated with another op.  This is implied by the colocation
      # attribute, but we propagate the device field for completeness.
      for op, coloc_op_list in colocation_pairs.items():
        coloc_device = None
        # Find any device in the list of colocated ops that have a
        # device, if it exists.  We assume that if multiple ops
        # have devices, they refer to the same device.  Otherwise, a
        # runtime error will occur since the colocation property
        # cannot be guaranteed.
        #
        # One possible improvement is to try to check for compatibility
        # of all devices in this list at import time here, which would
        # require implementing a compatibility function for device specs
        # in python.
        for coloc_op in coloc_op_list:
          if coloc_op.device:
            coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
            break
        if coloc_device:
          op._set_device(coloc_device)  # pylint: disable=protected-access

      # Treat input mappings that don't appear in the graph as an error,
      # because they are likely to be due to a typo.
      def _IsImportedNodeOutput(tensor_name):
        operation_name, output_index = _ParseTensorName(tensor_name)
        try:
          return output_index < len(name_to_op[operation_name].outputs)
        except KeyError:
          return False
      absent_input_keys = [
          k for k in frozenset(input_map.keys()).difference(used_input_keys)
          if not _IsImportedNodeOutput(k)]
      if absent_input_keys:
        raise ValueError(
            'Attempted to map inputs that were not found in graph_def: [%s]'
            % ', '.join(absent_input_keys))

      if return_elements is None:
        return None
      else:
        ret = []
        for name in return_elements:
          name = compat.as_str(name)
          if ':' in name:
            try:
              operation_name, output_index = _ParseTensorName(name)
              ret.append(name_to_op[operation_name].outputs[output_index])
            except (ValueError, KeyError, IndexError):
              raise ValueError(
                  'Requested return_element %r not found in graph_def.' % name)
          else:
            try:
              ret.append(name_to_op[name])
            except KeyError:
              raise ValueError(
                  'Requested return_element %r not found in graph_def.' % name)
        return ret
    # LINT.ThenChange(//tensorflow/core/graph/graph_constructor.cc)
