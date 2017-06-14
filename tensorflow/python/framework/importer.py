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

import contextlib
import copy

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import compat


# TODO(josh11b): SWIG the code from node_def_util instead of duplicating
# the logic here.
def _GetNodeAttr(node_def, attr_name):
  if attr_name not in node_def.attr:
    raise ValueError('Expected one attr with name %r in %s.'
                     % (attr_name, str(node_def)))
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


def _FindAttrInOpDef(attr_name, op_def):
  for attr_def in op_def.attr:
    if attr_name == attr_def.name:
      return attr_def
  return None


def import_graph_def(graph_def, input_map=None, return_elements=None,
                     name=None, op_dict=None, producer_op_list=None):
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
    op_dict: (Optional.) A dictionary mapping op type names to `OpDef` protos.
      Must contain an `OpDef` proto for each op type named in `graph_def`.
      If omitted, uses the `OpDef` protos registered in the global registry.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided, attrs
      for ops in `graph_def` that are not in `op_dict` that have their default
      value according to `producer_op_list` will be removed. This will allow
      some more `GraphDef`s produced by later binaries to be accepted by
      earlier binaries.

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
  # Type checks for inputs.
  if not isinstance(graph_def, graph_pb2.GraphDef):
    # `graph_def` could be a dynamically-created message, so try a duck-typed
    # approach
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('graph_def must be a GraphDef proto.')
  if input_map is None:
    input_map = {}
  else:
    if not (isinstance(input_map, dict)
            and all(isinstance(k, compat.bytes_or_text_types)
                    for k in input_map.keys())):
      raise TypeError('input_map must be a dictionary mapping strings to '
                      'Tensor objects.')
  if return_elements is not None:
    return_elements = tuple(return_elements)
    if not all(isinstance(x, compat.bytes_or_text_types)
               for x in return_elements):
      raise TypeError('return_elements must be a list of strings.')

  # Use a canonical representation for all tensor names.
  input_map = {_CanonicalInputName(k): v for k, v in input_map.items()}
  used_input_keys = set()

  name_to_op = {}

  if op_dict is None:
    op_dict = op_def_registry.get_registered_ops()

  if producer_op_list is None:
    producer_op_dict = None
  else:
    producer_op_dict = {op.name: op for op in producer_op_list.op}

  g = ops.get_default_graph()

  # Add any functions defined in `graph_def` to `g`
  if graph_def.library and graph_def.library.function:
    # Copy op_dict so we don't clobber the original
    op_dict = copy.copy(op_dict)
    # pylint: disable=protected-access
    # Note that we do not prepend `name` to the function name. The reasoning is
    # that function names are similar to op definition names, which currently do
    # not have a scoped name or namespace scheme.
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

    if not all(isinstance(v, ops.Tensor) for v in input_map.values()):
      if not scope:
        # The caller must have passed `name=''`.
        raise ValueError(
            'tf.import_graph_def() requires a non-empty `name` if `input_map` '
            'contains non-Tensor values. Try calling tf.convert_to_tensor() on '
            '`input_map` values before calling tf.import_graph_def().')
      with ops.name_scope('_inputs'):
        input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}

    # NOTE(mrry): We do this in two passes, because there may be a cycle in
    # `graph_def`.

    # 1. Add operations without their inputs.
    for node in graph_def.node:
      # Check to see if this op's name matches a previously seen op
      if node.name in name_to_op:
        raise ValueError('Duplicate name \'%s\' in GraphDef.' % node.name)
      # Set any default attr values that aren't present.
      if node.op not in op_dict:
        raise ValueError('No op named %s in defined operations.' % node.op)
      op_def = op_dict[node.op]
      for attr_def in op_def.attr:
        key = attr_def.name
        if attr_def.HasField('default_value'):
          value = node.attr[key]
          if value is None or value.WhichOneof('value') is None:
            node.attr[key].CopyFrom(attr_def.default_value)
      if producer_op_dict:
        # Remove any default attr values that aren't in op_def.
        if node.op in producer_op_dict:
          producer_op_def = producer_op_dict[node.op]
          # We make a copy of node.attr to iterate through since we
          # may modify node.attr inside the loop.
          for key in list(node.attr):
            if _FindAttrInOpDef(key, op_def) is None:
              # No attr_def in consumer, look in producer.
              attr_def = _FindAttrInOpDef(key, producer_op_def)
              if (attr_def and attr_def.HasField('default_value') and
                  node.attr[key] == attr_def.default_value):
                # Unknown attr had default value in producer, delete it
                # so it can be understood by consumer.
                del node.attr[key]

      output_types = _OutputTypes(node, op_dict)
      name_to_op[node.name] = g.create_op(
          node.op, [], output_types, name=node.name, attrs=node.attr,
          compute_shapes=False, compute_device=False,
          op_def=op_def)

    # Maps from a node to the op it is colocated with, if colocation
    # is specified in the attributes.
    colocation_pairs = {}
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
                colocation_pairs[op] = original_op
                # Don't apply this op's device function,
                # the colocation constraint will ensure
                # the proper device gets assigned at runtime.
                apply_device_function = False

            else:
              new_class_values.append(class_value)
          value.list.CopyFrom(attr_value_pb2.AttrValue.ListValue(
              s=new_class_values))

      # NOTE(mrry): We cannot use zip here because control inputs do not appear
      # in the list of input_types.
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
                    'Control input %r not found in graph_def.' % (input_name,)))
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
      if op._input_dtypes != input_types:
        raise ValueError(
            _InvalidNodeMessage(
                node,
                'Input types mismatch (expected %r but got %r)'
                % (', '.join(dtypes.as_dtype(x).name for x in input_types),
                   ', '.join(x.name for x in op._input_dtypes))))
      # pylint: enable=protected-access

      if not g._is_function(op.type):  # pylint: disable=protected-access
        # Execute shape inference for this op.
        # NOTE(mrry): If the graph contains a cycle, the full shape information
        # may not be available for this op's inputs.
        ops.set_shapes_for_outputs(op)
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
    for op, coloc_op in colocation_pairs.items():
      # If the colocation op has no device, even after a device
      # application, there's nothing to do here.
      if not coloc_op.device:
        continue
      coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
      op._set_device(coloc_device)  # pylint: disable=protected-access

    # Treat unused input mappings as an error, because they are likely to be
    # due to a typo.
    unused_input_keys = frozenset(input_map.keys()).difference(used_input_keys)
    if unused_input_keys:
      raise ValueError(
          'Attempted to map inputs that were not found in graph_def: [%s]'
          % ', '.join(unused_input_keys))

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
