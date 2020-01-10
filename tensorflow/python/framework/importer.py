"""A utility function for importing TensorFlow graphs."""
import contextlib

import tensorflow.python.platform

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import types as types_lib


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
    return [types_lib.as_dtype(dt).as_ref.as_datatype_enum for dt in types]
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


def import_graph_def(graph_def, input_map=None, return_elements=None,
                     name=None, op_dict=None):
  """Imports the TensorFlow graph in `graph_def` into the Python `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  [`Tensor`](#Tensor) and [`Operation`](#Operation) objects. See
  [`Graph.as_graph_def()`](#Graph.as_graph_def) for a way to create a
  `GraphDef` proto.

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
      `graph_def`. Defaults to `"import"`.
    op_dict: (Optional.) A dictionary mapping op type names to `OpDef` protos.
      Must contain an `OpDef` proto for each op type named in `graph_def`.
      If omitted, uses the `OpDef` protos registered in the global registry.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements'.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map' is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  # Type checks for inputs.
  if not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError('graph_def must be a GraphDef proto.')
  if input_map is None:
    input_map = {}
  else:
    if not (isinstance(input_map, dict)
            and all(isinstance(k, basestring) for k in input_map.keys())):
      raise TypeError('input_map must be a dictionary mapping strings to '
                      'Tensor objects.')
  if (return_elements is not None
      and not (isinstance(return_elements, (list, tuple))
               and all(isinstance(x, basestring) for x in return_elements))):
    raise TypeError('return_elements must be a list of strings.')

  # Use a canonical representation for all tensor names.
  input_map = {_CanonicalInputName(k): v for k, v in input_map.items()}
  used_input_keys = set()

  name_to_op = {}

  if op_dict is None:
    op_dict = op_def_registry.get_registered_ops()

  with ops.op_scope(input_map.values(), name, 'import'):
    g = ops.get_default_graph()

    with ops.name_scope('_inputs'):
      input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}

    # NOTE(mrry): We do this in two passes, because there may be a cycle in
    # `graph_def'.

    # 1. Add operations without their inputs.
    for node in graph_def.node:
      output_types = _OutputTypes(node, op_dict)
      with _MaybeDevice(node.device):
        name_to_op[node.name] = g.create_op(
            node.op, [], output_types, name=node.name, attrs=node.attr,
            compute_shapes=False)

    # 2. Add inputs to the operations.
    for node in graph_def.node:
      op = name_to_op[node.name]
      input_types = _InputTypes(node, op_dict)

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
            # (c) Input should be taken from an op in `graph_def'.
            operation_name, output_index = _ParseTensorName(input_name)
            try:
              source_op = name_to_op[operation_name]
              source_tensor = source_op.values()[output_index]
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
            raise ValueError(
                _InvalidNodeMessage(node, 'Input tensor %r %s'
                                    % (input_name, te.message)))

      # pylint: disable=protected_access
      if op._input_dtypes != input_types:
        raise ValueError(
            _InvalidNodeMessage(
                node,
                'Input types mismatch (expected %r but got %r)'
                % (", ".join(types_lib.as_dtype(x).name for x in input_types),
                   ", ".join(x.name for x in op._input_dtypes))))
      # pylint: enable=protected_access

      # Execute shape inference for this op.
      # NOTE(mrry): If the graph contains a cycle, the full shape information
      # may not be available for this op's inputs.
      ops.set_shapes_for_outputs(op)

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
