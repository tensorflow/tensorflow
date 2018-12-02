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

"""Class to hold a library of OpDefs and use it to create Brain operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib


def _Attr(op_def, name):
  for attr in op_def.attr:
    if attr.name == name:
      return attr
  raise TypeError("Inconsistent OpDef for '%s', missing attr '%s'" %
                  (op_def.name, name))


def _AttrValue(attr_protos, name):
  if name in attr_protos:
    return attr_protos[name]
  raise TypeError("Inconsistent OpDef, missing attr '%s' from '%s'." %
                  (name, attr_protos))


def _SatisfiesTypeConstraint(dtype, attr_def, param_name):
  if attr_def.HasField("allowed_values"):
    allowed_list = attr_def.allowed_values.list.type
    if dtype not in allowed_list:
      raise TypeError(
          "Value passed to parameter '%s' has DataType %s not in list of "
          "allowed values: %s" %
          (param_name, dtypes.as_dtype(dtype).name,
           ", ".join(dtypes.as_dtype(x).name for x in allowed_list)))


def _IsListParameter(arg):
  if arg.number_attr:
    return True
  elif arg.type_list_attr:
    return True
  return False


def _NumTypeFields(arg):
  num = 0
  if arg.type != types_pb2.DT_INVALID: num += 1
  if arg.type_attr: num += 1
  if arg.type_list_attr: num += 1
  return num


def _IsListValue(v):
  return isinstance(v, (list, tuple))


def _Flatten(l):
  """Converts [1, 2, [3, 4], [5]] to [1, 2, 3, 4, 5]."""
  # [1, 2, [3, 4], [5]] -> [[1], [2], [3, 4], [5]]
  l_of_l = [x if _IsListValue(x) else [x] for x in l]
  # [[1], [2], [3, 4], [5]] -> [1, 2, 3, 4, 5]
  return [item for sublist in l_of_l for item in sublist]


def _Restructure(l, structure):
  """Returns the elements of list l structured according to the given structure.

  A structure is represented by a list whose elements are either
  `None` or a non-negative integer. `None` corresponds to a single
  element in the output list, and an integer N corresponds to a nested
  list of length N.

  The function returns a data structure whose shape is given by
  `structure`, and whose elements are taken from `l`. If `structure`
  is a singleton, the function returns the single data structure
  implied by the 0th element of `structure`. For example:

      _Restructure(["foo", "bar", "baz", "qux"], [None, 2, None])
        -> ["foo", ["bar", "baz"], "qux"]

      _Restructure(["foo"], [None]) -> "foo"

      _Restructure(["foo"], [1]) -> ["foo"]

      _Restructure([], [0]) -> []

  Args:
    l: A list.
    structure: A list whose elements are either `None` or a non-negative
      integer.

  Returns:
    The elements of `l`, restructured according to `structure`. If
    `structure` is a list of length 1, this function returns the
    single data structure implied by `structure[0]`.

  """
  result = []
  current_index = 0
  for element in structure:
    if element is None:
      result.append(l[current_index])
      current_index += 1
    else:
      result.append(l[current_index:current_index+element])
      current_index += element

  if len(result) == 1:
    return result[0]
  else:
    return tuple(result)


def _MakeFloat(v, arg_name):
  if not isinstance(v, compat.real_types):
    raise TypeError("Expected float for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return float(v)


def _MakeInt(v, arg_name):
  if isinstance(v, six.string_types):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))
  try:
    return int(v)
  except (ValueError, TypeError):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))


def _MakeStr(v, arg_name):
  if not isinstance(v, compat.bytes_or_text_types):
    raise TypeError("Expected string for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return compat.as_bytes(v)  # Convert unicode strings to bytes.


def _MakeBool(v, arg_name):
  if not isinstance(v, bool):
    raise TypeError("Expected bool for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return v


def _MakeType(v, attr_def):
  try:
    v = dtypes.as_dtype(v).base_dtype
  except TypeError:
    raise TypeError("Expected DataType for argument '%s' not %s." %
                    (attr_def.name, repr(v)))
  i = v.as_datatype_enum
  _SatisfiesTypeConstraint(i, attr_def, param_name=attr_def.name)
  return i


def _MakeShape(v, arg_name):
  """Convert v into a TensorShapeProto."""
  # Args:
  #   v: A TensorShapeProto, a list of ints, or a tensor_shape.TensorShape.
  #   arg_name: String, for error messages.

  # Returns:
  #   A TensorShapeProto.
  if isinstance(v, tensor_shape_pb2.TensorShapeProto):
    for d in v.dim:
      if d.name:
        logging.warning("Warning: TensorShapeProto with a named dimension: %s",
                        str(v))
        break
    return v
  try:
    return tensor_shape.as_shape(v).as_proto()
  except TypeError as e:
    raise TypeError("Error converting %s to a TensorShape: %s" % (arg_name, e))
  except ValueError as e:
    raise ValueError("Error converting %s to a TensorShape: %s" % (arg_name, e))


def _MakeTensor(v, arg_name):
  """Ensure v is a TensorProto."""
  if isinstance(v, tensor_pb2.TensorProto):
    return v
  raise TypeError(
      "Don't know how to convert %s to a TensorProto for argument '%s'" %
      (repr(v), arg_name))


class _OpInfo(object):
  """All per-Op state we would like to precompute/validate."""

  def __init__(self, op_def):
    self.op_def = op_def
    # TODO(josh11b): SWIG the ValidateOpDef() function from C++ and call it
    # here, instead of these checks.
    for arg in list(op_def.input_arg) + list(op_def.output_arg):
      num_type_fields = _NumTypeFields(arg)
      if num_type_fields != 1:
        raise TypeError("Arg '%s' of '%s' must have one type field not %d" %
                        (arg.name, op_def.name, num_type_fields))
      if arg.type_attr:
        attr_type = _Attr(op_def, arg.type_attr).type
        if attr_type != "type":
          raise TypeError("Attr '%s' of '%s' used as a type_attr "
                          "but has type %s" %
                          (arg.type_attr, op_def.name, attr_type))
      if arg.type_list_attr:
        attr_type = _Attr(op_def, arg.type_list_attr).type
        if attr_type != "list(type)":
          raise TypeError(
              "Attr '%s' of '%s' used as a type_list_attr but has type %s" %
              (arg.type_attr, op_def.name, attr_type))
      if arg.number_attr:
        attr_type = _Attr(op_def, arg.number_attr).type
        if attr_type != "int":
          raise TypeError(
              "Attr '%s' of '%s' used as a number_attr but has type %s" %
              (arg.number_attr, op_def.name, attr_type))


# pylint: disable=g-doc-return-or-yield
@tf_contextlib.contextmanager
def _MaybeColocateWith(inputs):
  """A context manager for (maybe) colocating with a list of input tensors.

  Args:
    inputs: A list of `Tensor` or `Operation` objects.

  Returns:
    A context manager.
  """
  if not inputs:
    yield
  else:
    # NOTE(mrry): The `ops.colocate_with()` function accepts only a single
    # op or tensor, so we create one context manager per element in the list.
    with ops.colocate_with(inputs[0]), _MaybeColocateWith(inputs[1:]):
      yield
# pylint: enable=g-doc-return-or-yield


class OpDefLibrary(object):
  """Holds a collection of OpDefs, can add the corresponding Ops to a graph."""

  def __init__(self):
    self._ops = {}

  # pylint: disable=invalid-name
  def add_op(self, op_def):
    """Register an OpDef. May call apply_op with the name afterwards."""
    if not isinstance(op_def, op_def_pb2.OpDef):
      raise TypeError("%s is %s, not an op_def_pb2.OpDef" %
                      (op_def, type(op_def)))
    if not op_def.name:
      raise ValueError("%s missing name." % op_def)
    if op_def.name in self._ops:
      raise RuntimeError("Op name %s registered twice." % op_def.name)
    self._ops[op_def.name] = _OpInfo(op_def)

  def add_op_list(self, op_list):
    """Register the OpDefs from an OpList."""
    if not isinstance(op_list, op_def_pb2.OpList):
      raise TypeError("%s is %s, not an op_def_pb2.OpList" %
                      (op_list, type(op_list)))
    for op_def in op_list.op:
      self.add_op(op_def)

  def apply_op(self, op_type_name, name=None, **keywords):
    # pylint: disable=g-doc-args
    """Add a node invoking a registered Op to a graph.

    Example usage:
       # input1 and input2 can be Tensors or anything ops.convert_to_tensor()
       # will convert to a Tensor.
       op_def_library.apply_op("op", input1=input1, input2=input2)
       # Can specify a node name.
       op_def_library.apply_op("op", input1=input1, name="node_name")
       # Must use keyword arguments, with the names specified in the OpDef.
       op_def_library.apply_op("op", input_name=input, attr_name=attr)

    All attrs must either be inferred from an input or specified.
    (If inferred, the attr must not be specified.)  If an attr has a default
    value specified in the Op's OpDef, then you may pass None as the value
    of that attr to get the default.

    Args:
      op_type_name: string. Must match the name field of a registered Op.
      name: string. Optional name of the created op.
      **keywords: input Tensor and attr arguments specified by name,
        and optional parameters to pass when constructing the Operation.

    Returns:
      The Tensor(s) representing the output of the operation, or the Operation
      itself if there are no outputs.

    Raises:
      RuntimeError: On some errors.
      TypeError: On some errors.
      ValueError: On some errors.
    """
    output_structure, is_stateful, op = self._apply_op_helper(
        op_type_name, name, **keywords)
    if output_structure:
      outputs = op.outputs
      res = _Restructure(ops.convert_n_to_tensor(outputs), output_structure)
      if isinstance(res, list) and not res and is_stateful:
        return op
      else:
        return res
    else:
      return op

  def _apply_op_helper(self, op_type_name, name=None, **keywords):
    """Implementation of apply_op that returns output_structure, op."""
    op_info = self._ops.get(op_type_name, None)
    if op_info is None:
      raise RuntimeError("Unrecognized Op name " + op_type_name)
    op_def = op_info.op_def

    # Determine the graph context.
    try:
      # Need to flatten all the arguments into a list.
      # pylint: disable=protected-access
      g = ops._get_graph_from_inputs(_Flatten(keywords.values()))
      # pylint: enable=protected-access
    except AssertionError as e:
      raise RuntimeError(
          "Cannot determine graph for Op '%s' due to: %s"
          % (op_type_name, e.message))

    # Default name if not specified.
    if name is None:
      name = op_type_name

    # Check for deprecation
    deprecation_version = op_def.deprecation.version
    if deprecation_version:
      producer = g.graph_def_versions.producer
      if producer >= deprecation_version:
        raise NotImplementedError(
            ("Op %s is not available in GraphDef version %d. "
             "It has been removed in version %d. %s.") %
            (op_type_name, producer, deprecation_version,
             op_def.deprecation.explanation))

    # Fill in the list of default types for all "type" attrs.  This
    # will be used to choose a preferred dtype to convert to in the
    # absence of input type information.
    #
    # TODO(b/31302892): Currently the defaults don't work in the right
    # way if you have two inputs, one of whose type resolution depends
    # on the other.  Handling this will require restructuring this code
    # significantly.
    default_type_attr_map = {}
    for attr_def in op_def.attr:
      if attr_def.type != "type":
        continue
      key = attr_def.name
      if attr_def.HasField("default_value"):
        default_type_attr_map[key] = dtypes.as_dtype(
            attr_def.default_value.type)

    # Requires that op_def has passed validation (using the C++
    # ValidateOpDef() from ../framework/op_def_util.h).
    attrs = {}
    inputs = []
    input_types = []
    with g.as_default(), ops.name_scope(name) as scope:

      # Perform input type inference
      inferred_from = {}
      for input_arg in op_def.input_arg:
        input_name = input_arg.name
        if input_name in keywords:
          values = keywords.pop(input_name)
        elif input_name + "_" in keywords:
          # Handle the case where the name is a keyword or built-in
          # for Python so we use the name + _ instead.
          input_name += "_"
          values = keywords.pop(input_name)
        else:
          raise TypeError("No argument for input " + input_name)

        # Goals:
        # * Convert values to Tensors if it contains constants.
        # * Verify that values is a list if that matches the input_arg's
        #   type.
        # * If the input_arg's type is determined by attrs, either set
        #   those attrs and validate those attr values are legal (if
        #   they have not yet been set) or validate the input matches
        #   the type indicated by the attrs (if they have already been
        #   inferred via an earlier input).
        # * If the input_arg has an explicit type, make sure the input
        #   conforms.

        if _IsListParameter(input_arg):
          if not _IsListValue(values):
            raise TypeError(
                "Expected list for '%s' argument to '%s' Op, not %s." %
                (input_name, op_type_name, values))
          # In cases where we expect all elements of the list to have the
          # same dtype, try to cast non-Tensor elements to that type.
          dtype = None
          default_dtype = None
          if input_arg.type != types_pb2.DT_INVALID:
            dtype = input_arg.type
          elif input_arg.number_attr:
            if input_arg.type_attr in attrs:
              dtype = attrs[input_arg.type_attr]
            else:
              for t in values:
                if isinstance(t, ops.Tensor):
                  dtype = t.dtype
                  break

            # dtype still not found, prefer using the default dtype
            # from the attr.
            if dtype is None and input_arg.type_attr in default_type_attr_map:
              default_dtype = default_type_attr_map[input_arg.type_attr]

          try:
            if not input_arg.is_ref and dtype:
              dtype = dtypes.as_dtype(dtype).base_dtype
            values = ops.internal_convert_n_to_tensor(
                values,
                name=input_arg.name,
                dtype=dtype if dtype else None,
                preferred_dtype=default_dtype,
                as_ref=input_arg.is_ref)
            if input_arg.number_attr and len(
                set(v.dtype.base_dtype for v in values)) > 1:
              raise TypeError()  # All types should match.
          except (TypeError, ValueError):
            # What types does the conversion function think values have?
            observed_types = []
            for value in values:
              try:
                converted_value = ops.internal_convert_to_tensor(
                    value, as_ref=input_arg.is_ref)
                observed_types.append(converted_value.dtype.base_dtype.name)
              except (TypeError, ValueError):
                observed_types.append("<NOT CONVERTIBLE TO TENSOR>")
            observed = ", ".join(observed_types)

            prefix = (
                "Tensors in list passed to '%s' of '%s' Op have types [%s]" %
                (input_name, op_type_name, observed))
            if input_arg.number_attr:
              if input_arg.type != types_pb2.DT_INVALID:
                raise TypeError("%s that do not match expected type %s." %
                                (prefix, dtype.name))
              elif input_arg.type_attr in attrs:
                raise TypeError("%s that do not match type %s inferred from "
                                "earlier arguments." %
                                (prefix, dtype.name))
              else:
                raise TypeError("%s that don't all match." % prefix)
            else:
              raise TypeError(
                  "%s that are invalid. Tensors: %s" % (prefix, values))

          types = [x.dtype for x in values]
          inputs.extend(values)
        else:
          # In cases where we have an expected type, try to convert non-Tensor
          # arguments to that type.
          dtype = None
          default_dtype = None
          if input_arg.type != types_pb2.DT_INVALID:
            dtype = input_arg.type
          elif input_arg.type_attr in attrs:
            dtype = attrs[input_arg.type_attr]
          elif input_arg.type_attr in default_type_attr_map:
            # The dtype could not be inferred solely from the inputs,
            # so we prefer the attr's default, so code that adds a new attr
            # with a default is backwards compatible.
            default_dtype = default_type_attr_map[input_arg.type_attr]

          try:
            values = ops.internal_convert_to_tensor(
                values,
                name=input_arg.name,
                dtype=dtype,
                as_ref=input_arg.is_ref,
                preferred_dtype=default_dtype)
          except TypeError as err:
            if dtype is None:
              raise err
            else:
              raise TypeError(
                  "Expected %s passed to parameter '%s' of op '%s', got %s of "
                  "type '%s' instead." %
                  (dtypes.as_dtype(dtype).name, input_arg.name, op_type_name,
                   repr(values), type(values).__name__))
          except ValueError:
            # What type does convert_to_tensor think it has?
            try:
              observed = ops.internal_convert_to_tensor(
                  values, as_ref=input_arg.is_ref).dtype.name
            except ValueError as err:
              raise ValueError(
                  "Tried to convert '%s' to a tensor and failed. Error: %s" %
                  (input_name, err))
            prefix = ("Input '%s' of '%s' Op has type %s that does not match" %
                      (input_name, op_type_name, observed))
            if input_arg.type != types_pb2.DT_INVALID:
              raise TypeError("%s expected type of %s." %
                              (prefix, dtypes.as_dtype(input_arg.type).name))
            else:
              # Update the maps with the default, if needed.
              k = input_arg.type_attr
              if k in default_type_attr_map:
                if k not in attrs:
                  attrs[k] = default_type_attr_map[k]
                  if k not in inferred_from:
                    inferred_from[k] = "Default in OpDef"

              raise TypeError(
                  "%s type %s of argument '%s'." %
                  (prefix, dtypes.as_dtype(attrs[input_arg.type_attr]).name,
                   inferred_from[input_arg.type_attr]))

          types = [values.dtype]
          inputs.append(values)
        base_types = [x.base_dtype for x in types]

        if input_arg.number_attr:
          # <number-attr> * <type> or <number-attr> * <type-attr>
          if input_arg.number_attr in attrs:
            if len(values) != attrs[input_arg.number_attr]:
              raise ValueError(
                  "List argument '%s' to '%s' Op with length %d must match "
                  "length %d of argument '%s'." %
                  (input_name, op_type_name, len(values),
                   attrs[input_arg.number_attr],
                   inferred_from[input_arg.number_attr]))
          else:
            attrs[input_arg.number_attr] = len(values)
            inferred_from[input_arg.number_attr] = input_name
            num_attr = _Attr(op_def, input_arg.number_attr)
            if num_attr.has_minimum and len(values) < num_attr.minimum:
              raise ValueError(
                  "List argument '%s' to '%s' Op with length %d shorter "
                  "than minimum length %d." %
                  (input_name, op_type_name, len(values), num_attr.minimum))
          # All tensors must have the same base type.
          if any(bt != base_types[0] for bt in base_types):
            raise TypeError(
                "All tensors passed to '%s' of '%s' Op "
                "must have the same type." %
                (input_name, op_type_name))
          if input_arg.type != types_pb2.DT_INVALID:
            # <number-attr> * <type> case
            if base_types and base_types[0] != input_arg.type:
              assert False, "Unreachable"
          elif input_arg.type_attr in attrs:
            # <number-attr> * <type-attr> case, where <type-attr> already
            # has an inferred value.
            if base_types and base_types[0] != attrs[input_arg.type_attr]:
              assert False, "Unreachable"
          else:
            # <number-attr> * <type-attr> case, where we are now setting
            # the <type-attr> based on this input
            if not base_types:
              raise TypeError(
                  "Don't know how to infer type variable from empty input "
                  "list passed to input '%s' of '%s' Op." %
                  (input_name, op_type_name))
            attrs[input_arg.type_attr] = base_types[0]
            inferred_from[input_arg.type_attr] = input_name
            type_attr = _Attr(op_def, input_arg.type_attr)
            _SatisfiesTypeConstraint(base_types[0], type_attr,
                                     param_name=input_name)
        elif input_arg.type_attr:
          # <type-attr>
          attr_value = base_types[0]
          if input_arg.type_attr in attrs:
            if attrs[input_arg.type_attr] != attr_value:
              assert False, "Unreachable"
          else:
            for base_type in base_types:
              _SatisfiesTypeConstraint(base_type,
                                       _Attr(op_def, input_arg.type_attr),
                                       param_name=input_name)
            attrs[input_arg.type_attr] = attr_value
            inferred_from[input_arg.type_attr] = input_name
        elif input_arg.type_list_attr:
          # <type-list-attr>
          attr_value = base_types
          if input_arg.type_list_attr in attrs:
            if attrs[input_arg.type_list_attr] != attr_value:
              raise TypeError(
                  "Input '%s' of '%s' Op has type list of %s that does not "
                  "match type list %s of argument '%s'." %
                  (input_name, op_type_name,
                   ", ".join(dtypes.as_dtype(x).name for x in attr_value),
                   ", ".join(dtypes.as_dtype(x).name
                             for x in attrs[input_arg.type_list_attr]),
                   inferred_from[input_arg.type_list_attr]))
          else:
            for base_type in base_types:
              _SatisfiesTypeConstraint(base_type,
                                       _Attr(op_def, input_arg.type_list_attr),
                                       param_name=input_name)
            attrs[input_arg.type_list_attr] = attr_value
            inferred_from[input_arg.type_list_attr] = input_name
        else:
          # single Tensor with specified type
          if base_types[0] != input_arg.type:
            assert False, "Unreachable"

        if input_arg.is_ref:
          if not all(x._is_ref_dtype for x in types):  # pylint: disable=protected-access
            raise TypeError(
                ("'%s' Op requires that input '%s' be a mutable tensor "
                 "(e.g.: a tf.Variable)") % (op_type_name, input_name))
          input_types.extend(types)
        else:
          input_types.extend(base_types)

      # Process remaining attrs
      for attr in op_def.attr:
        # Skip attrs that have already had their values inferred
        if attr.name in attrs:
          if attr.name in keywords:
            raise TypeError(
                "Should not specify value for inferred attr '%s'." % attr.name)
          continue
        if attr.name in keywords:
          attrs[attr.name] = keywords.pop(attr.name)
        elif attr.name + "_" in keywords:
          # Attrs whose names match Python keywords have an extra '_'
          # appended, so we must check for that as well.
          attrs[attr.name] = keywords.pop(attr.name + "_")
        else:
          raise TypeError("No argument for attr " + attr.name)

      # Convert attr values to AttrValue protos.
      attr_protos = {}
      for attr_def in op_def.attr:
        key = attr_def.name
        value = attrs[key]
        attr_value = attr_value_pb2.AttrValue()
        if attr_def.HasField("default_value") and value is None:
          attr_value.CopyFrom(attr_def.default_value)
          attr_protos[key] = attr_value
          continue
        if attr_def.type.startswith("list("):
          if not _IsListValue(value):
            raise TypeError("Expected list for attr " + key)
          if attr_def.has_minimum:
            if len(value) < attr_def.minimum:
              raise ValueError("Attr '%s' of '%s' Op passed list of length %d "
                               "less than minimum %d." %
                               (key, op_type_name, len(value),
                                attr_def.minimum))
          attr_value.list.SetInParent()
        if attr_def.type == "string":
          attr_value.s = _MakeStr(value, key)
          if attr_def.HasField("allowed_values"):
            if attr_value.s not in attr_def.allowed_values.list.s:
              raise ValueError(
                  "Attr '%s' of '%s' Op passed string '%s' not in: \"%s\"." %
                  (key, op_type_name, compat.as_text(attr_value.s),
                   '", "'.join(map(compat.as_text,
                                   attr_def.allowed_values.list.s))))
        elif attr_def.type == "list(string)":
          attr_value.list.s.extend([_MakeStr(x, key) for x in value])
          if attr_def.HasField("allowed_values"):
            for x in attr_value.list.s:
              if x not in attr_def.allowed_values.list.s:
                raise ValueError(
                    "Attr '%s' of '%s' Op passed string '%s' not in: \"%s\"." %
                    (key, op_type_name, compat.as_text(x),
                     '", "'.join(map(compat.as_text,
                                     attr_def.allowed_values.list.s))))
        elif attr_def.type == "int":
          attr_value.i = _MakeInt(value, key)
          if attr_def.has_minimum:
            if attr_value.i < attr_def.minimum:
              raise ValueError(
                  "Attr '%s' of '%s' Op passed %d less than minimum %d." %
                  (key, op_type_name, attr_value.i, attr_def.minimum))
        elif attr_def.type == "list(int)":
          attr_value.list.i.extend([_MakeInt(x, key) for x in value])
        elif attr_def.type == "float":
          attr_value.f = _MakeFloat(value, key)
        elif attr_def.type == "list(float)":
          attr_value.list.f.extend([_MakeFloat(x, key) for x in value])
        elif attr_def.type == "bool":
          attr_value.b = _MakeBool(value, key)
        elif attr_def.type == "list(bool)":
          attr_value.list.b.extend([_MakeBool(x, key) for x in value])
        elif attr_def.type == "type":
          attr_value.type = _MakeType(value, attr_def)
        elif attr_def.type == "list(type)":
          attr_value.list.type.extend(
              [_MakeType(x, attr_def) for x in value])
        elif attr_def.type == "shape":
          attr_value.shape.CopyFrom(_MakeShape(value, key))
        elif attr_def.type == "list(shape)":
          attr_value.list.shape.extend(
              [_MakeShape(x, key) for x in value])
        elif attr_def.type == "tensor":
          attr_value.tensor.CopyFrom(_MakeTensor(value, key))
        elif attr_def.type == "list(tensor)":
          attr_value.list.tensor.extend(
              [_MakeTensor(x, key) for x in value])
        elif attr_def.type == "func":
          if isinstance(value, attr_value_pb2.NameAttrList):
            attr_value.func.CopyFrom(value)
          elif isinstance(value, compat.bytes_or_text_types):
            attr_value.func.name = value
          else:
            value.add_to_graph(ops.get_default_graph())
            attr_value.func.name = value.name
        else:
          raise TypeError("Unrecognized Attr type " + attr_def.type)

        attr_protos[key] = attr_value
      del attrs  # attrs is no longer authoritative, use attr_protos instead

      # Determine output types (possibly using attrs)
      output_types = []
      output_structure = []
      for arg in op_def.output_arg:
        types = []
        if arg.number_attr:
          n = _AttrValue(attr_protos, arg.number_attr).i
          if arg.type_attr:
            types = [_AttrValue(attr_protos, arg.type_attr).type] * n
          else:
            types = [arg.type] * n
          output_structure.append(n)
        elif arg.type_attr:
          t = _AttrValue(attr_protos, arg.type_attr)
          types = [t.type]
          output_structure.append(None)
        elif arg.type_list_attr:
          t = _AttrValue(attr_protos, arg.type_list_attr)
          types = t.list.type
          output_structure.append(len(types))
        else:
          types = [arg.type]
          output_structure.append(None)
        if arg.is_ref:
          types = [dtypes.as_dtype(x)._as_ref for x in types]  # pylint: disable=protected-access
        output_types.extend(types)

      if keywords:
        raise TypeError("apply_op() got unexpected keyword arguments: " +
                        ", ".join(sorted(keywords.keys())))

      # NOTE(mrry): We add an explicit colocation constraint between
      # the newly created op and any of its reference-typed inputs.
      must_colocate_inputs = [val for arg, val in zip(op_def.input_arg, inputs)
                              if arg.is_ref]
      with _MaybeColocateWith(must_colocate_inputs):
        # Add Op to graph
        op = g.create_op(op_type_name, inputs, output_types, name=scope,
                         input_types=input_types, attrs=attr_protos,
                         op_def=op_def)
      return output_structure, op_def.is_stateful, op

# pylint: enable=invalid-name
