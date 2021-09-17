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

from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib


def _Attr(op_def, name):
  for attr in op_def.attr:
    if attr.name == name:
      return attr
  raise TypeError(f"Inconsistent OpDef for '{op_def.name}', missing attr "
                  f"'{name}'")


def _AttrValue(attr_protos, name, op_type_name):
  if name in attr_protos:
    return attr_protos[name]
  raise TypeError(f"Inconsistent OpDef for '{op_type_name}', missing attr "
                  f"'{name}' from '{attr_protos}'.")


def _SatisfiesTypeConstraint(dtype, attr_def, param_name):
  if attr_def.HasField("allowed_values"):
    allowed_list = attr_def.allowed_values.list.type
    allowed_values = ", ".join(dtypes.as_dtype(x).name for x in allowed_list)
    if dtype not in allowed_list:
      raise TypeError(
          f"Value passed to parameter '{param_name}' has DataType "
          f"{dtypes.as_dtype(dtype).name} not in list of allowed values: "
          f"{allowed_values}")


def _SatisfiesLengthConstraint(length, attr_def, param_name, op_type_name):
  if attr_def.has_minimum and length < attr_def.minimum:
    raise ValueError(f"Attr '{param_name}' of '{op_type_name}' Op passed list "
                     f"of length {length} less than minimum "
                     f"{attr_def.minimum}.")


def _SatisfiesAllowedStringsConstraint(value, attr_def, arg_name, op_type_name):
  if value not in attr_def.allowed_values.list.s:
    allowed_values = '", "'.join(
        map(compat.as_text, attr_def.allowed_values.list.s))
    raise ValueError(f"Attr '{arg_name}' of '{op_type_name}' Op passed string "
                     f"'{compat.as_text(value)}' not in: \"{allowed_values}\".")


def _SatisfiesIntMinimumConstraint(value, attr_def, arg_name, op_type_name):
  if value < attr_def.minimum:
    raise ValueError(f"Attr '{arg_name}' of '{op_type_name}' Op passed {value} "
                     f"less than minimum {attr_def.minimum}.")


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
    raise TypeError(f"Expected float for argument '{arg_name}' not {repr(v)}.")
  return float(v)


def _MakeInt(v, arg_name):
  if isinstance(v, six.string_types):
    raise TypeError(f"Expected int for argument '{arg_name}' not {repr(v)}.")
  try:
    return int(v)
  except (ValueError, TypeError):
    raise TypeError(f"Expected int for argument '{arg_name}' not {repr(v)}.")


def _MakeStr(v, arg_name):
  if not isinstance(v, compat.bytes_or_text_types):
    raise TypeError(f"Expected string for argument '{arg_name}' not {repr(v)}.")
  return compat.as_bytes(v)  # Convert unicode strings to bytes.


def _MakeBool(v, arg_name):
  if not isinstance(v, bool):
    raise TypeError(f"Expected bool for argument '{arg_name}' not {repr(v)}.")
  return v


def _MakeType(v, arg_name):
  try:
    v = dtypes.as_dtype(v).base_dtype
  except TypeError:
    raise TypeError(f"Expected DataType for argument '{arg_name}' not "
                    f"{repr(v)}.")
  return v.as_datatype_enum


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
    raise TypeError(f"Error converting {repr(v)} (arg name = {arg_name}) to a "
                    f"TensorShape: {e}")
  except ValueError as e:
    raise TypeError(f"Error converting {repr(v)} (arg name = {arg_name}) to a "
                    f"TensorShape: {e}")


def _MakeTensor(v, arg_name):
  """Ensure v is a TensorProto."""
  if isinstance(v, tensor_pb2.TensorProto):
    return v
  raise TypeError(
      f"Don't know how to convert {repr(v)} to a TensorProto for argument "
      f"'{arg_name}'")


def _MakeFunc(v, arg_name):
  """Ensure v is a func."""
  if isinstance(v, attr_value_pb2.NameAttrList):
    return v
  if isinstance(v, compat.bytes_or_text_types):
    fn_attr = attr_value_pb2.NameAttrList(name=v)
  elif hasattr(v, "add_to_graph"):
    v.add_to_graph(ops.get_default_graph())
    if hasattr(v, "_as_name_attr_list"):
      fn_attr = v._as_name_attr_list  # pylint: disable=protected-access
    else:
      fn_attr = attr_value_pb2.NameAttrList(name=v.name)
  else:
    raise TypeError(f"Don't know how to convert {repr(v)} to a func for "
                    f"argument {arg_name}")
  return fn_attr


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


def apply_op(op_type_name, name=None, **keywords):  # pylint: disable=invalid-name
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
  output_structure, is_stateful, op, outputs = _apply_op_helper(
      op_type_name, name, **keywords)
  if output_structure:
    res = _Restructure(ops.convert_n_to_tensor(outputs), output_structure)
    if isinstance(res, list) and not res and is_stateful:
      return op
    else:
      return res
  else:
    return op


def _apply_op_helper(op_type_name, name=None, **keywords):  # pylint: disable=invalid-name
  """Implementation of apply_op that returns output_structure, op."""
  op_def = op_def_registry.get(op_type_name)
  if op_def is None:
    raise RuntimeError(f"Unrecognized Op name {op_type_name}")

  # Determine the graph context.
  try:
    # Need to flatten all the arguments into a list.
    # pylint: disable=protected-access
    g = ops._get_graph_from_inputs(_Flatten(keywords.values()))
    # pylint: enable=protected-access
  except AssertionError as e:
    raise RuntimeError(
        f"Cannot determine graph for Op '{op_type_name}' due to: {e.message}")

  # Default name if not specified.
  if name is None:
    name = op_type_name

  # Check for deprecation
  deprecation_version = op_def.deprecation.version
  if deprecation_version:
    producer = g.graph_def_versions.producer
    if producer >= deprecation_version:
      raise NotImplementedError(
          f"Op {op_type_name} is not available in GraphDef version {producer}. "
          f"It has been removed in version {deprecation_version}. "
          f"{op_def.deprecation.explanation}.")

  # Fill in the list of default types for all "type" attrs.  This
  # will be used to choose a preferred dtype to convert to in the
  # absence of input type information.
  #
  # TODO(b/31302892): Currently the defaults don't work in the right
  # way if you have two inputs, one of whose type resolution depends
  # on the other.  Handling this will require restructuring this code
  # significantly.
  default_type_attr_map = {}
  allowed_list_attr_map = {}
  for attr_def in op_def.attr:
    if attr_def.type != "type":
      continue
    key = attr_def.name
    if attr_def.HasField("default_value"):
      default_type_attr_map[key] = dtypes.as_dtype(
          attr_def.default_value.type)
    if attr_def.HasField("allowed_values"):
      allowed_list_attr_map[key] = attr_def.allowed_values.list.type

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
        raise TypeError(f"No argument for input {input_name} found in {op_def}")

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
              f"Expected list for '{input_name}' argument to '{op_type_name}' "
              f"Op, not {values}.")
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
          all_types = set(v.dtype.base_dtype for v in values)
          if input_arg.number_attr and len(all_types) > 1:
            # All types should match.
            raise TypeError(f"Not all types matched for {input_arg.name} for "
                            f"{op_type_name}. Got {all_types}")
        except (TypeError, ValueError):
          # What types does the conversion function think values have?
          observed_types = []
          for value in values:
            try:
              converted_value = ops.convert_to_tensor(
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
              raise TypeError(f"{prefix} that do not match expected type "
                              f"{dtype.name}.")
            elif input_arg.type_attr in attrs:
              raise TypeError(f"{prefix} that do not match type {dtype.name} "
                              "inferred from earlier arguments.")
            else:
              raise TypeError(f"{prefix} that don't all match.")
          else:
            raise TypeError(f"{prefix} that are invalid. Tensors: {values}")

        types = [x.dtype for x in values]
        inputs.extend(values)
      else:
        # In cases where we have an expected type, try to convert non-Tensor
        # arguments to that type.
        dtype = None
        default_dtype = None
        allowed_list = None
        if input_arg.type != types_pb2.DT_INVALID:
          dtype = input_arg.type
        elif input_arg.type_attr in attrs:
          dtype = attrs[input_arg.type_attr]
        elif input_arg.type_attr in default_type_attr_map:
          # The dtype could not be inferred solely from the inputs,
          # so we prefer the attr's default, so code that adds a new attr
          # with a default is backwards compatible.
          default_dtype = default_type_attr_map[input_arg.type_attr]
          allowed_list = allowed_list_attr_map.get(input_arg.type_attr)

        try:
          # First see if we can get a valid dtype with the default conversion
          # and see if it matches an allowed dtypes. Some ops like ConcatV2 may
          # not list allowed dtypes, in which case we should skip this.
          if dtype is None and allowed_list:
            inferred = None
            try:
              inferred = ops.convert_to_tensor(
                  values, name=input_arg.name, as_ref=input_arg.is_ref)
            except TypeError as err:
              # When converting a python object such as a list of Dimensions, we
              # need a dtype to be specified, thus tensor conversion may throw
              # an exception which we will ignore and try again below.
              pass

            # If we did not match an allowed dtype, try again with the default
            # dtype. This could be because we have an empty tensor and thus we
            # picked the wrong type.
            if inferred is not None and inferred.dtype in allowed_list:
              values = inferred
            else:
              values = ops.convert_to_tensor(
                  values,
                  name=input_arg.name,
                  as_ref=input_arg.is_ref,
                  preferred_dtype=default_dtype)
          else:
            values = ops.convert_to_tensor(
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
                f"Expected {dtypes.as_dtype(dtype).name} passed to parameter "
                f"'{input_arg.name}' of op '{op_type_name}', got "
                f"{repr(values)} of type '{type(values).__name__}' instead. "
                f"Error: {err}")
        except ValueError:
          # What type does convert_to_tensor think it has?
          try:
            observed = ops.convert_to_tensor(
                values, as_ref=input_arg.is_ref).dtype.name
          except ValueError as err:
            raise ValueError(
                f"Tried to convert '{input_name}' to a tensor and failed. "
                f"Error: {err}")
          prefix = ("Input '%s' of '%s' Op has type %s that does not match" %
                    (input_name, op_type_name, observed))
          if input_arg.type != types_pb2.DT_INVALID:
            raise TypeError(f"{prefix} expected type of "
                            f"{dtypes.as_dtype(input_arg.type).name}.")
          else:
            # Update the maps with the default, if needed.
            k = input_arg.type_attr
            if k in default_type_attr_map:
              if k not in attrs:
                attrs[k] = default_type_attr_map[k]
                if k not in inferred_from:
                  inferred_from[k] = "Default in OpDef"

            raise TypeError(
                f"{prefix} type "
                f"{dtypes.as_dtype(attrs[input_arg.type_attr]).name} of "
                f"argument '{inferred_from[input_arg.type_attr]}'.")

        types = [values.dtype]
        inputs.append(values)
      base_types = [x.base_dtype for x in types]

      if input_arg.number_attr:
        # <number-attr> * <type> or <number-attr> * <type-attr>
        if input_arg.number_attr in attrs:
          if len(values) != attrs[input_arg.number_attr]:
            raise ValueError(
                f"List argument '{input_name}' to '{op_type_name}' Op with "
                f"length {len(values)} must match length "
                f"{attrs[input_arg.number_attr]} of argument "
                f"'{inferred_from[input_arg.number_attr]}'.")
        else:
          attrs[input_arg.number_attr] = len(values)
          inferred_from[input_arg.number_attr] = input_name
          num_attr = _Attr(op_def, input_arg.number_attr)
          if num_attr.has_minimum and len(values) < num_attr.minimum:
            raise ValueError(
                f"List argument '{input_name}' to '{op_type_name}' Op with "
                f"length {len(values)} shorter than minimum length "
                f"{num_attr.minimum}.")
        # All tensors must have the same base type.
        if any(bt != base_types[0] for bt in base_types):
          raise TypeError(
              f"All tensors passed to '{input_name}' of '{op_type_name}' Op "
              f"must have the same type. Got {base_types} instead.")
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
            # If it's in default_type_attr_map, then wait to set it
            # (in "process remaining attrs", below).
            if input_arg.type_attr not in default_type_attr_map:
              raise TypeError(
                  "Don't know how to infer type variable from empty input "
                  f"list passed to input '{input_name}' of '{op_type_name}' "
                  "Op.")
          else:
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
            raise TypeError(
                f"Input '{input_name}' of '{op_type_name}' Op has type "
                f"{dtypes.as_dtype(attr_value).name} that does not match type "
                f"{dtypes.as_dtype(attrs[input_arg.type_attr]).name} of "
                f"argument '{inferred_from[input_arg.type_attr]}'.")
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
            actual_types = ", ".join(
                dtypes.as_dtype(x).name for x in attr_value)
            expected_types = ", ".join(
                dtypes.as_dtype(x).name
                for x in attrs[input_arg.type_list_attr])
            raise TypeError(
                f"Input '{input_name}' of '{op_type_name}' Op has type list of "
                f"{actual_types} that does not match type list {expected_types}"
                f" of argument '{inferred_from[input_arg.type_list_attr]}'.")
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
              f"'{op_type_name}' Op requires that input '{input_name}' be a "
              "mutable tensor (e.g.: a tf.Variable)")
        input_types.extend(types)
      else:
        input_types.extend(base_types)

    # Process remaining attrs
    for attr in op_def.attr:
      # Skip attrs that have already had their values inferred
      if attr.name in attrs:
        if attr.name in keywords:
          raise TypeError(
              f"Should not specify value for inferred attr '{attr.name}' for "
              f"{op_type_name}.")
        continue
      if attr.name in keywords:
        attrs[attr.name] = keywords.pop(attr.name)
      elif attr.name + "_" in keywords:
        # Attrs whose names match Python keywords have an extra '_'
        # appended, so we must check for that as well.
        attrs[attr.name] = keywords.pop(attr.name + "_")
      elif attr.name in default_type_attr_map:
        attrs[attr.name] = default_type_attr_map[attr.name]
        inferred_from.setdefault(attr.name, "Default in OpDef")
      else:
        raise TypeError(f"No argument found for attr {attr.name} for "
                        f"{op_type_name}")

    # Convert attr values to AttrValue protos.
    attr_protos = {}
    for attr_def in op_def.attr:
      key = attr_def.name
      value = attrs[key]

      if attr_def.HasField("default_value") and value is None:
        attr_value = attr_value_pb2.AttrValue()
        attr_value.CopyFrom(attr_def.default_value)
        attr_protos[key] = attr_value
        continue

      attr_value = value_to_attr_value(value, attr_def.type, key)
      if attr_def.type.startswith("list("):
        _SatisfiesLengthConstraint(len(value), attr_def, key, op_type_name)
      if attr_def.HasField("allowed_values"):
        if attr_def.type == "string":
          _SatisfiesAllowedStringsConstraint(attr_value.s, attr_def, key,
                                             op_type_name)
        elif attr_def.type == "list(string)":
          for value in attr_value.list.s:
            _SatisfiesAllowedStringsConstraint(value, attr_def, key,
                                               op_type_name)
      if attr_def.has_minimum and attr_def.type == "int":
        _SatisfiesIntMinimumConstraint(attr_value.i, attr_def, key,
                                       op_type_name)
      if attr_def.type == "type":
        _SatisfiesTypeConstraint(attr_value.type, attr_def, key)
      if attr_def.type == "list(type)":
        for value in attr_value.list.type:
          _SatisfiesTypeConstraint(value, attr_def, key)

      attr_protos[key] = attr_value
    del attrs  # attrs is no longer authoritative, use attr_protos instead

    # Determine output types (possibly using attrs)
    output_structure = []
    for arg in op_def.output_arg:
      if arg.number_attr:
        n = _AttrValue(attr_protos, arg.number_attr, op_type_name).i
        output_structure.append(n)
      elif arg.type_attr:
        t = _AttrValue(attr_protos, arg.type_attr, op_type_name)
        output_structure.append(None)
      elif arg.type_list_attr:
        t = _AttrValue(attr_protos, arg.type_list_attr, op_type_name)
        output_structure.append(len(t.list.type))
      else:
        output_structure.append(None)

    if keywords:
      all_keywords = ", ".join(sorted(keywords.keys()))
      raise TypeError(f"{op_type_name} got unexpected keyword arguments: "
                      f"{all_keywords}.")

    # NOTE(mrry): We add an explicit colocation constraint between
    # the newly created op and any of its reference-typed inputs.
    must_colocate_inputs = [val for arg, val in zip(op_def.input_arg, inputs)
                            if arg.is_ref]
    with _MaybeColocateWith(must_colocate_inputs):
      # Add Op to graph
      # pylint: disable=protected-access
      op = g._create_op_internal(op_type_name, inputs, dtypes=None,
                                 name=scope, input_types=input_types,
                                 attrs=attr_protos, op_def=op_def)

    # `outputs` is returned as a separate return value so that the output
    # tensors can the `op` per se can be decoupled so that the
    # `op_callbacks` can function properly. See framework/op_callbacks.py
    # for more details.
    outputs = op.outputs
    # Conditionally invoke tfdbg v2's op callback(s).
    if op_callbacks.should_invoke_op_callbacks():
      callback_outputs = op_callbacks.invoke_op_callbacks(
          op.node_def.op, tuple(op.inputs), attr_protos, tuple(outputs),
          op_name=op.name, graph=g)
      if callback_outputs is not None:
        outputs = callback_outputs

    return output_structure, op_def.is_stateful, op, outputs


def value_to_attr_value(value, attr_type, arg_name):  # pylint: disable=invalid-name
  """Encodes a Python value as an `AttrValue` proto message.

  Args:
    value: The value to convert.
    attr_type: The value type (string) -- see the AttrValue proto definition for
      valid strings.
    arg_name: Argument name (for error messages).

  Returns:
    An AttrValue proto message that encodes `value`.
  """
  attr_value = attr_value_pb2.AttrValue()

  if attr_type.startswith("list("):
    if not _IsListValue(value):
      raise TypeError(f"Expected list for attr {arg_name}, obtained "
                      f"{type(value).__name__} instead.")

  if attr_type == "string":
    attr_value.s = _MakeStr(value, arg_name)
  elif attr_type == "list(string)":
    attr_value.list.s.extend([_MakeStr(x, arg_name) for x in value])
  elif attr_type == "int":
    attr_value.i = _MakeInt(value, arg_name)
  elif attr_type == "list(int)":
    attr_value.list.i.extend([_MakeInt(x, arg_name) for x in value])
  elif attr_type == "float":
    attr_value.f = _MakeFloat(value, arg_name)
  elif attr_type == "list(float)":
    attr_value.list.f.extend([_MakeFloat(x, arg_name) for x in value])
  elif attr_type == "bool":
    attr_value.b = _MakeBool(value, arg_name)
  elif attr_type == "list(bool)":
    attr_value.list.b.extend([_MakeBool(x, arg_name) for x in value])
  elif attr_type == "type":
    attr_value.type = _MakeType(value, arg_name)
  elif attr_type == "list(type)":
    attr_value.list.type.extend([_MakeType(x, arg_name) for x in value])
  elif attr_type == "shape":
    attr_value.shape.CopyFrom(_MakeShape(value, arg_name))
  elif attr_type == "list(shape)":
    attr_value.list.shape.extend([_MakeShape(x, arg_name) for x in value])
  elif attr_type == "tensor":
    attr_value.tensor.CopyFrom(_MakeTensor(value, arg_name))
  elif attr_type == "list(tensor)":
    attr_value.list.tensor.extend([_MakeTensor(x, arg_name) for x in value])
  elif attr_type == "func":
    attr_value.func.CopyFrom(_MakeFunc(value, arg_name))
  elif attr_type == "list(func)":
    attr_value.list.func.extend([_MakeFunc(x, arg_name) for x in value])
  else:
    raise TypeError(f"Unrecognized Attr type {attr_type} for {arg_name}.")
  return attr_value


# The following symbols are used by op_def_util.cc.
_pywrap_utils.RegisterPyObject("tf.dtypes.DType", dtypes.DType)
_pywrap_utils.RegisterPyObject("tf.dtypes.as_dtype", dtypes.as_dtype)
_pywrap_utils.RegisterPyObject("tf.TensorShape", tensor_shape.TensorShape)
_pywrap_utils.RegisterPyObject("tf.as_shape", tensor_shape.as_shape)
_pywrap_utils.RegisterPyObject("tf.TensorProto", tensor_pb2.TensorProto)
_pywrap_utils.RegisterPyObject("text_format.Parse", text_format.Parse)
_pywrap_utils.RegisterPyObject("tf.convert_to_tensor", ops.convert_to_tensor)
