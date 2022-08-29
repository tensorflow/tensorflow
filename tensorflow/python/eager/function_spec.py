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
"""Defines an input type specification for tf.function."""

import functools
import itertools
import weakref

import numpy as np
import six

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

# Sentinel value used by with ConcreteFunction's structured signature to
# indicate that a non-tensor parameter should use the value that was
# specified when the concrete function was created.
BOUND_VALUE = object()


# TODO(b/214462107): Clean up and migrate to core/function when unblocked.
class FunctionSpec(object):
  """Specification of how to bind arguments to a function."""

  @classmethod
  def from_function_and_signature(cls, python_function,
                                  input_signature,
                                  is_pure=False,
                                  experimental_follow_type_hints=False,
                                  jit_compile=None):
    """Creates a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
      will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
    _validate_signature(input_signature)
    _validate_python_function(python_function, input_signature)

    fullargspec = tf_inspect.getfullargspec(python_function)
    # Checks if the `fullargspec` contains self or cls as its first argument.
    is_method = tf_inspect.isanytargetmethod(python_function)

    # Treat a wrapped partial function as a special case. For all arguments that
    # were overridden with keywords in the partial:
    #   - remove the corresponding arguments,
    #   - remove the corresponding keywords.
    _, unwrapped = tf_decorator.unwrap(python_function)
    if isinstance(unwrapped, functools.partial):
      # Also consider the Python3 case with kwonlydefaults.
      if fullargspec.defaults or fullargspec.kwonlydefaults:
        new_defaults = fullargspec.defaults
        new_args = fullargspec.args
        if fullargspec.defaults:
          # To be able to canonicalize the function properly, we want to ignore
          # default values that are overridden via a partial kwarg. For example:
          #
          #   def func(a, b, c, d=5, e=7):
          #     return a, b, c, d, e
          #   p_func = tf.function(functools.partial(func, 10, e=9))
          #
          # Here we want to drop from the defaults the parameter `e`. If we
          # forwarded the call to the partial function with a default for `e`
          # we would get an error for passing two values for one parameter.
          #
          # Note that this has a limitation: we can only override parameters at
          # the end of the parameter list.
          #
          # In this case we want to end up with 3 arguments (b, c, d) and 1
          # default value (5). We do this by constructing a mask where 0 stands
          # for a value that was overridden by a partial kwarg. The seemingly
          # complicated logic below does just that - for arguments (b, c, d, e)
          # we would get a mask (1, 1, 1, 0).
          old_args = fullargspec.args
          old_defaults = fullargspec.defaults

          no_default = object()
          num_args_without_defaults = len(old_args) - len(old_defaults)
          left_padding = tuple([no_default] * num_args_without_defaults)

          args_with_defaults = zip(old_args, left_padding + old_defaults)

          # Create a mask where 0 stands for args that had a partial kwarg
          # defined.
          non_keyword_defaults_mask = [
              0 if key in unwrapped.keywords else 1 for key in old_args
          ]
          # Keep only arguments and defaults that were not kwargs of partial.
          new_args_with_defaults = list(
              itertools.compress(args_with_defaults, non_keyword_defaults_mask))
          # Keep all args.
          new_args = [arg for arg, _ in new_args_with_defaults]
          # Keep only real default values.
          new_defaults = [
              default for _, default in new_args_with_defaults
              if default is not no_default
          ]
        fullargspec = tf_inspect.FullArgSpec(
            args=new_args,
            varargs=fullargspec.varargs,
            varkw=fullargspec.varkw,
            defaults=new_defaults,
            kwonlyargs=[],
            kwonlydefaults={},
            annotations=fullargspec.annotations)

    # Get the function's name.  Remove functools.partial wrappers if necessary.
    while isinstance(python_function, functools.partial):
      python_function = python_function.func
    name = getattr(python_function, "__name__", "f")

    return FunctionSpec(
        fullargspec,
        is_method,
        input_signature,
        is_pure=is_pure,
        jit_compile=jit_compile,
        experimental_follow_type_hints=experimental_follow_type_hints,
        name=name)

  def __init__(self,
               fullargspec,
               is_method,
               input_signature,
               is_pure=False,
               experimental_follow_type_hints=False,
               name=None,
               jit_compile=None):
    """Constructs a FunctionSpec describing a python function.

    Args:
      fullargspec: `tf_inspect.FullArgSpec` object describing the function.
      is_method: True if the function is a method.
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
    self._fullargspec = fullargspec
    self._is_method = is_method
    self._is_pure = is_pure
    self._jit_compile = jit_compile
    self._experimental_follow_type_hints = experimental_follow_type_hints

    # TODO(edloper): Include name when serializing for SavedModel?
    self._name = name or "f"

    if self._is_method:
      # Remove `self`: default arguments shouldn't be matched to it.
      # TODO(b/127938157): Should this error out if there is no arg to
      # be removed?
      args = fullargspec.args[1:]
    else:
      args = fullargspec.args

    # A cache mapping from argument name to index, for canonicalizing
    # arguments that are called in a keyword-like fashion.
    self._args_to_indices = {arg: i for i, arg in enumerate(args)}
    self._arg_names = args

    # A cache mapping from arg index to default value, for canonicalization.
    default_values = fullargspec.defaults
    offset = len(args) - len(default_values or [])
    self._arg_indices_to_default_values = {
        offset + index: default
        for index, default in enumerate(default_values or [])
    }
    self._arg_indices_no_default_values = set(range(len(args))) - set(
        self._arg_indices_to_default_values)

    _validate_signature(input_signature)
    if input_signature is None:
      self._input_signature = None
    else:
      self._input_signature = tuple(input_signature)
      self._flat_input_signature = tuple(nest.flatten(input_signature,
                                                      expand_composites=True))
    self.validate_input_signature_with_argspec()

  @property
  def fullargspec(self):
    return self._fullargspec

  @property
  def is_method(self):
    return self._is_method

  @property
  def args_to_indices(self):
    return self._args_to_indices

  @property
  def kwargs_to_include(self):
    return self._kwargs_to_include

  @property
  def input_signature(self):
    return self._input_signature

  @property
  def flat_input_signature(self):
    return self._flat_input_signature

  @property
  def is_pure(self):
    return self._is_pure

  @property
  def jit_compile(self):
    return self._jit_compile

  @property
  def arg_names(self):
    return self._arg_names

  @property
  def vararg_name(self):
    return self._fullargspec.varargs

  @property
  def varkw_name(self):
    return self._fullargspec.varkw

  def signature_summary(self, default_values=False):
    """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    args = list(self._arg_names)
    if default_values:
      for (i, default) in self._arg_indices_to_default_values.items():
        args[i] += "={}".format(default)
    if self._fullargspec.kwonlyargs:
      args.append("*")
      for arg_name in self._fullargspec.kwonlyargs:
        args.append(arg_name)
        if default_values and arg_name in self._fullargspec.kwonlydefaults:
          args[-1] += "={}".format(self._fullargspec.kwonlydefaults[arg_name])
    return f"{self._name}({', '.join(args)})"

  def validate_input_signature_with_argspec(self):
    """Checks the python_function's args to be valid against input_signature."""
    if self.input_signature is not None:
      arglen = len(self.input_signature)
      arg_names_len = len(self.arg_names)
      defaults = self.fullargspec.defaults or ()
      unbound_self_arg = 1 if (not self.is_method and arg_names_len > 0 and
                               self.arg_names[0] == "self") else 0
      if not all(d is BOUND_VALUE for d in defaults):
        default_arg_len = len(defaults)
        required_arg_len = arg_names_len - default_arg_len - unbound_self_arg
        # The input signature must cover all required function arguments.
        if arglen < required_arg_len:
          missing_tensor_specs = self.arg_names[
              arglen:required_arg_len]
          raise TypeError(
              f"The decorated tf.function has {required_arg_len} "
              f"required argument(s), but tf.function was only passed an "
              f"input_signature of length {arglen}. This covers {arglen} "
              f"required argument(s): {self.arg_names[:arglen]}, "
              f"but TensorSpecs are still required for the remaining "
              f"{len(missing_tensor_specs)} argument(s):"
              f" {missing_tensor_specs}.")

  def _convert_annotated_args_to_tensors(self, args, kwargs):
    """Attempts to autobox arguments annotated as tf.Tensor."""
    if self.input_signature is not None:
      return

    args = list(args)
    for i, arg in enumerate(args):
      # See
      # https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
      if i < len(self._fullargspec.args):
        annotation_key = self._fullargspec.args[i]
      else:
        annotation_key = self._fullargspec.varargs
      arg_annotation = self._fullargspec.annotations.get(annotation_key, None)

      # TODO(rahulkamat): Change to TensorLike (here ans below)
      if arg_annotation == ops.Tensor:
        args[i] = _to_tensor_or_tensor_spec(arg)

    for kw, v in kwargs.items():
      if kw in self._fullargspec.kwonlyargs or kw in self._fullargspec.args:
        annotation_key = kw
      else:
        annotation_key = self._fullargspec.varkw
      kwarg_annotation = self._fullargspec.annotations.get(annotation_key, None)
      if kwarg_annotation == ops.Tensor:
        kwargs[kw] = _to_tensor_or_tensor_spec(v)
    return tuple(args), kwargs

  def _validate_inputs(self, flat_inputs):
    """Raises an error if inputs contain illegal values."""
    for inp in flat_inputs:
      # TODO(b/183107079): Allow these once they're handled properly.
      if isinstance(inp, weakref.ref):
        raise ValueError(
            f"weakref input {inp} not supported for function {self._name}")

  def validate_inputs_with_signature(self, args, kwargs):
    """Checks args and kwargs against the specified input_signature."""
    if kwargs:
      raise ValueError("Cannot define a TensorFlow function from a Python "
                       "function with keyword arguments when "
                       "input_signature is provided, got keyword arguments "
                       f"({kwargs}) with input_signature "
                       f"({self.input_signature}).")
    if args:
      # If args are provided, they must match the input signature.
      if not is_same_structure(self.input_signature, args):
        raise ValueError("Structure of Python function inputs does not match "
                         f"input_signature: inputs ({args}), "
                         f"input_signature ({self.input_signature}).")
      flat_inputs = nest.flatten(args, expand_composites=True)
      if any(not isinstance(arg, (ops.Tensor, tensor_spec.DenseSpec,
                                  resource_variable_ops.BaseResourceVariable))
             for arg in flat_inputs):
        raise ValueError("When input_signature is provided, all inputs to "
                         "the Python function must be Tensors, Variables, "
                         "tf.TensorSpec or tf.VariableSpec objects.")
      if any(not spec.is_compatible_with(other)
             for spec, other in zip(self.flat_input_signature, flat_inputs)):
        raise ValueError("Python inputs incompatible with input_signature: "
                         f"inputs ({args}), input_signature "
                         f"({self.input_signature}).")

  def canonicalize_function_inputs(self, args, kwargs):
    """Canonicalizes `args` and `kwargs`.

    Canonicalize the inputs to the Python function using a `FunctionSpec`
    instance. In particular, we parse the varargs and kwargs that the
    original function was called with into a tuple corresponding to the
    Python function's positional (named) arguments and a dictionary
    corresponding to its kwargs.  Missing default arguments are added.

    If this `FunctionSpec` has an input signature, then it is used to convert
    arguments to tensors; otherwise, any inputs containing numpy arrays are
    converted to tensors.

    Additionally, any inputs containing numpy arrays are converted to Tensors.

    Args:
      args: The varargs this object was called with.
      kwargs: The keyword args this function was called with.

    Returns:
      A canonicalized ordering of the inputs, as well as full and filtered
      (Tensors and Variables only) versions of their concatenated flattened
      representations, represented by a tuple in the form (args, kwargs,
      flat_args, filtered_flat_args). Here: `args` is a full list of bound
      arguments, and `kwargs` contains only true keyword arguments, as opposed
      to named arguments called in a keyword-like fashion.

    Raises:
      ValueError: If a keyword in `kwargs` cannot be matched with a positional
        argument when an input signature is specified, or when the inputs
        do not conform to the input signature.
    """
    kwargs = {key: kwargs[key] for key in kwargs}
    if self._is_pure:
      args, kwargs = _convert_variables_to_tensors(args, kwargs)
    if self._experimental_follow_type_hints:
      args, kwargs = self._convert_annotated_args_to_tensors(args, kwargs)
    # Pre-calculate to reduce overhead
    arglen = len(args)
    if self._input_signature is not None:
      if arglen > len(self._input_signature):
        raise TypeError(f"{self.signature_summary()} has an input_signature "
                        f"{self._input_signature} which specifies "
                        f"{len(self._input_signature)} positional arguments, "
                        f"but got {arglen}.")
      for arg in six.iterkeys(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is None:
          raise TypeError(f"{self.signature_summary()} got unexpected keyword "
                          f"argument `{arg}`.")
        if index >= len(self._input_signature):
          raise TypeError(
              f"{self.signature_summary()} got keyword argument `{arg}` that "
              "was not included in input_signature.")

    if not kwargs:
      inputs = args
      if self._arg_indices_to_default_values:
        try:
          inputs += tuple(self._arg_indices_to_default_values[i]
                          for i in range(arglen, len(self._arg_names)))
        except KeyError:
          missing_args = [
              self._arg_names[i]
              for i in range(arglen, len(self._arg_names))
              if i not in self._arg_indices_to_default_values
          ]
          raise TypeError(f"{self.signature_summary()} missing required "
                          f"arguments: {', '.join(missing_args)}.")

      if self._fullargspec.kwonlydefaults:
        kwargs.update(self._fullargspec.kwonlydefaults)
    else:
      # Maps from index of arg to its corresponding value, according to `args`
      # and `kwargs`; seeded with the default values for the named args that
      # aren't in `args`.
      arg_indices_to_values = {
          index: default for index, default in six.iteritems(
              self._arg_indices_to_default_values) if index >= arglen
      }
      consumed_args = []
      missing_arg_indices = self._arg_indices_no_default_values - set(
          range(arglen))
      for arg, value in six.iteritems(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is not None:
          if index < arglen:
            raise TypeError(f"{self.signature_summary()} got two values for "
                            f"{arg!r}.")
          arg_indices_to_values[index] = value
          # These arguments in 'kwargs' might also belong to
          # positional arguments
          missing_arg_indices.discard(index)
          consumed_args.append(arg)
      for arg in consumed_args:
        # After this loop, `kwargs` will only contain keyword_only arguments,
        # and all positional_or_keyword arguments have been moved to `inputs`.
        kwargs.pop(arg)
      inputs = args + _deterministic_dict_values(arg_indices_to_values)
      # Exclude positional args with values
      if missing_arg_indices:
        missing_args = [self._arg_names[i] for i in sorted(missing_arg_indices)]
        if len(missing_args) == 1:
          raise TypeError(f"{self.signature_summary()} missing 1 required "
                          f"argument: {missing_args[0]}.")
        else:
          raise TypeError(f"{self.signature_summary()} missing required "
                          f"arguments: {', '.join(missing_args)}.")

      if kwargs and self._input_signature is not None:
        raise TypeError("Keyword arguments are not supported when "
                        "input_signature is provided. Signature: "
                        f"{self.signature_summary()}. Keyword arguments: "
                        f"{kwargs}.")

      if self._fullargspec.kwonlydefaults:
        for (kwarg, default) in self._fullargspec.kwonlydefaults.items():
          kwargs.setdefault(kwarg, default)

    if self._input_signature is None:
      inputs, flat_inputs, filtered_flat_inputs = _convert_numpy_inputs(inputs)
      kwargs, flat_kwargs, filtered_flat_kwargs = _convert_numpy_inputs(kwargs)
      flat_inputs += flat_kwargs
      filtered_flat_inputs += filtered_flat_kwargs
    else:
      inputs, flat_inputs, filtered_flat_inputs = convert_inputs_to_signature(
          inputs, self._input_signature, self._flat_input_signature)

    self._validate_inputs(flat_inputs)

    return inputs, kwargs, filtered_flat_inputs


def _validate_signature(signature):
  """Checks the input_signature to be valid."""
  if signature is None:
    return

  if not isinstance(signature, (tuple, list)):
    raise TypeError("input_signature must be either a tuple or a list, got "
                    f"{type(signature)}.")

  # TODO(xjun): Allow VariableSpec once we figure out API for de-aliasing.
  variable_specs = _get_variable_specs(signature)
  if variable_specs:
    raise TypeError(
        f"input_signature doesn't support VariableSpec, got {variable_specs}")

  if any(not isinstance(arg, tensor_spec.TensorSpec)
         for arg in nest.flatten(signature, expand_composites=True)):
    bad_args = [arg for arg in nest.flatten(signature, expand_composites=True)
                if not isinstance(arg, tensor_spec.TensorSpec)]
    raise TypeError("input_signature must be a possibly nested sequence of "
                    f"TensorSpec objects, got invalid args {bad_args} with "
                    f"types {list(six.moves.map(type, bad_args))}.")


def _validate_python_function(python_function, input_signature):
  """Checks the python_function to be valid against the input_signature."""
  if not callable(python_function):
    raise TypeError(f"{python_function} is not a callable object.")

  if input_signature is not None:
    fullargspec = tf_inspect.getfullargspec(python_function)
    if set(fullargspec.kwonlyargs) - set(fullargspec.kwonlydefaults or ()):
      nodefault_kwonlyargs = set(fullargspec.kwonlyargs)
      if fullargspec.kwonlydefaults is not None:
        nodefault_kwonlyargs -= set(fullargspec.kwonlydefaults)
      raise ValueError("Cannot build TF function from "
                       f"{python_function.__name__}: keyword-only arguments "
                       "must have default values when input_signature is "
                       "provided. Got keyword-only arguments without default "
                       f"values: {sorted(nodefault_kwonlyargs)}.")


def is_same_structure(structure1, structure2, check_values=False):
  """Check two structures for equality, optionally of types and of values."""
  try:
    nest.assert_same_structure(structure1, structure2, expand_composites=True)
  except (ValueError, TypeError):
    return False
  if check_values:
    flattened1 = nest.flatten(structure1, expand_composites=True)
    flattened2 = nest.flatten(structure2, expand_composites=True)
    # First check the types to avoid AttributeErrors.
    if any(type(f1) is not type(f2) for f1, f2 in zip(flattened1, flattened2)):
      return False
    return flattened1 == flattened2
  return True


def _to_tensor_or_tensor_spec(x):
  return (x if isinstance(x, (ops.Tensor, tensor_spec.TensorSpec)) else
          ops.convert_to_tensor(x))


def _deterministic_dict_values(dictionary):
  return tuple(dictionary[key] for key in sorted(dictionary))


def _convert_variables_to_tensors(args, kwargs):
  args = [_to_tensor_or_tensor_spec(x) for x in args]
  kwargs = {kw: _to_tensor_or_tensor_spec(x)
            for kw, x in kwargs.items()}
  return tuple(args), kwargs


def _convert_numpy_inputs(inputs):
  """Converts numpy array inputs to tensors."""
  # We assume that any CompositeTensors have already converted their components
  # from numpy arrays to Tensors, so we don't need to expand composites here for
  # the numpy array conversion. Instead, we do so because the flattened inputs
  # are eventually passed to ConcreteFunction()._call_flat, which requires
  # expanded composites.
  flat_inputs = nest.flatten(inputs, expand_composites=True)

  # Check for NumPy arrays in arguments and convert them to Tensors.
  # TODO(nareshmodi): Skip ndarray conversion to tensor altogether, perhaps
  # finding a way to store them directly in the cache key (currently not
  # possible since ndarrays are not hashable).
  need_packing = False
  filtered_flat_inputs = []
  for index, value in enumerate(flat_inputs):
    if isinstance(value,
                  (ops.Tensor, resource_variable_ops.BaseResourceVariable)):
      filtered_flat_inputs.append(value)
    elif hasattr(value, "__array__") and not (
        hasattr(value, "_should_act_as_resource_variable") or
        isinstance(value, (np.str_, type, composite_tensor.CompositeTensor))):
      # This case is equivalent to _is_ndarray(value) == True
      a = value.__array__()
      if not isinstance(a, np.ndarray):
        raise TypeError(f"The output of __array__ must be an np.ndarray, "
                        f"got {type(a)} from {value}.")
      flat_inputs[index] = constant_op.constant(a)
      filtered_flat_inputs.append(flat_inputs[index])
      need_packing = True
  if need_packing:
    return (nest.pack_sequence_as(
        structure=inputs, flat_sequence=flat_inputs,
        expand_composites=True), flat_inputs, filtered_flat_inputs)
  else:
    return inputs, flat_inputs, filtered_flat_inputs


def convert_inputs_to_signature(inputs, input_signature, flat_input_signature):
  """Converts inputs to pass into a function with an explicit signature."""

  def format_error_message(inputs, input_signature):
    return ("  inputs: (\n" + "    " + ",\n    ".join(str(i) for i in inputs) +
            ")\n" + "  input_signature: (\n" + "    " +
            ",\n    ".join(str(i) for i in input_signature) + ")")

  try:
    flatten_inputs = nest.flatten_up_to(
        input_signature,
        inputs[:len(input_signature)],
        expand_composites=True,
        check_types=False)  # lists are convert to tuples for `tf.data`.
  except ValueError:
    raise ValueError("Structure of Python function inputs does not match "
                     "input_signature:\n"
                     f"{format_error_message(inputs, input_signature)}.")

  need_packing = False
  for index, (value, spec) in enumerate(zip(flatten_inputs,
                                            flat_input_signature)):
    if (isinstance(spec, tensor_spec.TensorSpec) and
        not isinstance(value, tensor_spec.TensorSpec) and
        not _pywrap_utils.IsTensor(value)):
      try:
        flatten_inputs[index] = ops.convert_to_tensor(
            value, dtype_hint=spec.dtype)
        need_packing = True
      except ValueError:
        raise ValueError("When input_signature is provided, all inputs to "
                         "the Python function must be convertible to "
                         "tensors:\n"
                         f"{format_error_message(inputs, input_signature)}.")

  if any(not spec.is_compatible_with(other) for spec, other in zip(
      flat_input_signature,
      flatten_inputs)):
    raise ValueError("Python inputs incompatible with input_signature:\n"
                     f"{format_error_message(inputs, input_signature)}.")

  if need_packing:
    inputs = nest.pack_sequence_as(
        structure=input_signature,
        flat_sequence=flatten_inputs,
        expand_composites=True)

  flat_inputs = nest.flatten(inputs, expand_composites=True)

  return (inputs, flat_inputs, [
      t for t in flat_inputs
      if isinstance(t, (ops.Tensor, resource_variable_ops.BaseResourceVariable))
  ])


def _get_variable_specs(args):
  """Returns `VariableSpecs` from `args`."""
  variable_specs = []
  for arg in nest.flatten(args):
    if not isinstance(arg, type_spec.TypeSpec):
      continue
    if isinstance(arg, resource_variable_ops.VariableSpec):
      variable_specs.append(arg)
    elif not isinstance(arg, tensor_spec.TensorSpec):
      # arg is a CompositeTensor spec.
      variable_specs.extend(_get_variable_specs(arg._component_specs))  # pylint: disable=protected-access
  return variable_specs
