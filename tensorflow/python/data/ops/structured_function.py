# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for managing tf.data user-defined functions."""

import warnings

from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as eager_function

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import function_utils
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import variable_utils

autograph = lazy_loader.LazyLoader(
    "autograph", globals(),
    "tensorflow.python.autograph.impl.api")
# TODO(mdan): Create a public API for this.
autograph_ctx = lazy_loader.LazyLoader(
    "autograph_ctx", globals(),
    "tensorflow.python.autograph.core.ag_ctx")


def _should_pack(arg):
  """Determines whether the caller needs to pack the argument in a tuple.

  If user-defined function returns a list of tensors, `nest.flatten()` and
  `ops.convert_to_tensor()` and would conspire to attempt to stack those tensors
  into a single tensor because the tf.data version of `nest.flatten()` does
  not recurse into lists. Since it is more likely that the list arose from
  returning the result of an operation (such as `tf.numpy_function()`) that
  returns a list of not-necessarily-stackable tensors, we treat the returned
  value as a `tuple` instead. A user wishing to pack the return value into a
  single tensor can use an explicit `tf.stack()` before returning.

  Args:
    arg: argument to check

  Returns:
    Indication of whether the caller needs to pack the argument in a tuple.
  """
  return isinstance(arg, list)


def _should_unpack(arg):
  """Determines whether the caller needs to unpack the argument from a tuple.

  Args:
    arg: argument to check

  Returns:
    Indication of whether the caller needs to unpack the argument from a tuple.
  """
  return type(arg) is tuple  # pylint: disable=unidiomatic-typecheck


class StructuredFunctionWrapper():
  """A function wrapper that supports structured arguments and return values."""

  def __init__(self,
               func,
               transformation_name,
               dataset=None,
               input_classes=None,
               input_shapes=None,
               input_types=None,
               input_structure=None,
               add_to_graph=True,
               use_legacy_function=False,
               defun_kwargs=None):
    """Creates a new `StructuredFunctionWrapper` for the given function.

    Args:
      func: A function from a (nested) structure to another (nested) structure.
      transformation_name: Human-readable name of the transformation in which
        this function is being instantiated, for error messages.
      dataset: (Optional.) A `tf.data.Dataset`. If given, the structure of this
        dataset will be assumed as the structure for `func` arguments; otherwise
        `input_classes`, `input_shapes`, and `input_types` must be defined.
      input_classes: (Optional.) A (nested) structure of `type`. If given, this
        argument defines the Python types for `func` arguments.
      input_shapes: (Optional.) A (nested) structure of `tf.TensorShape`. If
        given, this argument defines the shapes and structure for `func`
        arguments.
      input_types: (Optional.) A (nested) structure of `tf.DType`. If given,
        this argument defines the element types and structure for `func`
        arguments.
      input_structure: (Optional.) A `Structure` object. If given, this argument
        defines the element types and structure for `func` arguments.
      add_to_graph: (Optional.) If `True`, the function will be added to the
        default graph, if it exists.
      use_legacy_function: (Optional.) A boolean that determines whether the
        function be created using `tensorflow.python.eager.function.defun`
        (default behavior) or `tensorflow.python.framework.function.Defun`
        (legacy behavior).
      defun_kwargs: (Optional.) A dictionary mapping string argument names to
        values. If supplied, will be passed to `function` as keyword arguments.

    Raises:
      ValueError: If an invalid combination of `dataset`, `input_classes`,
        `input_shapes`, and `input_types` is passed.
    """
    # pylint: disable=protected-access
    if input_structure is None:
      if dataset is None:
        if input_classes is None or input_shapes is None or input_types is None:
          raise ValueError("Either `dataset`, `input_structure` or all of "
                           "`input_classes`, `input_shapes`, and `input_types` "
                           "must be specified.")
        self._input_structure = structure.convert_legacy_structure(
            input_types, input_shapes, input_classes)
      else:
        if not (input_classes is None and input_shapes is None and
                input_types is None):
          raise ValueError("Either `dataset`, `input_structure` or all of "
                           "`input_classes`, `input_shapes`, and `input_types` "
                           "must be specified.")
        self._input_structure = dataset.element_spec
    else:
      if not (dataset is None and input_classes is None and
              input_shapes is None and input_types is None):
        raise ValueError("Either `dataset`, `input_structure`, or all of "
                         "`input_classes`, `input_shapes`, and `input_types` "
                         "must be specified.")
      self._input_structure = input_structure

    self._func = func

    if defun_kwargs is None:
      defun_kwargs = {}

    readable_transformation_name = transformation_name.replace(
        ".", "_")[:-2] if len(transformation_name) > 2 else ""

    func_name = "_".join(
        [readable_transformation_name,
         function_utils.get_func_name(func)])
    # Sanitize function name to remove symbols that interfere with graph
    # construction.
    for symbol in ["<", ">", "\\", "'", " "]:
      func_name = func_name.replace(symbol, "")

    ag_ctx = autograph_ctx.control_status_ctx()

    def wrapper_helper(*args):
      """Wrapper for passing nested structures to and from tf.data functions."""
      nested_args = structure.from_compatible_tensor_list(
          self._input_structure, args)
      if not _should_unpack(nested_args):
        nested_args = (nested_args,)
      ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
      ret = variable_utils.convert_variables_to_tensors(ret)
      if _should_pack(ret):
        ret = tuple(ret)

      try:
        self._output_structure = structure.type_spec_from_value(ret)
      except (ValueError, TypeError) as e:
        raise TypeError(f"Unsupported return value from function passed to "
                        f"{transformation_name}: {ret}.") from e
      return ret

    def trace_legacy_function(defun_kwargs):

      @function.Defun(*structure.get_flat_tensor_types(self._input_structure),
                      **defun_kwargs)
      def wrapped_fn(*args):
        ret = wrapper_helper(*args)
        return structure.to_tensor_list(self._output_structure, ret)

      return lambda: wrapped_fn

    def trace_py_function(defun_kwargs):
      # First we trace the function to infer the output structure.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def unused(*args):  # pylint: disable=missing-docstring,unused-variable
        ret = wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      _ = unused.get_concrete_function()

      def py_function_wrapper(*args):
        nested_args = structure.from_compatible_tensor_list(
            self._input_structure, args)
        if not _should_unpack(nested_args):
          nested_args = (nested_args,)
        ret = self._func(*nested_args)
        if _should_pack(ret):
          ret = tuple(ret)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      # Next we trace the function wrapped in `eager_py_func` to force eager
      # execution.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def wrapped_fn(*args):  # pylint: disable=missing-docstring
        return script_ops.eager_py_func(
            py_function_wrapper, args,
            structure.get_flat_tensor_types(self._output_structure))

      return wrapped_fn.get_concrete_function

    def trace_tf_function(defun_kwargs):
      # Note: wrapper_helper will apply autograph based on context.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def wrapped_fn(*args):  # pylint: disable=missing-docstring
        ret = wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      return wrapped_fn.get_concrete_function

    if use_legacy_function:
      defun_kwargs.update({"func_name": func_name + "_" + str(ops.uid())})
      fn_factory = trace_legacy_function(defun_kwargs)
    else:
      defun_kwargs.update({"func_name": func_name})
      defun_kwargs.update({"_tf_data_function": True})
      if debug_mode.DEBUG_MODE and str(dataset.element_spec)[0:16] != 'RaggedTensorSpec':
        fn_factory = trace_py_function(defun_kwargs)
      else:
        if def_function.functions_run_eagerly():
          warnings.warn(
              "Even though the `tf.config.experimental_run_functions_eagerly` "
              "option is set, this option does not apply to tf.data functions. "
              "To force eager execution of tf.data functions, please use "
              "`tf.data.experimental.enable_debug_mode()`.")
        fn_factory = trace_tf_function(defun_kwargs)

    self._function = fn_factory()
    # There is no graph to add in eager mode.
    add_to_graph &= not context.executing_eagerly()
    # There are some lifetime issues when a legacy function is not added to a
    # out-living graph. It's already deprecated so de-prioritizing the fix.
    add_to_graph |= use_legacy_function
    if add_to_graph:
      self._function.add_to_graph(ops.get_default_graph())

    if not use_legacy_function:
      outer_graph_seed = ops.get_default_graph().seed
      if outer_graph_seed and self._function.graph.seed == outer_graph_seed:
        if self._function.graph._seed_used:
          warnings.warn(
              "Seed %s from outer graph might be getting used by function %s, "
              "if the random op has not been provided any seed. Explicitly set "
              "the seed in the function if this is not the intended behavior." %
              (outer_graph_seed, func_name),
              stacklevel=4)

  @property
  def output_structure(self):
    return self._output_structure

  @property
  def output_classes(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def output_shapes(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def output_types(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def function(self):
    return self._function
