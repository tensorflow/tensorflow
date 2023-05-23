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
"""Tracing Compiler implementation."""

import collections
import contextlib
import threading
from typing import List

from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import concrete_function as concrete_function_lib
from tensorflow.python.eager.polymorphic_function import function_context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat
from tensorflow.python.util import lazy_loader


# Loaded lazily due to a circular dependency (roughly
# tf.function->autograph->->dataset->tf.function).
# TODO(b/133251390): Use a regular import.
ag_ctx = lazy_loader.LazyLoader(
    "ag_ctx",
    globals(),
    "tensorflow.python.autograph.core.ag_ctx",
)

_graph_building_time_counter = monitoring.Counter(
    "/tensorflow/core/tf_function/graph_building_time_usecs",
    "Time for tf.function to build a graph (us).",
)


# TODO(fmuham): Revamp the API of this class to be 100% compiler-focused.
class TracingCompiler:
  """Generates, caches and dispatchs traced Monomorphic Concrete Functions.

  The tracing is done using the Python source function with respect to inputs
  and other options specified by constructor.

  See the documentation for `tf.function` for more information on the semantics
  of defined functions.

  `TracingCompiler` class is thread-compatible meaning that minimal usage of
  tf.function (defining and calling) is thread-safe, but if users call other
  methods or invoke the base `python_function` themselves, external
  synchronization is necessary.

  In addition, TracingCompiler is not reentrant, so recursive functions need
  to call the wrapped function, not the wrapper.
  """

  def __init__(
      self,
      python_function,
      name,
      input_signature=None,
      attributes=None,
      autograph=True,
      autograph_options=None,
      reduce_retracing=False,
      jit_compile=None,
  ):
    """Initializes a `TracingCompiler`.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      input_signature: a possibly nested sequence of `TensorSpec` objects
        specifying the input signature of this function. If `None`, a separate
        function is instantiated for each inferred input signature.
      attributes: dict, extra keyword arguments that will be added as attribute
        of the function.
      autograph: whether to use autograph to compile `python_function`. See
        https://www.tensorflow.org/guide/autograph for more information.
      autograph_options: Experimental knobs to control behavior `when
        autograph=True`. See https://www.tensorflow.org/guide/autograph for more
        information.
      reduce_retracing: When True, `tf.function` uses
        `tf.types.experimental.TraceType` to trace supertypes of arguments to
        reduce the number of traces.
      jit_compile: Force-compile the function with XLA, cf. tf.function doc on
        jit_compile.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    self._python_function = python_function
    pure_function = attributes and attributes_lib.IMPLEMENTS in attributes
    self._function_type, self._default_values = (
        function_type_utils.make_function_type(python_function, input_signature)
    )
    self._is_pure = pure_function
    self._name = name
    self._autograph = autograph
    self._autograph_options = autograph_options
    self._reduce_retracing = reduce_retracing
    self._function_cache = function_cache.FunctionCache()

    self._function_attributes = attributes or {}
    for attribute in self._function_attributes:
      if attribute not in attributes_lib.TRACING_COMPILER_ALLOWLIST:
        raise ValueError(
            f"TracingCompiler does not support `{attribute}` as an attribute."
        )

    self.tracing_count = 0
    # Maintein a dict of all captures: identifier -> lambda function. It's used
    # to get runtime values for all captures during ConcreteFunction dispatch,
    self._func_captures = capture_container.FunctionCaptures()
    self._lock = threading.RLock()
    self._jit_compile = jit_compile

  def __call__(self, *args, **kwargs):
    """Calls a graph function specialized to the inputs."""
    with self._lock:
      (concrete_function, filtered_flat_args) = self._maybe_define_function(
          args, kwargs
      )
    return concrete_function._call_flat(
        filtered_flat_args, captured_inputs=concrete_function.captured_inputs
    )  # pylint: disable=protected-access

  @property
  def python_function(self):
    """Returns the wrapped Python function."""
    return self._python_function  # pylint: disable=protected-access

  @property
  def function_spec(self):
    return function_type_utils.FunctionSpec(
        self._function_type,
        self._default_values,
        self._is_pure,
        self._name,
        self._jit_compile,
    )

  @property
  def input_signature(self):
    """Returns the input signature."""
    return function_type_utils.to_input_signature(self._function_type)

  def _maybe_define_concrete_function(self, args, kwargs):
    if self.input_signature and not args and not kwargs:
      # TODO(b/215596825): Throw error here if multiple entries are defined.
      args = self.input_signature
      kwargs = {}

    return self._maybe_define_function(args, kwargs)

  def get_concrete_function(
      self, args=None, kwargs=None, bind_graph_to_function=False
  ):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    Args:
      args: inputs to specialize on. Can be concrete values (e.g. 1) or
        `tf.Tensor` or `tf.TensorSpec`.
      kwargs: keyword inputs to specialize on. Concrete values (e.g. 1) or
        `tf.Tensor` or `tf.TensorSpec`.
      bind_graph_to_function: Sets up FuncGraph to be deleted alongside
        ConcreteFunction.
    """
    args = args if args else ()
    kwargs = kwargs if kwargs else {}

    if self.input_signature and (args or kwargs):
      # Check to see if a valid type can be generated from the args, kwargs
      args, kwargs = function_type_utils.bind_function_inputs(
          args, kwargs, self._function_type, self._default_values
      )

    with self._lock:
      concrete_function, _ = self._maybe_define_concrete_function(args, kwargs)
      _set_arg_keywords(concrete_function)

    if not bind_graph_to_function:
      concrete_function._garbage_collector.release()  # pylint: disable=protected-access

    return concrete_function

  def _list_all_concrete_functions(
      self,
  ) -> List[concrete_function_lib.ConcreteFunction]:
    return self._function_cache.values()

  def _create_concrete_function(
      self, function_type, placeholder_mapping, func_graph
  ):
    """Create a `ConcreteFunction` from `args`, `kwargs`, and `func_graph`."""
    self.tracing_count += 1

    placeholder_context = trace_type.InternalPlaceholderContext(
        func_graph, placeholder_mapping
    )
    with func_graph.as_default():
      placeholder_bound_args = function_type.placeholder_arguments(
          placeholder_context
      )

    traced_func_graph = func_graph_module.func_graph_from_py_func(
        self._name,
        self._python_function,
        placeholder_bound_args.args,
        placeholder_bound_args.kwargs,
        None,
        func_graph=func_graph,
        arg_names=function_type_utils.to_arg_names(function_type),
        create_placeholders=False,
    )

    transform.apply_func_graph_transforms(traced_func_graph)

    concrete_function = concrete_function_lib.ConcreteFunction(
        traced_func_graph,
        self._function_attributes,
        # Tell the ConcreteFunction to clean up its graph once it goes out of
        # scope. This is not the default behavior since it gets used in some
        # places (like Keras) where the FuncGraph lives longer than the
        # ConcreteFunction.
        shared_func_graph=False,
        function_type=function_type
    )

    transform.call_concrete_function_callbacks(concrete_function)

    return concrete_function

  def _maybe_define_function(self, args, kwargs):
    """Gets a function for these inputs, defining it if necessary.

    Caller must hold self._lock.

    Args:
      args: The varargs for the Python function.
      kwargs: The keyword args for the Python function.

    Returns:
      A graph function corresponding to the input signature implied by args and
      kwargs, as well as filtered flattened inputs (only Tensors and Variables)
      that the object should be called with.

    Raises:
      ValueError: If inputs are incompatible with the input signature.
      TypeError: If the function inputs include non-hashable objects
      RuntimeError: If there's an internal bug (inconsistency) in handling
        shape relaxation retracing.
    """
    args, kwargs, filtered_flat_args = (
        function_type_utils.canonicalize_function_inputs(
            args,
            kwargs,
            self._function_type,
            self._default_values,
            self._is_pure,
        )
    )

    if self.input_signature is not None:
      args = (*self.input_signature, *args[len(self.input_signature) :])

    # Get runtime values of captures
    captures = self._func_captures.get_by_ref_snapshot()

    current_func_context = function_context.make_function_context()

    # cache_key_deletion_observer is useless here. It's based on all captures.
    # A new cache key will be built later when saving ConcreteFunction because
    # only active captures should be saved.
    lookup_func_type, lookup_func_context = (
        function_type_utils.make_canonicalized_monomorphic_type(
            args, kwargs, captures, self._function_type, self._default_values
        )
    )
    concrete_function = self._function_cache.lookup(
        current_func_context, lookup_func_type
    )
    if concrete_function is not None:
      return concrete_function, filtered_flat_args

    # Use a timer for graph building only if not already inside a function. This
    # avoids double counting graph building time for nested functions.
    with monitoring.MonitoredTimer(
        _graph_building_time_counter.get_cell()
    ) if not ops.inside_function() else contextlib.nullcontext():
      with trace.Trace("tf.function-graph_building"):
        logging.vlog(
            1,
            "Creating new FuncGraph for Python function %r (key: %r, %r)",
            self._python_function,
            current_func_context,
            lookup_func_type,
        )
        logging.vlog(
            2, "Python function signature [args: %s] [kwargs: %s]", args, kwargs
        )
        ag_status = (
            ag_ctx.Status.ENABLED if self._autograph else ag_ctx.Status.DISABLED
        )
        with ag_ctx.ControlStatusCtx(
            status=ag_status, options=self._autograph_options
        ):
          func_graph = func_graph_module.FuncGraph(self._name)
          if self.input_signature is None and self._reduce_retracing:
            target_func_type = self._function_cache.generalize(
                current_func_context, lookup_func_type
            )
          else:
            target_func_type = lookup_func_type
          placeholder_mapping = lookup_func_context.get_placeholder_mapping()
          concrete_function = self._create_concrete_function(
              target_func_type, placeholder_mapping, func_graph
          )

          # TODO(b/263520817): Remove access to private attribute.
          graph_capture_container = concrete_function.graph.function_captures
          # Maintain the list of all captures
          self._func_captures.merge_by_ref_with(graph_capture_container)
          # Get current active captures snapshot
          captures = graph_capture_container.get_by_ref_snapshot()

          # Create a cache_key with args and captures
          traced_func_type = _insert_capture_type(
              target_func_type, captures, lookup_func_context
          )

          self._function_cache.add(
              current_func_context, traced_func_type, concrete_function
          )

          return concrete_function, filtered_flat_args


def _insert_capture_type(original_func_type, captures, type_context):
  capture_types = collections.OrderedDict()
  for name, value in captures.items():
    capture_types[name] = trace_type.from_value(value, type_context)
  return function_type_lib.FunctionType(
      original_func_type.parameters.values(), capture_types
  )


def _set_arg_keywords(concrete_function):
  """Sets arg keywords for ConcreteFunction."""
  seen_names = set()
  concrete_function._arg_keywords = []  # pylint: disable=protected-access
  prefix_counts = {}
  graph = concrete_function.graph
  num_captures = len(graph.internal_captures + graph.deferred_internal_captures)
  num_positional = len(graph.inputs) - num_captures
  for arg in concrete_function.graph.inputs[:num_positional]:
    try:
      user_arg_name = compat.as_str(arg.op.get_attr("_user_specified_name"))
    except ValueError:
      user_arg_name = "tensor_arg"
    proposal = user_arg_name
    while proposal in seen_names:
      index = prefix_counts.get(user_arg_name, 1)
      proposal = "{}_{}".format(user_arg_name, index)
      prefix_counts[user_arg_name] = index + 1
    seen_names.add(proposal)
    concrete_function._arg_keywords.append(proposal)  # pylint: disable=protected-access
  # Anything can be a positional argument, in the same order as .inputs
  concrete_function._num_positional_args = num_positional  # pylint: disable=protected-access
