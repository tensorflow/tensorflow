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
"""Polymorphic Function implementation."""

import threading
import types as types_lib
from typing import List
import weakref

from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.eager import function_context
from tensorflow.python.eager import function_spec
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import monomorphic_function
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

# Loaded lazily due to a circular dependency (roughly
# tf.function->autograph->->dataset->tf.function).
# TODO(b/133251390): Use a regular import.
ag_ctx = lazy_loader.LazyLoader(
    "ag_ctx", globals(),
    "tensorflow.python.autograph.core.ag_ctx")

_graph_building_time_counter = monitoring.Counter(
    "/tensorflow/core/tf_function/graph_building_time_usecs",
    "Time for tf.function to build a graph (us).")


# TODO(mdan): Refactor this and clarify relationship with def_function.Function.
# Right now, def_function.Function is the higher level implementation.
class Function:
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `defun` for more information on the semantics of
  defined functions.

  `Function` class is thread-compatible meaning that minimal usage of defuns
  (defining and calling) is thread-safe, but if users call other methods or
  invoke the base `python_function` themselves, external synchronization is
  necessary.
  In addition, Function is not reentrant, so recursive functions need to call
  the wrapped function, not the wrapper.
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None,
               attributes=None,
               autograph=True,
               autograph_options=None,
               reduce_retracing=False,
               capture_by_value=None,
               jit_compile=None,
               experimental_follow_type_hints=False):
    """Initializes a `Function`.

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
      capture_by_value: Experimental. Whether to capture resource variables by
        value or reference. If None, will inherit from a parent context or
        default to False.
      jit_compile: Force-compile the function with XLA, cf.
        def_function.Function doc on jit_compile.
      experimental_follow_type_hints: See the documentation for `tf.function`.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    self._python_function = python_function
    pure_function = attributes and monomorphic_function.IMPLEMENTS_ATTRIBUTE_NAME in attributes
    self._function_spec = function_spec.FunctionSpec.from_function_and_signature(
        python_function,
        input_signature,
        is_pure=pure_function,
        experimental_follow_type_hints=experimental_follow_type_hints)
    self._name = name
    self._autograph = autograph
    self._autograph_options = autograph_options
    self._reduce_retracing = reduce_retracing
    self._function_cache = function_cache.FunctionCache()
    self._function_attributes = attributes or {}
    self._capture_by_value = capture_by_value
    self.tracing_count = 0
    # Maintein a dict of all captures: identifier -> lambda function. It's used
    # to get runtime values for all captures during ConcreteFunction dispatch,
    self._captures_container = func_graph_module.CapturesContainer()
    self._lock = threading.RLock()
    # _descriptor_cache is a of instance of a class to an instance-specific
    # `Function`, used to make sure defun-decorated methods create different
    # functions for each instance.
    self._descriptor_cache = weakref.WeakKeyDictionary()
    self._jit_compile = jit_compile
    self._experimental_follow_type_hints = experimental_follow_type_hints

  def __call__(self, *args, **kwargs):
    """Calls a graph function specialized to the inputs."""
    with self._lock:
      (concrete_function,
       filtered_flat_args) = self._maybe_define_function(args, kwargs)
    return concrete_function._call_flat(
        filtered_flat_args, captured_inputs=concrete_function.captured_inputs)  # pylint: disable=protected-access

  @property
  def python_function(self):
    """Returns the wrapped Python function."""
    return self._python_function  # pylint: disable=protected-access

  @property
  def function_spec(self):
    return self._function_spec

  @property
  def input_signature(self):
    """Returns the input signature."""
    return self._function_spec.input_signature

  def _maybe_define_concrete_function(self, args, kwargs):
    if self.input_signature and not args and not kwargs:
      # TODO(b/215596825): Throw error here if multiple entries are defined.
      args = self.input_signature
      kwargs = {}

    return self._maybe_define_function(args, kwargs)

  def _get_concrete_function_internal_garbage_collected(self, *args, **kwargs):
    """Returns a concrete function which cleans up its graph function."""
    with self._lock:
      concrete_function, _ = self._maybe_define_concrete_function(args, kwargs)
    return concrete_function

  def _get_concrete_function_internal(self, *args, **kwargs):
    """Bypasses error checking when getting a graph function."""
    concrete_function = self._get_concrete_function_internal_garbage_collected(
        *args, **kwargs)
    # We're returning this concrete function to someone, and they may keep a
    # reference to the FuncGraph without keeping a reference to the
    # ConcreteFunction object. So we won't clean up the reference cycles
    # manually and instead will leave them to Python's garbage collector.
    concrete_function._garbage_collector.release()  # pylint: disable=protected-access
    return concrete_function

  def _get_concrete_function_garbage_collected(self, *args, **kwargs):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    Unlike `get_concrete_function(...)`, the graph will be deleted when the
    returned function is deleted.  It's useful to avoid creating a reference
    cycle when you know for sure that the graph will be no longer used without
    the returned function.

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.
    """
    if self.input_signature:
      self._function_spec.validate_inputs_with_signature(args, kwargs)

    with self._lock:
      concrete_function, _ = self._maybe_define_concrete_function(args, kwargs)
      seen_names = set()
      captured = object_identity.ObjectIdentitySet(
          concrete_function.graph.internal_captures)
      # pylint: disable=protected-access
      concrete_function._arg_keywords = []
      prefix_counts = {}
      # pylint: enable=protected-access
      num_positional = 0
      for arg in concrete_function.graph.inputs:
        if arg in captured:
          break
        num_positional += 1
        user_arg_name = compat.as_str(arg.op.get_attr("_user_specified_name"))
        proposal = user_arg_name
        while proposal in seen_names:
          index = prefix_counts.get(user_arg_name, 1)
          proposal = "{}_{}".format(user_arg_name, index)
          prefix_counts[user_arg_name] = index + 1
        seen_names.add(proposal)
        concrete_function._arg_keywords.append(proposal)  # pylint: disable=protected-access
      # Anything can be a positional argument, in the same order as .inputs
      concrete_function._num_positional_args = num_positional  # pylint: disable=protected-access
      return concrete_function

  def get_concrete_function(self, *args, **kwargs):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    Args:
      *args: inputs to specialize on. Can be concrete values (e.g. 1) or
        `tf.Tensor` or `tf.TensorSpec`.
      **kwargs: keyword inputs to specialize on. Concrete values (e.g. 1) or
        `tf.Tensor` or `tf.TensorSpec`.
    """
    concrete_function = self._get_concrete_function_garbage_collected(
        *args, **kwargs)
    concrete_function._garbage_collector.release()  # pylint: disable=protected-access
    return concrete_function

  def _list_all_concrete_functions(
      self) -> List[monomorphic_function.ConcreteFunction]:
    return self._function_cache.values()

  def __get__(self, instance, owner):
    """Makes it possible to defun instance methods."""
    del owner
    # `instance` here is the instance that this `Function` was accessed through
    # e.g., for
    #
    #   class Foo:
    #
    #     @function.defun
    #     def bar(self):
    #       ...
    #
    #   foo = Foo()
    #   foo.bar()  # `foo.bar` is a `Function` instance
    #
    # then `instance` will be `foo` (and `owner` will be `Foo`).  We create a
    # new instance of `Function` here to allow different instances each
    # to create variables once, thereby allowing methods to be decorated with
    # defun. Keeps a cache to avoid retracing the function every time the
    # descriptor is accessed.
    if instance not in self._descriptor_cache:
      if instance is None:
        return self
      # If there is no instance-specific `Function` in the cache, we construct
      # an instance-specific `Function` that uses a weak reference to the
      # instance (so that the instance will be correctly gc'd).

      # And finally add the wrapped function to the description cache
      self._descriptor_cache[instance] = class_method_to_instance_method(
          self, instance)

    # Return the cached `Function` for the instance
    return self._descriptor_cache[instance]

  def _create_concrete_function(self, args, kwargs):
    """Create a `ConcreteFunction` from `args` and `kwargs`."""
    self.tracing_count += 1

    arglen = len(args)
    base_arg_names = self._function_spec.arg_names[:arglen]
    num_missing_args = arglen - len(self._function_spec.arg_names)
    missing_arg_names = [self._function_spec.vararg_name] * num_missing_args
    # Produce a list of missing args of the form ["arg_0", "arg_1", ...],
    # where arg is based on the self._function_spec.vararg_name.
    missing_arg_names = [
        "%s_%d" % (arg, i) for i, arg in enumerate(missing_arg_names)
    ]
    arg_names = base_arg_names + missing_arg_names
    concrete_function = monomorphic_function.ConcreteFunction(
        func_graph_module.func_graph_from_py_func(
            self._name,
            self._python_function,
            args,
            kwargs,
            None,
            autograph=self._autograph,
            autograph_options=self._autograph_options,
            arg_names=arg_names,
            capture_by_value=self._capture_by_value),
        self._function_attributes,
        spec=self.function_spec,
        # Tell the ConcreteFunction to clean up its graph once it goes out of
        # scope. This is not the default behavior since it gets used in some
        # places (like Keras) where the FuncGraph lives longer than the
        # ConcreteFunction.
        shared_func_graph=False)
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
        self._function_spec.canonicalize_function_inputs(args, kwargs))

    if self.input_signature is not None:
      args = self.input_signature
      kwargs = {}

    # Get runtime values of captures
    captures = self._captures_container.get_snapshot()

    # cache_key_deletion_observer is useless here. It's based on all captures.
    # A new cache key will be built later when saving ConcreteFunction because
    # only active captures should be saved.
    lookup_func_key, _ = function_context.make_cache_key((args, kwargs),
                                                         captures)
    concrete_function = self._function_cache.lookup(lookup_func_key, True)
    if concrete_function is not None:
      return concrete_function, filtered_flat_args

    with monitoring.MonitoredTimer(_graph_building_time_counter.get_cell()):
      with trace.Trace("tf.function-graph_building"):
        logging.vlog(1,
                     "Creating new FuncGraph for Python function %r (key: %r)",
                     self._python_function, lookup_func_key)
        logging.vlog(2, "Python function signature [args: %s] [kwargs: %s]",
                     args, kwargs)
        ag_status = (
            ag_ctx.Status.ENABLED
            if self._autograph else ag_ctx.Status.DISABLED)
        with ag_ctx.ControlStatusCtx(
            status=ag_status, options=self._autograph_options):
          if self.input_signature is None and self._reduce_retracing:
            generalized_func_key = self._function_cache.generalize(
                lookup_func_key)
            # Only get placeholders for arguments, not captures
            args, kwargs = generalized_func_key._placeholder_value()["args"]  # pylint: disable=protected-access

          concrete_function = self._create_concrete_function(args, kwargs)

          graph_capture_container = concrete_function.graph._capture_func_lib  # pylint: disable=protected-access
          # Maintain the list of all captures
          self._captures_container.update(graph_capture_container)
          # Get current active captures snapshot
          captures = graph_capture_container.get_snapshot()

          # Create a cache_key with args and captures
          traced_func_key, traced_func_deletion_observer = (
              function_context.make_cache_key((args, kwargs), captures))

          self._function_cache.add(traced_func_key,
                                   traced_func_deletion_observer,
                                   concrete_function)

          return concrete_function, filtered_flat_args


def register(func, *args, **kwargs):
  """Register a specialization of a `Function` into the graph.

  This won't actually call the function with the inputs, and only put the
  function definition into graph. Register function with different input param
  will result into multiple version of functions registered in graph.

  Args:
    func: the `Function` instance that generated by a @defun
    *args: input arguments for the Python function.
    **kwargs: input keyword arguments for the Python function.

  Returns:
    a `ConcreteFunction` object specialized to inputs and execution context.

  Raises:
    ValueError: When the input function is not a defun wrapped python function.
  """
  if not isinstance(func, Function):
    raise ValueError("Only defun function is allowed to be registered. "
                     f"Got {func} with type {type(func)}.")
  concrete_func = func.get_concrete_function(*args, **kwargs)
  concrete_func.add_to_graph()
  concrete_func.add_gradient_functions_to_graph()
  return concrete_func


def defun(func=None,
          input_signature=None,
          autograph=True,
          experimental_autograph_options=None,
          reduce_retracing=False):
  """Compiles a Python function into a callable TensorFlow graph.

  `defun` (short for "define function") compiles a Python function
  composed of TensorFlow operations into a callable that executes a `tf.Graph`
  containing those operations. The callable produced by `defun` contains only
  the subgraph of TensorFlow operations that were executed when the Python
  function was called with a particular input signature, defined as a list
  of the shapes and dtypes of the Python function's Tensor-valued arguments and
  the values of its non-Tensor Python objects.

  When eager execution is enabled, the ability to create graphs from Python
  functions makes it possible to incrementally trade off debuggability and
  interactivity for performance.  Functions compiled with `defun` cannot be
  inspected with `pdb`; however, executing a graph
  generated by `defun` sometimes takes less time and memory than eagerly
  executing the corresponding Python function, since specifying computations as
  graphs allows for optimizations like automatic buffer reuse and
  parallelization among ops. Note that executing a `defun`-compiled function
  incurs a small constant overhead, so eagerly executing sufficiently small
  Python functions might take less time than executing their corresponding
  `defun`-generated graphs.

  For a Python function to be compatible with `defun`, all of its arguments must
  be hashable Python objects or lists thereof. The function itself may not
  modify the list/map structure of its arguments. Additionally, it must return
  zero or more `tf.Tensor` objects. If the Python function returns
  a `tf.Variable`, its compiled version will return the value of that variable
  as a `tf.Tensor`.

  Executing a graph generated by `defun` respects device annotations (i.e.,
  all `with tf.device` directives present in a Python function will also be
  present in its corresponding graph), but it is not yet possible to execute the
  generated graphs across multiple machines.

  _Example Usage_

  ```python
  import tensorflow as tf

  tf.compat.v1.enable_eager_execution()

  # A simple example.
  def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  g = tf.contrib.eager.defun(f)

  x = tf.constant([[2.0, 3.0]])
  y = tf.constant([[3.0, -2.0]])

  # `f` and `g` will return the same value, but `g` will be executed as a
  # TensorFlow graph.
  assert f(x, y).numpy() == g(x, y).numpy()

  # `defun` is capable of compiling Python functions that close over Python
  # objects, including Tensors and Variables.
  @tf.contrib.eager.defun
  def h():
    return f(x, y)

  assert (h().numpy() == f(x, y).numpy()).all()

  # `defun` automatically lifts variables out of the graphs it creates,
  # allowing you to compile the `call` methods of `tf.keras.layers.Layer` and
  # `tf.keras.Model` objects.
  class MyModel(tf.keras.Model):

    def __init__(self, keep_probability=0.2):
      super().__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
      self.keep_probability = keep_probability

    @tf.contrib.eager.defun
    def call(self, inputs, training=True):
      x = self.dense2(self.dense1(inputs))
      if training:
        return tf.nn.dropout(x, self.keep_probability)
      else:
        return x

  model = MyModel()
  model(x, training=True)  # executes a graph, with dropout
  model(x, training=False) # executes a graph, without dropout

  # `defun`-compiled functions are differentiable.
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
  with tf.GradientTape() as tape:
    outputs = model(x)
  gradient = tape.gradient(outputs, model.trainable_variables)
  optimizer.apply_gradients((grad, var) for grad, var in zip(gradient,
                            model.trainable_variables))
  ```

  When using `defun`, there are subtleties regarding inputs, Python control
  flow, and variable creation that one should be aware of. For concreteness, let
  `f` be a Python function that returns zero or more `tf.Tensor` objects and
  let `F = defun(f)`. `F` builds a graph for each unique input signature it
  sees, Python control flow is baked into graphs, and operations related to
  variable initialization are automatically lifted out of the graphs that `F`
  generates and placed in the eager context if executing eagerly or into an
  outer graph otherwise.

  _Input Signatures_

  By default, `F = tf.contrib.eager.defun(f)` instantiates a separate graph
  for every unique sequence of the shapes and dtypes of Tensor arguments and
  the values of Python objects it is invoked with. For example, calling
  `F(tf.random.uniform([2])` will execute a different graph than
  `F(tf.random.uniform([3])` because the two inputs have different shapes.
  The first time that `F(*args, **kwargs)` is called with a particular sequence
  of Tensor shapes and dtypes and Python values, it constructs a graph by
  tracing the execution of `f(*args, **kwargs)`; this graph is bound to an
  input signature inferred from `(*args, **kwargs)` and cached for future reuse.

  NumPy arrays passed as inputs to `F` are converted to `tf.Tensor` objects
  before being passed to `f`, and are treated as Tensors for caching. This
  allows a function to be called multiple times with NumPy arrays having
  different values but the same shape and dtype without re-tracing each time.

  `tf.contrib.eager.defun` caches graphs for your convenience, letting you
  define TensorFlow functions without explicitly specifying their signatures.
  However, this policy is conservative and potentially expensive; for example,
  when different invocations of your function have differently-shaped Tensor
  inputs, this policy might generate more graph functions than necessary. To
  eliminate such costs, `tf.contrib.eager.defun` allows you to supply an
  optional `input_signature` argument specifying the shapes and dtypes of the
  inputs. In particular, the shapes may be partially unspecified, with `None`s
  in the unknown dimensions.  When an input signature is provided,
  `tf.contrib.eager.defun` will only instantiate a single graph for the
  decorated Python function. The following is an example:

  ```python
  import tensorflow as tf

  # The first `TensorSpec` below describes the shape and dtype of `words`,
  # and the second describes the shape and dtype of `another_tensor`. Note that
  # the last dimension of the `words` `TensorSpec` is left unspecified.
  @tf.contrib.eager.defun(input_signature=[
    tf.contrib.eager.TensorSpec(shape=[50, 300, None], dtype=tf.float32),
    tf.contrib.eager.TensorSpec(shape=[300, 100], dtype=tf.float32)
  ])
  def my_sequence_model(words, another_tensor):
    ...

  # Note how the third dimension of the first input can vary freely.
  words = tf.random.uniform(([50, 300, 10])
  second_input = tf.random.uniform([300, 100])
  my_sequence_model(words, second_input)

  words = tf.random.uniform(([50, 300, 20])
  my_sequence_model(words, second_input)

  # Passing an input with an incompatible shape will raise an error.
  words = tf.random.uniform(([50, 100, 20])
  my_sequence_model(words, second_input)  # <---- This will raise an error.

  ```

  Python functions that are compiled with an `input_signature` must only accept
  Tensors as arguments and must not take unnamed keyword arguments (**kwargs).

  _Tracing_

  Be aware that because `F` only logs TensorFlow operations, all the other
  Python code that `f` executes will only shape the _construction_ of the graphs
  that `F` executes: the Python code won't be executed when the graphs
  themselves are executed, though it will be executed every time the Python
  function is traced (and a given Python function might be traced multiple
  times, once for each input signature it is invoked with). For example, whereas
  the Python function

  ```python
  import tensorflow as tf
  import numpy as np

  tf.compat.v1.enable_eager_execution()

  def add_noise():
    return tf.eye(5) + np.random.randn(5, 5)
  ```

  will return a different output everytime it is invoked, the compiled function
  `compiled = tf.contrib.eager.defun(add_noise)` will return the same value
  every time it is called, since a particular random offset generated by NumPy
  will be inserted into the graph as a TensorFlow constant. The solution is to
  replace the call to `np.random.randn` with `tf.random.normal((5, 5))`.

  _Python Side-Effects_

  A corollary of the previous discussion on tracing is the following: If a
  Python function `f` has Python side-effects, then executing `f` multiple times
  will not necessarily be semantically equivalent to executing `F =
  tf.contrib.eager.defun(f)` multiple times; this difference is due to the fact
  that `defun` only captures the subgraph of TensorFlow operations that is
  constructed when `f` is called in a graph-building context.

  _Python Control Flow_

  The structure of many machine learning computations depend upon whether one is
  training or validating, and it is common to nest specialized logic under `if
  training:` blocks. By mapping each input signature to a unique graph, `defun`
  lets users transparently compile such code, as the following code snippet
  demonstrates:

  ```python
  import tensorflow as tf

  tf.compat.v1.enable_eager_execution()

  @tf.contrib.eager.defun
  def lossy_matmul(W, x, training=True):
    outputs = tf.matmul(W, x)
    if training:
      outputs = tf.nn.dropout(outputs, keep_probability=0.2)
    return outputs

  W = tf.random.normal((3, 5))
  x = tf.random.normal((5, 1))

  # Executes a graph that applies dropout.
  lossy_outputs = lossy_matmul(W, x, training=True)

  # Executes a graph that does not apply dropout.
  exact_outputs = lossy_matmul(W, x, training=False)
  ```

  _TensorFlow Control Flow_

  When `autograph` is `True`, data-dependent control flow is allowed as well.
  Control flow statements that depend on `Tensor` values are staged into
  corresponding TensorFlow ops. For example, the following code will work as
  expected:

  ```python
  @tf.contrib.eager.defun
  def dynamic_rnn_loop(cell, seq):
    state, output = cell.zero_state()
    for input in seq:
      state, output = cell(input, state)
    return output
  ```

  For more information see `tf.autograph`.

  _Variables_

  TensorFlow operations related to variable creation and initialization are
  automatically lifted out of the graphs generated by `defun`. In practice, this
  implies that variable creation and initialization only happen the first time
  `F` is called, and that variables are reused every time thereafter. Many
  TensorFlow APIs, like `tf.keras.layers.Layer` objects, create variables the
  first time they are called and reuse them thereafter. Automatic variable
  lifting makes it possible to compile these APIs without extra effort, at the
  cost of introducing a discrepancy between the semantics of executing Python
  functions and their corresponding compiled functions. For example:

  ```python
  import tensorflow as tf

  tf.compat.v1.enable_eager_execution()

  def fn():
    x = tf.Variable(0.0)
    x.assign_add(1.0)
    return x.read_value()

  # `fn` is a Python function, so x is created, initialized, and destroyed upon
  # every invocation
  assert fn().numpy() == fn().numpy() == 1.0

  compiled = tf.contrib.eager.defun(fn)

  # Compiling `fn` with `defun` hoists all variables outside of the generated
  # graph, so initialization happens exactly once.
  assert compiled().numpy() == 1.0
  assert compiled().numpy() == 2.0
  ```

  Finally, because each input signature is bound to a unique graph, if your
  Python function constructs `tf.Variable` objects, then each graph constructed
  for that Python function will reference a unique set of variables. To
  circumvent this problem, we recommend against compiling Python functions that
  create `tf.Variable` objects. Instead, Python functions should either
  lexically close over `tf.Variable` objects or accept them as arguments,
  preferably encapsulated in an object-oriented container. If you must create
  variables inside your Python function and you want each graph generated for it
  to reference the same set of variables, add logic to your Python function that
  ensures that variables are only created the first time it is called and are
  reused for every subsequent invocation; note that this is precisely what
  `tf.keras.layers.Layer` objects do, so we recommend using them to represent
  variable-bearing computations whenever possible.

  Args:
    func: function to be compiled. If `func` is None, returns a decorator that
      can be invoked with a single argument - `func`. The end result is
      equivalent to providing all the arguments up front. In other words,
      defun(input_signature=...)(func) is equivalent to defun(func,
      input_signature=...). The former allows the following use case:
      @tf.contrib.eager.defun(input_signature=...) def foo(...): ...
    input_signature: A possibly nested sequence of `tf.contrib.eager.TensorSpec`
      objects specifying the shapes and dtypes of the Tensors that will be
      supplied to this function. If `None`, a separate function is instantiated
      for each inferred input signature.  If a signature is specified, every
      input to `func` must be a `Tensor`, and `func` cannot accept `**kwargs`.
    autograph: Whether `func` should be compiled before constructing the graph.
      See https://www.tensorflow.org/guide/autograph for more information.
    experimental_autograph_options: Experimental knobs (in the form of a tuple
      of tensorflow.autograph.Feature values) to control behavior when
      autograph=True.
    reduce_retracing: When True, `tf.function` uses
      `tf.types.experimental.TraceType` to trace supertypes of arguments to
      reduce the number of traces.

  Returns:
     If `func` is not None, returns a callable that will execute the compiled
     function (and return zero or more `tf.Tensor` objects).
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a callable equivalent to the case above.

  Raises:
    TypeError: If `input_signature` is neither `None` nor a sequence of
      `tf.contrib.eager.TensorSpec` objects.
  """
  return defun_with_attributes(
      func=func,
      input_signature=input_signature,
      autograph=autograph,
      experimental_autograph_options=experimental_autograph_options,
      reduce_retracing=reduce_retracing)


@tf_export("__internal__.function.defun_with_attributes", v1=[])
def defun_with_attributes(func=None,
                          input_signature=None,
                          attributes=None,
                          autograph=True,
                          experimental_autograph_options=None,
                          jit_compile=None,
                          reduce_retracing=False,
                          experimental_follow_type_hints=False):
  """Compiles a Python function into a callable TensorFlow graph.

  This function supports adding extra function attributes. See detailed
  documentation in defun(). Currently this is not exposed in public API since we
  don't expect user to directly use attributes, and attribute won't work by
  itself. This assumption might change in future.

  Args:
    func: function to be compiled.
    input_signature: same as defun()'s input_signature.
    attributes: A dictionary of arguments which will be added to function def as
      attributes. Currently only support primitive types as value, and only
      allowlisted attribute name is allowed. Unallowlisted attribute name or
      unsupported value will result into ValueError. `func_name` is also one of
      the allowlisted argument which is a python string, and sets the name for
      this `ConcreteFunction` in the graph.
    autograph: same as defun()'s autograph.
    experimental_autograph_options: same as defun()'s
      experimental_autograph_options.
    jit_compile: same as defun()'s jit_compile.
    reduce_retracing: same as defun()'s reduce_retracing
    experimental_follow_type_hints: see `tf.function`.

  Returns:
    Same as the return value of defun, with attributes added to the function in
    graph.
  """

  # TODO(apassos): deal with captured global state. Deal with control flow.
  def decorated(function):
    try:
      if attributes:
        name = attributes.pop("func_name", function.__name__)
      else:
        name = function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        function,
        Function(
            function,
            name,
            input_signature=input_signature,
            attributes=attributes,
            autograph=autograph,
            autograph_options=experimental_autograph_options,
            jit_compile=jit_compile,
            reduce_retracing=reduce_retracing,
            experimental_follow_type_hints=experimental_follow_type_hints))

  # This code path is for the `foo = tfe.defun(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the
  #
  # @tfe.defun(...)
  # def foo(...):
  #    ...
  #
  # use case, which is equivalent to `foo = tfe.defun(...)(foo)`
  return decorated


# When a method is bound to objects of this type, it allows AutoGraph to
# recover a weak reference the original method's self pointer, so that it can
# execute it consistent with class_method_to_instance_method's
# bound_method_wrapper.
# TODO(b/119246461): This is not pretty. Use a descriptor instead?
class TfMethodTarget:
  """Binding target for methods replaced by function and defun."""

  __slots__ = ("weakrefself_target__", "weakrefself_func__")

  def __init__(self, target, original_python_function):
    self.weakrefself_target__ = target
    self.weakrefself_func__ = weakref.ref(original_python_function)

  @property
  def target(self):
    return self.weakrefself_target__()

  @property
  def target_class(self):
    true_self = self.weakrefself_target__()
    if tf_inspect.isclass(true_self):
      # Class method
      return true_self
    else:
      return true_self.__class__

  def call(self, args, kwargs):
    wrapped_fn = self.weakrefself_func__()
    return wrapped_fn(self.weakrefself_target__(), *args, **kwargs)


def class_method_to_instance_method(original_function, instance):
  """Constructs a new `Function` with `self` bound."""
  weak_instance = weakref.ref(instance)

  # Note: while we could bind to a weakref proxy instead, that causes the
  # bound method to be unhashable.
  bound_method = types_lib.MethodType(
      original_function.python_function,
      TfMethodTarget(weak_instance, original_function.python_function))

  # original_function is expected to be of one of the two `Function` types
  # (defined either in function.py or def_function.py).
  assert hasattr(original_function, "_name")
  assert hasattr(original_function, "_autograph")
  assert hasattr(original_function, "_function_spec")
  assert hasattr(original_function, "python_function")

  weak_bound_method_wrapper = None

  def bound_method_wrapper(*args, **kwargs):
    """Wraps either a dummy MethodType or a converted AutoGraph function."""
    # __wrapped__ allows AutoGraph to swap in a converted function.
    strong_bound_method_wrapper = weak_bound_method_wrapper()
    wrapped_fn = strong_bound_method_wrapper.__wrapped__

    if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
      # If __wrapped__ was not replaced, then call original_function.
      # TODO(mdan): For better consistency, use the wrapper's call().
      wrapped_fn = original_function.python_function
      return wrapped_fn(weak_instance(), *args, **kwargs)

    # If __wrapped__ was replaced, then it is always an unbound function.
    # However, the replacer is still responsible for attaching self properly.
    # TODO(mdan): Is it possible to do it here instead?
    return wrapped_fn(*args, **kwargs)

  weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)

  # pylint: disable=protected-access
  # We make a dummy MethodType object to generate the correct bound method
  # signature. The actual call is to a function with a weak reference to
  # `instance`.
  instance_func = type(original_function)(
      tf_decorator.make_decorator(bound_method, bound_method_wrapper),
      name=original_function._name,
      autograph=original_function._autograph,
      input_signature=original_function.input_signature,
      reduce_retracing=original_function._reduce_retracing,
      jit_compile=original_function._jit_compile)
  # pylint: enable=protected-access

  # We wrap the bound method with tf_decorator so inspection works correctly
  wrapped_instance_func = tf_decorator.make_decorator(bound_method,
                                                      instance_func)
  return wrapped_instance_func
