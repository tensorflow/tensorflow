# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=unidiomatic-typecheck
"""Prototype decorator for defining graph functions with eager semantics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import weakref

from tensorflow.python.eager import context
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export


class UnliftedInitializerVariable(resource_variable_ops.UninitializedVariable):
  """Variable which does not lift its initializer out of function context.

  Instances of this variable, when created, build a graph which runs their
  initializer inside a tf.cond(is_initialized) block.

  This can only be created inside a defun called from (eventually) eager
  mode. That is, non-function-building graphs are not supported.
  """

  def __init__(self,
               initial_value=None,
               trainable=None,
               caching_device=None,
               name=None,
               dtype=None,
               constraint=None,
               add_initializers_to=None,
               lifted_initializer_graph=None,
               synchronization=None,
               aggregation=None,
               shape=None,
               **unused_kwargs):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
       a Tensor) or float32 will be used (if it is a Python object convertible
       to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.
      add_initializers_to: if not None and not in legacy graph mode, the
        initializer tensor will be added to this map in addition to adding the
        assignment to the function.
      lifted_initializer_graph: FuncGraph to try to lift initializers to.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If called outside of a function definition.
    """
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
    if not ops.inside_function():
      # If we've been init_scope()d out of the function definition nothing to do
      # here; we can't really do the capturing or conditional logic.
      resource_variable_ops.ResourceVariable.__init__(
          self, initial_value=initial_value, trainable=trainable,
          caching_device=caching_device, name=name, dtype=dtype,
          constraint=constraint)
      return
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, trackable.CheckpointInitialValue):
      self._maybe_initialize_trackable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    with ops.name_scope(name, "Variable", []
                        if init_from_fn else [initial_value]) as scope_name:
      with ops.name_scope("Initializer"), ops.device(None):
        initial_value = ops.convert_to_tensor(
            initial_value() if init_from_fn else initial_value,
            name="initial_value", dtype=dtype)
      assert initial_value is not None

      # Don't use `shape or initial_value.shape` since TensorShape has
      # overridden `__bool__`.
      if shape is None:
        shape = initial_value.shape

    # Use the constructor for UninitializedVariable to start. Outside the name
    # scope so we don't double up the prefix.
    super(UnliftedInitializerVariable, self).__init__(
        trainable=trainable,
        caching_device=caching_device,
        name=name,
        shape=shape,
        dtype=initial_value.dtype,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        extra_handle_data=initial_value,
        **unused_kwargs)

    with ops.name_scope(scope_name):
      if self._in_graph_mode:
        with ops.init_scope():
          outer_graph = ops.get_default_graph()
        func_graph = ops.get_default_graph()
        function_placeholders = (
            func_graph.inputs + func_graph.internal_captures)
        placeholder_ops = set(
            [tensor.op for tensor in function_placeholders])
        lifted_initializer = lift_to_graph.lift_to_graph(
            [initial_value], outer_graph,
            disallowed_placeholders=placeholder_ops)[initial_value]
        with ops.init_scope():
          self._initial_value = lifted_initializer
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                resource_variable_ops.var_is_initialized_op(self._handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
              self._initializer_op = resource_variable_ops.assign_variable_op(
                  self._handle, lifted_initializer, name=n)
      elif context.executing_eagerly():
        # In this case, both current scope and init scope are eager.
        # Assign_variable_op will be executed immediately. So we don't need to
        # add it to "add_initializers_to" to lift it out.
        with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
          resource_variable_ops.assign_variable_op(
              self._handle, initial_value, name=n)
      else:
        # Init scope is eager but current scope is graph. We will lift out this
        # variable by addint it into "add_initializers_to".
        if add_initializers_to is not None:
          add_initializers_to[self] = initial_value
        def assign_fn():
          with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
            resource_variable_ops.assign_variable_op(
                self._handle,
                initial_value,
                name=n)
            # Returning values to keep tf.cond happy.
          return ops.convert_to_tensor(1)
        def not_assign_fn():
          return ops.convert_to_tensor(0)
        # Note: this cond is always guaranteed to run because we're inside a
        # defun which will insert automatic control dependencies. It will only
        # execute assign_fn if lifting failed.
        control_flow_ops.cond(
            resource_variable_ops.var_is_initialized_op(self._handle),
            not_assign_fn, assign_fn)


RUN_FUNCTIONS_EAGERLY = False


@tf_export("config.experimental_run_functions_eagerly")
def run_functions_eagerly(run_eagerly):
  """Enables / disables eager execution of `tf.function`s.

  After calling `tf.config.experimental_run_functions_eagerly(True)` all
  invocations of tf.function will run eagerly instead of running through a graph
  function.

  This can be useful for debugging or profiling.

  Similarly, calling `tf.config.experimental_run_functions_eagerly(False)` will
  revert the behavior of all functions to graph functions.

  Args:
    run_eagerly: Boolean. Whether to run functions eagerly.
  """
  global RUN_FUNCTIONS_EAGERLY
  RUN_FUNCTIONS_EAGERLY = bool(run_eagerly)


class FunctionDeleter(object):

  def __init__(self, func_graph):
    self.func_graph = func_graph

  def __del__(self):
    try:
      func_graph_module.dismantle_func_graph(self.func_graph)
    except:  # pylint: disable=bare-except
      # Note: bare except here because this can be noisy at shutdown time.
      pass


class Function(object):
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `tf.function` for more information on the semantics
  of defined functions.

  `Function` is thread-compatible.
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None,
               autograph=True,
               experimental_autograph_options=None,
               experimental_relax_shapes=False,
               experimental_compile=None):
    """Initializes a `Function`.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      input_signature: a possibly nested sequence of `TensorSpec` objects
        specifying the input signature of this function. If `None`, a separate
        function is instantiated for each inferred input signature.
      autograph: whether `python_function` should be converted to graph mode.
        See https://www.tensorflow.org/guide/autograph for more information.
      experimental_autograph_options: optional tuple of
        tensorflow.autograph.Feature values. Allows enabling additional
        conversion options when autograph is set to True.
      experimental_relax_shapes: When true, argument shapes may be relaxed to
        avoid unecessary retracing.
      experimental_compile: If false, execute the function in a regular way. The
        function is optimized by some graph rewrite passes (some ops might be
        clustered into a single op) and interpreted by the standard TensorFlow
        executor, which dispatches op kernels one by one as they become
        executable. Set it to false when directly running a multi-device
        function on TPUs (e.g. two TPU cores, one TPU core and its
        host CPU). If True, the function is compiled directly by XLA. XLA would
        fuse all the ops and emit more efficient code to run for some devices
        (e.g. TPU, XLA_GPU) and some use cases (e.g. dense tensor computation).
        It requires that the whole function is compilable by XLA. If None
        (default), compile the function with XLA when running on TPU and go
        through the regular function execution path when running on other
        devices.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    self._python_function = python_function
    self._function_spec = function_lib.FunctionSpec.from_function_and_signature(
        python_function, input_signature)
    self._autograph = autograph
    self._experimental_autograph_options = experimental_autograph_options
    self.experimental_relax_shapes = experimental_relax_shapes
    self._experimental_compile = experimental_compile
    self._created_variables = None
    self._stateful_fn = None
    self._stateless_fn = None
    self._descriptor_cache = weakref.WeakKeyDictionary()
    self._name = name

  def _defun_with_scope(self, scope):
    """Creates a defun wrapped inside a variable creator scope."""

    weak_wrapped_fn = None
    def wrapped_fn(*args, **kwds):
      """Wraps `self._python_function` in a variable creator scope."""
      # We register a variable creator with reduced priority. If an outer
      # variable creator is just modifying keyword arguments to the variable
      # constructor, this will work harmoniously. Since the `scope` registered
      # here actually creates the variable, it taking priority would otherwise
      # ignore the outer creator.
      #
      # If an outer variable creator calls the variable constructor manually,
      # for example creating a MirroredVariable, then they won't call our
      # creator. This means we won't be able to trace the initialization graph,
      # and so variable initializers can't depend on function arguments. This is
      # better than the alternative, tracing the initialization graph but giving
      # the user a variable type they didn't want.
      with ops.get_default_graph()._variable_creator_scope(scope, priority=50):  # pylint: disable=protected-access
        # __wrapped__ allows AutoGraph to swap in a converted function. We give
        # the function a weak reference to itself to avoid a reference cycle.
        return weak_wrapped_fn().__wrapped__(*args, **kwds)
    weak_wrapped_fn = weakref.ref(wrapped_fn)

    return self._defun(tf_decorator.make_decorator(
        self._python_function,
        wrapped_fn))

  def _defun(self, fn):
    """Returns a defun generated from the input function."""
    attributes = None
    if self._experimental_compile is not None:
      if self._experimental_compile:
        attributes = {"_XlaCompile": True}
      else:
        attributes = {"_XlaCompile": False}
    return function_lib.defun_with_attributes(
        fn,
        input_signature=self.input_signature,
        attributes=attributes,
        autograph=self._autograph,
        experimental_autograph_options=self._experimental_autograph_options,
        experimental_relax_shapes=self.experimental_relax_shapes)

  def _initialize(self, args, kwds, add_initializers_to=None):
    """Initializes, on the first call.

    Creates two `Function`s, one that will allow creation of variables
    and one that won't.

    Additionally runs a trace for the `Function` that allows creation
    of variables.

    Args:
      args: Arguments to the underlying python callable.
      kwds: Keyword arguments to the python callable.
      add_initializers_to: Where to collect variable initializers, if not None.
    """

    created_variables = []
    lifted_initializer_graph = func_graph_module.FuncGraph("initializer")

    def variable_capturing_scope(unused_next_creator, **kwds):
      """Creates UnliftedInitializerVariables and saves references to them."""
      v = UnliftedInitializerVariable(
          add_initializers_to=add_initializers_to,
          lifted_initializer_graph=lifted_initializer_graph, **kwds)
      created_variables.append(weakref.ref(v))
      return v

    self._created_variables = created_variables
    self._stateful_fn = self._defun_with_scope(variable_capturing_scope)
    self._stateful_fn._name = self._name  # pylint: disable=protected-access
    # Force the definition of the function for these arguments
    self._lifted_initializer_graph = lifted_initializer_graph
    self._graph_deleter = FunctionDeleter(self._lifted_initializer_graph)
    self._concrete_stateful_fn = (
        self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
            *args, **kwds))

    def invalid_creator_scope(*unused_args, **unused_kwds):
      """Disables variable creation."""
      raise ValueError(
          "tf.function-decorated function tried to create "
          "variables on non-first call.")

    self._stateless_fn = self._defun_with_scope(invalid_creator_scope)
    self._stateless_fn._name = self._name  # pylint: disable=protected-access

  def _decorate(self, decorator):
    """Allows the captured Python function to be decorated in place.

    This method is only safe to call when the Function has not been called by a
    user. It makes sense to use this method to push a decorator into the
    function rather than wrapping the function in the decorator.

    We use this in tf.Module to allow user annotated `tf.functions` to remain as
    `Function` objects but still automatically enter the Module name_scope
    when they are evaluated like all other methods.

    Args:
      decorator: A callable accepting a single argument which is the function
        to decorate and returning a callable result.

    Raises:
      ValueError: If the function has been called a ValueError is raised.
    """
    if self._stateful_fn is not None or self._stateless_fn is not None:
      raise ValueError(
          "Functions cannot be decorated after they have been traced.")

    self._python_function = decorator(self._python_function)
    self._function_spec = function_lib.FunctionSpec.from_function_and_signature(
        self._python_function, self.input_signature)

  def __call__(self, *args, **kwds):
    """Calls the graph function."""
    context.ensure_initialized()
    if RUN_FUNCTIONS_EAGERLY:
      return self._python_function(*args, **kwds)
    if self._created_variables:
      # In this case we have created variables on the first call, so we run the
      # defunned version which is guaranteed to never create variables.
      return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
    elif self._stateful_fn is not None:
      # In this case we have not created variables on the first call. So we can
      # run the first trace but we should fail if variables are created.
      results = self._stateful_fn(*args, **kwds)
      if self._created_variables:
        raise ValueError("Creating variables on a non-first call to a function"
                         " decorated with tf.function.")
      return results

    # This is the first call of __call__, so we have to initialize.
    initializer_map = object_identity.ObjectIdentityDictionary()
    self._initialize(args, kwds, add_initializers_to=initializer_map)
    if self._created_variables:
      try:
        # Attempt to initialize variables eagerly and without conds by lifting
        # out initialization graphs. This is the only initialization strategy
        # compatible with XLA at the moment.
        self._initialize_uninitialized_variables(initializer_map)
      except lift_to_graph.UnliftableError:
        pass  # Fall through to cond-based initialization.
      else:
        # Lifting succeeded, so variables are initialized and we can run the
        # stateless function.
        return self._stateless_fn(*args, **kwds)
    else:
      canon_args, canon_kwds = \
          self._stateful_fn._function_spec.canonicalize_function_inputs(  # pylint: disable=protected-access
              *args, **kwds)
      # If we did not create any variables the trace we have is good enough.
      return self._concrete_stateful_fn._filtered_call(canon_args, canon_kwds)  # pylint: disable=protected-access

    def fn_with_cond(*inner_args, **inner_kwds):
      """Conditionally runs initialization if it's needed."""
      condition = True
      for wr in self._created_variables:
        variable = wr()
        if variable is None:
          raise ValueError(
              "A tf.Variable created inside your tf.function has been"
              " garbage-collected. Your code needs to keep Python references"
              " to variables created inside `tf.function`s.\n"
              "\n"
              "A common way to raise this error is to create and return a"
              " variable only referenced inside your function:\n"
              "\n"
              "@tf.function\n"
              "def f():\n"
              "  v = tf.Variable(1.0)\n"
              "  return v\n"
              "\n"
              "v = f()  # Crashes with this error message!\n"
              "\n"
              "The reason this crashes is that @tf.function annotated"
              " function returns a **`tf.Tensor`** with the **value** of the"
              " variable when the function is called rather than the"
              " variable instance itself. As such there is no code holding a"
              " reference to the `v` created inside the function and Python"
              " garbage collects it.\n"
              "\n"
              "The simplest way to fix this issue is to create variables"
              " outside the function and capture them:\n"
              "\n"
              "v = tf.Variable(1.0)\n"
              "\n"
              "@tf.function\n"
              "def f():\n"
              "  return v\n"
              "\n"
              "f()  # <tf.Tensor: ... numpy=1.>\n"
              "v.assign_add(1.)\n"
              "f()  # <tf.Tensor: ... numpy=2.>")
        condition = math_ops.logical_and(
            condition, resource_variable_ops.var_is_initialized_op(
                variable.handle))
      # We want to call stateless_fn if possible because it avoids recomputing
      # potentially expensive initializers.
      return control_flow_ops.cond(
          condition,
          lambda: self._stateless_fn(*inner_args, **inner_kwds),
          functools.partial(self._concrete_stateful_fn._filtered_call,  # pylint: disable=protected-access
                            inner_args, inner_kwds))

    # We've created variables and are unable to lift the initialization graphs,
    # so we fall back to initializing with conds while running the function.
    canon_args, canon_kwds = \
        self._stateful_fn._function_spec.canonicalize_function_inputs(  # pylint: disable=protected-access
            *args, **kwds)
    return function_lib.defun(fn_with_cond)(*canon_args, **canon_kwds)

  @property
  def python_function(self):
    """The python function wrapped in this tf.function."""
    return self._python_function

  @property
  def input_signature(self):
    return self._function_spec.input_signature

  @property
  def function_spec(self):
    return self._function_spec

  def _initialize_uninitialized_variables(self, initializer_map):
    """Make and call a `ConcreteFunction` which initializes variables."""

    # Note: using defun here avoids an infinite recursion.
    @function_lib.defun
    def initialize_variables():
      op_map = object_identity.ObjectIdentityDictionary()
      for v, init in initializer_map.items():
        with ops.init_scope():
          if resource_variable_ops.var_is_initialized_op(v.handle):
            # Ignore variables which are already initialized at trace time.
            continue
        op_map = lift_to_graph.lift_to_graph(
            [init], ops.get_default_graph(), op_map=op_map)
        v.assign(op_map[init])

    with ops.init_scope():
      return initialize_variables.get_concrete_function()()

  def get_initialization_function(self, *args, **kwargs):
    """Returns a `ConcreteFunction` which initializes this function's variables.

    Requires that this function hasn't been accessed yet through either calling
    it or calling get_concrete_function. Fails if we cannot build an initializer
    function which does not depend on the concrete values of the inputs to this
    function.

    Note that running this function will overwrite any values currently assigned
    to variables, for example restores from a checkpoint.

    Args:
      *args: arguments to the underlying python callable.
      **kwargs: keyword arguments to the python callable.

    Returns:
      A `ConcreteFunction` object which initializes the variables of this
      function.

    Raises:
      RuntimeError: if called after the variables have been initialized.
    """
    if self._stateful_fn is not None:
      raise RuntimeError(
          "get_initialization_function cannot be called after the function "
          "has been used")
    # Here we trace the function, collect the initializers, and attempt to
    # extract them and run them eagerly. Fail only if we cannot do so.
    initializer_map = object_identity.ObjectIdentityDictionary()
    self._initialize(args, kwargs, add_initializers_to=initializer_map)

    # Note: using defun here avoids an infinite recursion.
    @function_lib.defun
    def initialize_variables():
      for v, init in initializer_map.items():
        v.assign(lift_to_graph.lift_to_graph(
            [init], ops.get_default_graph())[init])

    return initialize_variables.get_concrete_function()

  def _list_all_concrete_functions_for_serialization(self):
    """Returns all concrete functions for serialization.

    Returns:
      A list of instances of `Function`.
    """
    if self.input_signature is not None:
      self.get_concrete_function()
    concrete_functions = []
    # pylint: disable=protected-access
    if self._stateful_fn:
      concrete_functions.extend(
          self._stateful_fn._function_cache.all_values())
    if self._stateless_fn:
      concrete_functions.extend(
          self._stateless_fn._function_cache.all_values())
    # pylint: enable=protected-access
    seen_signatures = []
    for concrete_function in concrete_functions:
      signature = concrete_function.structured_input_signature
      flattened = nest.flatten(signature)
      if any(
          isinstance(arg, func_graph_module.UnknownArgument)
          for arg in flattened):
        logging.info("Unsupported signature for serialization: %s.", signature)
        continue
      equal_to_signature = functools.partial(
          function_lib.is_same_structure, signature, check_values=True)
      if not any(equal_to_signature(s) for s in seen_signatures):
        seen_signatures.append(signature)

    # Re-create concrete functions for these signatures. Re-creating ensures
    # that if the cache key has changed, the function will be traced again.
    concrete_functions = []
    for args, kwargs in seen_signatures:
      concrete_functions.append(self.get_concrete_function(*args, **kwargs))
    return concrete_functions

  def get_concrete_function(self, *args, **kwargs):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    If this `Function` was created with an `input_signature`, `args` and
    `kwargs` may be omitted. With an input signature there is only one
    concrete function associated with this `Function`.

    If there is no fixed `input_signature` associated with this
    `Function`, positional and keyword arguments to `get_concrete_function`
    follow the same rules as input signature specification, with `tf.TensorSpec`
    objects describing `tf.Tensor`s which will be passed to the concrete
    function.

    Each `tf.Tensor` argument to the concrete function must have a unique name,
    either because it is the only one associated with a named argument of the
    Python function or because an explicit `name=` was passed to its
    `tf.TensorSpec` object. These names become the argument names for the
    concrete function.

    Arguments to the concrete function may always be specified as keyword
    arguments, naming the Tensor input. Positional arguments may be used instead
    when each preceding argument to the Python function is a Tensor.

    ```python
    @tf.function
    def f(x):
      return x

    f_concrete = f.get_concrete_function(tf.TensorSpec([], tf.float64))
    f_concrete(tf.constant(1.))
    f_concrete(x=tf.constant(1.))
    ```

    Nested structures containing Tensors may be specified when retrieving
    concrete functions. Structures with multiple Tensors are expanded into
    multiple arguments of the concrete function. Since multiple concrete
    function arguments are associated with one argument to the original
    function, these Tensors must be named explicitly. Tensors in nested
    structures may not be passed using positional arguments when calling the
    concrete function.

    ```python
    f_concrete2 = f.get_concrete_function(
        (tf.TensorSpec(None, tf.float64, name="first"),
         tf.TensorSpec([], tf.float32, name="second")))
    # Keyword arguments are required when identifying Tensors in nested
    # structures.
    f_concrete2(first=tf.constant([1.]), second=tf.constant(0.))
    ```

    Functions with fixed input signatures have only one concrete function
    associated with them, which can be retrieved without specifying any
    arguments. As before Tensors must have unique names, either inferred from
    the argument names in the original Python function or specified
    explicitly.

    ```python
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float32)))
    def f_sig(y):
      return y

    f_sig_concrete = f.get_concrete_function()
    f_sig_concrete(tf.constant(1.))
    f_sig_concrete(y=tf.constant(1.))
    ```

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.

    Returns:
      A TensorFlow function which takes exactly one `tf.Tensor` per argument.

    Raises:
      ValueError: if this object has not yet been called on concrete values.
    """
    if self._stateful_fn is None:
      initializer_map = object_identity.ObjectIdentityDictionary()
      self._initialize(args, kwargs, add_initializers_to=initializer_map)
      self._initialize_uninitialized_variables(initializer_map)

    if self._created_variables:
      # In this case we have created variables on the first call, so we run the
      # defunned version which is guaranteed to never create variables.
      return self._stateless_fn.get_concrete_function(*args, **kwargs)
    elif self._stateful_fn is not None:
      # In this case we have not created variables on the first call. So we can
      # run the first trace but we should fail if variables are created.
      concrete = self._stateful_fn.get_concrete_function(*args, **kwargs)
      if self._created_variables:
        raise ValueError("Creating variables on a non-first call to a function"
                         " decorated with tf.function.")
      return concrete

  def __get__(self, instance, owner):
    """Makes it possible to defun instance methods."""
    del owner
    # `instance` here is the instance that this `Function` was accessed through
    # e.g., for
    #
    #   class Foo(object):
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
    # tf.function. Keeps a cache to avoid retracing the function every time the
    # descriptor is accessed.
    if instance not in self._descriptor_cache:
      if instance is None:
        return self
      self._descriptor_cache[instance] = (
          function_lib.class_method_to_instance_method(self, instance))
    return self._descriptor_cache[instance]


@tf_export("function")
def function(func=None,
             input_signature=None,
             autograph=True,
             experimental_autograph_options=None,
             experimental_relax_shapes=False,
             experimental_compile=None):
  """Creates a callable TensorFlow graph from a Python function.

  `function` constructs a callable that executes a TensorFlow graph
  (`tf.Graph`) created by tracing the TensorFlow operations in `func`.
  This allows the TensorFlow runtime to apply optimizations and exploit
  parallelism in the computation defined by `func`.

  _Example Usage_

  ```python
  def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  g = tf.function(f)

  x = tf.constant([[2.0, 3.0]])
  y = tf.constant([[3.0, -2.0]])

  # `f` and `g` will return the same value, but `g` will be executed as a
  # TensorFlow graph.
  assert f(x, y).numpy() == g(x, y).numpy()

  # Tensors and tf.Variables used by the Python function are captured in the
  # graph.
  @tf.function
  def h():
    return f(x, y)

  assert (h().numpy() == f(x, y).numpy()).all()

  # Data-dependent control flow is also captured in the graph. Supported
  # control flow statements include `if`, `for`, `while`, `break`, `continue`,
  # `return`.
  @tf.function
  def g(x):
    if tf.reduce_sum(x) > 0:
      return x * x
    else:
      return -x // 2

  # print and TensorFlow side effects are supported, but exercise caution when
  # using Python side effects like mutating objects, saving to files, etc.
  l = []

  @tf.function
  def g(x):
    for i in x:
      print(i)                              # Works
      tf.compat.v1.assign(v, i)                       # Works
      tf.compat.v1.py_func(lambda i: l.append(i))(i)  # Works
      l.append(i)                           # Caution! Doesn't work.
  ```

  Note that unlike other TensorFlow operations, we don't convert python
  numerical inputs to tensors. Moreover, a new graph is generated for each
  distinct python numerical value, for example calling `g(2)` and `g(3)` will
  generate two new graphs (while only one is generated if you call
  `g(tf.constant(2))` and `g(tf.constant(3))`). Therefore, python numerical
  inputs should be restricted to arguments that will have few distinct values,
  such as hyperparameters like the number of layers in a neural network. This
  allows TensorFlow to optimize each variant of the neural network.

  _Referencing `tf.Variable`s_

  The Python function `func` may reference stateful objects (such as
  `tf.Variable`).
  These are captured as implicit inputs to the callable returned by `function`.
  For example:

  ```python
  c = tf.Variable(0)

  @tf.function
  def f(x):
    c.assign_add(1)
    return x + tf.compat.v1.to_float(c)

  assert int(c) == 0
  assert f(1.0) == 2.0
  assert int(c) == 1
  assert f(1.0) == 3.0
  assert int(c) == 2
  ```

  `function` can be applied to methods of an object. For example:

  ```python
  class Dense(object):
    def __init__(self):
      self.W = tf.Variable(tf.compat.v1.glorot_uniform_initializer()((10, 10)))
      self.b = tf.Variable(tf.zeros(10))

    @tf.function
    def compute(self, x):
      return tf.matmul(x, self.W) + self.b

  d1 = Dense()
  d2 = Dense()
  x = tf.random.uniform((10, 10))
  # d1 and d2 are using distinct variables
  assert not (d1.compute(x).numpy() == d2.compute(x).numpy()).all()
  ```

  _Usage with `tf.keras`_

  The `call` methods of a `tf.keras.Model` subclass can be decorated with
  `function` in order to apply graph execution optimizations on it.
  For example:

  ```python
  class MyModel(tf.keras.Model):
    def __init__(self, keep_probability=0.2):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4)
      self.dense2 = tf.keras.layers.Dense(5)
      self.keep_probability = keep_probability

    @tf.function
    def call(self, inputs, training=True):
      y = self.dense2(self.dense1(inputs))
      if training:
        return tf.nn.dropout(y, self.keep_probability)
      else:
        return y

  model = MyModel()
  model(x, training=True)  # executes a graph, with dropout
  model(x, training=False) # executes a graph, without dropout
  ```

  _Input Signatures_

  `function` instantiates a separate graph for every unique set of input
  shapes and datatypes. For example, the following code snippet will result
  in three distinct graphs being traced, as each input has a different
  shape.

  ```python
  @tf.function
  def f(x): return tf.add(x, 1.)

  scalar = tf.constant(1.0)
  vector = tf.constant([1.0, 1.0])
  matrix = tf.constant([[3.0]])

  f(scalar)
  f(vector)
  f(matrix)
  ```

  An "input signature" can be optionally provided to `function` to control
  the graphs traced. The input signature specifies the shape and type of each
  `Tensor` argument to the function using a `tf.TensorSpec` object. For example,
  the following code snippet ensures that a single graph is created where the
  input `Tensor` is required to be a floating point tensor with no restrictions
  on shape.

  ```python
  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def f(x): return tf.add(x, 1.)
  ```

  When an `input_signature` is specified, the callable will convert the inputs
  to the specified TensorSpecs.

  _Tracing and staging_

  When `autograph` is `True`, all Python control flow that depends on `Tensor`
  values is staged into a TensorFlow graph. When `autograph` is `False`, the
  function is traced and control flow is not allowed to depend on data.

  Note that `function` only stages TensorFlow operations, all Python code that
  `func` executes and does not depend on data will shape the _construction_ of
  the graph.
  For example, consider the following:

  ```python
  import numpy as np

  def add_noise():
    return tf.eye(5) + np.random.randn(5, 5)

  traced = tf.function(add_noise)
  ```

  `add_noise()` will return a different output every time it is invoked.
  However, `traced()` will return the same value every time it is called,
  since a particular random value generated by the `np.random.randn` call will
  be inserted in the traced/staged TensorFlow graph as a constant. In this
  particular example, replacing `np.random.randn(5, 5)` with
  `tf.random.normal((5, 5))` will result in the same behavior for `add_noise()`
  and `traced()`.

  _Python Side-Effects_

  A corollary of the previous discussion on tracing is the following: If a
  Python function `func` has Python side-effects, then executing `func` multiple
  times may not be semantically equivalent to executing `F = tf.function(func)`
  multiple times; this difference is due to the fact that `function` only
  captures the subgraph of TensorFlow operations that is constructed when `func`
  is invoked to trace a graph.

  The same is true if code with Python side effects is used inside control flow,
  such as a loop. If your code uses side effects that are not intended to
  control graph construction, wrap them inside `tf.compat.v1.py_func`.

  _Retracing_

  A single tf.function object might need to map to multiple computation graphs
  under the hood. This should be visible only as performance (tracing graphs has
  a nonzero computational and memory cost) but should not affect the correctness
  of the program. A traced function should return the same result as it would
  when run eagerly, assuming no unintended Python side-effects.

  Calling a `tf.function` with tensor arguments of different dtypes should lead
  to at least one computational graph per distinct set of dtypes. Alternatively,
  always calling a `tf.function` with tensor arguments of the same shapes and
  dtypes and the same non-tensor arguments should not lead to additional
  retracings of your function.

  Other than that, TensorFlow reserves the right to retrace functions as many
  times as needed, to ensure that traced functions behave as they would when run
  eagerly and to provide the best end-to-end performance. For example, the
  behavior of how many traces TensorFlow will do when the function is repeatedly
  called with different python scalars as arguments is left undefined to allow
  for future optimizations.

  To control the tracing behavior, use the following tools:
   - different `tf.function` objects are guaranteed to not share traces; and
   - specifying a signature or using concrete function objects returned from
     get_concrete_function() guarantees that only one function graph will be
     built.

  Args:
    func: function to be compiled. If `func` is None, returns a decorator that
      can be invoked with a single argument - `func`. The end result is
      equivalent to providing all the arguments up front. In other words,
      `tf.function(input_signature=...)(func)` is equivalent to
      `tf.function(func, input_signature=...)`. The former can be used to
      decorate Python functions, for example:
        @tf.function(input_signature=...)
        def foo(...): ...
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects
      specifying the shapes and dtypes of the Tensors that will be supplied to
      this function. If `None`, a separate function is instantiated for each
      inferred input signature.  If input_signature is specified, every input to
      `func` must be a `Tensor`, and `func` cannot accept `**kwargs`.
    autograph: Whether autograph should be applied on `func` before tracing a
      graph. This allows for dynamic control flow (Python if's, loops etc.)
      in the traced graph. See https://www.tensorflow.org/guide/autograph for
        more information.
    experimental_autograph_options: Experimental knobs (in the form of a tuple
      of tensorflow.autograph.Feature values) to control behavior when
      autograph=True.
    experimental_relax_shapes: When true, argument shapes may be relaxed to
      avoid unecessary retracing.
    experimental_compile: If false, execute the function in a regular way. The
      function is optimized by some graph rewrite passes (some ops might be
      clustered into a single op) and interpreted by the standard TensorFlow
      executor, which dispatches op kernels one by one as they become
      executable. Set it to false when directly running a multi-device function
      on TPUs (e.g. two TPU cores, one TPU core and its host CPU). If True, the
      function is compiled directly by XLA (https://www.tensorflow.org/xla).
      XLA would fuse all the ops and emit more efficient code to run for some
      devices (e.g. TPU, XLA_GPU) and some use cases (e.g. dense tensor
      computation). It requires that the whole function is compilable by XLA
      (e.g. static tensor shape, a subset of operations, no string, compile-time
      constant input, etc). If None (default), compile the function with XLA
      when running on TPU and go through the regular function execution path
      when running on other devices. Note: TensorArrays on TPU don't work with
      standard TensorFlow executor.

  Returns:
     If `func` is not None, returns a callable that will execute the compiled
     function (and return zero or more `tf.Tensor` objects).
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a callable equivalent to the case above.

  Raises:
    TypeError: If `input_signature` is neither `None` nor a sequence of
      `TensorSpec` objects.
  """
  if input_signature is not None:
    function_lib.validate_signature(input_signature)

  def decorated(inner_function):
    try:
      name = inner_function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        inner_function,
        Function(
            inner_function,
            name,
            input_signature=input_signature,
            autograph=autograph,
            experimental_autograph_options=experimental_autograph_options,
            experimental_relax_shapes=experimental_relax_shapes,
            experimental_compile=experimental_compile))

  # This code path is for the `foo = tf.function(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the
  #
  # @tf.function(...)
  # def foo(...):
  #    ...
  #
  # use case, which is equivalent to `foo = tf.function(...)(foo)`
  return decorated
