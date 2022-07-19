# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Decorator to overrides the gradient for a function."""

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import tape as tape_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


VAR_OP_TYPES = [
    "VariableV2",
    "VarHandleOp",
]


@tf_export("custom_gradient")
def custom_gradient(f=None):
  """Decorator to define a function with a custom gradient.

  This decorator allows fine grained control over the gradients of a sequence
  for operations.  This may be useful for multiple reasons, including providing
  a more efficient or numerically stable gradient for a sequence of operations.

  For example, consider the following function that commonly occurs in the
  computation of cross entropy and log likelihoods:

  ```python
  def log1pexp(x):
    return tf.math.log(1 + tf.exp(x))
  ```

  Due to numerical instability, the gradient of this function evaluated at x=100
  is NaN.  For example:

  ```python
  x = tf.constant(100.)
  y = log1pexp(x)
  dy_dx = tf.gradients(y, x) # Will be NaN when evaluated.
  ```

  The gradient expression can be analytically simplified to provide numerical
  stability:

  ```python
  @tf.custom_gradient
  def log1pexp(x):
    e = tf.exp(x)
    def grad(upstream):
      return upstream * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad
  ```

  With this definition, the gradient `dy_dx` at `x = 100` will be correctly
  evaluated as 1.0.

  The variable `upstream` is defined as the upstream gradient. i.e. the gradient
  from all the layers or functions originating from this layer. The above
  example has no upstream functions, therefore `upstream = dy/dy = 1.0`.

  Assume that `x_i` is `log1pexp` in the forward pass `x_1 = x_1(x_0)`,
  `x_2 = x_2(x_1)`, ..., `x_i = x_i(x_i-1)`, ..., `x_n = x_n(x_n-1)`. By
  chain rule we know that `dx_n/dx_0 = dx_n/dx_n-1 * dx_n-1/dx_n-2 * ... *
  dx_i/dx_i-1 * ... * dx_1/dx_0`.

  In this case the gradient of our current function defined as
  `dx_i/dx_i-1 = (1 - 1 / (1 + e))`. The upstream gradient `upstream` would be
  `dx_n/dx_n-1 * dx_n-1/dx_n-2 * ... * dx_i+1/dx_i`. The upstream gradient
  multiplied by the current gradient is then passed downstream.

  In case the function takes multiple variables as input, the `grad`
  function must also return  the same number of variables.
  We take the function `z = x * y` as an example.

  >>> @tf.custom_gradient
  ... def bar(x, y):
  ...   def grad(upstream):
  ...     dz_dx = y
  ...     dz_dy = x
  ...     return upstream * dz_dx, upstream * dz_dy
  ...   z = x * y
  ...   return z, grad
  >>> x = tf.constant(2.0, dtype=tf.float32)
  >>> y = tf.constant(3.0, dtype=tf.float32)
  >>> with tf.GradientTape(persistent=True) as tape:
  ...   tape.watch(x)
  ...   tape.watch(y)
  ...   z = bar(x, y)
  >>> z
  <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
  >>> tape.gradient(z, x)
  <tf.Tensor: shape=(), dtype=float32, numpy=3.0>
  >>> tape.gradient(z, y)
  <tf.Tensor: shape=(), dtype=float32, numpy=2.0>

  Nesting custom gradients can lead to unintuitive results. The default
  behavior does not correspond to n-th order derivatives. For example

  ```python
  @tf.custom_gradient
  def op(x):
    y = op1(x)
    @tf.custom_gradient
    def grad_fn(dy):
      gdy = op2(x, y, dy)
      def grad_grad_fn(ddy):  # Not the 2nd order gradient of op w.r.t. x.
        return op3(x, y, dy, ddy)
      return gdy, grad_grad_fn
    return y, grad_fn
  ```

  The function `grad_grad_fn` will be calculating the first order gradient
  of `grad_fn` with respect to `dy`, which is used to generate forward-mode
  gradient graphs from backward-mode gradient graphs, but is not the same as
  the second order gradient of `op` with respect to `x`.

  Instead, wrap nested `@tf.custom_gradients` in another function:

  ```python
  @tf.custom_gradient
  def op_with_fused_backprop(x):
    y, x_grad = fused_op(x)
    def first_order_gradient(dy):
      @tf.custom_gradient
      def first_order_custom(unused_x):
        def second_order_and_transpose(ddy):
          return second_order_for_x(...), gradient_wrt_dy(...)
        return x_grad, second_order_and_transpose
      return dy * first_order_custom(x)
    return y, first_order_gradient
  ```

  Additional arguments to the inner `@tf.custom_gradient`-decorated function
  control the expected return values of the innermost function.

  The examples above illustrate how to specify custom gradients for functions
  which do not read from variables. The following example uses variables, which
  require special handling because they are effectively inputs of the forward
  function.

  >>> weights = tf.Variable(tf.ones([2]))  # Trainable variable weights
  >>> @tf.custom_gradient
  ... def linear_poly(x):
  ...   # Creating polynomial
  ...   poly = weights[1] * x + weights[0]
  ...
  ...   def grad_fn(dpoly, variables):
  ...     # dy/dx = weights[1] and we need to left multiply dpoly
  ...     grad_xs = dpoly * weights[1]  # Scalar gradient
  ...
  ...     grad_vars = []  # To store gradients of passed variables
  ...     assert variables is not None
  ...     assert len(variables) == 1
  ...     assert variables[0] is weights
  ...     # Manually computing dy/dweights
  ...     dy_dw = dpoly * tf.stack([x ** 1, x ** 0])
  ...     grad_vars.append(
  ...         tf.reduce_sum(tf.reshape(dy_dw, [2, -1]), axis=1)
  ...     )
  ...     return grad_xs, grad_vars
  ...   return poly, grad_fn
  >>> x = tf.constant([1., 2., 3.])
  >>> with tf.GradientTape(persistent=True) as tape:
  ...   tape.watch(x)
  ...   poly = linear_poly(x)
  >>> poly # poly = x + 1
  <tf.Tensor: shape=(3,),
    dtype=float32,
    numpy=array([2., 3., 4.], dtype=float32)>
  >>> tape.gradient(poly, x)  # conventional scalar gradient dy/dx
  <tf.Tensor: shape=(3,),
    dtype=float32,
    numpy=array([1., 1., 1.], dtype=float32)>
  >>> tape.gradient(poly, weights)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 3.], dtype=float32)>

  Above example illustrates usage of trainable variable `weights`.
  In the example, the inner `grad_fn` accepts an extra `variables` input
  parameter and also returns an extra `grad_vars` output. That extra argument
  is passed if the forward function reads any variables. You need to
  compute the gradient w.r.t. each of those `variables` and output it as a list
  of `grad_vars`. Note here that default value of `variables` is set to `None`
  when no variables are used in the forward function.

  It should be noted `tf.GradientTape` is still watching the forward pass of a
  `tf.custom_gradient`, and will use the ops it watches. As a consequence,
  calling `tf.function` while the tape is still watching leads
  to a gradient graph being built. If an op is used in `tf.function` without
  registered gradient, a `LookupError` will be raised.

  Users can insert `tf.stop_gradient` to customize this behavior. This
  is demonstrated in the example below. `tf.random.shuffle` does not have a
  registered gradient. As a result `tf.stop_gradient` is used to avoid the
  `LookupError`.

  ```python
  x = tf.constant([0.3, 0.5], dtype=tf.float32)

  @tf.custom_gradient
  def test_func_with_stop_grad(x):
    @tf.function
    def _inner_func():
      # Avoid exception during the forward pass
      return tf.stop_gradient(tf.random.shuffle(x))
      # return tf.random.shuffle(x)  # This will raise

    res = _inner_func()
    def grad(upstream):
      return upstream  # Arbitrarily defined custom gradient
    return res, grad

  with tf.GradientTape() as g:
    g.watch(x)
    res = test_func_with_stop_grad(x)

  g.gradient(res, x)
  ```

  See also `tf.RegisterGradient` which registers a gradient function for a
  primitive TensorFlow operation. `tf.custom_gradient` on the other hand allows
  for fine grained control over the gradient computation of a sequence of
  operations.

  Note that if the decorated function uses `Variable`s, the enclosing variable
  scope must be using 
  [ResourceVariables](https://www.tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables).

  Args:
    f: function `f(*x)` that returns a tuple `(y, grad_fn)` where:
       - `x` is a sequence of (nested structures of) `Tensor` inputs to the
         function.
       - `y` is a (nested structure of) `Tensor` outputs of applying TensorFlow
         operations in `f` to `x`.
       - `grad_fn` is a function with the signature `g(*grad_ys)` which returns
         a list of `Tensor`s the same size as (flattened) `x` - the derivatives
         of `Tensor`s in `y` with respect to the `Tensor`s in `x`.  `grad_ys` is
         a sequence of `Tensor`s the same size as (flattened) `y` holding the
         initial value gradients for each `Tensor` in `y`.

         In a pure mathematical sense, a vector-argument vector-valued function
         `f`'s derivatives should be its Jacobian matrix `J`. Here we are
         expressing the Jacobian `J` as a function `grad_fn` which defines how
         `J` will transform a vector `grad_ys` when left-multiplied with it
         (`grad_ys * J`, the vector-Jacobian product, or VJP). This functional
         representation of a matrix is convenient to use for chain-rule
         calculation (in e.g. the back-propagation algorithm).

         If `f` uses `Variable`s (that are not part of the
         inputs), i.e. through `get_variable`, then `grad_fn` should have
         signature `g(*grad_ys, variables=None)`, where `variables` is a list of
         the `Variable`s, and return a 2-tuple `(grad_xs, grad_vars)`, where
         `grad_xs` is the same as above, and `grad_vars` is a `list<Tensor>`
         with the derivatives of `Tensor`s in `y` with respect to the variables
         (that is, grad_vars has one Tensor per variable in variables).

  Returns:
    A function `h(x)` which returns the same value as `f(x)[0]` and whose
    gradient (as calculated by `tf.gradients`) is determined by `f(x)[1]`.
  """

  if f is None:
    return lambda f: custom_gradient(f=f)

  @Bind.decorator
  def decorated(wrapped, args, kwargs):
    """Decorated function with custom gradient."""
    if context.executing_eagerly():
      return _eager_mode_decorator(wrapped, args, kwargs)
    else:
      return _graph_mode_decorator(wrapped, args, kwargs)

  return tf_decorator.make_decorator(f, decorated(f))  # pylint: disable=no-value-for-parameter


class Bind:
  """When called evaluates `d(f, args, kwargs)` but supports binding `f`.

  >>> @Bind.decorator
  ... def my_decorator(f, args, kwargs):
  ...   print("my_decorator called with", args, kwargs)
  ...   return f(*args, **kwargs)

  >>> class Foo:
  ...   @my_decorator
  ...   def bar(self, a, b, c):
  ...     return a * b * c

  >>> Foo.bar(None, 1, 2, c=3)
  my_decorator called with (None, 1, 2) {'c': 3}
  6

  >>> foo = Foo()
  >>> foo.bar(1, 2, c=3)
  my_decorator called with (1, 2) {'c': 3}
  6
  """

  @classmethod
  def decorator(cls, d):
    return lambda f: Bind(f, d)

  def __init__(self, f, d):
    self._f = f
    self._d = d

  def __get__(self, instance, owner):
    if instance is not None:
      f = self._f.__get__(instance, owner)
      return tf_decorator.make_decorator(f, Bind(f, self._d))
    else:
      return self

  def __call__(self, *a, **k):
    return self._d(self._f, a, k)


def get_variable_by_name(var_name):
  """Given a variable name, retrieves a handle on the tensorflow Variable."""
  global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def _filter_fn(item):
    try:
      return var_name == item.op.name
    except AttributeError:
      # Collection items without operation are ignored.
      return False

  candidate_vars = list(filter(_filter_fn, global_vars))

  if len(candidate_vars) >= 1:
    # Filter out non-trainable variables.
    candidate_vars = [v for v in candidate_vars if v.trainable]
  else:
    raise ValueError("Unsuccessful at finding variable {}.".format(var_name))

  if len(candidate_vars) == 1:
    return candidate_vars[0]
  elif len(candidate_vars) > 1:
    raise ValueError(
        "Unsuccessful at finding trainable variable {}. "
        "Number of candidates: {}. "
        "Candidates: {}".format(var_name, len(candidate_vars), candidate_vars))
  else:
    # The variable is not trainable.
    return None


def _get_dependent_variables(input_ops, output_ops):
  """Finds variables involved in the subgraph between input_ops and output_ops.

  Args:
    input_ops: Flattened list of input ops
    output_ops: Flattened list of output ops

  Returns:
    A list of variables
  """

  # avoids the edge-case when input_ops == output_ops.
  output_ops = nest.map_structure(gen_array_ops.identity, output_ops)
  inbetween_ops = op_selector.get_backward_walk_ops(
      seed_ops=output_ops,
      stop_at_ts=input_ops,
      inclusive=False,
      only_differentiable=True)
  var_ops = (op for op in inbetween_ops if op.type in VAR_OP_TYPES)
  var_names = (op.name for op in var_ops)
  tf_vars = (get_variable_by_name(var_name) for var_name in var_names)
  tf_vars = [v for v in tf_vars if v is not None]
  return tf_vars


def generate_name():
  return "CustomGradient-%s" % ops.uid()


def _graph_mode_decorator(f, args, kwargs):
  """Implement custom gradient decorator for graph mode."""
  # TODO(rsepassi): Add support for kwargs
  if kwargs:
    raise ValueError(
        "The custom_gradient decorator currently supports keywords "
        "arguments only when eager execution is enabled.")
  name = generate_name()
  args = nest.map_structure(ops.convert_to_tensor, args)

  # Checking global and local variables attempts to ensure that no non-resource
  # Variables are added to the graph.
  current_var_scope = variable_scope.get_variable_scope()
  before_vars = set([
      v.ref() for v in current_var_scope.global_variables() +
      current_var_scope.local_variables()
  ])
  with tape_lib.VariableWatcher() as variable_watcher:
    result, grad_fn = f(*args)

  args = nest.flatten(args)
  flat_result = nest.flatten(result)
  flat_result_len = len(flat_result)

  after_vars = set([
      v.ref() for v in current_var_scope.global_variables() +
      current_var_scope.local_variables()
  ])
  new_vars = after_vars - before_vars
  new_vars_list = [v.deref() for v in new_vars]
  for v in new_vars_list:
    if not resource_variable_ops.is_resource_variable(v):
      raise TypeError(
          "All variables used by a function wrapped with @custom_gradient must "
          "be `ResourceVariable`s. Ensure that no `variable_scope` is created "
          "with `use_resource=False`.")

  # The variables that grad_fn needs to return gradients for are the set of
  # variables used that are *not* part of the inputs.
  variables_in_tape = frozenset([
      v.ref() for v in variable_watcher.watched_variables()
  ])

  graphs = {getattr(o, "graph", None) for o in flat_result}
  # Not all results may be tensors. However, we want to ensure all tensor
  # outputs are from the same graph and get a list of captured inputs for
  # variable search
  graphs.discard(None)  # Discard non-graph outputs
  if graphs:
    if len(graphs) > 1:
      raise ValueError(
          "All custom_gradient outputs should be from the same graph")
    output_graph = graphs.pop()
    filtered_input_tensors = []
    for i in args:
      if i.graph == output_graph:
        filtered_input_tensors.append(i)
  else:
    filtered_input_tensors = args

  variables_in_subgraph = frozenset([
      v.ref() for v in _get_dependent_variables(
          input_ops=filtered_input_tensors, output_ops=flat_result)
  ])
  variables = sorted(
      [v.deref() for v in variables_in_subgraph.union(variables_in_tape)],
      key=lambda v: v.name)

  grad_argspec = tf_inspect.getfullargspec(grad_fn)
  variables_in_signature = ("variables" in grad_argspec.args or
                            "variables" in grad_argspec.kwonlyargs or
                            grad_argspec.varkw)
  if variables and not variables_in_signature:
    raise TypeError(
        "@tf.custom_gradient grad_fn must accept keyword argument 'variables', "
        "since function uses variables: {}".format(variables))
  if variables_in_signature and not variables:
    # User seems to intend to use variables but none were captured.
    logging.vlog(
        1, "@custom_gradient grad_fn has 'variables' in signature, "
        "but no ResourceVariables were used on the forward pass.")

  all_tensors = flat_result + args + variables

  def tape_grad_fn(*result_grads):
    """Custom grad fn wrapper."""
    result_grads = result_grads[:flat_result_len]
    if variables:
      input_grads, variable_grads = grad_fn(*result_grads, variables=variables)
      if len(variable_grads) != len(variables):
        raise ValueError("Must return gradient for each variable from "
                         "@custom_gradient grad_fn.")
    else:
      input_grads = grad_fn(*result_grads)
      variable_grads = []

    # Need to return one value per input to the IdentityN, so pad the
    # gradients of the inputs of the custom_gradient function with the
    # gradients of the outputs as well.
    input_grads = nest.flatten(input_grads)
    return ([None] * flat_result_len) + input_grads + variable_grads

  @ops.RegisterGradient(name)
  def internal_grad_fn(unused_op, *result_grads):  # pylint: disable=unused-variable
    """Custom grad fn wrapper."""
    return tape_grad_fn(*result_grads)

  original_tensors = all_tensors
  with ops.get_default_graph().gradient_override_map({"IdentityN": name}):
    all_tensors = array_ops.identity_n(all_tensors)

  original_tensors = [ops.convert_to_tensor(x) for x in original_tensors]

  # Propagate handle data for happier shape inference for resource variables.
  for i, t in enumerate(original_tensors):
    if t.dtype == dtypes.resource and hasattr(t, "_handle_data"):
      all_tensors[i]._handle_data = t._handle_data  # pylint: disable=protected-access
  tape_lib.record_operation(
      f.__name__, all_tensors, original_tensors, tape_grad_fn)
  for ot, t in zip(original_tensors, all_tensors):
    handle_data_util.copy_handle_data(ot, t)
  return nest.pack_sequence_as(
      structure=result, flat_sequence=all_tensors[:flat_result_len])


def _eager_mode_decorator(f, args, kwargs):
  """Implement custom gradient decorator for eager mode."""
  with tape_lib.VariableWatcher() as variable_watcher:
    result, grad_fn = f(*args, **kwargs)
  args = nest.flatten(args)
  all_inputs = list(args) + list(kwargs.values())
  # The variables that grad_fn needs to return gradients for are the set of
  # variables used that are *not* part of the inputs.
  variables = [
      v.deref()  # pylint: disable=g-complex-comprehension
      for v in set(v.ref() for v in variable_watcher.watched_variables())
      if all(v.deref() is not i for i in all_inputs)
  ]
  grad_argspec = tf_inspect.getfullargspec(grad_fn)
  if (variables and ("variables" not in grad_argspec.args) and
      ("variables" not in grad_argspec.kwonlyargs) and
      not grad_argspec.varkw):
    raise TypeError(
        "@tf.custom_gradient grad_fn must accept keyword argument 'variables', "
        "since function uses variables: {}".format(variables))
  flat_result = nest.flatten(result)
  # TODO(apassos) consider removing the identity below.
  flat_result = [gen_array_ops.identity(x) for x in flat_result]

  input_tensors = [ops.convert_to_tensor(x) for x
                   in list(args) + list(variables)]

  recorded_inputs = input_tensors
  arg_count = len(args)

  def actual_grad_fn(*result_grads):
    """Custom grad fn wrapper."""
    if variables:
      input_grads, variable_grads = grad_fn(*result_grads, variables=variables)
      if len(variable_grads) != len(variables):
        raise ValueError("Must return gradient for each variable from "
                         "@custom_gradient grad_fn.")
    else:
      input_grads = grad_fn(*result_grads)
      variable_grads = []
    flat_grads = nest.flatten(input_grads)
    if len(flat_grads) != arg_count:
      raise ValueError(
          "custom_gradient function expected to return", arg_count,
          "gradients but returned", len(flat_grads), "instead.")
    return flat_grads + variable_grads

  tape_lib.record_operation(f.__name__, flat_result, recorded_inputs,
                            actual_grad_fn)
  flat_result = list(flat_result)
  return nest.pack_sequence_as(result, flat_result)


@tf_export("recompute_grad")
def recompute_grad(f):
  """Defines a function as a recompute-checkpoint for the tape auto-diff.

  Tape checkpointing is a technique to reduce the memory consumption of the
  auto-diff tape:

  - Without tape checkpointing operations and intermediate values are
  recorded to the tape for use in the backward pass.

  - With tape checkpointing, only the function call and its inputs are
  recorded. During back-propagation the `recompute_grad` custom gradient
  (`tf.custom_gradient`) recomputes the function under a localized Tape object.
  This recomputation of the function during backpropagation performs redundant
  calculation, but reduces the overall memory usage of the Tape.

  >>> y = tf.Variable(1.0)

  >>> def my_function(x):
  ...   tf.print('running')
  ...   z = x*y
  ...   return z

  >>> my_function_recompute = tf.recompute_grad(my_function)

  >>> with tf.GradientTape() as tape:
  ...   r = tf.constant(1.0)
  ...   for i in range(4):
  ...     r = my_function_recompute(r)
  running
  running
  running
  running

  >>> grad = tape.gradient(r, [y])
  running
  running
  running
  running

  Without `recompute_grad`, the tape contains all intermitate steps, and no
  recomputation is performed.

  >>> with tf.GradientTape() as tape:
  ...   r = tf.constant(1.0)
  ...   for i in range(4):
  ...     r = my_function(r)
  running
  running
  running
  running

  >>> grad = tape.gradient(r, [y])


  If `f` was a `tf.keras` `Model` or `Layer` object, methods and attributes
  such as `f.variables` are not available on the returned function `g`.
  Either keep a reference of `f` , or use `g.__wrapped__` for accessing
  these variables and methods.


  >>> def print_running_and_return(x):
  ...   tf.print("running")
  ...   return x

  >>> model = tf.keras.Sequential([
  ...   tf.keras.layers.Lambda(print_running_and_return),
  ...   tf.keras.layers.Dense(2)
  ... ])

  >>> model_recompute = tf.recompute_grad(model)

  >>> with tf.GradientTape(persistent=True) as tape:
  ...   r = tf.constant([[1,2]])
  ...   for i in range(4):
  ...     r = model_recompute(r)
  running
  running
  running
  running

  >>> grad = tape.gradient(r, model.variables)
  running
  running
  running
  running

  Alternatively, use the `__wrapped__` attribute to access the original
  model object.

  >>> grad = tape.gradient(r, model_recompute.__wrapped__.variables)
  running
  running
  running
  running


  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.

  Returns:
    A function `g` wrapping `f` that defines a custom gradient, which recomputes
    `f` on the backwards pass of a gradient call.
  """
  # TODO(cdfreeman) Add is_recomputing functionality from graph mode version

  @custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    current_var_scope = variable_scope.get_variable_scope()
    with tape_lib.stop_recording():
      result = f(*args, **kwargs)

    def grad_wrapper(*wrapper_args, variables=None):
      """Wrapper function to accomodate lack of kwargs in graph mode custom_gradient."""

      @custom_gradient
      def inner_recompute_grad(*dresult):
        """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
        # Gradient calculation for reverse mode autodiff.
        with backprop.GradientTape() as t:
          id_args = nest.map_structure(gen_array_ops.identity, args)
          # Tuple `dresult` should contain at least one tensor.
          assert len(dresult) >= 1

          if not context.executing_eagerly():
            # XLA doesn't respect `tf.control_dependencies`. The code block
            # below manually adds a data dependency to `dresult` to ensure
            # recomputation of `f(*args, **kwargs)` happens after `dresult`.

            # This works even if `dresult[0]` is a size 0 tensor as reduce_max
            # of a size 0 tensor returns -inf. Use reshape here to avoid reading
            # the entire `dresult[0]`.
            elem = math_ops.reduce_max(array_ops.reshape(dresult[0], [-1])[:1])
            # Cast elem to bool in case elem is NaN.
            elem_bool = math_ops.cast(elem, dtypes.bool)
            dresult_dep = array_ops.where_v2(
                elem_bool == elem_bool, 0., float("nan"))  # pylint: disable=comparison-with-itself
            id_args = nest.map_structure(
                lambda x: x + math_ops.cast(dresult_dep, x.dtype), id_args)

          t.watch(id_args)
          if variables is not None:
            t.watch(variables)
          with variable_scope.variable_scope(current_var_scope):
            recomputed_result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
          kw_vars = list(variables)
        grads = t.gradient(
            recomputed_result,
            list(id_args) + kw_vars,
            output_gradients=dresult,
            unconnected_gradients=UnconnectedGradients.ZERO)

        def transpose(*t_args, **t_kwargs):
          """Gradient function calculation for forward mode autodiff."""
          # Just throw an error since gradients / activations are not stored on
          # tape for recompute.
          raise NotImplementedError(
              "recompute_grad tried to transpose grad of {}. "
              "Consider not using recompute_grad in forward mode"
              "autodiff".format(f.__name__))

        return (grads[:len(id_args)], grads[len(id_args):]), transpose

      return inner_recompute_grad(*wrapper_args)

    return result, grad_wrapper

  return tf_decorator.make_decorator(f, inner)


@tf_export("grad_pass_through")
def grad_pass_through(f):
  """Creates a grad-pass-through op with the forward behavior provided in f.

  Use this function to wrap any op, maintaining its behavior in the forward
  pass, but replacing the original op in the backward graph with an identity.
  For example:

  ```python
  x = tf.Variable(1.0, name="x")
  z = tf.Variable(3.0, name="z")

  with tf.GradientTape() as tape:
    # y will evaluate to 9.0
    y = tf.grad_pass_through(x.assign)(z**2)
  # grads will evaluate to 6.0
  grads = tape.gradient(y, z)
  ```

  Another example is a 'differentiable' moving average approximation, where
  gradients are allowed to flow into the last value fed to the moving average,
  but the moving average is still used for the forward pass:

  ```python
  x = ... # Some scalar value
  # A moving average object, we don't need to know how this is implemented
  moving_average = MovingAverage()
  with backprop.GradientTape() as tape:
    # mavg_x will evaluate to the current running average value
    mavg_x = tf.grad_pass_through(moving_average)(x)
  grads = tape.gradient(mavg_x, x) # grads will evaluate to 1.0
  ```

  Args:
    f: function `f(*x)` that returns a `Tensor` or nested structure of `Tensor`
      outputs.

  Returns:
    A function `h(x)` which returns the same values as `f(x)` and whose
    gradients are the same as those of an identity function.
  """
  @custom_gradient
  def _grad_pass_through_op(*args, **kwargs):
    def grad(*args, **kwargs):
      variables = kwargs.get("variables")
      if variables is not None:
        # Variables involved in the wrapped op will not receive gradients.
        return args, [None] * len(variables)
      return args
    return f(*args, **kwargs), grad
  return tf_decorator.make_decorator(f, _grad_pass_through_op)
