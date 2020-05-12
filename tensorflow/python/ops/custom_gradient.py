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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import tape as tape_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


VAR_OP_TYPES = [
    "VariableV2",
    "VarHandleOp",
]


def copy_handle_data(source_t, target_t):
  """Copies HandleData for variant and resource type tensors if available.

  The CppShapeInferenceResult::HandleData proto contains information about the
  shapes and types of the element tensors of resource/variant type tensors.
  We need to copy this across function boundaries, i.e., when capturing a
  placeholder or when returning a function tensor as output. If we don't do this
  the element tensors will have unknown shapes, e.g., if a TensorList variant
  tensor is captured as a placeholder, elements popped from that list would have
  unknown shape.

  Args:
    source_t: The tensor to copy HandleData from.
    target_t: The tensor to copy HandleData to.
  """
  if (target_t.dtype == dtypes.resource or
      target_t.dtype == dtypes.variant):
    if isinstance(source_t, ops.EagerTensor):
      handle_data = source_t._handle_data  # pylint: disable=protected-access
    else:
      handle_data = resource_variable_ops.get_resource_handle_data(source_t)
    if (handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type):
      # pylint: disable=protected-access
      pywrap_tf_session.SetHandleShapeAndType(target_t.graph._c_graph,
                                              target_t._as_tf_output(),
                                              handle_data.SerializeToString())
      # pylint: enable=protected-access
      # Ensure that shapes and dtypes are propagated.
      shapes, types = zip(*[(pair.shape, pair.dtype)
                            for pair in handle_data.shape_and_type])
      ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
      shapes = [[d.size for d in s.dim]  # pylint: disable=g-complex-comprehension
                if not s.unknown_rank else None for s in shapes]
      pywrap_tf_session.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
          target_t._op._graph._c_graph,  # pylint: disable=protected-access
          target_t._as_tf_output(),  # pylint: disable=protected-access
          shapes,
          ranks,
          types)


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
  dy = tf.gradients(y, x) # Will be NaN when evaluated.
  ```

  The gradient expression can be analytically simplified to provide numerical
  stability:

  ```python
  @tf.custom_gradient
  def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
      return dy * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad
  ```

  With this definition, the gradient at x=100 will be correctly evaluated as
  1.0.

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

  See also `tf.RegisterGradient` which registers a gradient function for a
  primitive TensorFlow operation. `tf.custom_gradient` on the other hand allows
  for fine grained control over the gradient computation of a sequence of
  operations.

  Note that if the decorated function uses `Variable`s, the enclosing variable
  scope must be using `ResourceVariable`s.

  Args:
    f: function `f(*x)` that returns a tuple `(y, grad_fn)` where:
       - `x` is a sequence of `Tensor` inputs to the function.
       - `y` is a `Tensor` or sequence of `Tensor` outputs of applying
         TensorFlow operations in `f` to `x`.
       - `grad_fn` is a function with the signature `g(*grad_ys)` which returns
         a list of `Tensor`s - the derivatives of `Tensor`s in `y` with respect
         to the `Tensor`s in `x`.  `grad_ys` is a `Tensor` or sequence of
         `Tensor`s the same size as `y` holding the initial value gradients for
         each `Tensor` in `y`. In a pure mathematical sense, a vector-argument
         vector-valued function `f`'s derivatives should be its Jacobian matrix
         `J`. Here we are expressing the Jacobian `J` as a function `grad_fn`
         which defines how `J` will transform a vector `grad_ys` when
         left-multiplied with it (`grad_ys * J`). This functional representation
         of a matrix is convenient to use for chain-rule calculation
         (in e.g. the back-propagation algorithm).

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


class Bind(object):
  """When called evaluates `d(f, args, kwargs)` but supports binding `f`.

  >>> @Bind.decorator
  ... def my_decorator(f, args, kwargs):
  ...   print("my_decorator called with", args, kwargs)
  ...   return f(*args, **kwargs)

  >>> class Foo(object):
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

  candidate_vars = ops.get_collection(
      ops.GraphKeys.GLOBAL_VARIABLES, scope="{}:0".format(var_name))
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


def get_dependent_variables(input_ops, output_ops):
  """Finds variables involved in the subgraph b/w input_ops and output_ops."""

  # avoids the edge-case when input_ops == output_ops.
  output_ops = nest.map_structure(gen_array_ops.identity, output_ops)
  inbetween_ops = op_selector.get_backward_walk_ops(
      seed_ops=nest.flatten(output_ops),
      stop_at_ts=nest.flatten(input_ops),
      inclusive=False,
      only_differentiable=True)
  var_ops = (op for op in inbetween_ops if op.type in VAR_OP_TYPES)
  var_names = (op.name for op in var_ops)
  tf_vars = (get_variable_by_name(var_name) for var_name in var_names)
  tf_vars = [v for v in tf_vars if v is not None]
  return tf_vars


def _graph_mode_decorator(f, args, kwargs):
  """Implement custom gradient decorator for graph mode."""
  # TODO(rsepassi): Add support for kwargs
  if kwargs:
    raise ValueError(
        "The custom_gradient decorator currently supports keywords "
        "arguments only when eager execution is enabled.")
  name = "CustomGradient-%s" % ops.uid()
  args = [ops.convert_to_tensor(x) for x in args]

  # Checking global and local variables attempts to ensure that no non-resource
  # Variables are added to the graph.
  current_var_scope = variable_scope.get_variable_scope()
  before_vars = set([
      v.ref() for v in current_var_scope.global_variables() +
      current_var_scope.local_variables()
  ])
  with tape_lib.VariableWatcher() as variable_watcher:
    result, grad_fn = f(*args)
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
  inputs = args
  variables_in_tape = frozenset([
      v.ref() for v in variable_watcher.watched_variables()
  ]) - frozenset(v.ref() for v in inputs)
  variables_in_subgraph = frozenset([
      v.ref()
      for v in get_dependent_variables(input_ops=inputs, output_ops=result)
  ])
  variables = list(
      [v.deref() for v in variables_in_subgraph.union(variables_in_tape)])

  grad_argspec = tf_inspect.getfullargspec(grad_fn)
  variables_in_signature = ("variables" in grad_argspec.args or
                            grad_argspec.varkw)
  if variables and not variables_in_signature:
    raise TypeError("If using @custom_gradient with a function that "
                    "uses variables, then grad_fn must accept a keyword "
                    "argument 'variables'.")
  if variables_in_signature and not variables:
    # User seems to intend to use variables but none were captured.
    if not variable_scope.get_variable_scope().use_resource:
      raise TypeError("If using @custom_gradient with a function that "
                      "uses variables, the enclosing variable scope must "
                      "have use_resource=True.")
    else:
      logging.warn("@custom_gradient grad_fn has 'variables' in signature, but "
                   "no ResourceVariables were used on the forward pass.")
  flat_result = nest.flatten(result)
  flat_result_len = len(flat_result)

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
    copy_handle_data(ot, t)
  return nest.pack_sequence_as(
      structure=result, flat_sequence=all_tensors[:flat_result_len])


def _eager_mode_decorator(f, args, kwargs):
  """Implement custom gradient decorator for eager mode."""
  with tape_lib.VariableWatcher() as variable_watcher:
    result, grad_fn = f(*args, **kwargs)
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
      not grad_argspec.varkw):
    raise TypeError("If using @custom_gradient with a function that "
                    "uses variables, then grad_fn must accept a keyword "
                    "argument 'variables'.")
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
    return nest.flatten(input_grads) + variable_grads

  tape_lib.record_operation(f.__name__, flat_result, recorded_inputs,
                            actual_grad_fn)
  flat_result = list(flat_result)
  return nest.pack_sequence_as(result, flat_result)


@tf_export("recompute_grad")
def recompute_grad(f):
  """An eager-compatible version of recompute_grad.

  For f(*args, **kwargs), this supports gradients with respect to args or
  kwargs, but kwargs are currently only supported in eager-mode.
  Note that for keras layer and model objects, this is handled automatically.

  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.

  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.

  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  """
  # TODO(cdfreeman) Add is_recomputing functionality from graph mode version

  @custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    current_var_scope = variable_scope.get_variable_scope()
    with tape_lib.stop_recording():
      result = f(*args, **kwargs)
    @custom_gradient
    def grad(*dresult, **grad_kwargs):
      """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""

      def grad_eval():
        """Gradient function calculation for reverse mode autodiff."""
        variables = grad_kwargs.get("variables")
        with backprop.GradientTape() as t:
          id_args = [gen_array_ops.identity(x) for x in args]
          t.watch(id_args)
          if variables is not None:
            t.watch(variables)
          with ops.control_dependencies(dresult):
            with variable_scope.variable_scope(current_var_scope):
              result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
          kw_vars = list(variables)
        grads = t.gradient(result,
                           list(id_args) + kw_vars,
                           output_gradients=dresult)
        if len(grads) == 1 and None in grads:
          return 0
        return grads[:len(id_args)], grads[len(id_args):]

      def transpose(*t_args, **t_kwargs):
        """Gradient function calculation for forward mode autodiff."""
        # Just throw an error since gradients / activations are not stored on tape for recompute.
        raise NotImplementedError(
            "recompute_grad tried to transpose grad of {}. "
            "Consider not using recompute_grad in forward mode autodiff".format(
                f.__name__))

      return grad_eval(), transpose

    return result, grad

  return inner


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
