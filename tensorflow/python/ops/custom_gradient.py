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

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import tape as tape_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


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
    if handle_data is not None and handle_data.is_set:
      # pylint: disable=protected-access
      pywrap_tensorflow.SetHandleShapeAndType(target_t.graph._c_graph,
                                              target_t._as_tf_output(),
                                              handle_data.SerializeToString())
      # pylint: enable=protected-access
      # Ensure that shapes and dtypes are propagated.
      shapes, types = zip(*[(pair.shape, pair.dtype)
                            for pair in handle_data.shape_and_type])
      ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
      shapes = [[d.size for d in s.dim]
                if not s.unknown_rank else None for s in shapes]
      pywrap_tensorflow.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
          target_t._op._graph._c_graph,  # pylint: disable=protected-access
          target_t._as_tf_output(),  # pylint: disable=protected-access
          shapes, ranks, types)


@tf_export("custom_gradient")
def custom_gradient(f):
  """Decorator to define a function with a custom gradient.

  This decorator allows fine grained control over the gradients of a sequence
  for operations.  This may be useful for multiple reasons, including providing
  a more efficient or numerically stable gradient for a sequence of operations.

  For example, consider the following function that commonly occurs in the
  computation of cross entropy and log likelihoods:

  ```python
  def log1pexp(x):
    return tf.log(1 + tf.exp(x))
  ```

  Due to numerical instability, the gradient this function evaluated at x=100 is
  NaN.  For example:

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
    return tf.log(1 + e), grad
  ```

  With this definition, the gradient at x=100 will be correctly evaluated as
  1.0.

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

  def decorated(*args, **kwargs):
    """Decorated function with custom gradient."""
    if context.executing_eagerly():
      return _eager_mode_decorator(f, *args, **kwargs)
    else:
      return _graph_mode_decorator(f, *args, **kwargs)

  return tf_decorator.make_decorator(f, decorated)


def _graph_mode_decorator(f, *args, **kwargs):
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
  before_vars = set(current_var_scope.global_variables() +
                    current_var_scope.local_variables())
  with backprop.GradientTape() as tape:
    result, grad_fn = f(*args)
  after_vars = set(current_var_scope.global_variables() +
                   current_var_scope.local_variables())
  new_vars = after_vars - before_vars
  for v in new_vars:
    if not isinstance(v, resource_variable_ops.ResourceVariable):
      raise TypeError(
          "All variables used by a function wrapped with @custom_gradient must "
          "be `ResourceVariable`s. Ensure that no `variable_scope` is created "
          "with `use_resource=False`.")
  # The variables that grad_fn needs to return gradients for are the set of
  # variables used that are *not* part of the inputs.
  variables = list(set(tape.watched_variables()) - set(args))
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
  all_tensors = flat_result + args + variables

  def tape_grad_fn(*result_grads):
    """Custom grad fn wrapper."""
    result_grads = result_grads[:len(flat_result)]
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
    return ([None] * len(flat_result)) + input_grads + variable_grads

  @ops.RegisterGradient(name)
  def internal_grad_fn(unused_op, *result_grads):  # pylint: disable=unused-variable
    """Custom grad fn wrapper."""
    return tape_grad_fn(*result_grads)

  original_tensors = all_tensors
  with ops.get_default_graph().gradient_override_map({"IdentityN": name}):
    all_tensors = array_ops.identity_n(all_tensors)
  # Propagate handle data for happier shape inference for resource variables.
  for i, t in enumerate(original_tensors):
    if t.dtype == dtypes.resource and hasattr(t, "_handle_data"):
      all_tensors[i]._handle_data = t._handle_data  # pylint: disable=protected-access
  tape_lib.record_operation(
      f.__name__, all_tensors, original_tensors, tape_grad_fn)
  for ot, t in zip(original_tensors, all_tensors):
    copy_handle_data(ot, t)
  return nest.pack_sequence_as(
      structure=result, flat_sequence=all_tensors[:len(flat_result)])


def _eager_mode_decorator(f, *args, **kwargs):
  """Implement custom gradient decorator for eager mode."""
  with backprop.GradientTape() as tape:
    result, grad_fn = f(*args, **kwargs)
  all_inputs = list(args) + list(kwargs.values())
  # The variables that grad_fn needs to return gradients for are the set of
  # variables used that are *not* part of the inputs.
  variables = [v for v in set(tape.watched_variables()) if v not in all_inputs]
  grad_argspec = tf_inspect.getfullargspec(grad_fn)
  if (variables and ("variables" not in grad_argspec.args) and
      not grad_argspec.varkw):
    raise TypeError("If using @custom_gradient with a function that "
                    "uses variables, then grad_fn must accept a keyword "
                    "argument 'variables'.")
  flat_result = nest.flatten(result)
  # TODO(apassos) consider removing the identity below.
  flat_result = [gen_array_ops.identity(x) for x in flat_result]

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
    return nest.flatten(input_grads) + variable_grads

  input_tensors = [ops.convert_to_tensor(x) for x
                   in list(args) + list(variables)]
  tape_lib.record_operation(f.__name__, flat_result, input_tensors,
                            actual_grad_fn)
  flat_result = list(flat_result)
  return nest.pack_sequence_as(result, flat_result)
