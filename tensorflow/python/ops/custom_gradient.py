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

from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export


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

  See also @{tf.RegisterGradient} which registers a gradient function for a
  primitive TensorFlow operation. `tf.custom_gradient` on the other hand allows
  for fine grained control over the gradient computation of a sequence of
  operations.

  Args:
    f: function `f(x)` that returns a tuple `(y, grad_fn)` where:
       - `x` is a `Tensor` or sequence of `Tensor` inputs to the function.
       - `y` is a `Tensor` or sequence of `Tensor` outputs of applying
         TensorFlow
         operations in `f` to `x`.
       - `grad_fn` is a function with the signature `g(grad_ys)` which returns
         a list of `Tensor`s - the derivatives of `Tensor`s in `y` with respect
         to the `Tensor`s in `x.  `grad_ys` is a `Tensor` or sequence of
         `Tensor`s the same size as `y` holding the initial value gradients for
         each `Tensor` in `y`.

  Returns:
    A function `h(x)` which returns the same value as `f(x)[0]` and whose
    gradient (as calculated by @{tf.gradients}) is determined by `f(x)[1]`.
  """

  def decorated(*args, **kwargs):
    """Decorated function with custom gradient."""
    if not context.executing_eagerly():
      if kwargs:
        raise ValueError(
            "The custom_gradient decorator currently supports keywords "
            "arguments only when eager execution is enabled.")
      name = "CustomGradient-%s" % ops.uid()
      args = [ops.convert_to_tensor(x) for x in args]
      result, grad_fn = f(*args)
      flat_result = nest.flatten(result)
      all_tensors = flat_result + args

      @ops.RegisterGradient(name)
      def internal_grad_fn(unused_op, *result_grads):  # pylint: disable=unused-variable
        gradients = nest.flatten(grad_fn(*result_grads[:len(flat_result)]))
        # Need to return one value per input to the IdentityN, so pad the
        # gradients of the inputs of the custom_gradient function with the
        # gradients of the outputs as well.
        return ([None] * len(flat_result)) + gradients

      with ops.get_default_graph().gradient_override_map({"IdentityN": name}):
        all_tensors = array_ops.identity_n(all_tensors)
      return nest.pack_sequence_as(
          structure=result, flat_sequence=all_tensors[:len(flat_result)])

    input_tensors = [ops.convert_to_tensor(x) for x in args]

    result, grad_fn = f(*args, **kwargs)
    flat_result = nest.flatten(result)
    # TODO(apassos) consider removing the identity below.
    flat_result = [gen_array_ops.identity(x) for x in flat_result]

    def actual_grad_fn(*outputs):
      return nest.flatten(grad_fn(*outputs))

    tape.record_operation(f.__name__, flat_result, input_tensors,
                          actual_grad_fn)
    flat_result = list(flat_result)
    return nest.pack_sequence_as(result, flat_result)

  return tf_decorator.make_decorator(f, decorated)
