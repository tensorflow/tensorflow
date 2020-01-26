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
"""Implements the graph generation for computation of gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import image_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_ops  # pylint: disable=unused-import
from tensorflow.python.ops import logging_ops  # pylint: disable=unused-import
from tensorflow.python.ops import manip_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import optional_grad  # pylint: disable=unused-import
from tensorflow.python.ops import random_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["gradients"])
def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None,
              stop_gradients=None,
              unconnected_gradients=UnconnectedGradients.NONE):
  """Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.

  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.

  `gradients()` adds ops to the graph to output the derivatives of `ys` with
  respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
  each tensor is the `sum(dy/dx)` for y in `ys` and for x in `xs`.

  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).

  `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
  with respect to all `xs`. These tensors will not be backpropagated through,
  as though they had been explicitly disconnected using `stop_gradient`.  Among
  other things, this allows computation of partial derivatives as opposed to
  total derivatives. For example:

  ```python
  a = tf.constant(0.)
  b = 2 * a
  g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
  ```

  Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
  total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
  influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
  equivalent to:

  ```python
  a = tf.stop_gradient(tf.constant(0.))
  b = tf.stop_gradient(2 * a)
  g = tf.gradients(a + b, [a, b])
  ```

  `stop_gradients` provides a way of stopping gradient after the graph has
  already been constructed, as compared to `tf.stop_gradient` which is used
  during graph construction.  When the two approaches are combined,
  backpropagation stops at both `tf.stop_gradient` nodes and nodes in
  `stop_gradients`, whichever is encountered first.

  All integer tensors are considered constant with respect to all `xs`, as if
  they were included in `stop_gradients`.

  `unconnected_gradients` determines the value returned for each x in xs if it
  is unconnected in the graph to ys. By default this is None to safeguard
  against errors. Mathematically these gradients are zero which can be requested
  using the `'zero'` option. `tf.UnconnectedGradients` provides the
  following options and behaviors:

  ```python
  a = tf.ones([1, 2])
  b = tf.ones([3, 1])
  g1 = tf.gradients([b], [a], unconnected_gradients='none')
  sess.run(g1)  # [None]

  g2 = tf.gradients([b], [a], unconnected_gradients='zero')
  sess.run(g2)  # [array([[0., 0.]], dtype=float32)]
  ```

  Let us take one practical example which comes during the back propogation phase.
  This function is used to evaluate the derivatives of the cost function with
  respect to Weights `Ws` and Biases `bs`. Below sample implementation provides
  the exaplantion of what it is actually used for :

  ```python
  Ws = tf.constant(0.)
  bs = 2 * Ws
  cost = Ws + bs  # This is just an example. So, please ignore the formulas used.
  g = tf.gradients(cost, [Ws, bs])
  dCost_dW, dCost_db = g
  ```


  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
      `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'gradients'.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.
    gate_gradients: If True, add a tuple around the gradients returned
      for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.
    stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
      through.
    unconnected_gradients: Optional. Specifies the gradient value returned when
      the given input tensors are unconnected. Accepted values are constants
      defined in the class `tf.UnconnectedGradients` and the default value is
      `none`.

  Returns:
    A list of `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
    for y in `ys` and for x in `xs`.

  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.
    RuntimeError: if called in Eager mode.

  """
  # Creating the gradient graph for control flow mutates Operations.
  # _mutation_lock ensures a Session.run call cannot occur between creating and
  # mutating new ops.
  # pylint: disable=protected-access
  with ops.get_default_graph()._mutation_lock():
    return gradients_util._GradientsHelper(
        ys, xs, grad_ys, name, colocate_gradients_with_ops,
        gate_gradients, aggregation_method, stop_gradients,
        unconnected_gradients)
  # pylint: enable=protected-access


@tf_export("gradients", v1=[])
def gradients_v2(ys,  # pylint: disable=invalid-name
                 xs,
                 grad_ys=None,
                 name="gradients",
                 gate_gradients=False,
                 aggregation_method=None,
                 stop_gradients=None,
                 unconnected_gradients=UnconnectedGradients.NONE):
  """Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.

  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.

  `gradients()` adds ops to the graph to output the derivatives of `ys` with
  respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
  each tensor is the `sum(dy/dx)` for y in `ys`.

  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).

  `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
  with respect to all `xs`. These tensors will not be backpropagated through,
  as though they had been explicitly disconnected using `stop_gradient`.  Among
  other things, this allows computation of partial derivatives as opposed to
  total derivatives. For example:

  ```python
  a = tf.constant(0.)
  b = 2 * a
  g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
  ```

  Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
  total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
  influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
  equivalent to:

  ```python
  a = tf.stop_gradient(tf.constant(0.))
  b = tf.stop_gradient(2 * a)
  g = tf.gradients(a + b, [a, b])
  ```

  `stop_gradients` provides a way of stopping gradient after the graph has
  already been constructed, as compared to `tf.stop_gradient` which is used
  during graph construction.  When the two approaches are combined,
  backpropagation stops at both `tf.stop_gradient` nodes and nodes in
  `stop_gradients`, whichever is encountered first.

  All integer tensors are considered constant with respect to all `xs`, as if
  they were included in `stop_gradients`.

  `unconnected_gradients` determines the value returned for each x in xs if it
  is unconnected in the graph to ys. By default this is None to safeguard
  against errors. Mathematically these gradients are zero which can be requested
  using the `'zero'` option. `tf.UnconnectedGradients` provides the
  following options and behaviors:

  ```python
  a = tf.ones([1, 2])
  b = tf.ones([3, 1])
  g1 = tf.gradients([b], [a], unconnected_gradients='none')
  sess.run(g1)  # [None]

  g2 = tf.gradients([b], [a], unconnected_gradients='zero')
  sess.run(g2)  # [array([[0., 0.]], dtype=float32)]
  ```


  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
      `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'gradients'.
    gate_gradients: If True, add a tuple around the gradients returned
      for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.
    stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
      through.
    unconnected_gradients: Optional. Specifies the gradient value returned when
      the given input tensors are unconnected. Accepted values are constants
      defined in the class `tf.UnconnectedGradients` and the default value is
      `none`.

  Returns:
    A list of `sum(dy/dx)` for each x in `xs`.

  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.
    RuntimeError: if called in Eager mode.

  """
  # Creating the gradient graph for control flow mutates Operations.
  # _mutation_lock ensures a Session.run call cannot occur between creating and
  # mutating new ops.
  # pylint: disable=protected-access
  with ops.get_default_graph()._mutation_lock():
    return gradients_util._GradientsHelper(
        ys, xs, grad_ys, name, True, gate_gradients,
        aggregation_method, stop_gradients,
        unconnected_gradients)
  # pylint: enable=protected-access


# TODO(vrv): Make this available when we want to make it public.
def _hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.

  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.

  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.

  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.

  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.

  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.

  Raises:
    ValueError: `xs` and `v` have different length.

  """

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = gradients(ys, xs)

  assert len(grads) == length
  elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v)
      if grad_elem is not None
  ]

  # Second backprop
  return gradients(elemwise_products, xs)


@tf_export(v1=["hessians"])
def hessians(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None):
  """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.

  `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`.

  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.

  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.

  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  xs = gradients_util._AsList(xs)  # pylint: disable=protected-access
  kwargs = {
      "colocate_gradients_with_ops": colocate_gradients_with_ops,
      "gate_gradients": gate_gradients,
      "aggregation_method": aggregation_method
  }
  # Compute first-order derivatives and iterate for each x in xs.
  hessians = []
  _gradients = gradients(ys, xs, **kwargs)
  for gradient, x in zip(_gradients, xs):
    # change shape to one-dimension without graph branching
    gradient = array_ops.reshape(gradient, [-1])

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(x)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(x.dtype, n)
    ]
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    _, hessian = control_flow_ops.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1,
                           result.write(j, gradients(gradient[j], x)[0])),
        loop_vars
    )

    _shape = array_ops.shape(x)
    _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                          array_ops.concat((_shape, _shape), 0))
    hessians.append(_reshaped_hessian)
  return hessians


@tf_export("hessians", v1=[])
def HessiansV2(ys,
               xs,
               gate_gradients=False,
               aggregation_method=None,
               name="hessians"):
  return hessians(ys, xs, name=name, gate_gradients=gate_gradients,
                  aggregation_method=aggregation_method)


HessiansV2.__doc__ = hessians.__doc__
