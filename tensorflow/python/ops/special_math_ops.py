# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Arithmetic Operations that don't fit into math_ops due to dependencies.

To avoid circular dependencies, some math_ops should go here.
"""

import collections
import functools
import re
import string

import numpy as np
import opt_einsum


from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# TODO(b/27419586) Change docstring for required dtype of x once int allowed
@tf_export('math.lbeta', v1=['math.lbeta', 'lbeta'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('lbeta')
def lbeta(x, name=None):
  r"""Computes \\(ln(|Beta(x)|)\\), reducing along the last dimension.

  Given one-dimensional $z = [z_1,...,z_K]$, we define

  $$Beta(z) = \frac{\prod_j \Gamma(z_j)}{\Gamma(\sum_j z_j)},$$

  where $\Gamma$ is the gamma function.

  And for $n + 1$ dimensional $x$ with shape $[N_1, ..., N_n, K]$, we define

  $$lbeta(x)[i_1, ..., i_n] = \log{|Beta(x[i_1, ..., i_n, :])|}.$$

  In other words, the last dimension is treated as the $z$ vector.

  Note that if $z = [u, v]$, then

  $$Beta(z) = \frac{\Gamma(u)\Gamma(v)}{\Gamma(u + v)}
    = \int_0^1 t^{u-1} (1 - t)^{v-1} \mathrm{d}t,$$

  which defines the traditional bivariate beta function.

  If the last dimension is empty, we follow the convention that the sum over
  the empty set is zero, and the product is one.

  Args:
    x: A rank `n + 1` `Tensor`, `n >= 0` with type `float`, or `double`.
    name: A name for the operation (optional).

  Returns:
    The logarithm of \\(|Beta(x)|\\) reducing along the last dimension.
  """
  # In the event that the last dimension has zero entries, we return -inf.
  # This is consistent with a convention that the sum over the empty set 0, and
  # the product is 1.
  # This is standard.  See https://en.wikipedia.org/wiki/Empty_set.
  with ops.name_scope(name, 'lbeta', [x]):
    x = ops.convert_to_tensor(x, name='x')

    # Note reduce_sum([]) = 0.
    log_prod_gamma_x = math_ops.reduce_sum(math_ops.lgamma(x), axis=[-1])

    # Note lgamma(0) = infinity, so if x = []
    # log_gamma_sum_x = lgamma(0) = infinity, and
    # log_prod_gamma_x = lgamma(1) = 0,
    # so result = -infinity
    sum_x = math_ops.reduce_sum(x, axis=[-1])
    log_gamma_sum_x = math_ops.lgamma(sum_x)
    result = log_prod_gamma_x - log_gamma_sum_x

    return result


@tf_export('math.special.dawsn')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def dawsn(x, name=None):
  """Computes Dawson's integral of `x` element-wise.

  Dawson's integral is defined as `exp(-x**2)` times the integral of
  `exp(t**2)` from `0` to `x`, with the domain of definition all real numbers.

  Dawson's function is odd.
  >>> tf.math.special.dawsn([-1., -0.5, 0.5, 1.]).numpy()
  array([-0.5380795, -0.4244364, 0.4244364,  0.5380795], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.dawsn
  @end_compatibility
  """
  with ops.name_scope(name, 'dawsn', [x]):
    return gen_special_math_ops.dawsn(x)


@tf_export('math.special.expint')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def expint(x, name=None):
  """Computes the Exponential integral of `x` element-wise.

  The Exponential integral is defined as the integral of `exp(t) / t` from
  `-inf` to `x`, with the domain of definition all positive real numbers.

  >>> tf.math.special.expint([1., 1.1, 2.1, 4.1]).numpy()
  array([ 1.8951179,  2.1673784,  5.3332353, 21.048464], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.expi
  @end_compatibility
  """
  with ops.name_scope(name, 'expint', [x]):
    return gen_special_math_ops.expint(x)


@tf_export('math.special.fresnel_cos')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def fresnel_cos(x, name=None):
  """Computes Fresnel's cosine integral of `x` element-wise.

  The Fresnel cosine integral is defined as the integral of `cos(t^2)` from
  `0` to `x`, with the domain of definition all real numbers.

  The Fresnel cosine integral is odd.
  >>> tf.math.special.fresnel_cos([-1., -0.1, 0.1, 1.]).numpy()
  array([-0.7798934 , -0.09999753,  0.09999753,  0.7798934 ], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.fresnel second output.
  @end_compatibility
  """
  with ops.name_scope(name, 'fresnel_cos', [x]):
    return gen_special_math_ops.fresnel_cos(x)


@tf_export('math.special.fresnel_sin')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def fresnel_sin(x, name=None):
  """Computes Fresnel's sine integral of `x` element-wise.

  The Fresnel sine integral is defined as the integral of `sin(t^2)` from
  `0` to `x`, with the domain of definition all real numbers.

  >>> tf.math.special.fresnel_sin([-1., -0.1, 0.1, 1.]).numpy()
  array([-0.43825912, -0.00052359,  0.00052359,  0.43825912], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.fresnel first output.
  @end_compatibility
  """
  with ops.name_scope(name, 'fresnel_sin', [x]):
    return gen_special_math_ops.fresnel_sin(x)


@tf_export('math.special.spence')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def spence(x, name=None):
  """Computes Spence's integral of `x` element-wise.

  Spence's integral is defined as the integral of `log(t) / (1 - t)` from
  `1` to `x`, with the domain of definition all non-negative real numbers.

  >>> tf.math.special.spence([0.5, 1., 2., 3.]).numpy()
  array([ 0.58224034,  0.        , -0.82246685, -1.4367464], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.spence
  @end_compatibility
  """
  with ops.name_scope(name, 'spence', [x]):
    return gen_special_math_ops.spence(x)


@tf_export('math.bessel_i0', 'math.special.bessel_i0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i0(x, name=None):
  """Computes the Bessel i0 function of `x` element-wise.

  Modified Bessel function of order 0.

  It is preferable to use the numerically stabler function `i0e(x)` instead.

  >>> tf.math.special.bessel_i0([-1., -0.5, 0.5, 1.]).numpy()
  array([1.26606588, 1.06348337, 1.06348337, 1.26606588], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.i0
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_i0', [x]):
    return gen_special_math_ops.bessel_i0(x)


@tf_export('math.bessel_i0e', 'math.special.bessel_i0e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i0e(x, name=None):
  """Computes the Bessel i0e function of `x` element-wise.

  Modified Bessel function of order 0.

  >>> tf.math.special.bessel_i0e([-1., -0.5, 0.5, 1.]).numpy()
  array([0.46575961, 0.64503527, 0.64503527, 0.46575961], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.i0e
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_i0e', [x]):
    return gen_special_math_ops.bessel_i0e(x)


@tf_export('math.bessel_i1', 'math.special.bessel_i1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i1(x, name=None):
  """Computes the Bessel i1 function of `x` element-wise.

  Modified Bessel function of order 1.

  It is preferable to use the numerically stabler function `i1e(x)` instead.

  >>> tf.math.special.bessel_i1([-1., -0.5, 0.5, 1.]).numpy()
  array([-0.5651591 , -0.25789431,  0.25789431,  0.5651591 ], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.i1
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_i1', [x]):
    return gen_special_math_ops.bessel_i1(x)


@tf_export('math.bessel_i1e', 'math.special.bessel_i1e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i1e(x, name=None):
  """Computes the Bessel i1e function of `x` element-wise.

  Modified Bessel function of order 1.

  >>> tf.math.special.bessel_i1e([-1., -0.5, 0.5, 1.]).numpy()
  array([-0.20791042, -0.15642083,  0.15642083,  0.20791042], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.i1e
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_i1e', [x]):
    return gen_special_math_ops.bessel_i1e(x)


@tf_export('math.special.bessel_k0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k0(x, name=None):
  """Computes the Bessel k0 function of `x` element-wise.

  Modified Bessel function of order 0.

  It is preferable to use the numerically stabler function `k0e(x)` instead.

  >>> tf.math.special.bessel_k0([0.5, 1., 2., 4.]).numpy()
  array([0.92441907, 0.42102444, 0.11389387, 0.01115968], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.k0
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_k0', [x]):
    return gen_special_math_ops.bessel_k0(x)


@tf_export('math.special.bessel_k0e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k0e(x, name=None):
  """Computes the Bessel k0e function of `x` element-wise.

  Modified Bessel function of order 0.

  >>> tf.math.special.bessel_k0e([0.5, 1., 2., 4.]).numpy()
  array([1.52410939, 1.14446308, 0.84156822, 0.60929767], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.k0e
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_k0e', [x]):
    return gen_special_math_ops.bessel_k0e(x)


@tf_export('math.special.bessel_k1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k1(x, name=None):
  """Computes the Bessel k1 function of `x` element-wise.

  Modified Bessel function of order 1.

  It is preferable to use the numerically stabler function `k1e(x)` instead.

  >>> tf.math.special.bessel_k1([0.5, 1., 2., 4.]).numpy()
  array([1.65644112, 0.60190723, 0.13986588, 0.0124835 ], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.k1
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_k1', [x]):
    return gen_special_math_ops.bessel_k1(x)


@tf_export('math.special.bessel_k1e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k1e(x, name=None):
  """Computes the Bessel k1e function of `x` element-wise.

  Modified Bessel function of order 1.

  >>> tf.math.special.bessel_k1e([0.5, 1., 2., 4.]).numpy()
  array([2.73100971, 1.63615349, 1.03347685, 0.68157595], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.k1e
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_k1e', [x]):
    return gen_special_math_ops.bessel_k1e(x)


@tf_export('math.special.bessel_j0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_j0(x, name=None):
  """Computes the Bessel j0 function of `x` element-wise.

  Modified Bessel function of order 0.

  >>> tf.math.special.bessel_j0([0.5, 1., 2., 4.]).numpy()
  array([ 0.93846981,  0.76519769,  0.22389078, -0.39714981], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.j0
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_j0', [x]):
    return gen_special_math_ops.bessel_j0(x)


@tf_export('math.special.bessel_j1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_j1(x, name=None):
  """Computes the Bessel j1 function of `x` element-wise.

  Modified Bessel function of order 1.

  >>> tf.math.special.bessel_j1([0.5, 1., 2., 4.]).numpy()
  array([ 0.24226846,  0.44005059,  0.57672481, -0.06604333], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.j1
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_j1', [x]):
    return gen_special_math_ops.bessel_j1(x)


@tf_export('math.special.bessel_y0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_y0(x, name=None):
  """Computes the Bessel y0 function of `x` element-wise.

  Modified Bessel function of order 0.

  >>> tf.math.special.bessel_y0([0.5, 1., 2., 4.]).numpy()
  array([-0.44451873,  0.08825696,  0.51037567, -0.01694074], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.y0
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_y0', [x]):
    return gen_special_math_ops.bessel_y0(x)


@tf_export('math.special.bessel_y1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_y1(x, name=None):
  """Computes the Bessel y1 function of `x` element-wise.

  Modified Bessel function of order 1.

  >>> tf.math.special.bessel_y1([0.5, 1., 2., 4.]).numpy()
  array([-1.47147239, -0.78121282, -0.10703243,  0.39792571], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.y1
  @end_compatibility
  """
  with ops.name_scope(name, 'bessel_y1', [x]):
    return gen_special_math_ops.bessel_y1(x)


@ops.RegisterGradient('XlaEinsum')
def _einsum_grad(op, grad):
  equation = op.get_attr('equation')
  if isinstance(equation, bytes):
    equation = equation.decode()

  inputs, output = equation.split('->')
  left, right = inputs.split(',')

  return [
      gen_xla_ops.xla_einsum(
          grad,
          op.inputs[1],
          equation='{},{}->{}'.format(output, right, left),
          name=None),
      gen_xla_ops.xla_einsum(
          grad,
          op.inputs[0],
          equation='{},{}->{}'.format(output, left, right),
          name=None)
  ]


def _enclosing_tpu_context():
  # pylint: disable=protected-access
  context = ops.get_default_graph()._get_control_flow_context()
  # pylint: enable=protected-access
  while context is not None and not isinstance(
      context, control_flow_ops.XLAControlFlowContext):
    context = context.outer_context
  return context


@tf_export('einsum', 'linalg.einsum')
@dispatch.add_dispatch_support
def einsum(equation, *inputs, **kwargs):
  r"""Tensor contraction over specified indices and outer product.

  Einsum allows defining Tensors by defining their element-wise computation.
  This computation is defined by `equation`, a shorthand form based on Einstein
  summation. As an example, consider multiplying two matrices A and B to form a
  matrix C.  The elements of C are given by:

  $$ C_{i,k} = \sum_j A_{i,j} B_{j,k} $$

  or

  ```
  C[i,k] = sum_j A[i,j] * B[j,k]
  ```

  The corresponding einsum `equation` is:

  ```
  ij,jk->ik
  ```

  In general, to convert the element-wise equation into the `equation` string,
  use the following procedure (intermediate strings for matrix multiplication
  example provided in parentheses):

  1. remove variable names, brackets, and commas, (`ik = sum_j ij * jk`)
  2. replace "*" with ",", (`ik = sum_j ij , jk`)
  3. drop summation signs, and (`ik = ij, jk`)
  4. move the output to the right, while replacing "=" with "->". (`ij,jk->ik`)

  Note: If the output indices are not specified repeated indices are summed.
  So `ij,jk->ik` can be simplified to `ij,jk`.

  Many common operations can be expressed in this way.  For example:

  **Matrix multiplication**

  >>> m0 = tf.random.normal(shape=[2, 3])
  >>> m1 = tf.random.normal(shape=[3, 5])
  >>> e = tf.einsum('ij,jk->ik', m0, m1)
  >>> # output[i,k] = sum_j m0[i,j] * m1[j, k]
  >>> print(e.shape)
  (2, 5)

  Repeated indices are summed if the output indices are not specified.

  >>> e = tf.einsum('ij,jk', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]
  >>> print(e.shape)
  (2, 5)


  **Dot product**

  >>> u = tf.random.normal(shape=[5])
  >>> v = tf.random.normal(shape=[5])
  >>> e = tf.einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]
  >>> print(e.shape)
  ()

  **Outer product**

  >>> u = tf.random.normal(shape=[3])
  >>> v = tf.random.normal(shape=[5])
  >>> e = tf.einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]
  >>> print(e.shape)
  (3, 5)

  **Transpose**

  >>> m = tf.ones(2,3)
  >>> e = tf.einsum('ij->ji', m0)  # output[j,i] = m0[i,j]
  >>> print(e.shape)
  (3, 2)

  **Diag**

  >>> m = tf.reshape(tf.range(9), [3,3])
  >>> diag = tf.einsum('ii->i', m)
  >>> print(diag.shape)
  (3,)

  **Trace**

  >>> # Repeated indices are summed.
  >>> trace = tf.einsum('ii', m)  # output[j,i] = trace(m) = sum_i m[i, i]
  >>> assert trace == sum(diag)
  >>> print(trace.shape)
  ()

  **Batch matrix multiplication**

  >>> s = tf.random.normal(shape=[7,5,3])
  >>> t = tf.random.normal(shape=[7,3,2])
  >>> e = tf.einsum('bij,bjk->bik', s, t)
  >>> # output[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
  >>> print(e.shape)
  (7, 5, 2)

  This method does not support broadcasting on named-axes. All axes with
  matching labels should have the same length. If you have length-1 axes,
  use `tf.squeseze` or `tf.reshape` to eliminate them.

  To write code that is agnostic to the number of indices in the input
  use an ellipsis. The ellipsis is a placeholder for "whatever other indices
  fit here".

  For example, to perform a NumPy-style broadcasting-batch-matrix multiplication
  where the matrix multiply acts on the last two axes of the input, use:

  >>> s = tf.random.normal(shape=[11, 7, 5, 3])
  >>> t = tf.random.normal(shape=[11, 7, 3, 2])
  >>> e =  tf.einsum('...ij,...jk->...ik', s, t)
  >>> print(e.shape)
  (11, 7, 5, 2)

  Einsum **will** broadcast over axes covered by the ellipsis.

  >>> s = tf.random.normal(shape=[11, 1, 5, 3])
  >>> t = tf.random.normal(shape=[1, 7, 3, 2])
  >>> e =  tf.einsum('...ij,...jk->...ik', s, t)
  >>> print(e.shape)
  (11, 7, 5, 2)

  Args:
    equation: a `str` describing the contraction, in the same format as
      `numpy.einsum`.
    *inputs: the inputs to contract (each one a `Tensor`), whose shapes should
      be consistent with `equation`.
    **kwargs:
      - optimize: Optimization strategy to use to find contraction path using
        opt_einsum. Must be 'greedy', 'optimal', 'branch-2', 'branch-all' or
          'auto'. (optional, default: 'greedy').
      - name: A name for the operation (optional).

  Returns:
    The contracted `Tensor`, with shape determined by `equation`.

  Raises:
    ValueError: If
      - the format of `equation` is incorrect,
      - number of inputs or their shapes are inconsistent with `equation`.
  """
  return _einsum_v2(equation, *inputs, **kwargs)


def _einsum_v1(equation, *inputs, **kwargs):
  """Legacy implementation of einsum without using EinsumOp."""
  name = kwargs.pop('name', None)
  if kwargs:
    raise TypeError(
        f'Invalid keyword arguments for this function: '
        f'{", ".join([format(key) for key in sorted(list(kwargs.keys()))])}.'
        f' Expected: name.')
  with ops.name_scope(name, 'einsum', [equation, inputs]) as name:
    inputs = list(inputs)
    input_shapes = [x.shape for x in inputs]
    input_axis_labels, output_axis_labels = (
        _einsum_v1_parse_and_resolve_equation(equation, input_shapes))

    axis_labels = set(''.join(input_axis_labels) + output_axis_labels)

    for a in axis_labels:
      for input_labels in input_axis_labels:
        if (len(input_axis_labels) == 1 and input_labels.count(a) == 2 and
            input_labels == input_labels[::-1] and '->' not in equation):
          return math_ops.trace(inputs[0])
        if input_labels.count(a) > 1:
          raise ValueError(
              f'Subscript not supported: the axis {a} appears more than once'
              f' in {input_labels}.')
    for a in axis_labels:
      input_count = sum(1 for s in input_axis_labels if a in s)
      if input_count > 2 and a not in output_axis_labels:
        logging.warn(
            f'Falling back to exponential-space implementation of einsum()'
            f' because index {a} is summed over more than two inputs.')
        return _exponential_space_einsum_v1(equation, *inputs)

    # Use xla_einsum if executing on TPU and if the operation is a 2 input
    # einsum supported by XlaEinsumOp.
    if _enclosing_tpu_context() is not None and len(inputs) == 2:
      return gen_xla_ops.xla_einsum(
          inputs[0], inputs[1], input_axis_labels[0] + ',' +
          input_axis_labels[1] + '->' + output_axis_labels)
    temp = inputs[0]
    temp_axis_labels = input_axis_labels[0]
    for i in range(len(inputs) - 1):
      axes_to_sum = (
          set(temp_axis_labels) &
          set(input_axis_labels[i + 1]) - set(output_axis_labels))
      temp, temp_axis_labels = _einsum_v1_reduction(temp, temp_axis_labels,
                                                    inputs[i + 1],
                                                    input_axis_labels[i + 1],
                                                    axes_to_sum)

    missing_indices = set(temp_axis_labels) - set(output_axis_labels)
    if missing_indices:
      axis = [
          i for i, a in enumerate(temp_axis_labels)
          if a not in output_axis_labels
      ]
      temp = math_ops.reduce_sum(temp, axis=axis)
      temp_axis_labels = ''.join(
          a for a in temp_axis_labels if a in output_axis_labels)
    if sorted(temp_axis_labels) != sorted(output_axis_labels):
      raise ValueError(
          f'Invalid equation: {equation}. The computed and specified output '
          f'labels do not match: {temp_axis_labels} vs {output_axis_labels}.')

    perm = [temp_axis_labels.index(a) for a in output_axis_labels]
    return _transpose_if_necessary(temp, perm)


def _einsum_v1_parse_and_resolve_equation(equation, input_shapes):
  """Helper for einsum() that splits/resolves inputs & outputs.

  Args:
    equation: Equation string given as argument to einsum().
    input_shapes: List of the shapes of all inputs given to einsum()

  Returns:
    input_axis_labels, output_axis_labels where:
      input_axis_labels: List of length len(input_shapes) of strings
      representing the character label for each dimension of each given input,
      resolving any broadcast (...) axes,
    output_axis_labels: A string of character labels for each axes of output
      tensor, filling in missing output subscripts and broadcast axes.

  Raises:
    ValueError: If equation is in the uncorrect format, incorrect number of
      inputs given or broadcast axes "..." or output axes could not be resolved.
  """
  equation = equation.replace(' ', '')
  match = re.match('^([a-zA-Z,.]+)(->[a-zA-Z.]*)?$', equation)
  if not match:
    raise ValueError(f'Indices have incorrect format. Received: {equation}.')

  input_axis_labels = match.group(1).split(',')
  output_axis_labels = match.group(2)[2:] if match.group(2) else None

  if len(input_shapes) != len(input_axis_labels):
    raise ValueError(
        f'Got {len(input_shapes)} arguments for equation "{equation}", '
        f'expecting {len(input_axis_labels)}.')

  # Resolve Ellipsis
  # Assign axes labels for unspecified dimensions in inputs. Labels taken
  # from unused labels. Follow numpy einsum broadcasting conventions for
  # tensors of different length and unlabeled output.
  ellipsis_axes = ''
  if '...' in equation:
    unused = ''.join(
        c for c in string.ascii_letters if c not in ''.join(input_axis_labels))
    for i, ax in enumerate(input_axis_labels):
      if '...' in ax:
        parts = ax.split('...')
        if len(parts) != 2:
          raise ValueError(f'Unable to resolve ellipsis. '
                           f'Excess number found: {len(parts)-1} vs 1.')
        if input_shapes[i].ndims is None:
          raise ValueError('Unable to statically infer ellipsis axes. The '
                           'input shapes has a dynamic dimensionality.')
        n = input_shapes[i].ndims - len(''.join(parts))
        if n < 0:
          raise ValueError('Ellipses lengths do not match.')
        if len(unused) < n:
          raise ValueError(
              'Unable to resolve ellipsis, too many distinct labels.')
        replace_axes = unused[-n:] if n > 0 else ''
        input_axis_labels[i] = input_axis_labels[i].replace('...',
                                                            replace_axes)
        if len(replace_axes) > len(ellipsis_axes):
          ellipsis_axes = replace_axes

    if any('.' in ax for ax in input_axis_labels):
      raise ValueError(
          f'Period "." found outside of ellipsis in input {input_axis_labels}.')

    if output_axis_labels is not None:
      output_axis_labels = output_axis_labels.replace('...', ellipsis_axes)
      if '.' in output_axis_labels:
        raise ValueError(f'Period "." found outside of ellipsis in output '
                         f'{output_axis_labels}.')

  if output_axis_labels is None:
    # infer the output subscripts if not given, assume alphabetical order,
    # but always place ellipsis axes before given.
    axis_labels = set(''.join(input_axis_labels)) - set(ellipsis_axes)
    indices = ''.join(sorted(axis_labels))
    counts = {ax: 0 for ax in indices}
    for axes_ in input_axis_labels:
      for ax in axes_:
        if ax not in ellipsis_axes:
          counts[ax] += 1

    output_axis_labels = ellipsis_axes + ''.join(
        sorted(ax for ax in axis_labels if counts[ax] == 1))

  return input_axis_labels, output_axis_labels


def _einsum_v1_reduction(t0, t0_axis_labels, t1, t1_axis_labels, axes_to_sum):
  """Helper for einsum() that computes the result of a two-argument einsum().

  Args:
    t0: a `Tensor`
    t0_axis_labels: a string of axis labels.  This string's length must equal
      the rank of t0.
    t1: a `Tensor`
    t1_axis_labels: a string to axis labels.  This string's length must equal
      the rank of t1.
    axes_to_sum: set of labels of axes to be summed over

  Returns:
    A `Tensor` whose elements are obtained by summing, over all axes in
    `axes_to_sum`, the corresponding elements of `t0` and `t1`.

    For example, if t0_axis_labels == 'abijk', t1_axis_labels == 'acjkl', and
    axes_to_sum == {j,k}, this will return a tensor x where

      out[a,b,c,i,l] = sum_j sum_k t0[a,b,i,j,k] * t1[a,c,j,k,l]

  Raises:
    ValueError: if the rank of `t0` does not match the length of
      `t0_axis_labels`, or that of `t1` does not match the length of
      `t1_axis_labels`.
  """
  if len(t0_axis_labels) != len(t0.shape):
    raise ValueError(
        f'Tensor `t0` of rank {len(t0.shape)} does not match einsum reduction '
        f'of length {len(t0_axis_labels)}.')
  if len(t1_axis_labels) != len(t1.shape):
    raise ValueError(
        f'Tensor `t1` of rank {len(t1.shape)} does not match einsum reduction '
        f'of length {len(t1_axis_labels)}')

  # This function computes the result of a two-argument einsum() using batch
  # matrix multiplication.  This involves
  # 1. transposing t0 and t1 so that axes are in the correct order for
  #    batch matrix multiplication, and
  # 2. reshaping t0 and t1 so that they are both of rank 3.

  # First, we divide axes into three groups:
  #  * "preserved" axes are present in both inputs and the output
  #  * "summed" axes are present in both inputs but not the output
  #  * "broadcast" axes are present in exactly one input and the output
  #
  # As an example, if the einsum is abijk,acjkl->abcil, then "a" is a
  # preserved axis, "b" and "c" are broadcast axes, and "j" and "k" are
  # summed axes.
  assert all(a in t0_axis_labels and a in t1_axis_labels for a in axes_to_sum)
  preserved_axes = (set(t0_axis_labels) & set(t1_axis_labels)) - axes_to_sum
  broadcast_axes = {}
  for i, sym_list in enumerate([t0_axis_labels, t1_axis_labels]):
    broadcast_axes[i] = set(sym_list) - preserved_axes - axes_to_sum

  # Reorder the axes so that:
  # 1. preserved axes come first in both inputs
  # 2. in input 0, broadcast axes come next, followed by summed axes
  # 3. in input 1, summed axes come next, followed by broadcast axes
  def sort_key(input_index, a):
    if a in preserved_axes:
      return (-1, a)
    elif ((input_index == 0 and a in broadcast_axes[0]) or
          (input_index == 1 and a in axes_to_sum)):
      return (0, a)
    else:
      return (1, a)

  axis_labels = [t0_axis_labels, t1_axis_labels]
  sorted_axes = [
      sorted(sym_list, key=lambda a: sort_key(i, a))
      for i, sym_list in enumerate(axis_labels)
  ]
  inputs = [t0, t1]
  for i, axes_str in enumerate(axis_labels):
    perm = [axes_str.find(a) for a in sorted_axes[i]]
    inputs[i] = _transpose_if_necessary(inputs[i], perm)
  t0, t1 = inputs

  if not axes_to_sum:
    # In the special case where there are no axes to sum over, reduce to mul()
    # rather than to batch matrix multiplication.
    for _ in broadcast_axes[1]:
      t0 = array_ops.expand_dims(t0, -1)
    for _ in broadcast_axes[0]:
      t1 = array_ops.expand_dims(t1, len(preserved_axes))
    product = math_ops.multiply(t0, t1)
    product_axes = sorted_axes[0] + sorted_axes[1][len(preserved_axes):]
    return product, ''.join(product_axes)
  else:
    # Reduce to matmul().

    # Reshape both inputs so as to combine multiple broadcast axes
    # into a single axis, and combine multiple summed axes into a
    # single axis.

    t0_shape = _get_shape(t0)
    num_broadcast_elements_t0 = _total_size(
        t0_shape[len(preserved_axes):-len(axes_to_sum)])
    num_summed_elements = _total_size(t0_shape[-len(axes_to_sum):])
    new_shape = (
        t0_shape[:len(preserved_axes)] +
        [num_broadcast_elements_t0, num_summed_elements])
    t0 = _reshape_if_necessary(t0, new_shape)

    t1_shape = _get_shape(t1)
    num_broadcast_elements_t1 = _total_size(
        t1_shape[len(preserved_axes) + len(axes_to_sum):])
    new_shape = (
        t1_shape[:len(preserved_axes)] +
        [num_summed_elements, num_broadcast_elements_t1])
    t1 = _reshape_if_necessary(t1, new_shape)

    product = math_ops.matmul(t0, t1)

    # Undo compaction of broadcast axes
    uncompacted_shape = (
        t0_shape[:len(preserved_axes) + len(broadcast_axes[0])] +
        t1_shape[len(t1_shape) - len(broadcast_axes[1]):])
    product = _reshape_if_necessary(product, uncompacted_shape)

    product_axes = (
        sorted_axes[0][:len(preserved_axes) + len(broadcast_axes[0])] +
        sorted_axes[1][len(sorted_axes[1]) - len(broadcast_axes[1]):])

    return product, ''.join(product_axes)


def _transpose_if_necessary(tensor, perm):
  """Like transpose(), but avoids creating a new tensor if possible."""
  if perm != list(range(len(perm))):
    return array_ops.transpose(tensor, perm=perm)
  else:
    return tensor


def _reshape_if_necessary(tensor, new_shape):
  """Like reshape(), but avoids creating a new tensor if possible."""
  # Accept None as an alias for -1 in new_shape.
  new_shape = tuple(-1 if x is None else x for x in new_shape)
  cur_shape = tuple(x.value for x in tensor.shape.dims)
  if (len(new_shape) == len(cur_shape) and
      all(not isinstance(d1, ops.Tensor) and (d0 == d1 or d1 == -1)
          for d0, d1 in zip(cur_shape, new_shape))):
    return tensor
  else:
    return array_ops.reshape(tensor, new_shape)


def _get_shape(tensor):
  """Like get_shape().as_list(), but explicitly queries the shape of a tensor
  if necessary to ensure that the returned value contains no unknown value."""

  shape = tensor.shape.as_list()
  none_indices = [i for i, d in enumerate(shape) if d is None]
  if none_indices:
    # Query the shape if shape contains None values
    shape_tensor = array_ops.shape(tensor)
    for i in none_indices:
      shape[i] = shape_tensor[i]
  return shape


def _total_size(shape_values):
  """Given list of tensor shape values, returns total size.
  If shape_values contains tensor values (which are results of
  array_ops.shape), then it returns a scalar tensor.
  If not, it returns an integer."""

  result = 1
  for val in shape_values:
    result *= val
  return result


def _exponential_space_einsum_v1(equation, *inputs):
  """Fallback implementation that supports summing an index over > 2 inputs."""
  inputs = list(inputs)
  input_shapes = [x.shape for x in inputs]
  idx_in, idx_out = _einsum_v1_parse_and_resolve_equation(
      equation, input_shapes)

  idx_all = set(''.join(idx_in) + idx_out)
  indices = ''.join(sorted(idx_all))

  missing_idx = set(idx_out).difference(idx_all)
  if missing_idx:
    raise ValueError(f'Unknown output axes: {missing_idx}.')

  axis_order = {}
  for ax in indices:
    if ax not in idx_out:
      axis_order[ax] = len(axis_order)
  for ax in idx_out:
    axis_order[ax] = len(axis_order)

  # transpose inputs so axes are in order
  for i, (input_, axes_) in enumerate(zip(inputs, idx_in)):
    if input_.shape.ndims != len(axes_):
      raise ValueError(
          f'Input {i} with axes {axes_} has incorrect number of dimensions '
          f'(expected {len(axes_)}, got {input_.shape.ndims}).')

    sorted_idx = sorted(axes_, key=axis_order.get)

    if len(set(axes_)) != len(axes_):
      raise ValueError(
          f'Subscript not supported: an axis appears more than once: {axes_}.')

    if list(axes_) != sorted_idx:
      permuted = [axes_.find(ax) for ax in sorted_idx]
      inputs[i] = array_ops.transpose(input_, permuted)
      idx_in[i] = sorted_idx

  reduction_idx = []
  shapes = [[dim if dim else -1
             for dim in tensor.shape.as_list()]
            for tensor in inputs]

  # validate shapes for broadcasting
  for j, ax in enumerate(sorted(idx_all, key=axis_order.get)):
    dims = []
    for i, idx in enumerate(idx_in):
      if ax not in idx:
        shapes[i].insert(j, 1)
      else:
        dim = shapes[i][j]
        if isinstance(dim, int) and dim > 1:
          dims.append(dim)

    if len(set(dims)) > 1:
      raise ValueError(f'Dimension mismatch on axis: {ax}. '
                       f'Found {len(set(dims))}, expected 1.')

    if ax not in idx_out:
      reduction_idx.append(j)

  # reshape, multiply
  expanded_inputs = [
      array_ops.reshape(input_, shape) for input_, shape in zip(inputs, shapes)
  ]
  expanded_output = 1
  for input_ in expanded_inputs:
    expanded_output *= input_

  # contract
  return math_ops.reduce_sum(expanded_output, reduction_idx)


def _einsum_v2(equation, *inputs, **kwargs):
  """Implementation of einsum utilizing opt_einsum and EinsumOp."""
  name = kwargs.pop('name', None)
  optimize = kwargs.pop('optimize', 'greedy')
  if kwargs:
    raise TypeError(
        f'Invalid keyword arguments for einsum: {", ".join(kwargs)}. '
        f'Valid arguments: name, optimize, greedy.')

  with ops.name_scope(name, 'einsum', [equation, inputs]) as name:
    inputs = list(inputs)
    input_shapes = []
    for operand in inputs:
      if isinstance(operand.shape, tensor_shape.TensorShape):
        input_shapes.append(operand.shape.as_list() if operand.shape else None)
      else:
        input_shapes.append(list(operand.shape))
    # Validate and sanitize the equation and resolve static input shapes, as
    # opt_einsum requires that all shapes be a tuple of positive integers.
    # Also remove ellipsis from the equation as opt_einsum will replace them
    # with named labels. Then broadcasting between different shapes or ranks
    # wouldn't work. (E.g. [1, 1, 2] wouldn't broadcast with [3, 1]).
    resolved_equation, resolved_input_shapes, ellipsis_label = (
        _einsum_v2_parse_and_resolve_equation(equation, input_shapes))

    if len(inputs) <= 2:  # No need to call opt_einsum.
      # Replace back ellipses that were removed for opt_einsum.
      if ellipsis_label:
        resolved_equation = resolved_equation.replace(ellipsis_label, '...')
      return gen_linalg_ops.einsum(inputs, resolved_equation)

    # Send fully specified shapes to opt_einsum, since it cannot handle unknown
    # dimensions. For unknown dimensions, we guess that the dimension equals 1.
    # Instead of creating Tensors or NumPy arrays with the specified shape,
    # create a dummy `shaped` object with a `shape` property.
    shaped = collections.namedtuple('shaped', ['shape'])
    shaped_inputs = tuple(
        [shaped(tuple(shape)) for shape in resolved_input_shapes])
    # opt_einsum breaks down an n-ary einsum operation into n-1 binary einsums.
    # Obtain the sequence of equations and the indices of operands involved in
    # each einsum operation.
    indices_and_equations = _get_opt_einsum_contract_path(
        resolved_equation, shaped_inputs, optimize)
    for operand_indices, binary_equation in indices_and_equations:
      if ellipsis_label:
        # Replace back ellipses that were removed for opt_einsum.
        binary_equation = binary_equation.replace(ellipsis_label, '...')
      operands = list(map(inputs.pop, operand_indices))
      inputs.append(gen_linalg_ops.einsum(operands, binary_equation))
    return inputs[0]


def _get_opt_einsum_contract_path(equation, shaped_inputs_tuple, optimize):
  """Returns the (memoized) result of opt_einsum.contract_path."""
  # Note: We use einsum_call=True, which is an internal api for opt_einsum,
  # to get the contraction path without having opt_einsum perform the actual
  # contractions.
  _, contractions = opt_einsum.contract_path(
      equation,
      *shaped_inputs_tuple,
      optimize=optimize,
      einsum_call=True,
      use_blas=True)
  # Return a tuple so that the cached value is not mutable.
  indices_and_equations = tuple([(expr[0], expr[2]) for expr in contractions])
  return indices_and_equations


# Cache the possibly expensive opt_einsum.contract_path call using lru_cache
# from the Python3+ standard library.
_get_opt_einsum_contract_path = functools.lru_cache(maxsize=128)(
    _get_opt_einsum_contract_path)


def _einsum_v2_parse_and_resolve_equation(equation, input_shapes):
  """Helper which validates einsum equation and resolves input shapes."""
  resolved_equation = equation.replace(' ', '')
  ellipsis_label = None
  if '...' in equation:
    # Replace ellipsis ('...') with '0' for (a) ease of parsing and (b) to
    # prevent opt_einsum from resolving them into named labels; as it doesn't
    # support broadcasting.
    ellipsis_label = '0'
    if ellipsis_label in resolved_equation:
      raise ValueError(
          f'Invalid character "{ellipsis_label}" in equation: {equation}.')
    resolved_equation = resolved_equation.replace('...', ellipsis_label)

  # Ensure there are no non-alphanumeric characters in the equation, including
  # periods (`.`) outside of ellipses, in the equation. This is not a hard
  # requirement; except we use a special character '0' for ellipsis.
  allowed_labels = 'a-zA-Z'
  if ellipsis_label:
    allowed_labels += ellipsis_label
  match = re.match('^([{0},]*)(->[{0}]*)?$'.format(allowed_labels),
                   resolved_equation)
  if not match:
    raise ValueError(
        'Subscripts have incorrect format: {}'.format(resolved_equation))
  input_labels = match.group(1).split(',')
  output_labels = match.group(2)[2:] if match.group(2) else None

  if len(input_shapes) != len(input_labels):
    raise ValueError('Got {} inputs for equation "{}", expecting {}'.format(
        len(input_shapes), equation, len(input_labels)))

  # Special case: if there are no '->', then we create output subscripts from
  # labels appearing only once.
  if '->' not in resolved_equation:
    label_counts = collections.Counter(match.group(1))
    output_labels = ''.join([
        x for x in sorted(list(label_counts))
        if x != ',' and label_counts[x] == 1
    ])
    resolved_equation += '->' + output_labels
  # Validate output_labels.
  if output_labels and len(set(output_labels)) != len(output_labels):
    raise ValueError(
        'Output subscripts contain a label appearing more than once: {}'.format(
            equation))
  input_label_set = set(match.group(1))
  for label in output_labels:
    if label != ellipsis_label and label not in input_label_set:
      raise ValueError('Output subscripts contain the label {} not present '
                       'in the input subscripts.'.format(label))
  if ellipsis_label and output_labels:
    num_output_ellipses = output_labels.count(ellipsis_label)
    if num_output_ellipses > 1:
      raise ValueError(
          'Output subscripts contain multiple ellipsis: {}'.format(equation))

  # Early return if <= 2 inputs. Resolved shapes are not needed.
  if len(input_shapes) <= 2:
    return resolved_equation, None, ellipsis_label

  # Create a map from axis labels to known dimensions. This is used to infer
  # unknown dimensions if a known dimension also has the same label.
  label_to_dim = collections.defaultdict(lambda: 1)
  for i, (labels, shape) in enumerate(zip(input_labels, input_shapes)):
    if shape is None:
      continue
    ellipsis_start = labels.find(ellipsis_label) if ellipsis_label else -1
    if ellipsis_start != -1:  # This input contains an ellipsis.
      if ellipsis_start != labels.rfind(ellipsis_label):
        raise ValueError(f'Too many ellipses in input label '
                         f'{labels.replace(ellipsis_label, "...")}.')
      if len(labels) > len(shape) + 1:
        raise ValueError('Too many named labels in {}th subscript string of'
                         ' equation {} for input shape {} '.format(
                             i, equation, shape))
      ellipsis_end = ellipsis_start + len(shape) + 1 - len(labels)
      shape[ellipsis_start:ellipsis_end] = ([
          np.prod(
              list(filter(None, shape[ellipsis_start:ellipsis_end])),
              dtype=np.int64)
      ])
    else:
      # This input does not contain an ellipsis.
      if len(labels) != len(shape):
        raise ValueError(
            'Number of named labels in input #{} of equation {} '
            'must be equal to the number of dimensions in shape {}'.format(
                i, equation, shape))
    for dim, label in zip(shape, labels):
      if dim is not None:
        label_to_dim[label] = max(label_to_dim[label], dim)

  resolved_shapes = []
  for labels in input_labels:
    resolved_shapes.append([label_to_dim[label] for label in labels])
  return resolved_equation, resolved_shapes, ellipsis_label
