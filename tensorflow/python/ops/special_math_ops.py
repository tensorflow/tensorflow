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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import re

import numpy as np
import opt_einsum
import six

from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# TODO(b/27419586) Change docstring for required dtype of x once int allowed
@tf_export('math.lbeta', v1=['math.lbeta', 'lbeta'])
@deprecation.deprecated_endpoints('lbeta')
def lbeta(x, name=None):
  r"""Computes \\(ln(|Beta(x)|)\\), reducing along the last dimension.

  Given one-dimensional `z = [z_0,...,z_{K-1}]`, we define

  $$Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)$$

  And for `n + 1` dimensional `x` with shape `[N1, ..., Nn, K]`, we define
  $$lbeta(x)[i1, ..., in] = Log(|Beta(x[i1, ..., in, :])|)$$.

  In other words, the last dimension is treated as the `z` vector.

  Note that if `z = [u, v]`, then
  \\(Beta(z) = int_0^1 t^{u-1} (1 - t)^{v-1} dt\\), which defines the
  traditional bivariate beta function.

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


@tf_export('math.bessel_i0')
def bessel_i0(x, name=None):
  """Computes the Bessel i0 function of `x` element-wise.

  Modified Bessel function of order 0.

  It is preferable to use the numerically stabler function `i0e(x)` instead.

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
    return math_ops.exp(math_ops.abs(x)) * math_ops.bessel_i0e(x)


@tf_export('math.bessel_i1')
def bessel_i1(x, name=None):
  """Computes the Bessel i1 function of `x` element-wise.

  Modified Bessel function of order 1.

  It is preferable to use the numerically stabler function `i1e(x)` instead.

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
    return math_ops.exp(math_ops.abs(x)) * math_ops.bessel_i1e(x)


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


@tf_export('einsum', 'linalg.einsum')
def einsum(equation, *inputs, **kwargs):
  """Tensor contraction over specified indices and outer product.

  Einsum allows defining Tensors by defining their element-wise computation.
  This computation is defined by `equation`, a shorthand form based on Einstein
  summation. As an example, consider multiplying two matrices A and B to form a
  matrix C.  The elements of C are given by:

  ```
    C[i,k] = sum_j A[i,j] * B[j,k]
  ```

  The corresponding `equation` is:

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

  Many common operations can be expressed in this way.  For example:

  ```python
  # Matrix multiplication
  einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]

  # Dot product
  einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

  # Outer product
  einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]

  # Transpose
  einsum('ij->ji', m)  # output[j,i] = m[i,j]

  # Trace
  einsum('ii', m)  # output[j,i] = trace(m) = sum_i m[i, i]

  # Batch matrix multiplication
  einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
  ```

  To enable and control broadcasting, use an ellipsis.  For example, to perform
  batch matrix multiplication with NumPy-style broadcasting across the batch
  dimensions, use:

  ```python
  einsum('...ij,...jk->...ik', u, v)
  ```

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
  name = kwargs.pop('name', None)
  optimize = kwargs.pop('optimize', 'greedy')
  if kwargs:
    msg = 'Invalid keyword arguments for einsum: {}'
    raise TypeError(msg.format(', '.join(kwargs)))

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
        _einsum_parse_and_resolve_equation(equation, input_shapes))

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
# from the Python3 standard library.
if six.PY3:
  _get_opt_einsum_contract_path = functools.lru_cache(maxsize=128)(
      _get_opt_einsum_contract_path)


def _einsum_parse_and_resolve_equation(equation, input_shapes):
  """Helper which validates einsum equation and resolves input shapes."""
  resolved_equation = equation.replace(' ', '')
  ellipsis_label = None
  if '...' in equation:
    # Replace ellipsis ('...') with '0' for (a) ease of parsing and (b) to
    # prevent opt_einsum from resolving them into named labels; as it doesn't
    # support broadcasting.
    ellipsis_label = '0'
    if ellipsis_label in resolved_equation:
      raise ValueError('Invalid character "0" in equation: {}'.format(equation))
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
        raise ValueError('Too many ellipsis')
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
