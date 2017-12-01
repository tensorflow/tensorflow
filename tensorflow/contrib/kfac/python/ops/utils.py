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
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# Method used for inverting matrices.
POSDEF_INV_METHOD = "cholesky"
POSDEF_EIG_METHOD = "self_adjoint"


def set_global_constants(posdef_inv_method=None):
  """Sets various global constants used by the classes in this module."""
  global POSDEF_INV_METHOD

  if posdef_inv_method is not None:
    POSDEF_INV_METHOD = posdef_inv_method


class SequenceDict(object):
  """A dict convenience wrapper that allows getting/setting with sequences."""

  def __init__(self, iterable=None):
    self._dict = dict(iterable or [])

  def __getitem__(self, key_or_keys):
    if isinstance(key_or_keys, (tuple, list)):
      return list(map(self.__getitem__, key_or_keys))
    else:
      return self._dict[key_or_keys]

  def __setitem__(self, key_or_keys, val_or_vals):
    if isinstance(key_or_keys, (tuple, list)):
      for key, value in zip(key_or_keys, val_or_vals):
        self[key] = value
    else:
      self._dict[key_or_keys] = val_or_vals

  def items(self):
    return list(self._dict.items())


def tensors_to_column(tensors):
  """Converts a tensor or list of tensors to a column vector.

  Args:
    tensors: A tensor or list of tensors.

  Returns:
    The tensors reshaped into vectors and stacked on top of each other.
  """
  if isinstance(tensors, (tuple, list)):
    return array_ops.concat(
        tuple(array_ops.reshape(tensor, [-1, 1]) for tensor in tensors), axis=0)
  else:
    return array_ops.reshape(tensors, [-1, 1])


def column_to_tensors(tensors_template, colvec):
  """Converts a column vector back to the shape of the given template.

  Args:
    tensors_template: A tensor or list of tensors.
    colvec: A 2d column vector with the same shape as the value of
        tensors_to_column(tensors_template).

  Returns:
    X, where X is tensor or list of tensors with the properties:
     1) tensors_to_column(X) = colvec
     2) X (or its elements) have the same shape as tensors_template (or its
        elements)
  """
  if isinstance(tensors_template, (tuple, list)):
    offset = 0
    tensors = []
    for tensor_template in tensors_template:
      sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
      tensor = array_ops.reshape(colvec[offset:(offset + sz)],
                                 tensor_template.shape)
      tensors.append(tensor)
      offset += sz

    tensors = tuple(tensors)
  else:
    tensors = array_ops.reshape(colvec, tensors_template.shape)

  return tensors


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product two matrices."""
  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = array_ops.reshape(mat1, [m1, 1, n1, 1])
  m2, n2 = mat2.get_shape().as_list()
  mat2_rsh = array_ops.reshape(mat2, [1, m2, 1, n2])
  return array_ops.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def layer_params_to_mat2d(vector):
  """Converts a vector shaped like layer parameters to a 2D matrix.

  In particular, we reshape the weights/filter component of the vector to be
  2D, flattening all leading (input) dimensions. If there is a bias component,
  we concatenate it to the reshaped weights/filter component.

  Args:
    vector: A Tensor or pair of Tensors shaped like layer parameters.

  Returns:
    A 2D Tensor with the same coefficients and the same output dimension.
  """
  if isinstance(vector, (tuple, list)):
    w_part, b_part = vector
    w_part_reshaped = array_ops.reshape(w_part,
                                        [-1, w_part.shape.as_list()[-1]])
    return array_ops.concat(
        (w_part_reshaped, array_ops.reshape(b_part, [1, -1])), axis=0)
  else:
    return array_ops.reshape(vector, [-1, vector.shape.as_list()[-1]])


def mat2d_to_layer_params(vector_template, mat2d):
  """Converts a canonical 2D matrix representation back to a vector.

  Args:
    vector_template: A Tensor or pair of Tensors shaped like layer parameters.
    mat2d: A 2D Tensor with the same shape as the value of
        layer_params_to_mat2d(vector_template).

  Returns:
    A Tensor or pair of Tensors with the same coefficients as mat2d and the same
        shape as vector_template.
  """
  if isinstance(vector_template, (tuple, list)):
    w_part, b_part = mat2d[:-1], mat2d[-1]
    return array_ops.reshape(w_part, vector_template[0].shape), b_part
  else:
    return array_ops.reshape(mat2d, vector_template.shape)


def posdef_inv(tensor, damping):
  """Computes the inverse of tensor + damping * identity."""
  identity = linalg_ops.eye(tensor.shape.as_list()[0], dtype=tensor.dtype)
  damping = math_ops.cast(damping, dtype=tensor.dtype)
  return posdef_inv_functions[POSDEF_INV_METHOD](tensor, identity, damping)


def posdef_inv_matrix_inverse(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) directly."""
  return linalg_ops.matrix_inverse(tensor + damping * identity)


def posdef_inv_cholesky(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) with Cholesky."""
  chol = linalg_ops.cholesky(tensor + damping * identity)
  return linalg_ops.cholesky_solve(chol, identity)


def posdef_inv_eig(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) with eigendecomposition."""
  eigenvalues, eigenvectors = linalg_ops.self_adjoint_eig(
      tensor + damping * identity)
  return math_ops.matmul(
      eigenvectors / eigenvalues, eigenvectors, transpose_b=True)


posdef_inv_functions = {
    "matrix_inverse": posdef_inv_matrix_inverse,
    "cholesky": posdef_inv_cholesky,
    "eig": posdef_inv_eig,
}


def posdef_eig(mat):
  """Computes the eigendecomposition of a positive semidefinite matrix."""
  return posdef_eig_functions[POSDEF_EIG_METHOD](mat)


def posdef_eig_svd(mat):
  """Computes the singular values and left singular vectors of a matrix."""
  evals, evecs, _ = linalg_ops.svd(mat)

  return evals, evecs


def posdef_eig_self_adjoint(mat):
  """Computes eigendecomposition using self_adjoint_eig."""
  evals, evecs = linalg_ops.self_adjoint_eig(mat)
  evals = math_ops.abs(evals)  # Should be equivalent to svd approach.

  return evals, evecs


posdef_eig_functions = {
    "self_adjoint": posdef_eig_self_adjoint,
    "svd": posdef_eig_svd,
}


class SubGraph(object):
  """Defines a subgraph given by all the dependencies of a given set of outputs.
  """

  def __init__(self, outputs):
    self._members = set()

    self._recurse_add(outputs)

  def _recurse_add(self, nodes):
    for node in nodes:
      if node in self._members:
        continue
      self._members.add(node)

      if isinstance(node, ops.Tensor):
        self._recurse_add((node.op,))
      elif isinstance(node, ops.Operation):
        self._recurse_add(node.inputs)

  def is_member(self, node):
    """Check if 'node' is in this subgraph."""
    return node in self._members

  def variable_uses(self, var):
    """Computes number of times a variable is used."""
    return len(self._members.intersection(set(var.value().consumers())))

  def filter_list(self, node_list):
    """Filters 'node_list' to nodes in this subgraph."""
    filtered_list = []
    for node in node_list:
      if self.is_member(node):
        filtered_list.append(node)
    return filtered_list


def generate_random_signs(shape, dtype=dtypes.float32):
  """Generate a random tensor with {-1, +1} entries."""
  ints = random_ops.random_uniform(shape, maxval=2, dtype=dtypes.int32)
  return 2 * math_ops.cast(ints, dtype=dtype) - 1


def fwd_gradients(ys, xs, grad_xs=None, stop_gradients=None):
  """Compute forward-mode gradients."""
  # See b/37888268.

  # This version of forward-mode autodiff is based on code by Tim Cooijmans
  # and handles list arguments and certain special cases such as when the
  # ys doesn't depend on one or more of the xs, and when ops.IndexedSlices are
  # generated by the first gradients_impl.gradients call.

  us = [array_ops.zeros_like(y) + float("nan") for y in ys]
  dydxs = gradients_impl.gradients(
      ys, xs, grad_ys=us, stop_gradients=stop_gradients)

  # Deal with strange types that gradients_impl.gradients returns but can't
  # deal with.
  dydxs = [
      ops.convert_to_tensor(dydx)
      if isinstance(dydx, ops.IndexedSlices) else dydx for dydx in dydxs
  ]
  dydxs = [
      array_ops.zeros_like(x) if dydx is None else dydx
      for x, dydx in zip(xs, dydxs)
  ]

  dysdx = gradients_impl.gradients(dydxs, us, grad_ys=grad_xs)

  return dysdx

# TODO(b/69623235): Add a function for finding tensors that share gradients
# to eliminate redundant fisher factor computations.
