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
"""Utility functions for solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf


def create_operator(matrix):
  """Creates a linear operator from a rank-2 tensor."""

  linear_operator = collections.namedtuple(
      "LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])

  # TODO(rmlarsen): Handle SparseTensor.
  shape = matrix.get_shape()
  if shape.is_fully_defined():
    shape = shape.as_list()
  else:
    shape = tf.shape(matrix)
  return linear_operator(
      shape=shape,
      dtype=matrix.dtype,
      apply=lambda v: tf.matmul(matrix, v, adjoint_a=False),
      apply_adjoint=lambda v: tf.matmul(matrix, v, adjoint_a=True))


# TODO(rmlarsen): Measure if we should just call matmul.
def dot(x, y):
  return tf.reduce_sum(tf.conj(x) * y)


# TODO(rmlarsen): Implement matrix/vector norm op in C++ in core.
# We need 1-norm, inf-norm, and Frobenius norm.
def l2norm_squared(v):
  return tf.constant(2, dtype=v.dtype.base_dtype) * tf.nn.l2_loss(v)


def l2norm(v):
  return tf.sqrt(l2norm_squared(v))


def l2normalize(v):
  norm = l2norm(v)
  return v / norm, norm
