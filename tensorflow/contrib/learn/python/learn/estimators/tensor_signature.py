#  Copyright 2015 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""TensorSignature class and utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


class TensorSignature(collections.namedtuple(
    "TensorSignature", ["dtype", "shape", "is_sparse"])):
  """Signature of the `Tensor` object.

  Useful to check compatibility of tensors.

  Attributes:
    dtype: `DType` object.
    shape: `TensorShape` object.
  """

  def __new__(cls, tensor):
    if isinstance(tensor, ops.SparseTensor):
      return super(TensorSignature, cls).__new__(
          cls, dtype=tensor.values.dtype, shape=None, is_sparse=True)
    return super(TensorSignature, cls).__new__(
        cls, dtype=tensor.dtype, shape=tensor.get_shape(), is_sparse=False)

  def is_compatible_with(self, other):
    """Returns True if signatures are compatible."""
    if other.is_sparse:
      return self.is_sparse and self.dtype.is_compatible_with(other.dtype)
    return (self.dtype.is_compatible_with(other.dtype) and
            self.shape.is_compatible_with(other.shape) and not self.is_sparse)

  def get_placeholder(self):
    if self.is_sparse:
      return array_ops.sparse_placeholder(dtype=self.dtype)
    return array_ops.placeholder(dtype=self.dtype, shape=self.shape)


def tensors_compatible(tensors, signatures):
  """Check that tensors are compatible with signatures.

  Args:
    tensors: Dict of `Tensor` objects or single `Tensor` object.
    signatures: Dict of `TensorSignature` objects or
                single `TensorSignature` object.

  Returns:
    True if all tensors are compatible, False otherwise.
  """
  # Dict of Tensors as input.
  if isinstance(tensors, dict):
    if not isinstance(signatures, dict):
      return False
    for key in signatures:
      if key not in tensors:
        return False
      if not TensorSignature(tensors[key]).is_compatible_with(signatures[key]):
        return False
    return True

  # Single tensor as input.
  if isinstance(signatures, dict):
    return False
  return TensorSignature(tensors).is_compatible_with(signatures)


def create_signatures(tensors):
  """Creates TensorSignature objects for given tensors.

  Args:
    tensors: Dict of `Tensor` objects or single `Tensor`.

  Returns:
    Dict of `TensorSignature` objects or single `TensorSignature`.
  """
  if isinstance(tensors, dict):
    return {
        key: TensorSignature(tensors[key]) for key in tensors}
  return TensorSignature(tensors)


def create_placeholders_from_signatures(signatures):
  """Creates placeholders from given signatures.

  Args:
    signatures: Dict of `TensorSignature` objects or single `TensorSignature`.

  Returns:
    Dict of `tf.placeholder` objects or single `tf.placeholder`.
  """
  if not isinstance(signatures, dict):
    return signatures.get_placeholder()
  return {
      key: signatures[key].get_placeholder()
      for key in signatures}
