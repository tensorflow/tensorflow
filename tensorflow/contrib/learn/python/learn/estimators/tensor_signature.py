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
"""TensorSignature class and utilities (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops


class TensorSignature(
    collections.namedtuple("TensorSignature", ["dtype", "shape", "is_sparse"])):
  """Signature of the `Tensor` object.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Useful to check compatibility of tensors.

  Example:

  ```python
  examples = tf.compat.v1.placeholder(...)
  inputs = {'a': var_a, 'b': var_b}
  signatures = tensor_signature.create_signatures(inputs)
  result = tensor_signature.create_example_parser_from_signatures(
      signatures, examples)
  self.assertTrue(tensor_signature.tensors_compatible(result, signatures))
  ```

  Attributes:
    dtype: `DType` object.
    shape: `TensorShape` object.
  """

  def __new__(cls, tensor):
    if isinstance(tensor, sparse_tensor.SparseTensor):
      return super(TensorSignature, cls).__new__(
          cls, dtype=tensor.values.dtype, shape=None, is_sparse=True)
    return super(TensorSignature, cls).__new__(
        cls, dtype=tensor.dtype, shape=tensor.get_shape(), is_sparse=False)

  def is_compatible_with(self, other):
    """Returns True if signatures are compatible."""

    def _shape_is_compatible_0dim(this, other):
      """Checks that shapes are compatible skipping dim 0."""
      other = tensor_shape.as_shape(other)
      # If shapes are None (unknown) they may be compatible.
      if this.dims is None or other.dims is None:
        return True
      if this.ndims != other.ndims:
        return False
      for dim, (x_dim, y_dim) in enumerate(zip(this.dims, other.dims)):
        if dim == 0:
          continue
        if not x_dim.is_compatible_with(y_dim):
          return False
      return True

    if other.is_sparse:
      return self.is_sparse and self.dtype.is_compatible_with(other.dtype)
    return (self.dtype.is_compatible_with(other.dtype) and
            _shape_is_compatible_0dim(self.shape, other.shape) and
            not self.is_sparse)

  def get_placeholder(self):
    if self.is_sparse:
      return array_ops.sparse_placeholder(dtype=self.dtype)
    return array_ops.placeholder(
        dtype=self.dtype, shape=[None] + list(self.shape[1:]))

  def get_feature_spec(self):
    dtype = self.dtype
    # Convert, because example parser only supports float32, int64 and string.
    if dtype == dtypes.int32:
      dtype = dtypes.int64
    if dtype == dtypes.float64:
      dtype = dtypes.float32
    if self.is_sparse:
      return parsing_ops.VarLenFeature(dtype=dtype)
    return parsing_ops.FixedLenFeature(shape=self.shape[1:], dtype=dtype)


def tensors_compatible(tensors, signatures):
  """Check that tensors are compatible with signatures.

  Args:
    tensors: Dict of `Tensor` objects or single `Tensor` object.
    signatures: Dict of `TensorSignature` objects or single `TensorSignature`
      object.

  Returns:
    True if all tensors are compatible, False otherwise.
  """
  # Dict of Tensors as input.
  if tensors is None:
    return signatures is None

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
  if signatures is None or isinstance(signatures, dict):
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
    return {key: TensorSignature(tensors[key]) for key in tensors}
  if tensors is None:
    return None
  return TensorSignature(tensors)


def create_placeholders_from_signatures(signatures):
  """Creates placeholders from given signatures.

  Args:
    signatures: Dict of `TensorSignature` objects or single `TensorSignature`,
      or `None`.

  Returns:
    Dict of `tf.compat.v1.placeholder` objects or single
    `tf.compat.v1.placeholder`, or `None`.
  """
  if signatures is None:
    return None
  if not isinstance(signatures, dict):
    return signatures.get_placeholder()
  return {key: signatures[key].get_placeholder() for key in signatures}


def create_example_parser_from_signatures(signatures,
                                          examples_batch,
                                          single_feature_name="feature"):
  """Creates example parser from given signatures.

  Args:
    signatures: Dict of `TensorSignature` objects or single `TensorSignature`.
    examples_batch: string `Tensor` of serialized `Example` proto.
    single_feature_name: string, single feature name.

  Returns:
    features: `Tensor` or `dict` of `Tensor` objects.
  """
  feature_spec = {}
  if not isinstance(signatures, dict):
    feature_spec[single_feature_name] = signatures.get_feature_spec()
  else:
    feature_spec = {
        key: signatures[key].get_feature_spec() for key in signatures
    }
  features = parsing_ops.parse_example(examples_batch, feature_spec)
  if not isinstance(signatures, dict):
    # Returns single feature, casts if needed.
    features = features[single_feature_name]
    if not signatures.dtype.is_compatible_with(features.dtype):
      features = math_ops.cast(features, signatures.dtype)
    return features
  # Returns dict of features, casts if needed.
  for name in features:
    if not signatures[name].dtype.is_compatible_with(features[name].dtype):
      features[name] = math_ops.cast(features[name], signatures[name].dtype)
  return features
