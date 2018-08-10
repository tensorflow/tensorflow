# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""An Optional type for representing potentially missing values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops


class Optional(object):
  """Wraps a nested structure of tensors that may/may not be present at runtime.

  An `Optional` can represent the result of an operation that may fail as a
  value, rather than raising an exception and halting execution. For example,
  `tf.contrib.data.get_next_as_optional` returns an `Optional` that either
  contains the next value from a `tf.data.Iterator` if one exists, or a "none"
  value that indicates the end of the sequence has been reached.
  """

  @abc.abstractmethod
  def has_value(self, name=None):
    """Returns a tensor that evaluates to `True` if this optional has a value.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.bool`.
    """
    raise NotImplementedError("Optional.has_value()")

  @abc.abstractmethod
  def get_value(self, name=None):
    """Returns a nested structure of values wrapped by this optional.

    If this optional does not have a value (i.e. `self.has_value()` evaluates
    to `False`), this operation will raise `tf.errors.InvalidArgumentError`
    at runtime.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A nested structure of `tf.Tensor` and/or `tf.SparseTensor` objects.
    """
    raise NotImplementedError("Optional.get_value()")

  @abc.abstractproperty
  def output_classes(self):
    """Returns the class of each component of this optional.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of this optional.
    """
    raise NotImplementedError("Optional.output_classes")

  @abc.abstractproperty
  def output_shapes(self):
    """Returns the shape of each component of this optional.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of this optional.
    """
    raise NotImplementedError("Optional.output_shapes")

  @abc.abstractproperty
  def output_types(self):
    """Returns the type of each component of this optional.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of this optional.
    """
    raise NotImplementedError("Optional.output_types")

  @staticmethod
  def from_value(value):
    """Returns an `Optional` that wraps the given value.

    Args:
      value: A nested structure of `tf.Tensor` and/or `tf.SparseTensor` objects.

    Returns:
      An `Optional` that wraps `value`.
    """
    # TODO(b/110122868): Consolidate this destructuring logic with the
    # similar code in `Dataset.from_tensors()`.
    with ops.name_scope("optional") as scope:
      with ops.name_scope("value"):
        value = nest.pack_sequence_as(value, [
            sparse_tensor_lib.SparseTensor.from_value(t)
            if sparse_tensor_lib.is_sparse(t) else ops.convert_to_tensor(
                t, name="component_%d" % i)
            for i, t in enumerate(nest.flatten(value))
        ])

      encoded_value = nest.flatten(sparse.serialize_sparse_tensors(value))
      output_classes = sparse.get_classes(value)
      output_shapes = nest.pack_sequence_as(
          value, [t.get_shape() for t in nest.flatten(value)])
      output_types = nest.pack_sequence_as(
          value, [t.dtype for t in nest.flatten(value)])

    return _OptionalImpl(
        gen_dataset_ops.optional_from_value(encoded_value, name=scope),
        output_shapes, output_types, output_classes)

  @staticmethod
  def none_from_structure(output_shapes, output_types, output_classes):
    """Returns an `Optional` that has no value.

    NOTE: This method takes arguments that define the structure of the value
    that would be contained in the returned `Optional` if it had a value.

    Args:
      output_shapes: A nested structure of `tf.TensorShape` objects
        corresponding to each component of this optional.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of this optional.
      output_classes: A nested structure of Python `type` objects corresponding
        to each component of this optional.

    Returns:
      An `Optional` that has no value.
    """
    return _OptionalImpl(gen_dataset_ops.optional_none(), output_shapes,
                         output_types, output_classes)


class _OptionalImpl(Optional):
  """Concrete implementation of `tf.contrib.data.Optional`.

  NOTE(mrry): This implementation is kept private, to avoid defining
  `Optional.__init__()` in the public API.
  """

  def __init__(self, variant_tensor, output_shapes, output_types,
               output_classes):
    # TODO(b/110122868): Consolidate the structure validation logic with the
    # similar logic in `Iterator.from_structure()` and
    # `Dataset.from_generator()`.
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    output_shapes = nest.map_structure_up_to(
        output_types, tensor_shape.as_shape, output_shapes)
    nest.assert_same_structure(output_types, output_shapes)
    nest.assert_same_structure(output_types, output_classes)
    self._variant_tensor = variant_tensor
    self._output_shapes = output_shapes
    self._output_types = output_types
    self._output_classes = output_classes

  def has_value(self, name=None):
    return gen_dataset_ops.optional_has_value(self._variant_tensor, name=name)

  def get_value(self, name=None):
    # TODO(b/110122868): Consolidate the restructuring logic with similar logic
    # in `Iterator.get_next()` and `StructuredFunctionWrapper`.
    with ops.name_scope(name, "OptionalGetValue",
                        [self._variant_tensor]) as scope:
      return sparse.deserialize_sparse_tensors(
          nest.pack_sequence_as(
              self._output_types,
              gen_dataset_ops.optional_get_value(
                  self._variant_tensor,
                  name=scope,
                  output_types=nest.flatten(
                      sparse.as_dense_types(self._output_types,
                                            self._output_classes)),
                  output_shapes=nest.flatten(
                      sparse.as_dense_shapes(self._output_shapes,
                                             self._output_classes)))),
          self._output_types, self._output_shapes, self._output_classes)

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types
