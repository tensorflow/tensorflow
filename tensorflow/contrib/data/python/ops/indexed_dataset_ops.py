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
"""Python wrappers for indexed datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


class MaterializedIndexedDataset(object):
  """MaterializedIndexedDataset is highly experimental!
  """

  def __init__(self, materialized_resource, materializer, output_classes,
               output_types, output_shapes):
    self._materialized_resource = materialized_resource
    self._materializer = materializer
    self._output_classes = output_classes
    self._output_types = output_types
    self._output_shapes = output_shapes

  @property
  def initializer(self):
    if self._materializer is not None:
      return self._materializer
    raise ValueError("MaterializedDataset does not have a materializer")

  def get(self, index):
    """Get retrieves a value (or set of values) from the IndexedDataset.

    Args:
      index: A uint64 scalar or vector tensor with the indices to retrieve.

    Returns:
      A tensor containing the values corresponding to `index`.
    """
    # TODO(saeta): nest.pack_sequence_as(...)
    return ged_ops.experimental_indexed_dataset_get(
        self._materialized_resource,
        index,
        output_types=nest.flatten(
            sparse.as_dense_types(self._output_types, self._output_classes)),
        output_shapes=nest.flatten(
            sparse.as_dense_types(self._output_shapes, self._output_classes)))


class IndexedDataset(dataset_ops.Dataset):
  """IndexedDataset is highly experimental!
  """

  def __init__(self):
    pass

  def materialize(self, shared_name=None, container=None):
    """Materialize creates a MaterializedIndexedDataset.

    IndexedDatasets can be combined through operations such as TBD. Therefore,
    they are only materialized when absolutely required.

    Args:
      shared_name: a string for the shared name to use for the resource.
      container: a string for the container to store the resource.

    Returns:
      A MaterializedIndexedDataset.
    """
    if container is None:
      container = ""
    if shared_name is None:
      shared_name = ""
    materialized_resource = (
        ged_ops.experimental_materialized_index_dataset_handle(
            container=container,
            shared_name=shared_name,
            output_types=nest.flatten(
                sparse.as_dense_types(self.output_types, self.output_classes)),
            output_shapes=nest.flatten(
                sparse.as_dense_types(self.output_shapes,
                                      self.output_classes))))

    with ops.colocate_with(materialized_resource):
      materializer = ged_ops.experimental_indexed_dataset_materialize(
          self._as_variant_tensor(), materialized_resource)
    return MaterializedIndexedDataset(materialized_resource, materializer,
                                      self.output_classes, self.output_types,
                                      self.output_shapes)

  @abc.abstractproperty
  def output_types(self):
    """Returns the type of each component of an element of this IndexedDataset.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this IndexedDataset.
    """
    raise NotImplementedError("IndexedDataset.output_types")

  @abc.abstractproperty
  def output_classes(self):
    """Returns the class of each component of an element of this IndexedDataset.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this IndexedDataset.
    """
    raise NotImplementedError("IndexedDataset.output_classes")

  @abc.abstractproperty
  def output_shapes(self):
    """Returns the shape of each component of an element of this IndexedDataset.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this IndexedDataset.
    """
    raise NotImplementedError("IndexedDataset.output_shapes")

  @abc.abstractmethod
  def _as_variant_tensor(self):
    """Creates a `tf.variant` `tf.Tensor` representing this IndexedDataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this
      IndexedDataset.
    """
    raise NotImplementedError("IndexedDataset._as_variant_tensor")


class IdentityIndexedDataset(IndexedDataset):
  """IdentityIndexedDataset is a trivial indexed dataset used for testing.
  """

  def __init__(self, size):
    super(IdentityIndexedDataset, self).__init__()
    # TODO(saeta): Verify _size is a scalar!
    self._size = ops.convert_to_tensor(size, dtype=dtypes.uint64, name="size")

  @property
  def output_types(self):
    return dtypes.uint64

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  def _as_variant_tensor(self):
    return ged_ops.experimental_identity_indexed_dataset(self._size)

  def _inputs(self):
    return []
