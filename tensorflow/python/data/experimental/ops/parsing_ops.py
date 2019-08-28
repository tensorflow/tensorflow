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
"""Experimental `dataset` API for parsing example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util.tf_export import tf_export


class _ParseExampleDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that parses `example` dataset into a `dict` dataset."""

  def __init__(self, input_dataset, features, num_parallel_calls):
    self._input_dataset = input_dataset
    if not structure.are_compatible(
        input_dataset.element_spec,
        tensor_spec.TensorSpec([None], dtypes.string)):
      raise TypeError("Input dataset should be a dataset of vectors of strings")
    self._num_parallel_calls = num_parallel_calls
    # pylint: disable=protected-access
    self._features = parsing_ops._prepend_none_dimension(features)
    # sparse_keys and dense_keys come back sorted here.
    (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
     dense_shapes) = parsing_ops._features_to_raw_params(
         self._features, [
             parsing_ops.VarLenFeature, parsing_ops.SparseFeature,
             parsing_ops.FixedLenFeature, parsing_ops.FixedLenSequenceFeature
         ])
    # TODO(b/112859642): Pass sparse_index and sparse_values for SparseFeature.
    (_, dense_defaults_vec, sparse_keys, sparse_types, dense_keys, dense_shapes,
     dense_shape_as_shape) = parsing_ops._process_raw_parameters(
         None, dense_defaults, sparse_keys, sparse_types, dense_keys,
         dense_types, dense_shapes)
    # pylint: enable=protected-access
    self._sparse_keys = sparse_keys
    self._sparse_types = sparse_types
    self._dense_keys = dense_keys
    self._dense_defaults = dense_defaults_vec
    self._dense_shapes = dense_shapes
    self._dense_types = dense_types
    input_dataset_shape = dataset_ops.get_legacy_output_shapes(
        self._input_dataset)
    dense_output_shapes = [input_dataset_shape.concatenate(shape)
                           for shape in dense_shape_as_shape]
    sparse_output_shapes = [input_dataset_shape.concatenate([None])
                            for _ in range(len(sparse_keys))]

    output_shapes = dict(
        zip(self._dense_keys + self._sparse_keys,
            dense_output_shapes + sparse_output_shapes))
    output_types = dict(
        zip(self._dense_keys + self._sparse_keys,
            self._dense_types + self._sparse_types))
    output_classes = dict(
        zip(self._dense_keys + self._sparse_keys,
            [ops.Tensor for _ in range(len(self._dense_defaults))] +
            [sparse_tensor.SparseTensor for _ in range(len(self._sparse_keys))
            ]))
    self._element_spec = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)

    variant_tensor = (
        gen_experimental_dataset_ops.parse_example_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._num_parallel_calls,
            self._dense_defaults,
            self._sparse_keys,
            self._dense_keys,
            self._sparse_types,
            self._dense_shapes,
            **self._flat_structure))
    super(_ParseExampleDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


# TODO(b/111553342): add arguments names and example names as well.
@tf_export("data.experimental.parse_example_dataset")
def parse_example_dataset(features, num_parallel_calls=1):
  """A transformation that parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized `Example` protos given in `serialized`. We refer
  to `serialized` as a batch with `batch_size` many entries of individual
  `Example` protos.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`,
  `SparseFeature`, and `FixedLenFeature` objects. Each `VarLenFeature`
  and `SparseFeature` is mapped to a `SparseTensor`, and each
  `FixedLenFeature` is mapped to a `Tensor`. See `tf.io.parse_example` for more
  details about feature dictionaries.

  Args:
   features: A `dict` mapping feature keys to `FixedLenFeature`,
     `VarLenFeature`, and `SparseFeature` values.
   num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of parsing processes to call in parallel.

  Returns:
    A dataset transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if features argument is None.
  """
  if features is None:
    raise ValueError("Missing: features was %s." % features)

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    out_dataset = _ParseExampleDataset(dataset, features, num_parallel_calls)
    if any(
        isinstance(feature, parsing_ops.SparseFeature)
        for _, feature in features.items()
    ):
      # pylint: disable=protected-access
      # pylint: disable=g-long-lambda
      out_dataset = out_dataset.map(
          lambda x: parsing_ops._construct_sparse_tensors_for_sparse_features(
              features, x), num_parallel_calls=num_parallel_calls)
    return out_dataset

  return _apply_fn
