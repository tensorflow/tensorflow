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
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util.tf_export import tf_export


class _ParseExampleDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that parses `example` dataset into a `dict` dataset."""

  def __init__(self, input_dataset, features, num_parallel_calls,
               deterministic):
    self._input_dataset = input_dataset
    if not structure.are_compatible(
        input_dataset.element_spec,
        tensor_spec.TensorSpec([None], dtypes.string)):
      raise TypeError("Input dataset should be a dataset of vectors of "
                      f"strings. Instead it is `{input_dataset.element_spec}`.")
    self._num_parallel_calls = num_parallel_calls
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"
    # pylint: disable=protected-access
    self._features = parsing_ops._prepend_none_dimension(features)
    # TODO(b/112859642): Pass sparse_index and sparse_values for SparseFeature
    params = parsing_ops._ParseOpParams.from_features(self._features, [
        parsing_ops.VarLenFeature, parsing_ops.SparseFeature,
        parsing_ops.FixedLenFeature, parsing_ops.FixedLenSequenceFeature,
        parsing_ops.RaggedFeature
    ])
    # pylint: enable=protected-access
    self._sparse_keys = params.sparse_keys
    self._sparse_types = params.sparse_types
    self._ragged_keys = params.ragged_keys
    self._ragged_value_types = params.ragged_value_types
    self._ragged_split_types = params.ragged_split_types
    self._dense_keys = params.dense_keys
    self._dense_defaults = params.dense_defaults_vec
    self._dense_shapes = params.dense_shapes_as_proto
    self._dense_types = params.dense_types
    input_dataset_shape = dataset_ops.get_legacy_output_shapes(
        self._input_dataset)

    self._element_spec = {}

    for (key, value_type) in zip(params.sparse_keys, params.sparse_types):
      self._element_spec[key] = sparse_tensor.SparseTensorSpec(
          input_dataset_shape.concatenate([None]), value_type)

    for (key, value_type, dense_shape) in zip(params.dense_keys,
                                              params.dense_types,
                                              params.dense_shapes):
      self._element_spec[key] = tensor_spec.TensorSpec(
          input_dataset_shape.concatenate(dense_shape), value_type)

    for (key, value_type, splits_type) in zip(params.ragged_keys,
                                              params.ragged_value_types,
                                              params.ragged_split_types):
      self._element_spec[key] = ragged_tensor.RaggedTensorSpec(
          input_dataset_shape.concatenate([None]), value_type, 1, splits_type)

    variant_tensor = (
        gen_experimental_dataset_ops.parse_example_dataset_v2(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._num_parallel_calls,
            self._dense_defaults,
            self._sparse_keys,
            self._dense_keys,
            self._sparse_types,
            self._dense_shapes,
            deterministic=self._deterministic,
            ragged_keys=self._ragged_keys,
            ragged_value_types=self._ragged_value_types,
            ragged_split_types=self._ragged_split_types,
            **self._flat_structure))
    super(_ParseExampleDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


# TODO(b/111553342): add arguments names and example names as well.
@tf_export("data.experimental.parse_example_dataset")
def parse_example_dataset(features, num_parallel_calls=1, deterministic=None):
  """A transformation that parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized `Example` protos given in `serialized`. We refer
  to `serialized` as a batch with `batch_size` many entries of individual
  `Example` protos.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`,
  `SparseTensor`, and `RaggedTensor` objects. `features` is a dict from keys to
  `VarLenFeature`, `RaggedFeature`, `SparseFeature`, and `FixedLenFeature`
  objects. Each `VarLenFeature` and `SparseFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each
  `FixedLenFeature` is mapped to a `Tensor`. See `tf.io.parse_example` for more
  details about feature dictionaries.

  Args:
   features: A `dict` mapping feature keys to `FixedLenFeature`,
     `VarLenFeature`, `RaggedFeature`, and `SparseFeature` values.
   num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of parsing processes to call in parallel.
   deterministic: (Optional.) A boolean controlling whether determinism
      should be traded for performance by allowing elements to be produced out
      of order if some parsing calls complete faster than others. If
      `deterministic` is `None`, the
      `tf.data.Options.deterministic` dataset option (`True` by default) is used
      to decide whether to produce elements deterministically.

  Returns:
    A dataset transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if features argument is None.
  """
  if features is None:
    raise ValueError("Argument `features` is required, but not specified.")

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    out_dataset = _ParseExampleDataset(dataset, features, num_parallel_calls,
                                       deterministic)
    if any(
        isinstance(feature, parsing_ops.SparseFeature) or
        (isinstance(feature, parsing_ops.RaggedFeature) and feature.partitions)
        for feature in features.values()):
      # pylint: disable=protected-access
      # pylint: disable=g-long-lambda
      out_dataset = out_dataset.map(
          lambda x: parsing_ops._construct_tensors_for_composite_features(
              features, x),
          num_parallel_calls=num_parallel_calls)
    return out_dataset

  return _apply_fn
