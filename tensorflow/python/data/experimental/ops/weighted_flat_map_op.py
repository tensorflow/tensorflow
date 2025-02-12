# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `weighted_flat_map`."""

from typing import Optional, Sequence, Union

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest as tf_nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _weighted_flat_map(  # pylint: disable=unused-private-name
    input_datasets: Sequence[dataset_ops.DatasetV2],
    weights: Optional[Sequence[Union[float, tensor.Tensor]]] = None,
    name: Optional[str] = None) -> dataset_ops.DatasetV2:
  """A `Dataset` that fetches elements from `input_datasets` and flattens them.

  This operation combines elements from multiple datasets into a flattened
  dataset. Elements are read in proportion to the `weights` assigned to each
  input dataset. All requested elements from a dataset are read before reading
  the elements from the next dataset.

  For example, suppose we have 2 datasets:

  # TODO(wilsin): Make the following code testable after the API is released.
  dataset1 = tf.data.Dataset.range(0, 10)
  dataset2 = tf.data.Dataset.range(10, 20),

  Suppose that we call `weighted_flat_map` from these 2 datasets with the
  following weights:

  dataset = tf.data.Dataset.weighted_flat_map([dataset1, dataset2], [0.5, 1.0])

  Then, the outcome of the elements is:
  # [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

  Args:
    input_datasets: A non-empty list of `tf.data.Dataset` objects with
      compatible structure.
    weights: (Optional.) A list or Tensor of `len(datasets)` non-zero
      floating-point values where `weights[i]` represents the probability to
      sample from `datasets[i]`, or a `tf.data.Dataset` object where each
      element is such a list. Defaults to a uniform distribution across
      `datasets`.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A dataset that reads elements from all its inputs, reading the requested
    elements from an input according to the weight before proceeding to the next
    input. The number of elements read from an input is in proportion to its
    weight given in `weights`.

  Raises:
    TypeError: if the `datasets` or `weights` arguments have the wrong type.
    ValueError:
      - if `input_datasets` has less than 2 datasets.
      - if `weights` is specified and does not match the length of
        `input_datasets`.
    InvalidArgumentError:
      - if any of the `input_datasets` has an unknown or infinite cardinality.
      - if any of the `weights` has a value that is less than or equal to 0.0
  """
  return _WeightedFlatMap(input_datasets, weights, name=name)


class _WeightedFlatMap(dataset_ops.DatasetV2):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(
      self,
      input_datasets: Sequence[dataset_ops.DatasetV2],
      weights: Optional[Sequence[Union[float, tensor.Tensor]]] = None,
      name: Optional[str] = None):
    if not input_datasets:
      raise ValueError("Invalid `datasets`. `datasets` should not be empty.")
    if len(input_datasets) < 2:
      raise ValueError(
          "Invalid `datasets`. `datasets` should have at least two datasets.")
    self._input_datasets = input_datasets
    self._name = name

    def common_supertype(a, b):
      result = a.most_specific_common_supertype([b])
      if result is None:
        raise TypeError(f"No common supertype of {a} and {b}.")
      return result

    self._structure = input_datasets[0].element_spec
    for dataset in input_datasets[1:]:
      try:
        self._structure = tf_nest.map_structure(
            common_supertype, self._structure, dataset.element_spec
        )
      except (TypeError, ValueError) as e:
        raise TypeError(
            "Incompatible dataset elements:\n"
            f"  {input_datasets[0].element_spec} vs. "
            f"  {dataset.element_spec}"
        ) from e

    if weights is None:
      weights = [1.0] * len(input_datasets)
    else:
      if isinstance(weights, tensor.Tensor):
        if not weights.shape.is_compatible_with([len(input_datasets)]):
          raise ValueError(
              "Invalid `weights`. The shape of `weights` "
              "should be compatible with `[len(datasets)]` "
              f"but is {weights.shape}."
          )
      else:
        if len(input_datasets) != len(weights):
          raise ValueError(
              "Invalid `weights`. `weights` should have the "
              "same length as `datasets` but got "
              f"`len(weights)={len(weights)}` vs. "
              f"`len(datasets)={len(input_datasets)}`."
          )
      weights = [
          ops.convert_to_tensor(w, preferred_dtype=dtypes.float64)
          for w in weights
      ]
      for weight in weights:
        if weight.dtype not in (dtypes.float32, dtypes.float64):
          raise TypeError(
              "Invalid `weights`. `weights` type must be either "
              "`tf.float32` or `tf.float64` but is "
              f"{weight.dtype}."
          )

    # pylint: disable=protected-access
    variant_tensor = ged_ops.weighted_flat_map_dataset(
        [dataset._variant_tensor for dataset in self._input_datasets],
        weights,
        **self._common_args,
    )
    super().__init__(variant_tensor)

  def _inputs(self):
    return self._input_datasets

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.weighted_flat_map()"
