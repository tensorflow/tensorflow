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
"""Python API for `index_flat_map` dataset, which supports global shuffling."""

from typing import Any, Callable, Optional, Sequence, Union

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

_IndexType = Union[int, tensor.Tensor]


def index_flat_map(  # pylint: disable=unused-private-name
    input_dataset: dataset_ops.Dataset,
    map_func: Callable[[Any], tensor.Tensor],
    index_map_func: Callable[[_IndexType], tuple[_IndexType, _IndexType]],
    *,
    output_cardinality: _IndexType = dataset_ops.UNKNOWN,
    name: Optional[str] = None) -> dataset_ops.Dataset:
  """A variant of flat_map that supports global shuffling.

  In addition to a `map_func`, the user needs to provide an `index_map_fn`.
  Given an index in the flattened dataset, the `index_map_fn` returns an
  (element_index, offset) tuple that represents the index of the element in the
  unflattened dataset, and the offset in the unflattened element.

  For example, users could flatten a dataset as following:

  def _split(element: str) -> tensor.Tensor:
    return tf.strings.split(element, " ")

  def _index_map_func(flattened_index: int) -> tuple[int, int]:
    '''Returns an (element_index, offset) tuple for the requested element index.

    For a `flattened_index` that represents an element index in the output
    dataset, this function returns a tuple that represents the index of the
    element in the unflattened dataset, and the offset in that element.
    See the example below on how to implement this function.
    '''

  dataset = tf.data.Dataset.from_tensor_slices([["0 1", "2 3 4 5", "6 7", "8"])
  dataset = index_flat_map_op.index_flat_map(dataset, _split, _index_map_func)
  for x in dataset:
    print(x)  # Produces "0", "1", "2", "3", "4", "5", "6", "7", "8".

  Given an input of 5, the `index_map_func` should return (1, 3), which means
  that the target element in the result dataset is in the 2nd element in the
  original dataset ("2 3 4 5") with an offset of 3, which is "5".

  Args:
    input_dataset: The input dataset.
    map_func: A function mapping a dataset element to a Tensor of the mapped
      elements.
    index_map_func: Given an index in the flattened dataset, returns a
      (element index, offset) tuple that represents the index of the element in
      the unflattened dataset, and the offset in the unflattened element.
    output_cardinality: Cardinality of the output dataset. Can be an int or a
      Tensor of int64. Required if the dataset is globally shuffled because the
      cardinality cannot be inferred as `map_func` produces a varied number of
      output elements from each input element.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A new `Dataset` with the transformation applied as described above.

  Raises:
    errors.InvalidArgumentError: If `index_map_func` does not return a tuple of
      two integers, or the returned offset is out of range.
  """
  return _IndexFlatMapDataset(
      input_dataset, map_func, index_map_func, output_cardinality, name)


class _IndexFlatMapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(
      self,
      input_dataset: dataset_ops.Dataset,
      map_func: Callable[[Any], tensor.Tensor],
      index_map_func: Callable[[_IndexType], tuple[_IndexType, _IndexType]],
      output_cardinality: _IndexType = dataset_ops.UNKNOWN,
      name: str = None):

    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        transformation_name=f"{self._transformation_name()}.map_func",
        dataset=input_dataset)
    self._index_map_func = structured_function.StructuredFunctionWrapper(
        index_map_func,
        transformation_name=f"{self._transformation_name()}.index_map_func",
        input_structure=tensor_spec.TensorSpec([], dtypes.int64))
    self._output_cardinality = ops.convert_to_tensor(
        output_cardinality, dtype=dtypes.int64)
    self._name = name
    variant_tensor = ged_ops.index_flat_map_dataset(
        input_dataset._variant_tensor,
        self._map_func.function.captured_inputs,
        self._index_map_func.function.captured_inputs,
        map_func=self._map_func.function,
        index_map_func=self._index_map_func.function,
        output_cardinality=self._output_cardinality,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self) -> Any:
    output_structure = self._map_func.output_structure
    # If the `map_func` returns a Python list of lists, then `output_structure`
    # contains s sequence of `TensorSpecs`, each representing the structure of
    # the inner lists.
    if isinstance(output_structure, Sequence):
      return output_structure[0]
    # If the `map_func` returns a Tensor of nested lists, then each mapped
    # element is one stacked Tensor.
    if output_structure.shape.rank > 1:
      return tensor_spec.TensorSpec(
          shape=output_structure.shape[1:], dtype=output_structure.dtype)
    # `map_func` returns a list of scalars.
    return tensor_spec.TensorSpec(shape=[], dtype=output_structure.dtype)

  def _functions(self) -> list[structured_function.StructuredFunctionWrapper]:
    return [self._map_func, self._index_map_func]

  def _transformation_name(self) -> str:
    return "Dataset.index_flat_map()"
