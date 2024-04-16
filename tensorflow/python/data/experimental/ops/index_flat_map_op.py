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

from typing import Any, Callable, Optional, Union

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

_IndexType = Union[int, tensor.Tensor]


def index_flat_map(  # pylint: disable=unused-private-name
    input_dataset: dataset_ops.Dataset,
    map_func: Callable[[Any], tensor.Tensor],
    index_map_func: Callable[[_IndexType], tuple[_IndexType, _IndexType]],
    name: Optional[str] = None) -> dataset_ops.Dataset:
  """A variant of flat_map that supports global shuffling.

  In addition to a `map_func`, the user needs to provide an `index_map_fn`.
  Given an index in the flattened dataset, the `index_map_fn` returns a
  (element index, offset) tuple that represents the index of the element in the
  unflattened dataset, and the offset in the unflattened element.

  For example, users could flatten a dataset as following:

  def _split(element: str) -> tensor.Tensor:
    return tf.strings.split(element, " ")

  def _index_map_func(flattened_index: int) -> tuple[int, int]:
    # See index_flat_map_test.py for an example of how to implement this.
    # ...

  dataset = tf.data.Dataset.from_tensor_slices([["0 1", "2 3 4 5", "6 7", "8"])
  dataset = index_flat_map_op.index_flat_map(dataset, _split, _index_map_func)
  for x in dataset:
    print(x)  # Produces "0", "1", "2", "3", "4", "5", "6", "7", "8".

  Given an input of 5, the `index_map_func` should return (1, 3), which means
  that the target element in the result dataset is in the 2nd element in the
  original dataset ("2 3 4 5") with an offset of 3, wchi is "5".

  Args:
    input_dataset: The input dataset.
    map_func: A function mapping a dataset element to a Tensor of the mapped
      elements.
    index_map_func: Given an index in the flattened dataset, returns a
      (element index, offset) tuple that represents the index of the element in
      the unflattened dataset, and the offset in the unflattened element.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A new `Dataset` with the transformation applied as described above.
  """
  return _IndexFlatMapDataset(input_dataset, map_func, index_map_func, name)


class _IndexFlatMapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(
      self,
      input_dataset: dataset_ops.Dataset,
      map_func: Callable[[Any], tensor.Tensor],
      index_map_func: Callable[[_IndexType], tuple[_IndexType, _IndexType]],
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
    self._name = name
    variant_tensor = ged_ops.index_flat_map_dataset(
        input_dataset._variant_tensor,
        self._map_func.function.captured_inputs,
        self._index_map_func.function.captured_inputs,
        map_func=self._map_func.function,
        index_map_func=self._index_map_func.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  # TODO(b/325112575): Make sure this works if `map_func` returns lists.
  @property
  def element_spec(self) -> Any:
    return tensor_spec.TensorSpec(
        shape=[],
        dtype=self._map_func.output_structure.dtype)

  def _functions(self) -> list[structured_function.StructuredFunctionWrapper]:
    return [self._map_func, self._index_map_func]

  def _transformation_name(self) -> str:
    return "Dataset.index_flat_map()"
