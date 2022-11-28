# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.Dataset.interleave`."""

import warnings

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _interleave(  # pylint: disable=unused-private-name
    input_dataset,
    map_func,
    cycle_length=None,
    block_length=None,
    num_parallel_calls=None,
    deterministic=None,
    name=None):
  """See `Dataset.interleave()` for details."""
  if block_length is None:
    block_length = 1

  if cycle_length is None:
    cycle_length = dataset_ops.AUTOTUNE

  if num_parallel_calls is None or dataset_ops.DEBUG_MODE:
    if deterministic is not None and not dataset_ops.DEBUG_MODE:
      warnings.warn("The `deterministic` argument has no effect unless the "
                    "`num_parallel_calls` argument is specified.")
    return _InterleaveDataset(
        input_dataset, map_func, cycle_length, block_length, name=name)
  else:
    return _ParallelInterleaveDataset(
        input_dataset,
        map_func,
        cycle_length,
        block_length,
        num_parallel_calls,
        deterministic=deterministic,
        name=name)


class _InterleaveDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that interleaves the result of transformed inputs."""

  def __init__(self,
               input_dataset,
               map_func,
               cycle_length,
               block_length,
               name=None):
    """See `Dataset.interleave()` for details."""

    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
      raise TypeError(
          "The `map_func` argument must return a `Dataset` object. Got "
          f"{dataset_ops.get_type(self._map_func.output_structure)!r}.")
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._name = name
    variant_tensor = gen_dataset_ops.interleave_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        f=self._map_func.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.interleave()"


class _ParallelInterleaveDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and interleaves the result.
  """

  def __init__(self,
               input_dataset,
               map_func,
               cycle_length,
               block_length,
               num_parallel_calls,
               buffer_output_elements=dataset_ops.AUTOTUNE,
               prefetch_input_elements=dataset_ops.AUTOTUNE,
               deterministic=None,
               name=None):
    """See `Dataset.interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
      raise TypeError(
          "The `map_func` argument must return a `Dataset` object. Got "
          f"{dataset_ops.get_type(self._map_func.output_structure)!r}.")
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._buffer_output_elements = ops.convert_to_tensor(
        buffer_output_elements,
        dtype=dtypes.int64,
        name="buffer_output_elements")
    self._prefetch_input_elements = ops.convert_to_tensor(
        prefetch_input_elements,
        dtype=dtypes.int64,
        name="prefetch_input_elements")

    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    if deterministic is None:
      deterministic_string = "default"
    elif deterministic:
      deterministic_string = "true"
    else:
      deterministic_string = "false"

    self._name = name
    variant_tensor = gen_dataset_ops.parallel_interleave_dataset_v4(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        self._buffer_output_elements,
        self._prefetch_input_elements,
        self._num_parallel_calls,
        f=self._map_func.function,
        deterministic=deterministic_string,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.interleave()"
