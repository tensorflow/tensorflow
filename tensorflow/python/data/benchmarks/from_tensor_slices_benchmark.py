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
"""Benchmarks for `tf.data.Dataset.from_tensor_slices()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_dataset_ops


class SingleThreadedFlatMapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.flat_map()` for details."""
    self._input_dataset = input_dataset
    self._map_func = dataset_ops.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        defun_kwargs={"_executor": "SINGLE_THREADED_EXECUTOR"})
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._flat_structure)
    super(SingleThreadedFlatMapDataset, self).__init__(input_dataset,
                                                       variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "SingleThreadedFlatMapDataset"


# TODO(b/119837791): Add eager benchmarks.
class FromTensorSlicesBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.from_tensor_slices()`."""

  def benchmark_slice_repeat_batch(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100
    num_elements = input_size * num_epochs // batch_size

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data).repeat(
            num_epochs).batch(batch_size))

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        name="slice_repeat_batch_input_%d_batch_%d" % (input_size, batch_size))

  def benchmark_reshape_slice_repeat(self):
    input_size = 10000
    reshape_dim = [100, 100]
    num_epochs = 100

    num_elements = num_epochs * reshape_dim[0]

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(
            input_data.reshape(*reshape_dim)).repeat(num_epochs))

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        name="reshape_slice_repeat_input_%d" % input_size,
    )

  def benchmark_slice_repeat_sparse(self):
    non_zeros_per_row_values = [0, 1, 5, 10, 100]
    num_rows_values = [32, 64, 128, 1024]

    for non_zeros_per_row in non_zeros_per_row_values:
      tensor = sparse_tensor.SparseTensor(
          indices=np.arange(non_zeros_per_row, dtype=np.int64)[:, np.newaxis],
          values=np.arange(non_zeros_per_row, dtype=np.int64),
          dense_shape=[1000])

      for num_rows in num_rows_values:

        # TODO(b/147153744): Function-valued attributes with their own
        # attributes are currently only supported in graph mode.
        @def_function.function
        def make_dataset():
          batched = dataset_ops.Dataset.from_tensors(
              tensor).repeat(num_rows).batch(num_rows)  # pylint: disable=cell-var-from-loop
          batched_tensor = get_single_element.get_single_element(batched)

          dataset = dataset_ops.Dataset.from_tensors(batched_tensor).repeat()
          return SingleThreadedFlatMapDataset(
              dataset, dataset_ops.Dataset.from_tensor_slices)

        self.run_and_report_benchmark(
            make_dataset(),
            num_elements=100000,
            iters=5,
            name="slice_repeat_sparse_elements_per_row_%d_num_rows_%d" % (
                non_zeros_per_row, num_rows))

  def benchmark_slice_batch_cache_repeat(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100
    num_elements = input_size * num_epochs // batch_size

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data).batch(
            batch_size).cache().repeat(num_epochs))

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        name="slice_batch_cache_repeat_input_%d_batch_%d" % (input_size,
                                                             batch_size))


if __name__ == "__main__":
  benchmark_base.test.main()
