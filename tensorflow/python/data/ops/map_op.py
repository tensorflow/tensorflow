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
"""The implementation of `tf.data.Dataset.map`."""

import warnings

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _map_v2(
    input_dataset,  # pylint: disable=unused-private-name
    map_func,
    num_parallel_calls=None,
    deterministic=None,
    synchronous=None,
    use_unbounded_threadpool=None,
    name=None,
):
  """See `Dataset.map()` for details."""
  if num_parallel_calls is None or debug_mode.DEBUG_MODE:
    if deterministic is not None and not debug_mode.DEBUG_MODE:
      warnings.warn(
          "The `deterministic` argument has no effect unless the "
          "`num_parallel_calls` argument is specified."
      )
    return _MapDataset(
        input_dataset,
        map_func,
        preserve_cardinality=True,
        force_synchronous=False if synchronous is None else synchronous,
        name=name,
    )
  else:
    if synchronous:
      raise ValueError(
          "`synchronous` is not supported with `num_parallel_calls`, but"
          " `num_parallel_calls` was set to ",
          num_parallel_calls,
      )
    return _ParallelMapDataset(
        input_dataset,
        map_func,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
        preserve_cardinality=True,
        use_unbounded_threadpool=use_unbounded_threadpool,
        name=name)


def _map_v1(
    input_dataset,  # pylint: disable=unused-private-name
    map_func,
    num_parallel_calls=None,
    deterministic=None,
    synchronous=None,
    use_unbounded_threadpool=None,  # pylint: disable=unused-argument
):
  """See `Dataset.map()` for details."""
  if num_parallel_calls is None or debug_mode.DEBUG_MODE:
    return dataset_ops.DatasetV1Adapter(
        _MapDataset(
            input_dataset,
            map_func,
            preserve_cardinality=False,
            force_synchronous=False if synchronous is None else synchronous,
        )
    )
  else:
    if synchronous:
      raise ValueError(
          "`synchronous` is not supported with `num_parallel_calls`, but"
          " `num_parallel_calls` was set to ",
          num_parallel_calls,
      )
    return dataset_ops.DatasetV1Adapter(
        _ParallelMapDataset(
            input_dataset,
            map_func,
            num_parallel_calls,
            deterministic,
            preserve_cardinality=False,
            use_unbounded_threadpool=False))


def _map_v1_with_legacy_function(  # pylint: disable=unused-private-name
    input_dataset,
    map_func,
    num_parallel_calls=None,
    deterministic=None,
    synchronous=False,
):
  """See `Dataset.map()` for details."""
  if num_parallel_calls is None:
    if deterministic is not None:
      warnings.warn("The `deterministic` argument has no effect unless the "
                    "`num_parallel_calls` argument is specified.")
    return dataset_ops.DatasetV1Adapter(
        _MapDataset(
            input_dataset,
            map_func,
            force_synchronous=synchronous,
            preserve_cardinality=False,
            use_legacy_function=True,
        )
    )
  else:
    if synchronous:
      raise ValueError(
          "`synchronous` is not supported with `num_parallel_calls`, but"
          " `num_parallel_calls` was set to ",
          num_parallel_calls,
      )
    return dataset_ops.DatasetV1Adapter(
        _ParallelMapDataset(
            input_dataset,
            map_func,
            num_parallel_calls,
            deterministic,
            preserve_cardinality=False,
            use_legacy_function=True,
            use_unbounded_threadpool=False))


class _MapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(
      self,
      input_dataset,
      map_func,
      force_synchronous=False,
      use_inter_op_parallelism=True,
      preserve_cardinality=True,
      use_legacy_function=False,
      name=None,
  ):
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._preserve_cardinality = preserve_cardinality
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    self._force_synchronous = force_synchronous
    self._name = name
    variant_tensor = gen_dataset_ops.map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        force_synchronous=self._force_synchronous,
        **self._common_args
    )
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"


class _ParallelMapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over elements in its input in parallel."""

  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               deterministic,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False,
               use_unbounded_threadpool=False,
               name=None):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"
    self._preserve_cardinality = preserve_cardinality
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    self._use_unbounded_threadpool = use_unbounded_threadpool
    self._name = name
    variant_tensor = gen_dataset_ops.parallel_map_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        num_parallel_calls=self._num_parallel_calls,
        deterministic=self._deterministic,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        use_unbounded_threadpool=self._use_unbounded_threadpool,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"
