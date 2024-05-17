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
"""The implementation of `tf.data.Dataset.snapshot`."""

import multiprocessing

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _snapshot(input_dataset,  # pylint: disable=unused-private-name
              path,
              compression="AUTO",
              reader_func=None,
              shard_func=None,
              name=None):
  """See `Dataset.snapshot()` for details."""

  project_func = None
  if shard_func is None:
    input_dataset = input_dataset.enumerate(name=name)
    # This sets the amount of parallelism based on the number of CPU cores on
    # the machine where this Python code is executed, which may differ from
    # the number of CPU cores where the input pipeline graph is actually
    # executed (e.g. remote Cloud TPU workers).
    local_shard_func = lambda index, _: index % multiprocessing.cpu_count()
    project_func = lambda _, elem: elem
  else:
    local_shard_func = shard_func
  dataset = _SnapshotDataset(
      input_dataset=input_dataset,
      path=path,
      compression=compression,
      reader_func=reader_func,
      # This will not do the right thing where the graph is built on a
      # different machine than the executor (e.g. Cloud TPUs).
      shard_func=local_shard_func,
      name=name)
  if project_func is not None:
    dataset = dataset.map(project_func, name=name)
  return dataset


class _SnapshotDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A dataset that allows saving and re-use of already processed data."""

  def __init__(self,
               input_dataset,
               path,
               shard_func,
               compression=None,
               reader_func=None,
               pending_snapshot_expiry_seconds=None,
               use_legacy_function=False,
               name=None):

    if reader_func is None:
      reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
          lambda x: x,
          cycle_length=multiprocessing.cpu_count(),
          num_parallel_calls=dataset_ops.AUTOTUNE)

    self._input_dataset = input_dataset
    self._path = path
    self._compression = compression

    self._reader_func = structured_function.StructuredFunctionWrapper(
        reader_func,
        self._transformation_name() + ".reader_func",
        # Dataset of datasets of input elements
        input_structure=dataset_ops.DatasetSpec(
            dataset_ops.DatasetSpec(input_dataset.element_spec)),
        use_legacy_function=use_legacy_function)
    self._shard_func = structured_function.StructuredFunctionWrapper(
        shard_func,
        self._transformation_name() + ".shard_func",
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)

    if ((not self._shard_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int32))) and
        (not self._shard_func.output_structure.is_compatible_with(
            tensor_spec.TensorSpec([], dtypes.int64)))):
      raise TypeError(f"Invalid `shard_func`. `shard_func` must return "
                      f"`tf.int64` scalar tensor but its return type is "
                      f"{self._shard_func.output_structure}.")

    self._name = name
    variant_tensor = ged_ops.snapshot_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        path,
        self._reader_func.function.captured_inputs,
        self._shard_func.function.captured_inputs,
        compression=compression,
        reader_func=self._reader_func.function,
        shard_func=self._shard_func.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._reader_func, self._shard_func]

  def _transformation_name(self):
    return "Dataset.snapshot()"
