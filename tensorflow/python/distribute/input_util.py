# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utils to create distributed datasets based on TF version."""

from tensorflow.python import tf2
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1


def get_distributed_dataset(dataset,
                            input_workers,
                            strategy,
                            num_replicas_in_sync=None,
                            input_context=None,
                            options=None,
                            build=True):
  """Returns a distributed dataset from the given tf.data.Dataset instance.

  This is a common function that is used by all strategies to return a
  distributed dataset. The distributed dataset instance returned is different
  depending on if we are in a TF 1 or TF 2 context. The distributed dataset
  instances returned differ from each other in the APIs supported by each of
  them.

  Args:
    dataset: a tf.data.Dataset instance.
    input_workers: an InputWorkers object which specifies devices on which
      iterators should be created.
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
      handle last partial batch.
    num_replicas_in_sync: Optional integer. If this is not None, the value is
      used to decide how to rebatch datasets into smaller batches so that the
      total batch size for each step (across all workers and replicas) adds up
      to `dataset`'s batch size.
    input_context: `InputContext` for sharding. Only pass this in for between
      graph multi-worker cases where there is only one `input_worker`. In these
      cases, we will shard based on the `input_pipeline_id` and
      `num_input_pipelines` in the `InputContext`.
    options: Default is None. `tf.distribute.InputOptions` used to control
      options on how this dataset is distributed.
    build: whether to build underlying datasets when a DistributedDataset is
      created. This is only useful for `ParameterServerStrategy` now.

  Returns:
    A distributed dataset instance.
  """
  if tf2.enabled():
    return input_lib.DistributedDataset(
        input_workers,
        strategy,
        dataset,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context,
        build=build,
        options=options)
  else:
    return input_lib_v1.DistributedDatasetV1(
        dataset,
        input_workers,
        strategy,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context,
        options=options)


def get_distributed_datasets_from_function(dataset_fn,
                                           input_workers,
                                           input_contexts,
                                           strategy,
                                           options=None,
                                           build=True):
  """Returns a distributed dataset from the given input function.

  This is a common function that is used by all strategies to return a
  distributed dataset. The distributed dataset instance returned is different
  depending on if we are in a TF 1 or TF 2 context. The distributed dataset
  instances returned differ from each other in the APIs supported by each of
  them.

  Args:
    dataset_fn: a function that returns a tf.data.Dataset instance.
    input_workers: an InputWorkers object which specifies devices on which
      iterators should be created.
    input_contexts: A list of `InputContext` instances to be passed to call(s)
      to `dataset_fn`. Length and order should match worker order in
      `worker_device_pairs`.
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
      handle last partial batch.
    options: Default is None. `tf.distribute.InputOptions` used to control
      options on how this dataset is distributed.
    build: whether to build underlying datasets when a
      `DistributedDatasetFromFunction` is created. This is only useful for
      `ParameterServerStrategy` now.

  Returns:
    A distributed dataset instance.

  Raises:
    ValueError: if `options.experimental_replication_mode` and
    `options.experimental_place_dataset_on_device` are not consistent
  """
  if (options is not None and options.experimental_replication_mode !=
      input_lib.InputReplicationMode.PER_REPLICA and
      options.experimental_place_dataset_on_device):
    raise ValueError(
        "When `experimental_place_dataset_on_device` is set for dataset "
        "placement, you must also specify `PER_REPLICA` for the "
        "replication mode")

  if (options is not None and options.experimental_replication_mode
      == input_lib.InputReplicationMode.PER_REPLICA and
      options.experimental_fetch_to_device and
      options.experimental_place_dataset_on_device):
    raise ValueError(
        "`experimental_place_dataset_on_device` can not be set to True "
        "when experimental_fetch_to_device is True and "
        "replication mode is set to `PER_REPLICA`")

  if tf2.enabled():
    return input_lib.DistributedDatasetsFromFunction(
        input_workers,
        strategy,
        input_contexts=input_contexts,
        dataset_fn=dataset_fn,
        options=options,
        build=build,
    )
  else:
    return input_lib_v1.DistributedDatasetsFromFunctionV1(
        input_workers, strategy, input_contexts, dataset_fn, options)
