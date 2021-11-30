# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""API for using the tf.data service.

This module contains:

1. tf.data server implementations for running the tf.data service.
2. APIs for registering datasets with the tf.data service and reading from
   the registered datasets.

The tf.data service provides the following benefits:

- Horizontal scaling of tf.data input pipeline processing to solve input
  bottlenecks.
- Data coordination for distributed training. Coordinated reads
  enable all replicas to train on similar-length examples across each global
  training step, improving step times in synchronous training.
- Dynamic balancing of data across training replicas.

>>> dispatcher = tf.data.experimental.service.DispatchServer()
>>> dispatcher_address = dispatcher.target.split("://")[1]
>>> worker = tf.data.experimental.service.WorkerServer(
...     tf.data.experimental.service.WorkerConfig(
...         dispatcher_address=dispatcher_address))
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.apply(tf.data.experimental.service.distribute(
...     processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
...     service=dispatcher.target))
>>> print(list(dataset.as_numpy_iterator()))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## Setup

This section goes over how to set up the tf.data service.

### Run tf.data servers

The tf.data service consists of one dispatch server and `n` worker servers.
tf.data servers should be brought up alongside your training jobs, then brought
down when the jobs are finished.
Use `tf.data.experimental.service.DispatchServer` to start a dispatch server,
and `tf.data.experimental.service.WorkerServer` to start worker servers. Servers
can be run in the same process for testing purposes, or scaled up on separate
machines.

See https://github.com/tensorflow/ecosystem/tree/master/data_service for an
example of using Google Kubernetes Engine (GKE) to manage the tf.data service.
Note that the server implementation in
[tf_std_data_server.py](https://github.com/tensorflow/ecosystem/blob/master/data_service/tf_std_data_server.py)
is not GKE-specific, and can be used to run the tf.data service in other
contexts.

### Custom ops

If your dataset uses custom ops, these ops need to be made available to tf.data
servers by calling
[load_op_library](https://www.tensorflow.org/api_docs/python/tf/load_op_library)
from the dispatcher and worker processes at startup.

## Usage

Users interact with tf.data service by programmatically registering their
datasets with tf.data service, then creating datasets that read from the
registered datasets. The
[register_dataset](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service/register_dataset)
function registers a dataset, then the
[from_dataset_id](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service/from_dataset_id)
function creates a new dataset which reads from the registered dataset.
The
[distribute](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service/distribute)
function wraps `register_dataset` and `from_dataset_id` into a single convenient
transformation which registers its input dataset and then reads from it.
`distribute` enables tf.data service to be used with a one-line code change.
However, it assumes that the dataset is created and consumed by the same entity
and this assumption might not always be valid or desirable. In particular, in
certain scenarios, such as distributed training, it might be desirable to
decouple the creation and consumption of the dataset (via `register_dataset`
and `from_dataset_id` respectively) to avoid having to create the dataset on
each of the training workers.

### Example

#### `distribute`

To use the `distribute` transformation, apply the transformation after the
prefix of your input pipeline that you would like to be executed using tf.data
service (typically at the end).

```
dataset = ...  # Define your dataset here.
# Move dataset processing from the local machine to the tf.data service
dataset = dataset.apply(
    tf.data.experimental.service.distribute(
        processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
        service=FLAGS.tf_data_service_address,
        job_name="shared_job"))
# Any transformations added after `distribute` will be run on the local machine.
dataset = dataset.prefetch(1)
```

The above code will create a tf.data service "job", which iterates through the
dataset to generate data. To share the data from a job across multiple clients
(e.g. when using TPUStrategy or MultiWorkerMirroredStrategy), set a common
`job_name` across all clients.

#### `register_dataset` and `from_dataset_id`

`register_dataset` registers a dataset with the tf.data service, returning a
dataset id for the registered dataset. `from_dataset_id` creates a dataset that
reads from the registered dataset. These APIs can be used to reduce dataset
building time for distributed training. Instead of building the dataset on all
training workers, we can build the dataset just once and then register the
dataset using `register_dataset`. Then all workers can call `from_dataset_id`
without needing to build the dataset themselves.

```
dataset = ...  # Define your dataset here.
dataset_id = tf.data.experimental.service.register_dataset(
    service=FLAGS.tf_data_service_address,
    dataset=dataset)
# Use `from_dataset_id` to create per-worker datasets.
per_worker_datasets = {}
for worker in workers:
  per_worker_datasets[worker] = tf.data.experimental.service.from_dataset_id(
      processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
      service=FLAGS.tf_data_service_address,
      dataset_id=dataset_id,
      job_name="shared_job")
```

### Processing Modes

`processing_mode` specifies how to shard a dataset among tf.data service
workers. tf.data service supports `OFF`, `DYNAMIC`, `FILE`, `DATA`,
`FILE_OR_DATA`, `HINT` sharding policies.

OFF: No sharding will be performed. The entire input dataset will be processed
independently by each of the tf.data service workers. For this reason, it is
important to shuffle data (e.g. filenames) non-deterministically, so that each
worker will process the elements of the dataset in a different order. This mode
can be used to distribute datasets that aren't splittable.

If a worker is added or restarted during ShardingPolicy.OFF processing, the
worker will instantiate a new copy of the dataset and begin producing data from
the beginning.

#### Dynamic Sharding

DYNAMIC: In this mode, tf.data service divides the dataset into two components:
a source component that generates "splits" such as filenames, and a processing
component that takes splits and outputs dataset elements. The source component
is executed in a centralized fashion by the tf.data service dispatcher, which
generates different splits of input data. The processing component is executed
in a parallel fashion by the tf.data service workers, each operating on a
different set of input data splits.

For example, consider the following dataset:

```
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(TFRecordDataset)
dataset = dataset.map(preprocess_fn)
dataset = dataset.batch(batch_size)
dataset = dataset.apply(
    tf.data.experimental.service.distribute(
        processing_mode=tf.data.experimental.service.ShardingPolicy.DYNAMIC,
        ...))
```

The `from_tensor_slices` will be run on the dispatcher, while the `interleave`,
`map`, and `batch` will be run on tf.data service workers. The workers will pull
filenames from the dispatcher for processing. To process a dataset with
dynamic sharding, the dataset must have a splittable source, and all of
its transformations must be compatible with splitting. While most sources and
transformations support splitting, there are exceptions, such as custom datasets
which may not implement the splitting API. Please file a Github issue if you
would like to use distributed epoch processing for a currently unsupported
dataset source or transformation.

If no workers are restarted during training, dynamic sharding mode will visit
every example exactly once. If workers are restarted during training, the splits
they were processing will not be fully visited. The dispatcher maintains a
cursor through the dataset's splits. Assuming fault tolerance is enabled (See
"Fault Tolerance" below), the dispatcher will store cursor state in write-ahead
logs so that the cursor can be restored in case the dispatcher is restarted
mid-training. This provides an at-most-once visitation guarantee in the presence
of server restarts.

#### Static Sharding

The following are static sharding policies. The semantics are similar to
`tf.data.experimental.AutoShardPolicy`. These policies require:

  * The tf.data service cluster is configured with a fixed list of workers
    in DispatcherConfig.
  * Each client only reads from the local tf.data service worker.

If a worker is restarted while performing static sharding, the worker will
begin processing its shard again from the beginning.

FILE: Shards by input files (i.e. each worker will get a fixed set of files to
process). When this option is selected, make sure that there is at least as
many files as workers. If there are fewer input files than workers, a runtime
error will be raised.

DATA: Shards by elements produced by the dataset. Each worker will process the
whole dataset and discard the portion that is not for itself. Note that for
this mode to correctly partition the dataset elements, the dataset needs to
produce elements in a deterministic order.

FILE_OR_DATA: Attempts FILE-based sharding, falling back to DATA-based
sharding on failure.

HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
placeholder to replace with `shard(num_workers, worker_index)`.

For backwards compatibility, `processing_mode` may also be set to the strings
`"parallel_epochs"` or `"distributed_epoch"`, which are respectively equivalent
to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.

### Coordinated Data Read

By default, when multiple consumers read from the same job, they receive data on
a first-come first-served basis. In some use cases, it is advantageous to
coordinate the consumers. At each step, consumers read data from the same
worker.

For example, the tf.data service can be used to coordinate example sizes across
a cluster during synchronous training, so that during each step all replicas
train on similar-sized elements. To achieve this, define a dataset which
generates rounds of `num_consumers` consecutive similar-sized batches, then
enable coordinated reads by setting `consumer_index` and `num_consumers`.

NOTE: To keep consumers in sync, coordinated reads require that the dataset have
infinite cardinality. You can get this by adding `.repeat()` at the end of the
dataset definition.

### Jobs

A tf.data service "job" refers to the process of reading from a dataset managed
by the tf.data service, using one or more data consumers. Jobs are created when
iterating over datasets that read from tf.data service. The data produced by a
job is determined by (1) dataset associated with the job and (2) the job's
processing mode. For example, if a job is created for the dataset
`Dataset.range(5)`, and the processing mode is `ShardingPolicy.OFF`, each
tf.data worker will produce the elements `{0, 1, 2, 3, 4}` for the job,
resulting in the
job producing `5 * num_workers` elements. If the processing mode is
`ShardingPolicy.DYNAMIC`, the job will only produce `5` elements.

One or more consumers can consume data from a job. By default, jobs are
"anonymous", meaning that only the consumer which created the job can read from
it. To share the output of a job across multiple consumers, you can set a common
`job_name`.

### Fault Tolerance

By default, the tf.data dispatch server stores its state in-memory, making it a
single point of failure during training. To avoid this, pass
`fault_tolerant_mode=True` when creating your `DispatchServer`. Dispatcher
fault tolerance requires `work_dir` to be configured and accessible from the
dispatcher both before and after restart (e.g. a GCS path). With fault tolerant
mode enabled, the dispatcher will journal its state to the work directory so
that no state is lost when the dispatcher is restarted.

WorkerServers may be freely restarted, added, or removed during training. At
startup, workers will register with the dispatcher and begin processing all
outstanding jobs from the beginning.

### Usage with tf.distribute

tf.distribute is the TensorFlow API for distributed training. There are
several ways to use tf.data with tf.distribute:
`strategy.experimental_distribute_dataset`,
`strategy.distribute_datasets_from_function`, and (for PSStrategy)
`coordinator.create_per_worker_dataset`. The following sections give code
examples for each.

In general we recommend using
`tf.data.experimental.service.{register_dataset,from_dataset_id}` over
`tf.data.experimental.service.distribute` for two reasons:

- The dataset only needs to be constructed and optimized once, instead of once
  per worker. This can significantly reduce startup time, because the current
  `experimental_distribute_dataset` and `distribute_datasets_from_function`
  implementations create and optimize worker datasets sequentially.
- If a dataset depends on lookup tables or variables that are only present on
  one host, the dataset needs to be registered from that host. Typically this
  only happens when resources are placed on the chief or worker 0. Registering
  the dataset from the chief will avoid issues with depending on remote
  resources.

#### strategy.experimental_distribute_dataset

Nothing special is required when using
`strategy.experimental_distribute_dataset`, just apply `register_dataset` and
`from_dataset_id` as above, making sure to specify a `job_name` so that all
workers consume from the same tf.data service job.

```
dataset = ...  # Define your dataset here.
dataset_id = tf.data.experimental.service.register_dataset(
    service=FLAGS.tf_data_service_address,
    dataset=dataset)
dataset = tf.data.experimental.service.from_dataset_id(
    processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
    service=FLAGS.tf_data_service_address,
    dataset_id=dataset_id,
    job_name="shared_job")

dataset = strategy.experimental_distribute_dataset(dataset)
```

#### strategy.distribute_datasets_from_function

First, make sure the dataset produced by the `dataset_fn` does not depend on the
`input_context` for the training worker on which it is run. Instead of each
worker building its own (sharded) dataset, one worker should register an
unsharded dataset, and the remaining workers should consume data from that
dataset.

```
dataset = dataset_fn()
dataset_id = tf.data.experimental.service.register_dataset(
    service=FLAGS.tf_data_service_address,
    dataset=dataset)

def new_dataset_fn(input_context):
  del input_context
  return tf.data.experimental.service.from_dataset_id(
      processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
      service=FLAGS.tf_data_service_address,
      dataset_id=dataset_id,
      job_name="shared_job")

dataset = strategy.distribute_datasets_from_function(new_dataset_fn)
```

#### coordinator.create_per_worker_dataset

`create_per_worker_dataset` works the same as
`distribute_datasets_from_function`.

```
dataset = dataset_fn()
dataset_id = tf.data.experimental.service.register_dataset(
    service=FLAGS.tf_data_service_address,
    dataset=dataset)

def new_dataset_fn(input_context):
  del input_context
  return tf.data.experimental.service.from_dataset_id(
      processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
      service=FLAGS.tf_data_service_address,
      dataset_id=dataset_id,
      job_name="shared_job")

dataset = coordinator.create_per_worker_dataset(new_dataset_fn)
```

## Limitations

- Python-based data processing: Datasets which use Python-based data processing
  (e.g. `tf.py_function`, `tf.numpy_function`, or
  `tf.data.Dataset.from_generator`) are currently not supported.
- Non-Serializable Resources: Datasets may only depend on TF resources that
  support serialization. Serialization is currently supported for lookup
  tables and variables. If your dataset depends on a TF resource that cannot be
  serialized, please file a Github issue.
- Remote Resources: If a dataset depends on a resource, the dataset must be
  registered from the same process that created the resource (e.g. the "chief"
  job of ParameterServerStrategy).
"""

from tensorflow.python.data.experimental.ops.data_service_ops import distribute
from tensorflow.python.data.experimental.ops.data_service_ops import from_dataset_id
from tensorflow.python.data.experimental.ops.data_service_ops import register_dataset
from tensorflow.python.data.experimental.ops.data_service_ops import ShardingPolicy
from tensorflow.python.data.experimental.service.server_lib import DispatcherConfig
from tensorflow.python.data.experimental.service.server_lib import DispatchServer
from tensorflow.python.data.experimental.service.server_lib import WorkerConfig
from tensorflow.python.data.experimental.service.server_lib import WorkerServer
