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
2. A `distribute` dataset transformation that moves a dataset's preprocessing
   to happen in the tf.data service.

The tf.data service offers a way to improve training speed when the host
attached to a training device can't keep up with the data consumption of the
model. For example, suppose a host can generate 100 examples/second, but the
model can process 200 examples/second. Training speed could be doubled by using
the tf.data service to generate 200 examples/second.

## Before using the tf.data service

There are a few things to do before using the tf.data service to speed up
training.

### Understand processing_mode

The tf.data service uses a cluster of workers to prepare data for training your
model. The `processing_mode` argument to
`tf.data.experimental.service.distribute` describes how to leverage multiple
workers to process the input dataset. Currently, there are two processing modes
to choose from: "distributed_epoch" and "parallel_epochs".

"distributed_epoch" means that the dataset will be split across all tf.data
service workers. The dispatcher produces "splits" for the dataset and sends them
to workers for further processing. For example, if a dataset begins with a list
of filenames, the dispatcher will iterate through the filenames and send the
filenames to tf.data workers, which will perform the rest of the dataset
transformations on those files. "distributed_epoch" is useful when your model
needs to see each element of the dataset exactly once, or if it needs to see the
data in a generally-sequential order. "distributed_epoch" only works for
datasets with splittable sources, such as `Dataset.from_tensor_slices`,
`Dataset.list_files`, or `Dataset.range`.

"parallel_epochs" means that the entire input dataset will be processed
independently by each of the tf.data service workers. For this
reason, it is important to shuffle data (e.g. filenames) non-deterministically,
so that each worker will process the elements of the dataset in a different
order. "parallel_epochs" can be used to distribute datasets that aren't
splittable.

### Measure potential impact

Before using the tf.data service, it is useful to first measure the potential
performance improvement. To do this, add

```
dataset = dataset.take(1).cache().repeat()
```

at the end of your dataset, and see how it affects your model's step time.
`take(1).cache().repeat()` will cache the first element of your dataset and
produce it repeatedly. This should make the dataset very fast, so that the model
becomes the bottleneck and you can identify the ideal model speed. With enough
workers, the tf.data service should be able to achieve similar speed.

## Running the tf.data service

tf.data servers should be brought up alongside your training jobs, and brought
down when the jobs are finished. The tf.data service uses one `DispatchServer`
and any number of `WorkerServers`. See
https://github.com/tensorflow/ecosystem/tree/master/data_service for an example
of using Google Kubernetes Engine (GKE) to manage the tf.data service. The
server implementation in
[tf_std_data_server.py](https://github.com/tensorflow/ecosystem/blob/master/data_service/tf_std_data_server.py)
is not GKE-specific, and can be used to run the tf.data service in other
contexts.

### Fault tolerance

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

## Using the tf.data service from your training job

Once you have a tf.data service cluster running, take note of the dispatcher IP
address and port. To connect to the service, you will use a string in the format
"grpc://<dispatcher_address>:<dispatcher_port>".

```
# Create the dataset however you were before using the tf.data service.
dataset = your_dataset_factory()

service = "grpc://{}:{}".format(dispatcher_address, dispatcher_port)
# This will register the dataset with the tf.data service cluster so that
# tf.data workers can run the dataset to produce elements. The dataset returned
# from applying `distribute` will fetch elements produced by tf.data workers.
dataset = dataset.apply(tf.data.experimental.service.distribute(
    processing_mode="parallel_epochs", service=service))
```

Below is a toy example that you can run yourself.

>>> dispatcher = tf.data.experimental.service.DispatchServer()
>>> dispatcher_address = dispatcher.target.split("://")[1]
>>> worker = tf.data.experimental.service.WorkerServer(
...     tf.data.experimental.service.WorkerConfig(
...         dispatcher_address=dispatcher_address))
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.apply(tf.data.experimental.service.distribute(
...     processing_mode="parallel_epochs", service=dispatcher.target))
>>> print(list(dataset.as_numpy_iterator()))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

See the documentation of `tf.data.experimental.service.distribute` for more
details about using the `distribute` transformation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops.data_service_ops import distribute
from tensorflow.python.data.experimental.ops.data_service_ops import from_dataset_id
from tensorflow.python.data.experimental.ops.data_service_ops import register_dataset
from tensorflow.python.data.experimental.service.server_lib import DispatcherConfig
from tensorflow.python.data.experimental.service.server_lib import DispatchServer
from tensorflow.python.data.experimental.service.server_lib import WorkerConfig
from tensorflow.python.data.experimental.service.server_lib import WorkerServer
