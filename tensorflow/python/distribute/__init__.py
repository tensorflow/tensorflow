# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for running a computation across multiple devices.

The intent of this library is that you can write an algorithm in a stylized way
and it will be usable with a variety of different `tf.distribute.Strategy`
implementations. Each descendant will implement a different strategy for
distributing the algorithm across multiple devices/machines.  Furthermore, these
changes can be hidden inside the specific layers and other library classes that
need special treatment to run in a distributed setting, so that most users'
model definition code can run unchanged. The `tf.distribute.Strategy` API works
the same way with eager and graph execution.

*Guides*

* [TensorFlow v2.x](https://www.tensorflow.org/guide/distributed_training)
* [TensorFlow
v1.x](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/distribute_strategy.ipynb)

*Tutorials*

* [Distributed Training
Tutorials](https://www.tensorflow.org/tutorials/distribute/)

  The tutorials cover how to use `tf.distribute.Strategy` to do distributed
  training with native Keras APIs, custom training loops,
  and Estimator APIs. They also cover how to save/load model when using
  `tf.distribute.Strategy`.

*Glossary*

* _Data parallelism_ is where we run multiple copies of the model
  on different slices of the input data. This is in contrast to
  _model parallelism_ where we divide up a single copy of a model
  across multiple devices.
  Note: we only support data parallelism for now, but
  hope to add support for model parallelism in the future.
* A _device_ is a CPU or accelerator (e.g. GPUs, TPUs) on some machine that
  TensorFlow can run operations on (see e.g. `tf.device`). You may have multiple
  devices on a single machine, or be connected to devices on multiple
  machines. Devices used to run computations are called _worker devices_.
  Devices used to store variables are _parameter devices_. For some strategies,
  such as `tf.distribute.MirroredStrategy`, the worker and parameter devices
  will be the same (see mirrored variables below). For others they will be
  different. For example, `tf.distribute.experimental.CentralStorageStrategy`
  puts the variables on a single device (which may be a worker device or may be
  the CPU), and `tf.distribute.experimental.ParameterServerStrategy` puts the
  variables on separate machines called _parameter servers_ (see below).
* A _replica_ is one copy of the model, running on one slice of the
  input data. Right now each replica is executed on its own
  worker device, but once we add support for model parallelism
  a replica may span multiple worker devices.
* A _host_ is the CPU device on a machine with worker devices, typically
  used for running input pipelines.
* A _worker_ is defined to be the physical machine(s) containing the physical
  devices (e.g. GPUs, TPUs) on which the replicated computation is executed. A
  worker may contain one or more replicas, but contains at least one
  replica. Typically one worker will correspond to one machine, but in the case
  of very large models with model parallelism, one worker may span multiple
  machines. We typically run one input pipeline per worker, feeding all the
  replicas on that worker.
* _Synchronous_, or more commonly _sync_, training is where the updates from
  each replica are aggregated together before updating the model variables. This
  is in contrast to _asynchronous_, or _async_ training, where each replica
  updates the model variables independently. You may also have replicas
  partitioned into groups which are in sync within each group but async between
  groups.
* _Parameter servers_: These are machines that hold a single copy of
  parameters/variables, used by some strategies (right now just
  `tf.distribute.experimental.ParameterServerStrategy`). All replicas that want
  to operate on a variable retrieve it at the beginning of a step and send an
  update to be applied at the end of the step. These can in principle support
  either sync or async training, but right now we only have support for async
  training with parameter servers. Compare to
  `tf.distribute.experimental.CentralStorageStrategy`, which puts all variables
  on a single device on the same machine (and does sync training), and
  `tf.distribute.MirroredStrategy`, which mirrors variables to multiple devices
  (see below).

* _Replica context_ vs. _Cross-replica context_ vs _Update context_

  A _replica context_ applies
  when you execute the computation function that was called with `strategy.run`.
  Conceptually, you're in replica context when executing the computation
  function that is being replicated.

  An _update context_ is entered in a `tf.distribute.StrategyExtended.update`
  call.

  An _cross-replica context_ is entered when you enter a `strategy.scope`. This
  is useful for calling `tf.distribute.Strategy` methods which operate across
  the replicas (like `reduce_to()`). By default you start in a _replica context_
  (the "default single _replica context_") and then some methods can switch you
  back and forth.

* _Distributed value_: Distributed value is represented by the base class
  `tf.distribute.DistributedValues`. `tf.distribute.DistributedValues` is useful
  to represent values on multiple devices, and it contains a map from replica id
  to values. Two representative types of `tf.distribute.DistributedValues`
  are `tf.types.experimental.PerReplica` and `tf.types.experimental.Mirrored`
  values.

  `PerReplica` values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a distributed dataset
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.distribute_datasets_from_function`. They are also the
  typical result returned by `tf.distribute.Strategy.run`.

  `Mirrored` values are like `PerReplica` values, except we know that the value
  on all replicas are the same. `Mirrored` values are kept synchronized by the
  distribution strategy in use, while `PerReplica` values are left
  unsynchronized. `Mirrored` values typically represent model weights. We can
  safely read a `Mirrored` value in a cross-replica context by using the value
  on any replica, while PerReplica values can only be read within a replica
  context.

* _Unwrapping_ and _merging_: Consider calling a function `fn` on multiple
  replicas, like `strategy.run(fn, args=[w])` with an
  argument `w` that is a `tf.distribute.DistributedValues`. This means `w` will
  have a map taking replica id `0` to `w0`, replica id `1` to `w1`, etc.
  `strategy.run()` unwraps `w` before calling `fn`, so it calls `fn(w0)` on
  device `d0`, `fn(w1)` on device `d1`, etc.  It then merges the return
  values from `fn()`, which leads to one common object if the returned values
  are the same object from every replica, or a `DistributedValues` object
  otherwise.

* _Reductions_ and _all-reduce_: A _reduction_ is a method of aggregating
  multiple values into one value, like "sum" or "mean". If a strategy is doing
  sync training, we will perform a reduction on the gradients to a parameter
  from all replicas before applying the update. _All-reduce_ is an algorithm for
  performing a reduction on values from multiple devices and making the result
  available on all of those devices.

* _Mirrored variables_: These are variables that are created on multiple
  devices, where we keep the variables in sync by applying the same
  updates to every copy. Mirrored variables are created with
  `tf.Variable(...synchronization=tf.VariableSynchronization.ON_WRITE...)`.
  Normally they are only used in synchronous training.

* _SyncOnRead variables_

  _SyncOnRead variables_ are created by
  `tf.Variable(...synchronization=tf.VariableSynchronization.ON_READ...)`, and
  they are created on multiple devices. In replica context, each
  component variable on the local replica can perform reads and writes without
  synchronization with each other. When the
  _SyncOnRead variable_ is read in cross-replica context, the values from
  component variables are aggregated and returned.

  _SyncOnRead variables_ bring a lot of custom configuration difficulty to the
  underlying logic, so we do not encourage users to instantiate and use
  _SyncOnRead variable_ on their own. We have mainly used _SyncOnRead
  variables_ for use cases such as batch norm and metrics. For performance
  reasons, we often don't need to keep these statistics in sync every step and
  they can be accumulated on each replica independently. The only time we want
  to sync them is reporting or checkpointing, which typically happens in
  cross-replica context. _SyncOnRead variables_ are also often used by advanced
  users who want to control when variable values are aggregated. For example,
  users sometimes want to maintain gradients independently on each replica for a
  couple of steps without aggregation.

* _Distribute-aware layers_

  Layers are generally called in a replica context, except when defining a
  Keras functional model. `tf.distribute.in_cross_replica_context` will let you
  determine which case you are in. If in a replica context,
  the `tf.distribute.get_replica_context` function will return the default
  replica context outside a strategy scope, `None` within a strategy scope, and
  a `tf.distribute.ReplicaContext` object inside a strategy scope and within a
  `tf.distribute.Strategy.run` function. The `ReplicaContext` object has an
  `all_reduce` method for aggregating across all replicas.


Note that we provide a default version of `tf.distribute.Strategy` that is
used when no other strategy is in scope, that provides the same API with
reasonable default behavior.

API docstring: tensorflow.distribute
"""
