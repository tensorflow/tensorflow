# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Various classes representing distributed inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest


def get_distributed_dataset(dataset,
                            input_workers,
                            strategy,
                            split_batch_by=None,
                            input_context=None):
  """Returns a wrapped tf.data.DatasetV1 or tf.data.DatasetV2 instance.

  This is a common function that is used by all strategies to return the right
  tf.data.Dataset wrapped instance depending on the `dataset` argument type.

  Args:
    dataset: a tf.data.DatasetV1 or tf.data.DatasetV2 instance.
    input_workers: an InputWorkers object which specifies devices on which
        iterators should be created.
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    split_batch_by: Optional integer. If present, we "split" each batch of the
        dataset by `split_batch_by` value.
    input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.

  Returns:
    A wrapped tf.data.DatasetV1 or tf.data.DatasetV2 instance.
  """
  # We create a DistributedDataset if TF 2.x is enabled. This is to allow us to
  # expose a subset of APIs on the dataset and create a DistributedIterator vs
  # a DistributedIteratorV1.
  # In TF 2 we condition on being in eager/tf.function since the distributed
  # dataset and iterator we create is only supported in eager/tf.function.
  # TODO(b/143568310): Condition only on TF 2 vs TF 1 consistent with tf.data.
  if tf2.enabled() and ops.executing_eagerly_outside_functions():
    return DistributedDataset(
        dataset,
        input_workers,
        strategy,
        split_batch_by=split_batch_by,
        input_context=input_context)
  else:
    return DistributedDatasetV1(
        dataset,
        input_workers,
        strategy,
        split_batch_by=split_batch_by,
        input_context=input_context)


def get_distributed_datasets_from_function(dataset_fn,
                                           input_workers,
                                           input_contexts,
                                           strategy):
  """Returns a wrapped tf.data.DatasetV1 or tf.data.DatasetV2 instance.

  This is a common function that is used by all strategies to return the right
  tf.data.Dataset wrapped instance depending on if we are in graph or eager
  mode.

  Args:
    dataset_fn: a function that returns a tf.data.DatasetV1 or tf.data.DatasetV2
        instance.
    input_workers: an InputWorkers object which specifies devices on which
        iterators should be created.
    input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `dataset_fn`. Length and order should match worker order in
        `worker_device_pairs`.
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.

  Returns:
    A wrapped tf.data.DatasetV1 or tf.data.DatasetV2 instance.
  """
  # We create a DistributedDataset if TF 2.x is enabled. This is to allow us to
  # expose a subset of APIs on the dataset and create a DistributedIterator vs
  # a DistributedIteratorV1.
  # In TF 2 we condition on being in eager/tf.function since the distributed
  # dataset and iterator we create is only supported in eager/tf.function.
  # TODO(b/143568310): Condition only on TF 2 vs TF 1 consistent with tf.data.
  if tf2.enabled() and ops.executing_eagerly_outside_functions():
    return DistributedDatasetsFromFunction(
        dataset_fn,
        input_workers,
        input_contexts,
        strategy)
  else:
    return DistributedDatasetsFromFunctionV1(
        dataset_fn,
        input_workers,
        input_contexts,
        strategy)


class InputWorkers(object):
  """A 1-to-many mapping from input worker devices to compute devices."""

  def __init__(self, worker_device_pairs):
    """Initialize an `InputWorkers` object.

    Args:
      worker_device_pairs: A sequence of pairs:
        `(input device, a tuple of compute devices fed by that input device)`.
    """
    self._worker_device_pairs = worker_device_pairs
    self._input_worker_devices = tuple(d for d, _ in self._worker_device_pairs)
    self._fed_devices = tuple(tuple(device_util.canonicalize(d) for d in f)
                              for _, f in self._worker_device_pairs)

  @property
  def num_workers(self):
    return len(self._input_worker_devices)

  @property
  def worker_devices(self):
    return self._input_worker_devices

  def compute_devices_for_worker(self, worker_index):
    return self._fed_devices[worker_index]

  def __repr__(self):
    devices = self.worker_devices
    debug_repr = ",\n".join("  %d %s: %s" %
                            (i, devices[i], self._fed_devices[i])
                            for i in range(len(devices)))
    return "%s:{\n%s}" % (self.__class__.__name__, debug_repr)

  def serialize(self):
    return self._worker_device_pairs

  def deserialize(self, worker_device_pairs):
    return InputWorkers(worker_device_pairs)


def _get_next_as_optional(iterator, strategy, name=None):
  """Returns an empty dataset indicator and the next input from the iterator."""
  replicas = []
  worker_has_values = []
  worker_devices = []
  for i, worker in enumerate(iterator._input_workers.worker_devices):  # pylint: disable=protected-access
    if name is not None:
      d = tf_device.DeviceSpec.from_string(worker)
      new_name = "%s_%s_%d" % (name, d.job, d.task)
    else:
      new_name = None

    with ops.device(worker):
      worker_has_value, next_element = (
          iterator._iterators[i].get_next_as_list(new_name))  # pylint: disable=protected-access
      # Collective all-reduce requires explict devices for inputs.
      with ops.device("/cpu:0"):
        # Converting to integers for all-reduce.
        worker_has_value = math_ops.cast(worker_has_value, dtypes.int32)
        worker_devices.append(worker_has_value.device)
        worker_has_values.append(worker_has_value)
      # Make `replicas` a flat list of values across all replicas.
      replicas.append(next_element)

  # Run an all-reduce to see whether any worker has values.
  # TODO(b/131423105): we should be able to short-cut the all-reduce in some
  # cases.
  if getattr(strategy.extended, "_support_per_replica_values", True):
    # Slight hack: `reduce` expects a `PerReplica`, so we pass it one, even
    # though it doesn't actually have a value per replica.
    worker_has_values = values.PerReplica(worker_has_values)
    global_has_value = strategy.reduce(
        reduce_util.ReduceOp.SUM, worker_has_values, axis=None)
  else:
    assert len(worker_has_values) == 1
    global_has_value = worker_has_values[0]
  global_has_value = array_ops.reshape(
      math_ops.cast(global_has_value, dtypes.bool), [])
  return global_has_value, replicas


def _get_static_shape(iterators):
  """Returns a boolean indicating if the input is fully defined."""
  static_shape = True
  for iterator in iterators:
    if not isinstance(iterator, (_SingleWorkerOwnedDatasetIterator,
                                 _SingleWorkerDatasetIterator)):
      continue
    flattened_shapes = nest.flatten(iterator.output_shapes)
    for output_shape in flattened_shapes:
      if not output_shape.is_fully_defined():
        static_shape = False
        break
    return static_shape


class DistributedIteratorBase(object):
  """Common implementation for all input iterators."""

  def __init__(self, input_workers, iterators, strategy):
    static_shape = _get_static_shape(iterators)

    # TODO(b/133073708): we currently need a flag to control the usage because
    # there is a performance difference between get_next() and
    # get_next_as_optional(). And we only enable get_next_as_optional when the
    # output shapes are not static.
    #
    # TODO(yuefengz): Currently `experimental_enable_get_next_as_optional` is
    # always set to False in CollectiveAllReduceStrategy. We want to have a way
    # to distinguish multi workers/single worker between graph, so we can enable
    # the behavior in single worker case.
    #
    # TODO(rxsang): We want to always enable the get_next_as_optional behavior
    # when user passed input_fn instead of dataset.
    if getattr(
        strategy.extended, "experimental_enable_get_next_as_optional", False):
      self._enable_get_next_as_optional = not static_shape
    else:
      self._enable_get_next_as_optional = False

    assert isinstance(input_workers, InputWorkers)
    if not input_workers.worker_devices:
      raise ValueError("Should have at least one worker for input iterator.")

    self._iterators = iterators
    self._input_workers = input_workers
    self._strategy = strategy

  def next(self):
    return self.__next__()

  def __next__(self):
    try:
      return self.get_next()
    except errors.OutOfRangeError:
      raise StopIteration

  def __iter__(self):
    return self

  def get_next(self, name=None):
    """Returns the next input from the iterator for all replicas."""
    if not self._enable_get_next_as_optional:
      replicas = []
      for i, worker in enumerate(self._input_workers.worker_devices):
        if name is not None:
          d = tf_device.DeviceSpec.from_string(worker)
          new_name = "%s_%s_%d" % (name, d.job, d.task)
        else:
          new_name = None
        with ops.device(worker):
          # Make `replicas` a flat list of values across all replicas.
          replicas.extend(
              self._iterators[i].get_next_as_list_static_shapes(new_name))
      return values.regroup(replicas)

    out_of_range_replicas = []
    def out_of_range_fn(worker_index, device):
      """This function will throw an OutOfRange error."""
      # As this will be only called when there is no data left, so calling
      # get_next() will trigger an OutOfRange error.
      data = self._iterators[worker_index].get_next(device)
      out_of_range_replicas.append(data)
      return data

    global_has_value, replicas = _get_next_as_optional(self, self._strategy)
    results = []
    for i, worker in enumerate(self._input_workers.worker_devices):
      with ops.device(worker):
        devices = self._input_workers.compute_devices_for_worker(i)
        for j, device in enumerate(devices):
          with ops.device(device):
            # pylint: disable=undefined-loop-variable
            # pylint: disable=cell-var-from-loop
            # It is fine for the lambda to capture variables from the loop as
            # the lambda is executed in the loop as well.
            result = control_flow_ops.cond(
                global_has_value,
                lambda: replicas[i][j],
                lambda: out_of_range_fn(i, device),
                strict=True,
            )
            # pylint: enable=cell-var-from-loop
            # pylint: enable=undefined-loop-variable
            results.append(result)
    replicas = results

    # Some dimensions in `replicas` will become unknown after we conditionally
    # return the real tensors or the dummy tensors. We fix the input shapes by
    # using the shapes from `out_of_range_replicas` because it is calling
    # get_next() inside.
    flattened_replicas = nest.flatten(replicas)
    for i, replica_data in enumerate(nest.flatten(out_of_range_replicas)):
      for target, source in zip(
          nest.flatten(flattened_replicas[i], expand_composites=True),
          nest.flatten(replica_data, expand_composites=True)):
        target.set_shape(source.get_shape())
      # `SparseTensor` shape is not determined by the shape of its component
      # tensors. Rather, its shape depends on a tensor's values.
      if sparse_tensor.is_sparse(replica_data) and replica_data.get_shape():
        dense_shape = replica_data.get_shape()
        with ops.device(flattened_replicas[i].op.device):
          # For partially defined shapes, fill in missing values from tensor.
          if not dense_shape.is_fully_defined():
            dense_shape = array_ops.stack([
                flattened_replicas[i].dense_shape[j] if dim is None else dim
                for j, dim in enumerate(dense_shape.as_list())
            ])
          flattened_replicas[i] = sparse_tensor.SparseTensor(
              indices=flattened_replicas[i].indices,
              values=flattened_replicas[i].values,
              dense_shape=dense_shape)
    replicas = nest.pack_sequence_as(replicas, flattened_replicas)

    return values.regroup(replicas)


class DistributedIteratorV1(DistributedIteratorBase):
  """Input Iterator for tf.data.DatasetV1."""

  # We need a private initializer method for re-initializing multidevice
  # iterators when used with Keras training loops. If we don't reinitialize the
  # iterator we run into memory leak issues (b/123315763).
  @property
  def _initializer(self):
    init_ops = []
    for it in self._iterators:
      init_ops.extend(it.initialize())
    return control_flow_ops.group(init_ops)

  # TODO(anjalisridhar): Move to using `initializer` instead to be consistent
  # with tf.data iterator APIs.
  def initialize(self):
    """Initialize underlying iterators.

    Returns:
      A list of any initializer ops that should be run.
    """
    return self._initializer

  @property
  def initializer(self):
    return self.initialize()

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_classes(self):
    return self._iterators[0].output_classes

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_shapes(self):
    return self._iterators[0].output_shapes

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_types(self):
    return self._iterators[0].output_types

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  def get_iterator(self, worker):
    for i, w in enumerate(self._input_workers.worker_devices):
      if worker == w:
        return self._iterators[i]
    return None

  @property
  def element_spec(self):
    """The type specification of an element of this iterator."""
    return self._element_spec


class DistributedIteratorSpec(type_spec.TypeSpec):
  """Type specification for `DistributedIterator`."""

  __slots__ = ["_input_workers", "_element_spec", "_strategy"]

  def __init__(self, input_workers, element_spec, strategy):
    # We don't want to allow deserialization of this class because we don't
    # serialize the strategy object. Currently the only places where
    # _deserialize is called is when we save/restore using SavedModels.
    if isinstance(input_workers, tuple):
      raise NotImplementedError("DistributedIteratorSpec does not have support "
                                "for deserialization.")
    else:
      self._input_workers = input_workers
      self._element_spec = element_spec
      self._strategy = strategy

  @property
  def value_type(self):
    return DistributedIterator

  def _serialize(self):
    # We cannot serialize the strategy object so we convert it to an id that we
    # can use for comparison.
    return (self._input_workers.serialize(),
            self._element_spec, id(self._strategy))

  def _deserialize(self):
    raise ValueError("Deserialization is currently unsupported for "
                     "DistributedIteratorSpec.")

  @staticmethod
  def _is_compatible(a, b):
    """Returns true if the given type serializations compatible."""
    if type(a) is not type(b):
      return False
    if isinstance(a, tuple):
      return (len(a) == len(b) and
              all(DistributedIteratorSpec._is_compatible(x, y) for (x, y) in
                  zip(a, b)))
    if isinstance(a, dict):
      return (len(a) == len(b) and sorted(a.keys()) == sorted(b.keys()) and all(
          DistributedIteratorSpec._is_compatible(a[k], b[k]) for k in a.keys()))
    if isinstance(a, (type_spec.TypeSpec, tensor_shape.TensorShape,
                      dtypes.DType)):
      return a.is_compatible_with(b)
    return a == b

  # Overriding this method so that we can merge and reconstruct the spec object
  def most_specific_compatible_type(self, other):
    """Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    """
    # pylint: disable=protected-access
    if type(self) is not type(other):
      raise ValueError("No TypeSpec is compatible with both %s and %s" %
                       (self, other))
    if not self._is_compatible(self._input_workers.serialize(),
                               other._input_workers.serialize()):
      raise ValueError("_input_workers is not compatible with both %s "
                       "and %s" % (self, other))
    if self._element_spec != other._element_spec:
      raise ValueError("_element_spec is not compatible with both %s "
                       "and %s" % (self, other))
    if id(self._strategy) != id(other._strategy):
      raise ValueError("tf.distribute strategy is not compatible with both %s "
                       "and %s" % (self, other))
    return DistributedIteratorSpec(self._input_workers, self._element_spec,
                                   self._strategy)

  @property
  def _component_specs(self):
    specs = []
    worker_device_pairs = self._input_workers._worker_device_pairs  # pylint: disable=protected-access
    for i in range(len(worker_device_pairs)):
      input_device, compute_devices = worker_device_pairs[i]
      specs.append(_SingleWorkerDatasetIteratorSpec(input_device,
                                                    compute_devices,
                                                    element_spec=
                                                    self._element_spec))
    return specs

  def _to_components(self, value):
    return value._iterators  # pylint: disable=protected-access

  def _from_components(self, components):
    return DistributedIterator(input_workers=self._input_workers,
                               iterators=None,
                               components=components,
                               element_spec=self._element_spec,
                               strategy=self._strategy)

  @staticmethod
  def from_value(value):
    # pylint: disable=protected-access
    return DistributedIteratorSpec(value._input_workers, value._element_spec,
                                   value._strategy)


class DistributedIterator(DistributedIteratorBase,
                          composite_tensor.CompositeTensor):
  """Input Iterator for tf.data.DatasetV2."""

  def __init__(self, input_workers=None, iterators=None, strategy=None,
               components=None, element_spec=None):
    if input_workers is None:
      raise ValueError("`input_workers` should be "
                       "provided.")

    error_message = ("Either `input_workers` or "
                     "both `components` and `element_spec` need to be "
                     "provided.")

    if iterators is None:
      if (components is None or element_spec is None):
        raise ValueError(error_message)
      self._element_spec = element_spec
      self._input_workers = input_workers
      self._iterators = components
      static_shape = _get_static_shape(self._iterators)
      self._strategy = strategy
      if getattr(
          strategy.extended, "experimental_enable_get_next_as_optional", False):
        self._enable_get_next_as_optional = not static_shape
      else:
        self._enable_get_next_as_optional = False
    else:
      if (components is not None and element_spec is not None):
        raise ValueError(error_message)

      super(DistributedIterator, self).__init__(input_workers, iterators,
                                                strategy)

  @property
  def element_spec(self):
    return self._element_spec

  @property
  def _type_spec(self):
    return DistributedIteratorSpec(self._input_workers,
                                   self.element_spec,
                                   self._strategy)


class _IterableInput(object):
  """Base class for iterable inputs for distribution strategies."""

  def __init__(self, input_workers):
    assert isinstance(input_workers, InputWorkers)
    self._input_workers = input_workers

  def __iter__(self):
    raise NotImplementedError("must be implemented in descendants")

  def reduce(self, initial_state, reduce_fn):
    """Execute a `reduce_fn` over all the elements of the input."""
    iterator = iter(self)
    has_data, data = _get_next_as_optional(iterator, self._strategy)

    def cond(has_data, data, state):
      del data, state  # Unused.
      return has_data

    def loop_body(has_data, data, state):
      """Executes `reduce_fn` in a loop till the dataset is empty."""
      del has_data  # Unused.
      # data is list of lists here. where each list corresponds to one worker.
      # TODO(b/130570614): Add support for the multiworker and TPU pods use
      # case.
      if self._input_workers.num_workers == 1:
        data = data[0]
      else:
        raise ValueError("Dataset iteration within a tf.function is"
                         " not supported for multiple workers.")
      state = reduce_fn(state, values.regroup(data))
      has_data, data = _get_next_as_optional(iterator, self._strategy)
      return has_data, data, state

    has_data, data, final_state = control_flow_ops.while_loop(
        cond, loop_body, [has_data, data, initial_state], parallel_iterations=1)
    return final_state


class DistributedDataset(_IterableInput):
  """Wrapped tf.data.DatasetV2 that supports prefetching to multiple devices."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               split_batch_by=None,
               input_context=None):
    """Distribute the dataset on all workers.

    If `split_batch_by` is not None, we "split" each batch of the dataset by
    `split_batch_by` value.

    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      split_batch_by: Optional integer. If present, we "split" each batch of the
        dataset by `split_batch_by` value.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
    """
    super(DistributedDataset, self).__init__(input_workers=input_workers)
    # We clone and shard the dataset on each worker. The current setup tries to
    # shard the dataset by files if possible so that each worker sees a
    # different subset of files. If that is not possible, will attempt to shard
    # the final input such that each worker will run the entire preprocessing
    # pipeline and only receive its own shard of the dataset.
    if split_batch_by:
      try:
        # pylint: disable=protected-access
        with ops.colocate_with(dataset._variant_tensor):
          dataset = distribute._RebatchDataset(dataset, split_batch_by)
          # Add a prefetch to pipeline rebatching for performance.
          # TODO(rachelim): Instead of inserting an extra prefetch stage here,
          # leverage static graph rewrites to insert _RebatchDataset before
          # the final `prefetch` if it exists.
          dataset = dataset.prefetch(split_batch_by)
      except errors.InvalidArgumentError as e:
        if "without encountering a batch" in str(e):
          six.reraise(
              ValueError,
              ValueError(
                  "Call the `batch` method on the input Dataset in order to be "
                  "able to split your input across {} replicas.\n Please "
                  "the tf.distribute.Strategy guide. {}".format(
                      split_batch_by, e)),
              sys.exc_info()[2])
        else:
          raise

    # TODO(b/138745411): Remove once stateful transformations are supported.
    options = dataset_ops.Options()
    options.experimental_distribute._make_stateless = True  # pylint: disable=protected-access
    dataset = dataset.with_options(options)

    self._cloned_datasets = []
    if input_context:
      # Between-graph where we rely on the input_context for sharding
      assert input_workers.num_workers == 1
      dataset = input_ops.auto_shard_dataset(dataset,
                                             input_context.num_input_pipelines,
                                             input_context.input_pipeline_id)
      self._cloned_datasets.append(dataset)
    else:
      replicated_ds = distribute.replicate(dataset,
                                           input_workers.worker_devices)
      for i, worker in enumerate(input_workers.worker_devices):
        with ops.device(worker):
          cloned_dataset = replicated_ds[worker]
          cloned_dataset = cloned_dataset.with_options(dataset.options())
          cloned_dataset = input_ops.auto_shard_dataset(
              cloned_dataset, len(input_workers.worker_devices), i)
          self._cloned_datasets.append(cloned_dataset)

    self._input_workers = input_workers
    self._strategy = strategy
    self._element_spec = _create_distributed_tensor_spec(self._strategy,
                                                         dataset.element_spec)  # pylint: disable=protected-access

  def __iter__(self):
    if not (context.executing_eagerly() or
            ops.get_default_graph().building_function):
      raise RuntimeError("__iter__() is only supported inside of tf.function "
                         "or when eager execution is enabled.")

    worker_iterators = _create_iterators_per_worker(self._cloned_datasets,
                                                    self._input_workers)
    iterator = DistributedIterator(self._input_workers, worker_iterators,
                                   self._strategy)
    iterator._element_spec = self.element_spec  # pylint: disable=protected-access
    return iterator

  @property
  def element_spec(self):
    """The type specification of an element of this dataset."""
    return self._element_spec


class DistributedDatasetV1(DistributedDataset):
  """Wrapped tf.data.DatasetV1 that supports prefetching to multiple devices."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               split_batch_by=None,
               input_context=None):
    self._input_workers = input_workers
    super(DistributedDatasetV1, self).__init__(
        dataset,
        input_workers,
        strategy,
        split_batch_by=split_batch_by,
        input_context=input_context)

  def make_one_shot_iterator(self):
    """Get a one time use iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use `for ... in dataset:` to iterate
    over the dataset or `iter` to create an iterator.
    over the dataset or `iter` to create an iterator.

    Returns:
      A DistributedIteratorV1 instance.
    """
    return self._make_one_shot_iterator()

  def _make_one_shot_iterator(self):
    """Get an iterator for DistributedDatasetV1."""
    # Graph mode with one shot iterator is disabled because we have to call
    # `initialize` on the iterator which is only required if we are using a
    # tf.distribute strategy.
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    return self._get_iterator()

  def make_initializable_iterator(self):
    """Get an initializable iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use
    `tf.compat.v1.data.make_initializable_iterator(dataset)` to create an
    initializable iterator.

    Returns:
      A DistributedIteratorV1 instance.
    """
    return self._make_initializable_iterator()

  def _make_initializable_iterator(self, shared_name=None):  # pylint: disable=unused-argument
    """Get an initializable iterator for DistributedDatasetV1."""
    # Eager mode generates already initialized iterators. Hence we cannot create
    # an initializable iterator.
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `iter()` instead.")
    return self._get_iterator()

  def _get_iterator(self):
    worker_iterators = _create_iterators_per_worker(self._cloned_datasets,
                                                    self._input_workers,
                                                    graph_and_eager=True)
    iterator = DistributedIteratorV1(self._input_workers, worker_iterators,
                                     self._strategy)
    iterator._element_spec = self.element_spec  # pylint: disable=protected-access
    return iterator

  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")


# TODO(priyag): Add other replication modes.
class DistributedDatasetsFromFunction(_IterableInput):
  """Inputs created from dataset function."""

  def __init__(self, dataset_fn, input_workers, input_contexts, strategy):
    """Makes an iterable from datasets created by the given function.

    Args:
      dataset_fn: A function that returns a `Dataset` given an `InputContext`.
      input_workers: an `InputWorkers` object.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `dataset_fn`. Length and order should match worker order in
        `worker_device_pairs`.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    """
    super(DistributedDatasetsFromFunction, self).__init__(
        input_workers=input_workers)

    if input_workers.num_workers != len(input_contexts):
      raise ValueError(
          "Number of input workers (%d) is not same as number of "
          "input_contexts (%d)" %
          (input_workers.num_workers, len(input_contexts)))

    self._dataset_fn = dataset_fn
    self._input_workers = input_workers
    self._input_contexts = input_contexts
    self._strategy = strategy
    self._element_spec = None

    super(DistributedDatasetsFromFunction, self).__init__(
        input_workers=input_workers)

  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      iterators, element_spec = _create_iterators_per_worker_with_input_context(
          self._input_contexts, self._input_workers, self._dataset_fn)
      iterator = DistributedIterator(self._input_workers, iterators,
                                     self._strategy)
      self._element_spec = _create_distributed_tensor_spec(self._strategy,
                                                           element_spec)
      iterator._element_spec = self._element_spec  # pylint: disable=protected-access
      return iterator

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")

  @property
  def element_spec(self):
    """The type specification of an element of this dataset."""
    if self._element_spec is None:
      raise ValueError("You must create an iterator before calling "
                       "`element_spec` on the distributed dataset or iterator. "
                       "This is because the dataset function is not called "
                       "before an iterator is created.")

    return self._element_spec


class DistributedDatasetsFromFunctionV1(DistributedDatasetsFromFunction):
  """Inputs created from dataset function."""

  def _make_initializable_iterator(self, shared_name=None):
    """Get an initializable iterator for DistributedDatasetsFromFunctionV1."""
    del shared_name  # Unused
    # Eager mode generates already initialized iterators. Hence we cannot create
    # an initializable iterator.
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `iter()` instead.")
    return self._get_iterator()

  def _make_one_shot_iterator(self):
    """Get an iterator for iterating over DistributedDatasetsFromFunctionV1."""
    # Graph mode with one shot iterator is disabled because we have to call
    # `initialize` on the iterator which is only required if we are using a
    # tf.distribute strategy.
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    return self._get_iterator()

  def _get_iterator(self):
    iterators, element_spec = _create_iterators_per_worker_with_input_context(
        self._input_contexts, self._input_workers, self._dataset_fn)
    iterator = DistributedIteratorV1(self._input_workers, iterators,
                                     self._strategy)
    self._element_spec = _create_distributed_tensor_spec(self._strategy,
                                                         element_spec)
    iterator._element_spec = self._element_spec  # pylint: disable=protected-access
    return iterator

  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")


# TODO(anjalisridhar): This class will be soon be removed in favor of newer
# APIs.
class InputFunctionIterator(DistributedIteratorV1):
  """Iterator created from input function."""

  def __init__(self, input_fn, input_workers, input_contexts, strategy):
    """Make an iterator for input provided via an input function.

    Currently implements PER_WORKER mode, in which the `input_fn` is called
    once on each worker.

    TODO(priyag): Add other replication modes.

    Args:
      input_fn: Input function that returns a `tf.data.Dataset` object.
      input_workers: an `InputWorkers` object.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `input_fn`. Length and order should match worker order in
        `worker_device_pairs`.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    """
    assert isinstance(input_workers, InputWorkers)
    if input_workers.num_workers != len(input_contexts):
      raise ValueError(
          "Number of input workers (%d) is not same as number of "
          "input_contexts (%d)" %
          (input_workers.num_workers, len(input_contexts)))

    iterators = []
    for i, ctx in enumerate(input_contexts):
      worker = input_workers.worker_devices[i]
      with ops.device(worker):
        result = input_fn(ctx)
        devices = input_workers.compute_devices_for_worker(i)
        if isinstance(result, dataset_ops.DatasetV2):
          iterator = _SingleWorkerDatasetIterator(result, worker, devices)
        elif callable(result):
          iterator = _SingleWorkerCallableIterator(result, worker, devices)
        else:
          raise ValueError(
              "input_fn must return a tf.data.Dataset or a callable.")
        iterators.append(iterator)

    super(InputFunctionIterator, self).__init__(input_workers, iterators,
                                                strategy)


# TODO(anjalisridhar): This class will soon be removed and users should move
# to using DistributedIterator.
class DatasetIterator(DistributedIteratorV1):
  """Iterator created from input dataset."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               split_batch_by=None,
               input_context=None):
    """Make an iterator for the dataset on given devices.

    If `split_batch_by` is not None, we "split" each batch of the
    dataset by `split_batch_by` value.

    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      split_batch_by: Optional integer. If present, we "split" each batch of the
        dataset by `split_batch_by` value.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
    """
    dist_dataset = DistributedDatasetV1(
        dataset,
        input_workers,
        strategy,
        split_batch_by=split_batch_by,
        input_context=input_context)
    worker_iterators = _create_iterators_per_worker(
        dist_dataset._cloned_datasets, input_workers, graph_and_eager=True)  # pylint: disable=protected-access
    super(DatasetIterator, self).__init__(
        input_workers,
        worker_iterators,  # pylint: disable=protected-access
        strategy)
    self._element_spec = dist_dataset.element_spec


def _dummy_tensor_fn(value_structure):
  """A function to create dummy tensors from `value_structure`."""

  def create_dummy_tensor(spec):
    """Create a dummy tensor with possible batch dimensions set to 0."""
    if isinstance(spec, ragged_tensor.RaggedTensorSpec):
      # Splice out the ragged dimensions.
      # pylint: disable=protected-access
      feature_shape = spec._shape[:1].concatenate(
          spec._shape[(1 + spec._ragged_rank):])
      feature_type = spec._dtype
      # pylint: enable=protected-access
    else:
      feature_shape = spec.shape
      feature_type = spec.dtype
    # Ideally we should set the batch dimension to 0, however as in
    # DistributionStrategy we don't know the batch dimension, we try to
    # guess it as much as possible. If the feature has unknown dimensions, we
    # will set them to 0. If the feature shape is already static, we guess the
    # first dimension as batch dimension and set it to 0.
    dims = ([dim if dim is not None else 0 for dim in feature_shape.as_list()]
            if feature_shape else [])
    if dims and (isinstance(spec, ragged_tensor.RaggedTensorSpec) or
                 feature_shape.is_fully_defined()):
      dims[0] = tensor_shape.Dimension(0)

    if isinstance(spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensor(
          values=array_ops.zeros(0, feature_type),
          indices=array_ops.zeros((0, len(dims)), dtypes.int64),
          dense_shape=dims)

    # Create the dummy tensor.
    dummy_tensor = array_ops.zeros(tensor_shape.TensorShape(dims), feature_type)
    if isinstance(spec, ragged_tensor.RaggedTensorSpec):
      # Reinsert the ragged dimensions with size 0.
      # pylint: disable=protected-access
      row_splits = array_ops.zeros(1, spec._row_splits_dtype)
      dummy_tensor = ragged_tensor.RaggedTensor.from_nested_row_splits(
          dummy_tensor, (row_splits,) * spec._ragged_rank, validate=False)
      # pylint: enable=protected-access
    return dummy_tensor

  return nest.map_structure(create_dummy_tensor, value_structure)


class _SingleWorkerDatasetIteratorBase(object):
  """Iterator for a single `tf.data.Dataset`."""

  def __init__(self, dataset, worker, devices):
    """Create iterator for the `dataset` to fetch data to worker's `devices` .

    A `MultiDeviceIterator`  or `OwnedMultiDeviceIterator` is used to prefetch
    input to the devices on the given worker.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
    """
    self._dataset = dataset
    self._worker = worker
    self._devices = devices
    self._element_spec = dataset.element_spec
    self._make_iterator()

  def _make_iterator(self):
    raise NotImplementedError("must be implemented in descendants")

  def get_next(self, device, name=None):
    """Get next element for the given device."""
    del name
    with ops.device(self._worker):
      return self._iterator.get_next(device)

  def get_next_as_list_static_shapes(self, name=None):
    """Get next element from the underlying iterator.

    Runs the iterator get_next() within a device scope. Since this doesn't use
    get_next_as_optional(), is is considerably faster than get_next_as_list()
    (but can only be used when the shapes are static).

    Args:
      name: not used.

    Returns:
      A list consisting of the next data from each device.
    """
    del name
    with ops.device(self._worker):
      return self._iterator.get_next()

  def get_next_as_list(self, name=None):
    """Get next element from underlying iterator.

    If there is no data left, a list of dummy tensors with possible batch
    dimensions set to 0 will be returned. Use of get_next_as_optional() and
    extra logic adds overhead compared to get_next_as_list_static_shapes(), but
    allows us to handle non-static shapes.

    Args:
      name: not used.

    Returns:
      A boolean tensor indicates whether there is any data in next element and
      the real data as the next element or a list of dummy tensors if no data
      left.
    """
    del name
    with ops.device(self._worker):
      data_list = self._iterator.get_next_as_optional()
      result = []
      for i, data in enumerate(data_list):
        # Place the condition op in the same device as the data so the data
        # doesn't need to be sent back to the worker.
        with ops.device(self._devices[i]):
          # Data will be fetched in order, so we only need to check if the first
          # replica has value to see whether there is data left for this single
          # worker.
          if i == 0:
            worker_has_value = data.has_value()

          # pylint: disable=unnecessary-lambda
          # pylint: disable=cell-var-from-loop
          real_data = control_flow_ops.cond(
              data.has_value(),
              lambda: data.get_value(),
              lambda: _dummy_tensor_fn(data.value_structure),
              strict=True,
          )
          result.append(real_data)
          # pylint: enable=cell-var-from-loop
          # pylint: enable=unnecessary-lambda

      return worker_has_value, result


class _SingleWorkerDatasetIteratorSpec(type_spec.TypeSpec):
  """Type specification for `_SingleWorkerOwnedDatasetIterator`."""

  __slots__ = ["_worker", "_devices", "_element_spec"]

  def __init__(self, worker, devices, element_spec):
    self._worker = worker
    self._devices = devices
    self._element_spec = element_spec

  @property
  def value_type(self):
    return _SingleWorkerOwnedDatasetIterator

  def _serialize(self):
    return (self._worker, tuple(self._devices), self._element_spec)

  @property
  def _component_specs(self):
    specs = []
    specs.append(multi_device_iterator_ops.MultiDeviceIteratorSpec(
        self._devices, self._worker, element_spec=self._element_spec))
    return specs

  def _to_components(self, value):
    return [value._iterator]  # pylint: disable=protected-access

  def _from_components(self, components):
    return _SingleWorkerOwnedDatasetIterator(
        dataset=None,
        worker=self._worker,
        devices=self._devices,
        components=components,
        element_spec=self._element_spec)

  @staticmethod
  def from_value(value):
    # pylint: disable=protected-access
    return _SingleWorkerDatasetIteratorSpec(value._worker, value._devices,
                                            value._element_spec)


class _SingleWorkerOwnedDatasetIterator(_SingleWorkerDatasetIteratorBase,
                                        composite_tensor.CompositeTensor):
  """Iterator for a DistributedDataset instance."""

  def __init__(self, dataset=None, worker=None, devices=None, components=None,
               element_spec=None):
    """Create iterator for the `dataset` to fetch data to worker's `devices` .

    `OwnedMultiDeviceIterator` is used to prefetch input to the devices on the
    given worker. The lifetime of this iterator is tied to the encompassing
    python object. Once we go out of scope of the python object or return from
    a tf.function the underlying iterator resource is deleted.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
      components: Tensor components to construct the
        _SingleWorkerOwnedDatasetIterator from.
      element_spec: A nested structure of `TypeSpec` objects that represents the
      type specification of elements of the iterator.
    """
    if worker is None or devices is None:
      raise ValueError("Both `worker` and `devices` should be provided")

    error_message = ("Either `dataset` or both `components` and `element_spec` "
                     "need to be provided.")

    if dataset is None:
      if (components is None or element_spec is None):
        raise ValueError(error_message)
      self._element_spec = element_spec
      self._worker = worker
      self._devices = devices
      self._iterator = components[0]
    else:
      if (components is not None or element_spec is not None):
        raise ValueError(error_message)
      super(_SingleWorkerOwnedDatasetIterator, self).__init__(dataset, worker,
                                                              devices)

  def _make_iterator(self):
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
      self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
          self._dataset, self._devices)

  @property
  def element_spec(self):
    return self._element_spec

  @property
  def _type_spec(self):
    return _SingleWorkerDatasetIteratorSpec(self._worker, self._devices,
                                            self._element_spec)

  @property
  def output_classes(self):
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._element_spec)


class _SingleWorkerDatasetIterator(_SingleWorkerDatasetIteratorBase):
  """Iterator for a single DistributedDatasetV1 instance."""

  def _make_iterator(self):
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
      self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
          self._dataset, self._devices)

  def initialize(self):
    """Initialize underlying iterator.

    In eager execution, this simply recreates the underlying iterator.
    In graph execution, it returns the initializer ops for the underlying
    iterator.

    Returns:
      A list of any initializer ops that should be run.
    """
    if ops.executing_eagerly_outside_functions():
      self._iterator._eager_reset()  # pylint: disable=protected-access
      return []
    else:
      return [self._iterator.initializer]

  @property
  def output_classes(self):
    return dataset_ops.get_legacy_output_classes(self._iterator)

  @property
  def output_shapes(self):
    return dataset_ops.get_legacy_output_shapes(self._iterator)

  @property
  def output_types(self):
    return dataset_ops.get_legacy_output_types(self._iterator)


class _SingleWorkerCallableIterator(object):
  """Iterator for a single tensor-returning callable."""

  def __init__(self, fn, worker, devices):
    self._fn = fn
    self._worker = worker
    self._devices = devices

  def get_next(self, device, name=None):
    """Get next element for the given device from the callable."""
    del device, name
    with ops.device(self._worker):
      return self._fn()

  def get_next_as_list_static_shapes(self, name=None):
    """Get next element from the callable."""
    del name
    with ops.device(self._worker):
      data_list = [self._fn() for _ in self._devices]
      return data_list

  def get_next_as_list(self, name=None):
    """Get next element from the callable."""
    del name
    with ops.device(self._worker):
      data_list = [self._fn() for _ in self._devices]
      return constant_op.constant(True), data_list

  def initialize(self):
    # TODO(petebu) Should this throw an exception instead?
    return []


def _create_iterators_per_worker(worker_datasets, input_workers,
                                 graph_and_eager=False):
  """Create a multidevice iterator on each of the workers."""
  assert isinstance(input_workers, InputWorkers)

  assert len(worker_datasets) == len(input_workers.worker_devices)
  iterators = []
  for i, worker in enumerate(input_workers.worker_devices):
    with ops.device(worker):
      worker_devices = input_workers.compute_devices_for_worker(i)
      # We need an additional graph_and_eager condition to test for when we
      # create a DistributedDatasetV1 in TF 2.x and graph mode.
      # TODO(b/143568310): Condition only on graph vs eager consistent with
      # tf.data.
      if (tf2.enabled() and ops.executing_eagerly_outside_functions() and
          not graph_and_eager):
        iterator = _SingleWorkerOwnedDatasetIterator(worker_datasets[i], worker,
                                                     worker_devices)
      else:
        iterator = _SingleWorkerDatasetIterator(worker_datasets[i], worker,
                                                worker_devices)
      iterators.append(iterator)
  return iterators


def _create_iterators_per_worker_with_input_context(input_contexts,
                                                    input_workers,
                                                    dataset_fn,
                                                    graph_and_eager=False):
  """Create a multidevice iterator per workers given a dataset function."""
  iterators = []
  element_specs = []
  for i, ctx in enumerate(input_contexts):
    worker = input_workers.worker_devices[i]
    with ops.device(worker):
      dataset = dataset_fn(ctx)
      element_specs.append(dataset.element_spec)
      # TODO(b/138745411): Remove once stateful transformations are supported.
      options = dataset_ops.Options()
      options.experimental_distribute._make_stateless = True  # pylint: disable=protected-access
      dataset = dataset.with_options(options)
      devices = input_workers.compute_devices_for_worker(i)
      # We need an additional graph_and_eager condition to test for when we
      # create a DistributedDatasetV1 in TF 2.x and graph mode.
      # TODO(b/143568310): Condition only on graph vs eager consistent with
      # tf.data.
      if (tf2.enabled() and ops.executing_eagerly_outside_functions() and
          not graph_and_eager):
        iterator = _SingleWorkerOwnedDatasetIterator(dataset, worker,
                                                     devices)
      else:
        iterator = _SingleWorkerDatasetIterator(dataset, worker,
                                                devices)
      iterators.append(iterator)
  return iterators, dataset.element_spec


# TODO(sourabhbajaj): Remove this in lieu of distributed datasets
def _get_batched_dataset(d):
  """Get the batched dataset from `d`."""
  # pylint: disable=protected-access
  if isinstance(d, dataset_ops.DatasetV1Adapter):
    d = d._dataset

  if isinstance(d, (dataset_ops.BatchDataset, batching._MapAndBatchDataset)):
    return d
  elif isinstance(d, (dataset_ops.PrefetchDataset,
                      dataset_ops._OptionsDataset)):
    return _get_batched_dataset(d._input_dataset)

  raise ValueError(
      "Unable to get batched dataset from the input dataset. `batch` "
      "`map_and_batch` need to be the last operations on the dataset. "
      "The batch operations can be followed by a prefetch.")


def _get_batched_dataset_attributes(d):
  """Get `batch_size`, `drop_remainder` of dataset."""
  # pylint: disable=protected-access
  assert isinstance(d,
                    (dataset_ops.BatchDataset, batching._MapAndBatchDataset))
  if isinstance(d, dataset_ops.BatchDataset):
    batch_size = d._batch_size
    drop_remainder = d._drop_remainder
  elif isinstance(d, batching._MapAndBatchDataset):
    batch_size = d._batch_size_t
    drop_remainder = d._drop_remainder_t
  # pylint: enable=protected-access

  if tensor_util.is_tensor(batch_size):
    batch_size = tensor_util.constant_value(batch_size)

  if tensor_util.is_tensor(drop_remainder):
    drop_remainder = tensor_util.constant_value(drop_remainder)

  return batch_size, drop_remainder


# TODO(sourabhbajaj): Remove this in lieu of distributed datasets
def _get_dataset_attributes(dataset):
  """Get the underlying attributes from the dataset object."""
  # pylint: disable=protected-access

  # First, get batch_size and drop_remainder from the dataset. We need
  # to walk back the dataset creation process and find the batched version in
  # order to get the attributes.
  batched_dataset = _get_batched_dataset(dataset)
  batch_size, drop_remainder = _get_batched_dataset_attributes(batched_dataset)

  # Second, prefetch buffer should be get from the original dataset.
  prefetch_buffer = None
  if isinstance(dataset, dataset_ops.PrefetchDataset):
    prefetch_buffer = dataset._buffer_size
  elif (isinstance(dataset, dataset_ops.DatasetV1Adapter)
        and isinstance(dataset._dataset, dataset_ops.PrefetchDataset)):
    prefetch_buffer = dataset._dataset._buffer_size

  return batch_size, drop_remainder, prefetch_buffer


class MultiStepContext(object):
  """A context object that can be used to capture things when running steps.

  This context object is useful when running multiple steps at a time using the
  `experimental_run_steps_on_iterator` API. For e.g. it allows the user's step
  function to specify which outputs to emit at what frequency. Currently it
  supports capturing output from the last step, as well as capturing non tensor
  outputs.  In the future it will be augmented to support other use cases such
  as output each N steps.
  """

  def __init__(self):
    """Initialize an output context.

    Returns:
      A context object.
    """
    self._last_step_outputs = {}
    self._last_step_outputs_reduce_ops = {}
    self._non_tensor_outputs = {}

  @property
  def last_step_outputs(self):
    """A dictionary consisting of outputs to be captured on last step.

    Keys in the dictionary are names of tensors to be captured, as specified
    when `set_last_step_output` is called.
    Values in the dictionary are the tensors themselves. If
    `set_last_step_output` was called with a `reduce_op` for this output,
    then the value is the reduced value.

    Returns:
      A dictionary with last step outputs.
    """
    return self._last_step_outputs

  def _set_last_step_outputs(self, outputs):
    """Replace the entire dictionary of last step outputs."""
    if not isinstance(outputs, dict):
      raise ValueError("Need a dictionary to set last_step_outputs.")
    self._last_step_outputs = outputs

  def set_last_step_output(self, name, output, reduce_op=None):
    """Set `output` with `name` to be outputted from the last step.

    Args:
      name: String, name to identify the output. Doesn't need to match tensor
        name.
      output: The tensors that should be outputted with `name`. See below for
        actual types supported.
      reduce_op: Reduction method to use to reduce outputs from multiple
        replicas. Required if `set_last_step_output` is called in a replica
        context. Optional in cross_replica_context.
        When present, the outputs from all the replicas are reduced using the
        current distribution strategy's `reduce` method. Hence, the type of
        `output` must be what's supported by the corresponding `reduce` method.
        For e.g. if using MirroredStrategy and reduction is set, output
        must be a `PerReplica` value.
        The reduce method is also recorded in a dictionary
        `_last_step_outputs_reduce_ops` for later interpreting of the
        outputs as already reduced or not.
    """
    if distribution_strategy_context.in_cross_replica_context():
      self._last_step_outputs_reduce_ops[name] = reduce_op
      if reduce_op is None:
        self._last_step_outputs[name] = output
      else:
        distribution = distribution_strategy_context.get_strategy()
        self._last_step_outputs[name] = distribution.reduce(reduce_op, output,
                                                            axis=None)
    else:
      assert reduce_op is not None
      def merge_fn(distribution, value):
        self._last_step_outputs[name] = distribution.reduce(reduce_op, value,
                                                            axis=None)
        # Setting this inside the `merge_fn` because all replicas share the same
        # context object, so it's more robust to set it only once (even if all
        # the replicas are trying to set the same value).
        self._last_step_outputs_reduce_ops[name] = reduce_op

      distribution_strategy_context.get_replica_context().merge_call(
          merge_fn, args=(output,))

  @property
  def non_tensor_outputs(self):
    """A dictionary consisting of any non tensor outputs to be captured."""
    return self._non_tensor_outputs

  def set_non_tensor_output(self, name, output):
    """Set `output` with `name` to be captured as a non tensor output."""
    if distribution_strategy_context.in_cross_replica_context():
      self._non_tensor_outputs[name] = output
    else:
      def merge_fn(distribution, value):
        # NOTE(priyag): For non tensor outputs, we simply return all the values
        # in a list as reduction doesn't make sense on non tensors.
        self._non_tensor_outputs[name] = (
            distribution.experimental_local_results(value))
      distribution_strategy_context.get_replica_context().merge_call(
          merge_fn, args=(output,))


def _create_distributed_tensor_spec(strategy, tensor_spec):
  """Create a `tf.TypeSpec` for a given strategy and input `tensor_spec`.

  Args:
    strategy: The given `tf.distribute` strategy.
    tensor_spec: `tf.TensorSpec` of a given value. The batch dimension of the
      shape should be None if you have partial batches.

  Returns:
    A `tf.TypeSpec` that matches the values produced by a given strategy. This
    can be a `tf.TensorSpec` or a `PerRelicaSpec`.
  """
  num_replicas = len(strategy.extended.worker_devices)

  # If the number of devices used in the strategy is just 1 then we return
  # the tensor_spec as is.
  if num_replicas == 1:
    return tensor_spec

  # If the number of devices is greater than 1 then we assume the input to
  # tf.function is a per replica type.
  def _get_value_per_replica(tensor_spec_per_input):
    value_specs = [tensor_spec_per_input for _ in range(num_replicas)]
    return values.PerReplicaSpec(*value_specs)

  return nest.map_structure(_get_value_per_replica, tensor_spec)

