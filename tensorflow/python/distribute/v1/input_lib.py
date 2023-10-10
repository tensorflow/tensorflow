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
"""Various classes representing distributed inputs."""

from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated


class DistributedDatasetV1(input_lib.DistributedDataset):
  """Distributed dataset that supports prefetching to multiple devices."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               num_replicas_in_sync=None,
               input_context=None,
               options=None):
    self._input_workers = input_workers
    super(DistributedDatasetV1, self).__init__(
        input_workers,
        strategy,
        dataset,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context,
        options=options)

  def make_one_shot_iterator(self):
    """Get a one time use iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use `for ... in dataset:` to iterate
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
                                                    self._options)
    cardinality = input_lib._cardinality(self._cloned_datasets[0])  # pylint: disable=protected-access
    iterator = DistributedIteratorV1(self._input_workers, worker_iterators,
                                     self._strategy, cardinality,
                                     self._enable_get_next_as_optional)
    iterator._element_spec = self.element_spec  # pylint: disable=protected-access

    # When async eager is enabled, sometimes the iterator may not finish
    # initialization before passing to a multi device function, add a sync point
    # here to make sure all underlying iterators are initialized.
    if context.executing_eagerly():
      context.async_wait()

    return iterator

  # pylint: disable=non-iterator-returned
  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")

  # pylint: enable=non-iterator-returned


class DistributedDatasetsFromFunctionV1(
    input_lib.DistributedDatasetsFromFunction):
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
    iterators = _create_iterators_per_worker(self._datasets,
                                             self._input_workers, self._options)
    cardinality = input_lib._cardinality(self._datasets[0])  # pylint: disable=protected-access
    iterator = DistributedIteratorV1(self._input_workers, iterators,
                                     self._strategy, cardinality,
                                     self._enable_get_next_as_optional)
    iterator._element_spec = self._element_spec  # pylint: disable=protected-access

    # When async eager is enabled, sometimes the iterator may not finish
    # initialization before passing to a multi device function, add a sync point
    # here to make sure all underlying iterators are initialized.
    if context.executing_eagerly():
      context.async_wait()

    return iterator

  # pylint: disable=non-iterator-returned
  def __iter__(self):
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")

  # pylint: enable=non-iterator-returned


class DistributedIteratorV1(input_lib.DistributedIteratorBase):
  """Input Iterator for a distributed dataset."""

  # We need a private initializer method for re-initializing multidevice
  # iterators when used with Keras training loops. If we don't reinitialize the
  # iterator we run into memory leak issues (b/123315763).
  @property
  def _initializer(self):
    init_ops = []
    for it in self._iterators:
      init_ops.extend(it.initialize())
    return control_flow_ops.group(init_ops)

  @deprecated(None, "Use the iterator's `initializer` property instead.")
  def initialize(self):
    """Initialize underlying iterators.

    Returns:
      A list of any initializer ops that should be run.
    """
    return self._initializer

  @property
  def initializer(self):
    """Returns a list of ops that initialize the iterator."""
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


class DatasetIterator(DistributedIteratorV1):
  """Iterator created from input dataset."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               num_replicas_in_sync=None,
               input_context=None):
    """Make an iterator for the dataset on given devices.

    If `num_replicas_in_sync` is not None, we split each batch of the dataset
    into `num_replicas_in_sync` smaller batches, to be distributed among that
    worker's replicas, so that the batch size for a global step (across all
    workers and replicas) is as expected.

    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      num_replicas_in_sync: Optional integer. If this is not None, the value is
        used to decide how to rebatch datasets into smaller batches so that the
        total batch size for each step (across all workers and replicas) adds up
        to `dataset`'s batch size.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
    """
    dist_dataset = DistributedDatasetV1(
        dataset,
        input_workers,
        strategy,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context)
    # pylint: disable=protected-access
    worker_iterators = _create_iterators_per_worker(
        dist_dataset._cloned_datasets, input_workers)
    super(DatasetIterator,
          self).__init__(input_workers, worker_iterators, strategy,
                         dist_dataset.cardinality,
                         dist_dataset._enable_get_next_as_optional)
    self._element_spec = dist_dataset.element_spec
    # pylint: enable=protected-access


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
    assert isinstance(input_workers, input_lib.InputWorkers)
    if input_workers.num_workers != len(input_contexts):
      raise ValueError("Number of input workers (%d) is not same as number of "
                       "input_contexts (%d)" %
                       (input_workers.num_workers, len(input_contexts)))

    iterators = []
    for i, ctx in enumerate(input_contexts):
      worker = input_workers.worker_devices[i]
      with ops.device(worker):
        result = input_fn(ctx)
        devices = input_workers.compute_devices_for_worker(i)
        if isinstance(result, data_types.DatasetV2):
          iterator = _SingleWorkerDatasetIterator(result, worker, devices)
        elif callable(result):
          iterator = _SingleWorkerCallableIterator(result, worker, devices)
        else:
          raise ValueError(
              "input_fn must return a tf.data.Dataset or a callable.")
        iterators.append(iterator)

    super(InputFunctionIterator, self).__init__(
        input_workers,
        iterators,
        strategy,
        cardinality=cardinality_lib.UNKNOWN,
        enable_get_next_as_optional=False)
    self._enable_get_next_as_optional = False


class _SingleWorkerDatasetIterator(input_lib._SingleWorkerDatasetIteratorBase):  # pylint: disable=protected-access
  """Iterator for a single DistributedDatasetV1 instance."""

  def _make_iterator(self):
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
      if self._options is not None:
        self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset,
            self._devices,
            max_buffer_size=self._options.experimental_per_replica_buffer_size,
            prefetch_buffer_size=self._options
            .experimental_per_replica_buffer_size)
      else:
        self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset,
            self._devices,
        )

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

  def get_next_as_list(self, name=None):
    """Get next element from the callable."""
    del name
    with ops.device(self._worker):
      data_list = [self._fn() for _ in self._devices]
      return data_list

  def get_next_as_optional_list(self):
    with ops.device(self._worker):
      data_list = [
          optional_ops.Optional.from_value(self._fn()) for _ in self._devices
      ]
      return data_list

  def initialize(self):
    # TODO(petebu) Should this throw an exception instead?
    return []


def _create_iterators_per_worker(worker_datasets, input_workers, options=None):
  """Create a multidevice iterator on each of the workers."""
  assert isinstance(input_workers, input_lib.InputWorkers)
  assert len(worker_datasets) == len(input_workers.worker_devices)
  iterators = []
  for i, worker in enumerate(input_workers.worker_devices):
    with ops.device(worker):
      worker_devices = input_workers.compute_devices_for_worker(i)
      iterator = _SingleWorkerDatasetIterator(
          worker_datasets[i],  # pylint: disable=protected-access
          worker,
          worker_devices,
          options)
      iterators.append(iterator)
  return iterators
