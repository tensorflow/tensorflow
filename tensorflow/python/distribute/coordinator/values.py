# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Important value classes relevant to `ClusterCoordinator`.

This is currently under development and the API is subject to change.
"""

import enum
import threading

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


class RemoteValueStatus(enum.Enum):
  """The status of a `RemoteValue` object.

  A `RemoteValue` object can have three states:
    1) not ready: no value, no non-retryable error and not aborted;
    2) aborted: i.e. the execution of function was aborted because of task
       failure, but can be retried;
    3) ready: i.e. has value or has non-tryable error;

  The initial state of a `RemoteValue` is "not ready". When its corresponding
  closure has
  been executed at least once, it will become aborted or ready. The state
  transitions are:
    1) not ready -> 2) aborted:
      when the corresponding closure is aborted due to worker failure, and the
      worker failure is not immediately handled.
    1) not ready -> 3) ready:
      when the corresponding closure has been executed successfully.
    2) aborted -> 3) ready:
      when the `RemoteValue` is rebuilt by rerunning the corresponding closure
      and the closure has been executed successfully.
    3) ready -> 2) aborted:
      when the corresponding closure had been executed successfully but later
      the corresponding remote worker failed. This is currently only implemented
      for resource `RemoteValue` like iterators.
  """
  NOT_READY = "NOT_READY"
  ABORTED = "ABORTED"
  READY = "READY"


@tf_export("distribute.experimental.coordinator.RemoteValue",
           "distribute.coordinator.RemoteValue", v1=[])
class RemoteValue(object):
  """An asynchronously available value of a scheduled function.

  This class is used as the return value of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` where
  the underlying value becomes available at a later time once the function has
  been executed.

  Using `tf.distribute.experimental.coordinator.RemoteValue` as an input to
  a subsequent function scheduled with
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` is
  currently not supported.

  Example:

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

  with strategy.scope():
    v1 = tf.Variable(initial_value=0.0)
    v2 = tf.Variable(initial_value=1.0)

  @tf.function
  def worker_fn():
    v1.assign_add(0.1)
    v2.assign_sub(0.2)
    return v1.read_value() / v2.read_value()

  result = coordinator.schedule(worker_fn)
  # Note that `fetch()` gives the actual result instead of a `tf.Tensor`.
  assert result.fetch() == 0.125

  for _ in range(10):
    # `worker_fn` will be run on arbitrary workers that are available. The
    # `result` value will be available later.
    result = coordinator.schedule(worker_fn)
  ```
  """

  def fetch(self):
    """Wait for the result of `RemoteValue` and return the numpy result.

    This makes the value concrete by copying the remote value to local.

    Returns:
      The numpy array structure of the actual output of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single value, or a structure of values, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
    raise NotImplementedError("Must be implemented in subclasses.")

  def get(self):
    """Wait for the result of `RemoteValue` and return the tensor result.

    This makes the value concrete by copying the remote tensor to local.

    Returns:
      The actual output (in the form of `tf.Tensor`s) of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single Tensor, or a structure of Tensors, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
    raise NotImplementedError("Must be implemented in subclasses.")


# TODO(yuefengz): create an implementation for resource RemoteValue which needs
# to remember the closure object while a normal RemoteValue doesn't.
class RemoteValueImpl(RemoteValue):
  """Implementation of `RemoteValue`."""

  def __init__(self, closure, type_spec):  # pylint: disable=super-init-not-called
    """Initializes a `RemoteValueImpl`.

    Args:
      closure: The closure from which the `RemoteValue` is created.
      type_spec: The type spec for this `RemoteValue` which is used to trace
        functions that take this `RemoteValue` as input.
    """
    self._closure = closure
    self._type_spec = type_spec
    self._values = None
    self._has_fetched_to_local = False
    self._has_fetched_to_local_lock = threading.Lock()
    self._fetched_tensors = None
    self._error = None
    self._status_available_event = threading.Event()
    self._status = RemoteValueStatus.NOT_READY

  def _set_aborted(self):
    self._status = RemoteValueStatus.ABORTED
    self._values = None
    self._error = None

    # Wake up any waiting thread and clear the event.
    self._status_available_event.set()

  def _rebuild_on(self, worker):
    self._status_available_event.clear()
    # TODO(yuefengz): we may need to rebuild its inputs as well.
    self._closure.execute_on(worker)

  def _set_values(self, tensors):
    self._status = RemoteValueStatus.READY
    self._values = tensors
    self._error = None
    self._status_available_event.set()

  def _set_error(self, exception):
    self._status = RemoteValueStatus.READY
    self._values = None
    self._error = exception
    self._status_available_event.set()

  def _get_values(self):
    self._status_available_event.wait()
    return self._values

  def _get_error(self):
    self._status_available_event.wait()
    return self._error

  def _wait_and_maybe_error(self):
    self._status_available_event.wait()
    if self._status is RemoteValueStatus.ABORTED:
      raise errors.CancelledError(
          None, None,
          "The corresponding function is aborted. Please reschedule the "
          "function.")
    if self._error is not None:
      raise self._error

  def fetch(self):
    # TODO(rchao): Discuss the possibility of letting users perform `numpy`
    # themselves at API graduation.
    return nest.map_structure(
        lambda x: x.numpy() if hasattr(x, "numpy") else x, self.get())

  def get(self):
    self._wait_and_maybe_error()

    with self._has_fetched_to_local_lock:
      if not self._has_fetched_to_local:

        def copy_tensor(composite_tensor_obj):
          """Copy a remote tensor to local (coordinator)."""
          if isinstance(composite_tensor_obj, input_lib.DistributedIterator):
            # A DistributedIterator cannot be copied to local; users should not
            # access that anyway.
            return composite_tensor_obj

          with ops.device("/job:%s" % context.get_server_def().job_name):
            # Copying to local (the coordinator) with `tf.device`.
            return array_ops.identity(composite_tensor_obj)

        if self._values is not None:
          # When `self._values` is `None`, it indicates the associated function
          # does not have a return value.
          self._fetched_tensors = nest.map_structure(copy_tensor, self._values)
        self._has_fetched_to_local = True

    return self._fetched_tensors


@tf_export("distribute.experimental.coordinator.PerWorkerValues",
           "distribute.coordinator.PerWorkerValue", v1=[])
class PerWorkerValues(composite_tensor.CompositeTensor):
  """A container that holds a list of values, one value per worker.

  `tf.distribute.experimental.coordinator.PerWorkerValues` contains a collection
  of values, where each of the values is located on its corresponding worker,
  and upon being used as one of the `args` or `kwargs` of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule()`, the
  value specific to a worker will be passed into the function being executed at
  that corresponding worker.

  Currently, the only supported path to create an object of
  `tf.distribute.experimental.coordinator.PerWorkerValues` is through calling
  `iter` on a `ClusterCoordinator.create_per_worker_dataset`-returned
  distributed dataset instance. The mechanism to create a custom
  `tf.distribute.experimental.coordinator.PerWorkerValues` is not yet supported.
  """

  def __init__(self, values):
    for v in values:
      if not isinstance(v, RemoteValue):
        raise AssertionError(
            "`PerWorkerValues` should only take `RemoteValue`s.")
    self._values = tuple(values)

  @property
  def _type_spec(self):
    return PerWorkerValuesTypeSpec(
        self._values[0]._type_spec,  # pylint: disable=protected-access
        type(self))


class PerWorkerValuesTypeSpec(type_spec_lib.TypeSpec):
  """TypeSpec for PerWorkerValues.

  It only support tracing a function using a PerWorkerValues.
  """

  def __init__(self, value_spec, descendant_type):
    assert value_spec
    self._value_spec = value_spec
    self._descendant_type = descendant_type

  def _serialize(self):
    return (self._value_spec,)

  @property
  def value_type(self):
    return self._descendant_type

  def most_specific_common_supertype(self, others):
    raise NotImplementedError(
        "most_specific_common_supertype is not implemented")

  @property
  def _component_specs(self):
    return self._value_spec

  def _to_components(self, value):
    return self._value_spec

  def _from_components(self, value):
    return value


class PerWorkerDatasetFromDatasetFunction(object):
  """Represents worker-distributed datasets created from dataset function."""

  def __init__(self, dataset_fn, coordinator):
    """Makes an iterable from datasets created by the given function.

    Args:
      dataset_fn: A function that returns a `Dataset`.
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """

    def disallow_variable_creation(next_creator, **kwargs):
      raise ValueError("Creating variables in `dataset_fn` is not allowed.")

    if isinstance(dataset_fn, def_function.Function):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = dataset_fn.get_concrete_function()
    elif not isinstance(dataset_fn, tf_function.ConcreteFunction):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = def_function.function(dataset_fn).get_concrete_function()
    self._dataset_fn = dataset_fn
    self._coordinator = coordinator
    self._element_spec = None

  def __iter__(self):
    # We would like users to create iterators outside `tf.function`s so that we
    # can track them.
    if (not context.executing_eagerly() or
        ops.get_default_graph().building_function):
      raise RuntimeError(
          "__iter__() is not supported inside of tf.function or in graph mode.")

    def _create_per_worker_iterator():
      dataset = self._dataset_fn()
      return iter(dataset)

    # If PerWorkerDatasetFromDatasetFunction.__iter__ is called multiple
    # times, for the same object it should only create and register resource
    # once. Using object id to distinguish different iterator resources.
    per_worker_iterator = self._coordinator._create_per_worker_resources(
        _create_per_worker_iterator)

    # Setting type_spec of each RemoteValue so that functions taking these
    # RemoteValues as inputs can be traced.
    for iterator_remote_value in per_worker_iterator._values:
      iterator_remote_value._type_spec = (
          input_lib.get_iterator_spec_from_dataset(
              self._coordinator.strategy, self._dataset_fn.structured_outputs))

    return PerWorkerDistributedIterator(per_worker_iterator._values)

  @property
  def element_spec(self):
    """The type specification of an element of this dataset.

    This property is subject to change without notice.
    """
    if not isinstance(self._dataset_fn, tf_function.ConcreteFunction):
      raise NotImplementedError(
          "`element_spec` is not supported when the `dataset_fn` is not "
          "a `ConcreteFunction`.")
    return self._dataset_fn.structured_outputs.element_spec


def serialize_dataset_to_graph(dataset):
  dataset = dataset._apply_debug_options()  # pylint: disable=protected-access
  graph_def = gen_dataset_ops.dataset_to_graph_v2(
      dataset._variant_tensor,  # pylint: disable=protected-access
      external_state_policy=ExternalStatePolicy.WARN.value,
      strip_device_assignment=True)
  return graph_def


class _RemoteDataset(dataset_ops.DatasetSource):
  """Creates a dataset given a graph def."""

  def __init__(self, graph_def, element_spec):
    self._elem_spec = element_spec
    variant_tensor = ged_ops.dataset_from_graph(graph_def)
    super(_RemoteDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._elem_spec


def deserialize_dataset_from_graph(graph_def, element_spec):
  return _RemoteDataset(graph_def, element_spec)


class PerWorkerDatasetFromDataset(PerWorkerDatasetFromDatasetFunction):
  """Represents worker-distributed datasets created from a dataset."""

  def __init__(self, dataset, coordinator):
    """Makes an iterable from datasets created by the given dataset.

    It creates a dataset_fn which deserializes a dataset from a graph under the
    hood.

    Args:
      dataset: A tf.data.Dataset, a DistributedDataset or a
        DistributedDatasetsFromFunction
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """
    if isinstance(dataset, input_lib.DistributedDataset):
      original_dataset = dataset._original_dataset
      serialized = serialize_dataset_to_graph(original_dataset)

      def dataset_fn():
        deserialized = deserialize_dataset_from_graph(
            serialized, original_dataset.element_spec)
        dataset.build(dataset_to_replace=deserialized)
        return dataset
    elif isinstance(dataset, input_lib.DistributedDatasetsFromFunction):
      def dataset_fn():
        dataset.build()
        return dataset
    elif isinstance(dataset, dataset_ops.Dataset):
      serialized = serialize_dataset_to_graph(dataset)

      def dataset_fn():
        return deserialize_dataset_from_graph(serialized, dataset.element_spec)
    else:
      raise ValueError("Unexpected dataset type!")

    super(PerWorkerDatasetFromDataset, self).__init__(dataset_fn, coordinator)


def get_per_worker_dataset(dataset_or_dataset_fn, coordinator):
  if callable(dataset_or_dataset_fn):
    return PerWorkerDatasetFromDatasetFunction(dataset_or_dataset_fn,
                                               coordinator)
  else:
    return PerWorkerDatasetFromDataset(dataset_or_dataset_fn, coordinator)


class PerWorkerDistributedIterator(PerWorkerValues):
  """Distributed iterator for `ClusterCoordinator`."""

  def __next__(self):
    return self.get_next()

  def get_next(self, name=None):
    """Returns the next input from the iterator for all replicas."""
    raise NotImplementedError("Iterating over an `AsyncDistributedIterator` "
                              "is not supported right now.")
