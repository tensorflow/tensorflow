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
"""Python wrapper for prefetching_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops


class _PerDeviceGenerator(dataset_ops.DatasetV2):
  """A `dummy` generator dataset."""

  def __init__(self, shard_num, multi_device_iterator_resource, incarnation_id,
               source_device, element_structure):
    self._structure = element_structure

    multi_device_iterator_string_handle = (
        gen_dataset_ops.multi_device_iterator_to_string_handle(
            multi_device_iterator_resource))

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(autograph=False)  # Pure graph code.
    def _init_func():
      return multi_device_iterator_string_handle

    init_func_concrete = _init_func._get_concrete_function_internal()  # pylint: disable=protected-access

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(autograph=False)  # Pure graph code.
    def _remote_init_func():
      return functional_ops.remote_call(
          target=source_device,
          args=init_func_concrete.captured_inputs,
          Tout=[dtypes.string],
          f=init_func_concrete)

    self._init_func = _remote_init_func._get_concrete_function_internal()  # pylint: disable=protected-access
    self._init_captured_args = self._init_func.captured_inputs

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        autograph=False)  # Pure graph code.
    def _next_func(string_handle):
      # pylint: disable=protected-access
      multi_device_iterator = (
          gen_dataset_ops.multi_device_iterator_from_string_handle(
              string_handle=string_handle,
              output_types=self._structure._flat_types,
              output_shapes=self._structure._flat_shapes))
      return gen_dataset_ops.multi_device_iterator_get_next_from_shard(
          multi_device_iterator=multi_device_iterator,
          shard_num=shard_num,
          incarnation_id=incarnation_id,
          output_types=self._structure._flat_types,
          output_shapes=self._structure._flat_shapes)

    next_func_concrete = _next_func._get_concrete_function_internal()  # pylint: disable=protected-access

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun_with_attributes(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        attributes={"experimental_ints_on_device": True},
        autograph=False)  # Pure graph code.
    def _remote_next_func(string_handle):
      return functional_ops.remote_call(
          target=source_device,
          args=[string_handle] + next_func_concrete.captured_inputs,
          Tout=self._structure._flat_types,  # pylint: disable=protected-access
          f=next_func_concrete)

    self._next_func = _remote_next_func._get_concrete_function_internal()  # pylint: disable=protected-access
    self._next_captured_args = self._next_func.captured_inputs

    self._incarnation_id_index = -1
    for i, arg in enumerate(self._next_captured_args):
      if arg == incarnation_id:
        self._incarnation_id_index = i

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        autograph=False)  # Pure graph code.
    def _finalize_func(unused_string_handle):
      return array_ops.constant(0, dtypes.int64)

    finalize_func_concrete = _finalize_func._get_concrete_function_internal()  # pylint: disable=protected-access

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        autograph=False)  # Pure graph code.
    def _remote_finalize_func(string_handle):
      return functional_ops.remote_call(
          target=source_device,
          args=[string_handle] + finalize_func_concrete.captured_inputs,
          Tout=[dtypes.int64],
          f=finalize_func_concrete)

    self._finalize_func = _remote_finalize_func._get_concrete_function_internal(  # pylint: disable=protected-access
    )
    self._finalize_captured_args = self._finalize_func.captured_inputs

    variant_tensor = gen_dataset_ops.generator_dataset(
        self._init_captured_args,
        self._next_captured_args,
        self._finalize_captured_args,
        init_func=self._init_func,
        next_func=self._next_func,
        finalize_func=self._finalize_func,
        **dataset_ops.flat_structure(self))
    super(_PerDeviceGenerator, self).__init__(variant_tensor)

  def _inputs(self):
    # TODO(b/116506223): Determine which datasets should be used as inputs here.
    return []

  @property
  def _element_structure(self):
    return self._structure


class _ReincarnatedPerDeviceGenerator(dataset_ops.DatasetV2):
  """Creates a _PerDeviceGenerator-like dataset with a new incarnation_id.

  Re-uses the functions from the provided per_device_dataset and just switches
  out the function argument corresponding to the incarnation_id.
  """

  def __init__(self, per_device_dataset, incarnation_id):
    # pylint: disable=protected-access
    self._structure = per_device_dataset._structure

    self._init_func = per_device_dataset._init_func
    self._init_captured_args = self._init_func.captured_inputs

    self._next_func = per_device_dataset._next_func
    self._next_captured_args = per_device_dataset._next_captured_args
    # The captured arguments to the next_func are string_handle, incarnation_id.
    # We update the incarnation id to the new one.
    self._next_captured_args[
        per_device_dataset._incarnation_id_index] = incarnation_id

    self._finalize_func = per_device_dataset._finalize_func
    self._finalize_captured_args = per_device_dataset._finalize_captured_args

    variant_tensor = gen_dataset_ops.generator_dataset(
        self._init_captured_args,
        self._next_captured_args,
        self._finalize_captured_args,
        init_func=self._init_func,
        next_func=self._next_func,
        finalize_func=self._finalize_func,
        **dataset_ops.flat_structure(self))
    super(_ReincarnatedPerDeviceGenerator, self).__init__(variant_tensor)

  def _inputs(self):
    # TODO(b/116506223): Determine which datasets should be used as inputs here.
    return []

  @property
  def _element_structure(self):
    return self._structure


class MultiDeviceIterator(object):
  """An iterator over multiple devices."""

  def __init__(self,
               dataset,
               devices,
               max_buffer_size=1,
               prefetch_buffer_size=1,
               source_device="/cpu:0"):
    """Constructs a MultiDeviceIterator.

    Args:
      dataset: The input dataset to be iterated over.
      devices: The list of devices to fetch data to.
      max_buffer_size: Maximum size of the host side per device buffer to keep.
      prefetch_buffer_size: if > 1, then we setup a buffer on each device
        to prefetch into.
      source_device: The host device to place the `dataset` on.

      In order to prevent deadlocks, if the prefetch_buffer_size is greater
      than the max_buffer_size, we set the max_buffer_size to
      prefetch_buffer_size.
    """
    options = dataset_ops.Options()
    options.experimental_distribute.num_devices = len(devices)
    dataset = dataset.with_options(options)
    self._dataset = dataset._apply_options()  # pylint: disable=protected-access
    self._experimental_slack = dataset.options().experimental_slack
    self._devices = devices
    self._source_device = source_device
    self._source_device_tensor = ops.convert_to_tensor(source_device)
    self._max_buffer_size = max_buffer_size
    self._prefetch_buffer_size = prefetch_buffer_size

    if self._prefetch_buffer_size > self._max_buffer_size:
      self._max_buffer_size = self._prefetch_buffer_size

    # Create the MultiDeviceIterator.
    with ops.device(self._source_device):
      # TODO(b/121378567): Get rid of this shared_name hack.
      shared_name = ""
      if context.executing_eagerly():
        shared_name = context.shared_name()
      self._multi_device_iterator_resource = (
          gen_dataset_ops.multi_device_iterator(
              devices=self._devices,
              shared_name=shared_name,
              container="",
              **dataset_ops.flat_structure(self._dataset)))
      if context.executing_eagerly():
        # Delete the resource when this object is deleted
        self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
            handle=self._multi_device_iterator_resource,
            handle_device=self._source_device)

      # The incarnation ID is used to ensure consistency between the per-device
      # iterators and the multi-device iterator.
      self._incarnation_id = gen_dataset_ops.multi_device_iterator_init(
          self._dataset._variant_tensor,  # pylint: disable=protected-access
          self._multi_device_iterator_resource,
          max_buffer_size=self._max_buffer_size)

    self._prototype_device_datasets = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        ds = _PerDeviceGenerator(
            i, self._multi_device_iterator_resource, self._incarnation_id,
            self._source_device_tensor, self._dataset._element_structure)  # pylint: disable=protected-access
        self._prototype_device_datasets.append(ds)

    # TODO(rohanj): Explore the possibility of the MultiDeviceIterator to
    # initialize the device side of the pipeline. This would allow the
    # MultiDeviceIterator to choose, for example, to move some transformations
    # into the device side from its input. It might be useful in rewriting.
    # Create the per device iterators.
    self._device_iterators = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        ds = self._create_device_dataset(i)
        if context.executing_eagerly():
          self._device_iterators.append(dataset_ops.make_one_shot_iterator(ds))
        else:
          self._device_iterators.append(
              dataset_ops.make_initializable_iterator(ds))

    if not context.executing_eagerly():
      device_iterator_initializers = [
          iterator.initializer for iterator in self._device_iterators
      ]
      self._initializer = control_flow_ops.group(*device_iterator_initializers)

  def _create_device_dataset(self, i):
    """Uses _prototype_device_datasets[i] to build a dataset for the device."""
    ds = self._prototype_device_datasets[i]
    ds = _ReincarnatedPerDeviceGenerator(ds, self._incarnation_id)
    if self._prefetch_buffer_size > 0:
      if self._experimental_slack:
        ds = dataset_ops.PrefetchDataset(
            ds, self._prefetch_buffer_size, slack_period=1)
      else:
        ds = ds.prefetch(self._prefetch_buffer_size)
    # TODO(jsimsa): Enable auto-tuning and optimizations when supported for
    # non-CPU devices.
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False
    ds = ds.with_options(options)
    return ds

  def get_next(self, device=None):
    """Returns the next element given a `device`, else returns all in a list."""
    if device is not None:
      index = self._devices.index(device)
      return self._device_iterators[index].get_next()

    result = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        result.append(self._device_iterators[i].get_next())
    return result

  def get_next_as_optional(self):
    result = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        result.append(iterator_ops.get_next_as_optional(
            self._device_iterators[i]))
    return result

  @property
  def initializer(self):
    if context.executing_eagerly():
      return control_flow_ops.no_op()
    return self._initializer

  def _eager_reset(self):
    """Resets the MultiDeviceIterator in eager mode."""
    if not context.executing_eagerly():
      raise ValueError("Eager reset is only supported in eager mode.")
    # pylint: disable=protected-access
    self._incarnation_id = gen_dataset_ops.multi_device_iterator_init(
        self._dataset._variant_tensor,
        self._multi_device_iterator_resource,
        max_buffer_size=self._max_buffer_size)
    for i, device in enumerate(self._devices):
      with ops.device(device):
        ds = self._create_device_dataset(i)
        # Reset the device iterator resources with the new dataset.
        ds_variant = ds._variant_tensor
        gen_dataset_ops.make_iterator(
            ds_variant, self._device_iterators[i]._iterator_resource)

  @property
  def _element_structure(self):
    return dataset_ops.get_structure(self._dataset)
