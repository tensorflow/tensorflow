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
"""Extension of prefetching_ops to support more than one device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.util import nest


# pylint: disable=protected-access
class _PrefetchToDeviceIterator(object):
  """A replacement for @{tf.data.Iterator} that prefetches to another device."""

  def __init__(self, input_dataset, devices, buffer_size):
    self._input_dataset = input_dataset
    self._get_next_call_count = 0
    self._devices = devices
    input_iterator = input_dataset.make_one_shot_iterator()
    input_iterator_handle = input_iterator.string_handle()

    @function.Defun(dtypes.string)
    def _prefetch_fn(handle):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          handle, input_iterator.output_types, input_iterator.output_shapes,
          input_iterator.output_classes)
      return remote_iterator.get_next()

    target_device = gen_dataset_ops.iterator_get_device(
        input_iterator._iterator_resource)
    self._buffering_resources = []
    for device in nest.flatten(self._devices):
      with ops.device(device):
        buffer_resource_handle = prefetching_ops.function_buffering_resource(
            f=_prefetch_fn,
            target_device=target_device,
            string_arg=input_iterator_handle,
            buffer_size=buffer_size)
        self._buffering_resources.append(buffer_resource_handle)

  def get_next(self, name=None):
    """See @{tf.data.Iterator.get_next}."""
    self._get_next_call_count += 1
    if self._get_next_call_count > iterator_ops.GET_NEXT_CALL_WARNING_THRESHOLD:
      warnings.warn(iterator_ops.GET_NEXT_CALL_WARNING_MESSAGE)

    flat_result = []
    # TODO(priyag): This will fail if the input size (typically number of
    # batches) is not divisible by number of devices.
    # How do we handle that more gracefully / let the user know?
    for buffer_resource in self._buffering_resources:
      flat_ret = gen_dataset_ops.function_buffering_resource_get_next(
          buffer_resource,
          output_types=data_nest.flatten(sparse.as_dense_types(
              self.output_types, self.output_classes)), name=name)

      ret = sparse.deserialize_sparse_tensors(
          data_nest.pack_sequence_as(self.output_types, flat_ret),
          self.output_types, self.output_shapes, self.output_classes)

      for tensor, shape in zip(
          data_nest.flatten(ret), data_nest.flatten(self.output_shapes)):
        if isinstance(tensor, ops.Tensor):
          tensor.set_shape(shape)
      flat_result.append(ret)

    return nest.pack_sequence_as(self._devices, flat_result)

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types
# pylint: enable=protected-access


class _PrefetchToDeviceDataset(dataset_ops.Dataset):
  """A `Dataset` whose iterator prefetches elements to other device(s)."""

  def __init__(self, input_dataset, devices, buffer_size):
    self._input_dataset = input_dataset
    self._devices = devices
    self._buffer_size = buffer_size if buffer_size is not None else 1

  def make_one_shot_iterator(self):
    return _PrefetchToDeviceIterator(self._input_dataset, self._devices,
                                     self._buffer_size)

  def make_initializable_iterator(self, shared_name=None):
    raise NotImplementedError("`prefetch_to_devices()` is not currently "
                              "compatible with initializable iterators. Use "
                              "`make_one_shot_iterator()` instead.")

  def _as_variant_tensor(self):
    # TODO(mrry): Raise this error earlier (e.g. when one of the Dataset
    # transformation methods is called.
    # TODO(mrry): Investigate support for chaining further transformations after
    # the prefetch, including GPU support.
    raise NotImplementedError("`prefetch_to_devices()` must be the last "
                              "transformation in a dataset pipeline.")

  # TODO(priyag): Fix the output types, shapes and classes to match the result
  # of get_next (which has the additional nesting layer of devices now).
  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_classes(self):
    return self._input_dataset.output_classes


def prefetch_to_devices(devices, buffer_size=None):
  """A transformation that prefetches dataset values to the given `devices`.

  NOTE: Although the transformation creates a @{tf.data.Dataset}, the
  transformation must be the final `Dataset` in the input pipeline.

  Args:
    devices: A nested structure of devices on which to prefetch the data. It can
      be a single device name, or a tuple or list of device names.
    buffer_size: (Optional.) The number of elements to buffer on each device.
      Defaults to an automatically chosen value.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return _PrefetchToDeviceDataset(dataset, devices, buffer_size)

  return _apply_fn
