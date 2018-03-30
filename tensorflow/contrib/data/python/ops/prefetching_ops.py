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

import warnings

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops


# TODO(rohanj): Add a python class that constructs resource in the __init__
# method and provides a get_next() that calls the prefetch op.
def function_buffering_resource(string_arg,
                                target_device,
                                f,
                                buffer_size,
                                container="",
                                shared_name=None,
                                name=None):
  if shared_name is None:
    shared_name = ""
  return gen_dataset_ops.function_buffering_resource(
      string_arg=string_arg,
      target_device=target_device,
      shared_name=shared_name,
      f=f,
      buffer_size=buffer_size,
      container=container,
      name=name)


def function_buffering_resource_get_next(function_buffer_resource,
                                         output_types,
                                         name=None):
  return gen_dataset_ops.function_buffering_resource_get_next(
      function_buffer_resource=function_buffer_resource,
      output_types=output_types,
      name=name)


def function_buffering_resource_reset(function_buffer_resource, name=None):
  return gen_dataset_ops.function_buffering_resource_reset(
      function_buffer_resource=function_buffer_resource, name=name)


# pylint: disable=protected-access
class _PrefetchToDeviceIterator(object):
  """A replacement for @{tf.data.Iterator} that prefetches to another device."""

  def __init__(self, input_dataset, device, buffer_size):
    self._input_dataset = input_dataset
    self._get_next_call_count = 0
    input_iterator = input_dataset.make_one_shot_iterator()
    input_iterator_handle = input_iterator.string_handle()

    @function.Defun(dtypes.string)
    def _prefetch_fn(handle):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          handle, input_iterator.output_types, input_iterator.output_shapes,
          input_iterator.output_classes)
      return remote_iterator.get_next()

    with ops.device(device):
      self._buffering_resource = function_buffering_resource(
          f=_prefetch_fn,
          target_device=gen_dataset_ops.iterator_get_device(
              input_iterator._iterator_resource),
          string_arg=input_iterator_handle,
          buffer_size=buffer_size)

  def get_next(self, name=None):
    """See @{tf.data.Iterator.get_next}."""
    self._get_next_call_count += 1
    if self._get_next_call_count > iterator_ops.GET_NEXT_CALL_WARNING_THRESHOLD:
      warnings.warn(iterator_ops.GET_NEXT_CALL_WARNING_MESSAGE)

    flat_ret = gen_dataset_ops.function_buffering_resource_get_next(
        self._buffering_resource,
        output_types=nest.flatten(sparse.as_dense_types(
            self.output_types, self.output_classes)), name=name)

    ret = sparse.deserialize_sparse_tensors(
        nest.pack_sequence_as(self.output_types, flat_ret),
        self.output_types, self.output_shapes, self.output_classes)

    for tensor, shape in zip(
        nest.flatten(ret), nest.flatten(self.output_shapes)):
      if isinstance(tensor, ops.Tensor):
        tensor.set_shape(shape)

    return ret

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
  """A `Dataset` whose iterator prefetches elements to another device."""

  def __init__(self, input_dataset, device, buffer_size):
    self._input_dataset = input_dataset
    self._device = device
    self._buffer_size = buffer_size if buffer_size is not None else 1

  def make_one_shot_iterator(self):
    return _PrefetchToDeviceIterator(self._input_dataset, self._device,
                                     self._buffer_size)

  def make_initializable_iterator(self, shared_name=None):
    raise NotImplementedError("`prefetch_to_device()` is not currently "
                              "compatible with initializable iterators. Use "
                              "`make_one_shot_iterator()` instead.")

  def _as_variant_tensor(self):
    # TODO(mrry): Raise this error earlier (e.g. when one of the Dataset
    # transformation methods is called.
    # TODO(mrry): Investigate support for chaining further transformations after
    # the prefetch, including GPU support.
    raise NotImplementedError("`prefetch_to_device()` must be the last "
                              "transformation in a dataset pipeline.")

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_classes(self):
    return self._input_dataset.output_classes


def prefetch_to_device(device, buffer_size=None):
  """A transformation that prefetches dataset values to the given `device`.

  NOTE: Although the transformation creates a @{tf.data.Dataset}, the
  transformation must be the final `Dataset` in the input pipeline.

  Args:
    device: A string. The name of a device to which elements will be prefetched.
    buffer_size: (Optional.) The number of elements to buffer on `device`.
      Defaults to an automatically chosen value.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return _PrefetchToDeviceDataset(dataset, device, buffer_size)

  return _apply_fn
