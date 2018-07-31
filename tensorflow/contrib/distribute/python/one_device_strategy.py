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
"""Class OneDeviceStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.distribute.python import values
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import distribute as distribute_lib


# TODO(josh11b): Replace asserts in this file with if ...: raise ...


class OneDeviceStrategy(distribute_lib.DistributionStrategy):
  """A distribution strategy for running on a single device."""
  # TODO(josh11b): Do we wrap values in types to generate errors if you are
  # doing something that won't work with other DistributionStrategy
  # implementations?

  def __init__(self, device, prefetch_on_device=None):
    super(OneDeviceStrategy, self).__init__()
    self._device = device
    self._prefetch_on_device = prefetch_on_device
    self._default_device = device

  def _create_variable(self, next_creator, *args, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      with ops.device(self._device):
        return next_creator(*args, **kwargs)
    if isinstance(colocate_with, six.string_types):
      with ops.device(colocate_with):
        return next_creator(*args, **kwargs)
    if (isinstance(colocate_with, list) and len(colocate_with) == 1 and
        isinstance(colocate_with[0], six.string_types)):
      with ops.device(colocate_with[0]):
        return next_creator(*args, **kwargs)
    with ops.colocate_with(colocate_with):
      return next_creator(*args, **kwargs)

  def distribute_dataset(self, dataset_fn):
    return values.PerDeviceDataset(
        self._call_dataset_fn(dataset_fn), [self._device],
        self._prefetch_on_device)

  def _broadcast(self, tensor, destinations):
    return tensor

  def _call_for_each_tower(self, fn, *args, **kwargs):
    # We don't run `fn` in multiple threads in OneDeviceStrategy.
    kwargs.pop("run_concurrently", None)
    with ops.device(self._device), _OneDeviceTowerContext(self):
      return fn(*args, **kwargs)

  def map(self, map_over, fn, *args, **kwargs):
    with ops.device(self._device):
      return values.MapOutput([fn(m, *args, **kwargs) for m in map_over])

  def _reduce(self, aggregation, value, destinations):
    if not isinstance(value, values.MapOutput):
      return value
    l = value.get()
    assert l
    with ops.device(self._device):
      if aggregation == vs.VariableAggregation.SUM:
        return math_ops.add_n(l)
      elif aggregation == vs.VariableAggregation.MEAN:
        return math_ops.add_n(l) / len(l)
      else:
        assert False

  def _update(self, var, fn, *args, **kwargs):
    with ops.device(self._device), distribute_lib.UpdateContext(self._device):
      return fn(var, *args, **kwargs)

  def _update_non_slot(self, colocate_with, fn, *args, **kwargs):
    del colocate_with
    with ops.device(self._device), distribute_lib.UpdateContext(self._device):
      return fn(*args, **kwargs)

  def read_var(self, tower_local_var):
    """Read the aggregate value of a tower-local variable."""
    return array_ops.identity(tower_local_var)

  def _unwrap(self, value):
    return [value]

  @property
  def is_single_tower(self):
    return True

  @property
  def num_towers(self):
    return 1

  @property
  def worker_devices(self):
    return [self._device]

  @property
  def parameter_devices(self):
    return [self._device]

  def non_slot_devices(self, var_list):
    del var_list
    return [self._device]

  def _worker_device_index(self):
    return 0


class _OneDeviceTowerContext(distribute_lib.TowerContext):

  def __init__(self, distribution_strategy):
    distribute_lib.TowerContext.__init__(
        self, distribution_strategy, tower_id=0)

  @property
  def device(self):
    return self._distribution_strategy.worker_devices[0]
