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
"""Device-related support functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops


def canonicalize(d, default=None):
  """Canonicalize device string.

  If d has missing components, the rest would be deduced from the `default`
  argument or from '/replica:0/task:0/device:CPU:0'. For example:
    If d = '/cpu:0', default='/job:worker/task:1', it returns
      '/job:worker/replica:0/task:1/device:CPU:0'.
    If d = '/cpu:0', default='/job:worker', it returns
      '/job:worker/replica:0/task:0/device:CPU:0'.
    If d = '/gpu:0', default=None, it returns
      '/replica:0/task:0/device:GPU:0'.

  Note: This uses "job:localhost" as the default if executing eagerly.

  Args:
    d: a device string or tf.config.LogicalDevice
    default: a string for default device if d doesn't have all components.

  Returns:
    a canonicalized device string.
  """
  if isinstance(d, context.LogicalDevice):
    d = tf_device.DeviceSpec.from_string(d.name)
  else:
    d = tf_device.DeviceSpec.from_string(d)

  assert d.device_type is None or d.device_type == d.device_type.upper(), (
      "Device type '%s' must be all-caps." % (d.device_type,))
  # Fill in missing device fields using defaults.
  result = tf_device.DeviceSpec(
      replica=0, task=0, device_type="CPU", device_index=0)
  if ops.executing_eagerly_outside_functions():
    # Try to deduce job, replica and task in case it's in a multi worker setup.
    # TODO(b/151452748): Using list_logical_devices is not always safe since it
    # may return remote devices as well, but we're already doing this elsewhere.
    host_cpu = tf_device.DeviceSpec.from_string(
        config.list_logical_devices("CPU")[0].name)
    if host_cpu.job:
      result = result.make_merged_spec(host_cpu)
    else:
      # The default job is localhost if eager execution is enabled
      result = result.replace(job="localhost")
  if default:
    # Overrides any defaults with values from the default device if given.
    result = result.make_merged_spec(
        tf_device.DeviceSpec.from_string(default))

  # Apply `d` last, so that it's values take precedence over the defaults.
  result = result.make_merged_spec(d)
  return result.to_string()


def resolve(d):
  """Canonicalize `d` with current device as default."""
  return canonicalize(d, default=current())


class _FakeNodeDef(object):
  """A fake NodeDef for _FakeOperation."""

  def __init__(self):
    self.op = ""
    self.name = ""


class _FakeOperation(object):
  """A fake Operation object to pass to device functions."""

  def __init__(self):
    self.device = ""
    self.type = ""
    self.name = ""
    self.node_def = _FakeNodeDef()

  def _set_device(self, device):
    self.device = ops._device_string(device)  # pylint: disable=protected-access

  def _set_device_from_string(self, device_str):
    self.device = device_str


def current():
  """Return a string (not canonicalized) for the current device."""
  # TODO(josh11b): Work out how this function interacts with ops.colocate_with.
  if ops.executing_eagerly_outside_functions():
    d = context.context().device_name
  else:
    op = _FakeOperation()
    ops.get_default_graph()._apply_device_functions(op)  # pylint: disable=protected-access
    d = op.device
  return d


def get_host_for_device(device):
  """Returns the corresponding host device for the given device."""
  spec = tf_device.DeviceSpec.from_string(device)
  return tf_device.DeviceSpec(
      job=spec.job, replica=spec.replica, task=spec.task,
      device_type="CPU", device_index=0).to_string()


def local_devices_from_num_gpus(num_gpus):
  """Returns device strings for local GPUs or CPU."""
  return (tuple("/device:GPU:%d" % i for i in range(num_gpus)) or
          ("/device:CPU:0",))
