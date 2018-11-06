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
    d: a device string.
    default: a string for default device if d doesn't have all components.

  Returns:
    a canonicalized device string.
  """
  d = tf_device.DeviceSpec.from_string(d)
  assert d.device_type is None or d.device_type == d.device_type.upper(), (
      "Device type '%s' must be all-caps." % (d.device_type,))
  # Fill in missing device fields using defaults.
  result = tf_device.DeviceSpec(
      replica=0, task=0, device_type="CPU", device_index=0)
  if ops.executing_eagerly_outside_functions():
    result.job = "localhost"
  if default:
    result.merge_from(tf_device.DeviceSpec.from_string(default))
  result.merge_from(d)
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


def current():
  """Return a string (not canonicalized) for the current device."""
  # TODO(josh11b): Work out how this function interacts with ops.colocate_with.
  ctx = context.context()
  if ctx.executing_eagerly():
    d = ctx.device_name
  else:
    op = _FakeOperation()
    ops.get_default_graph()._apply_device_functions(op)  # pylint: disable=protected-access
    d = op.device
  return d
