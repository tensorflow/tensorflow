# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Class to represent a device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading
from tensorflow.python.util.tf_export import tf_export


@tf_export("DeviceSpec")
class DeviceSpec(object):
  """Represents a (possibly partial) specification for a TensorFlow device.

  `DeviceSpec`s are used throughout TensorFlow to describe where state is stored
  and computations occur. Using `DeviceSpec` allows you to parse device spec
  strings to verify their validity, merge them or compose them programmatically.

  Example:

  ```python
  # Place the operations on device "GPU:0" in the "ps" job.
  device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  with tf.device(device_spec):
    # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
    my_var = tf.Variable(..., name="my_variable")
    squared_var = tf.square(my_var)
  ```

  If a `DeviceSpec` is partially specified, it will be merged with other
  `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
  components defined in inner scopes take precedence over those defined in
  outer scopes.

  ```python
  with tf.device(DeviceSpec(job="train", )):
    with tf.device(DeviceSpec(job="ps", device_type="GPU", device_index=0):
      # Nodes created here will be assigned to /job:ps/device:GPU:0.
    with tf.device(DeviceSpec(device_type="GPU", device_index=1):
      # Nodes created here will be assigned to /job:train/device:GPU:1.
  ```

  A `DeviceSpec` consists of 5 components -- each of
  which is optionally specified:

  * Job: The job name.
  * Replica: The replica index.
  * Task: The task index.
  * Device type: The device type string (e.g. "CPU" or "GPU").
  * Device index: The device index.
  """

  def __init__(self, job=None, replica=None, task=None, device_type=None,
               device_index=None):
    """Create a new `DeviceSpec` object.

    Args:
      job: string.  Optional job name.
      replica: int.  Optional replica index.
      task: int.  Optional task index.
      device_type: Optional device type string (e.g. "CPU" or "GPU")
      device_index: int.  Optional device index.  If left
        unspecified, device represents 'any' device_index.
    """
    self.job = job
    self.replica = replica
    self.task = task
    if device_type == "cpu" or device_type == "gpu":
      # For backwards compatibility only, we support lowercase variants of
      # cpu and gpu but turn them into uppercase here.
      self.device_type = device_type.upper()
    else:
      self.device_type = device_type
    self.device_index = device_index
    self._hash = hash(self.to_string())

  def _clear(self):
    self._job = None
    self._replica = None
    self._task = None
    self.device_type = None
    self.device_index = None

  @property
  def job(self):
    return self._job

  @job.setter
  def job(self, job):
    if job is not None:
      self._job = str(job)
    else:
      self._job = None

  @property
  def replica(self):
    return self._replica

  @replica.setter
  def replica(self, replica):
    if replica is not None:
      self._replica = int(replica)
    else:
      self._replica = None

  @property
  def task(self):
    return self._task

  @task.setter
  def task(self, task):
    if task is not None:
      self._task = int(task)
    else:
      self._task = None

  def parse_from_string(self, spec):
    """Parse a `DeviceSpec` name into its components.

    Args:
      spec: a string of the form
       /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
      or
       /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
      as cpu and gpu are mutually exclusive.
      All entries are optional.

    Returns:
      The `DeviceSpec`.

    Raises:
      ValueError: if the spec was not valid.
    """
    self._clear()
    splits = [x.split(":") for x in spec.split("/")]
    for y in splits:
      ly = len(y)
      if y:
        # NOTE(touts): we use the property getters here.
        if ly == 2 and y[0] == "job":
          self.job = y[1]
        elif ly == 2 and y[0] == "replica":
          self.replica = y[1]
        elif ly == 2 and y[0] == "task":
          self.task = y[1]
        elif ((ly == 1 or ly == 2) and
              ((y[0].upper() == "GPU") or (y[0].upper() == "CPU"))):
          if self.device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          self.device_type = y[0].upper()
          if ly == 2 and y[1] != "*":
            self.device_index = int(y[1])
        elif ly == 3 and y[0] == "device":
          if self.device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          self.device_type = y[1]
          if y[2] != "*":
            self.device_index = int(y[2])
        elif ly and y[0] != "":  # pylint: disable=g-explicit-bool-comparison
          raise ValueError("Unknown attribute: '%s' in '%s'" % (y[0], spec))

    return self

  def merge_from(self, dev):
    """Merge the properties of "dev" into this `DeviceSpec`.

    Args:
      dev: a `DeviceSpec`.
    """
    if dev.job is not None:
      self.job = dev.job
    if dev.replica is not None:
      self.replica = dev.replica
    if dev.task is not None:
      self.task = dev.task
    if dev.device_type is not None:
      self.device_type = dev.device_type
    if dev.device_index is not None:
      self.device_index = dev.device_index

  def to_string(self):
    """Return a string representation of this `DeviceSpec`.

    Returns:
      a string of the form
      /job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>.
    """
    dev = ""
    if self.job is not None:
      dev += "/job:" + self.job
    if self.replica is not None:
      dev += "/replica:" + str(self.replica)
    if self.task is not None:
      dev += "/task:" + str(self.task)
    if self.device_type is not None:
      device_index_string = "*"
      if self.device_index is not None:
        device_index_string = str(self.device_index)
      dev += "/device:%s:%s" % (self.device_type, device_index_string)
    return dev

  @staticmethod
  def from_string(spec):
    """Construct a `DeviceSpec` from a string.

    Args:
      spec: a string of the form
       /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
      or
       /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
      as cpu and gpu are mutually exclusive.
      All entries are optional.

    Returns:
      A DeviceSpec.
    """
    return DeviceSpec().parse_from_string(spec)

  def __eq__(self, other):
    return self.to_string() == other.to_string()

  def __hash__(self):
    return self._hash


def check_valid(spec):
  """Check that a device spec is valid.

  Args:
    spec: a string.

  Raises:
    An exception if the spec is invalid.
  """
  # Construct a DeviceSpec.  It will assert a failure if spec is invalid.
  DeviceSpec.from_string(spec)


def canonical_name(device):
  """Returns a canonical name for the given `DeviceSpec` or device name."""
  if device is None:
    return ""
  if isinstance(device, DeviceSpec):
    return device.to_string()
  else:
    device = DeviceSpec.from_string(device)
    return device.to_string()


# Cache from DeviceSpec objects to their corresponding device functions.
# This cache is maintained for correctness, not performance: it makes it
# possible to compare the device function stacks belonging to different
# graphs in a meaningful way.
_cached_device_functions = {}
_cached_device_specs = {}
_cache_lock = threading.Lock()


def merge_device(spec):
  """Returns a device function that merges devices specifications.

  This can be used to merge partial specifications of devices. The
  innermost setting for a device field takes precedence. For example:

    with tf.device(merge_device("/device:GPU:0"))
      # Nodes created here have device "/device:GPU:0"
      with tf.device(merge_device("/job:worker")):
        # Nodes created here have device "/job:worker/device:GPU:0"
        with tf.device(merge_device("/device:CPU:0")):
          # Nodes created here have device "/job:worker/device:CPU:0"
          with tf.device(merge_device("/job:ps")):
            # Nodes created here have device "/job:ps/device:CPU:0"

  Args:
    spec: A `DeviceSpec` or a device spec string (partially) describing the
      device that should be used for all nodes created in the scope of
      the returned device function's with block.

  Returns:
    A device function with the above-described behavior.

  Raises:
    ValueError: if the spec was not valid.
  """
  with _cache_lock:
    if not isinstance(spec, DeviceSpec):
      cached_device_spec = _cached_device_specs.get(spec, None)
      if cached_device_spec is None:
        device_spec = DeviceSpec.from_string(spec or "")
        _cached_device_specs[spec] = device_spec
        spec = device_spec
      else:
        spec = cached_device_spec
    cached_function = _cached_device_functions.get(spec, None)
    if cached_function is not None:
      return cached_function

    def _device_function(node_def):
      current_device = DeviceSpec.from_string(node_def.device or "")
      copy_spec = copy.copy(spec)
      copy_spec.merge_from(current_device)  # current_device takes precedence.
      return copy_spec

    _cached_device_functions[spec] = _device_function
    return _device_function
