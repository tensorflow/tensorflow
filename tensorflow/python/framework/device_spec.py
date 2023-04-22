# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe

# EPU represents for TPU embedding for now. Subject to change in future.
_VALID_DEVICE_TYPES = frozenset({"CPU", "GPU", "TPU", "CUSTOM", "EPU"})


# ==============================================================================
# == Global Implementation Details =============================================
# ==============================================================================
_STRING_TO_COMPONENTS_CACHE = {}
_COMPONENTS_TO_STRING_CACHE = {}


def _as_str_or_none(inp):
  return None if inp is None else str(inp)


def _as_int_or_none(inp):
  return None if inp is None else int(inp)


def _as_device_str_or_none(device_type):
  # For backwards compatibility only, we support lowercase variants of
  # cpu and gpu but turn them into uppercase here.
  if device_type in ("cpu", "gpu"):
    return device_type.upper()
  return _as_str_or_none(device_type)


@tf_export("DeviceSpec", v1=[])
class DeviceSpecV2(object):
  """Represents a (possibly partial) specification for a TensorFlow device.

  `DeviceSpec`s are used throughout TensorFlow to describe where state is stored
  and computations occur. Using `DeviceSpec` allows you to parse device spec
  strings to verify their validity, merge them or compose them programmatically.

  Example:

  ```python
  # Place the operations on device "GPU:0" in the "ps" job.
  device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  with tf.device(device_spec.to_string()):
    # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
    my_var = tf.Variable(..., name="my_variable")
    squared_var = tf.square(my_var)
  ```

  With eager execution disabled (by default in TensorFlow 1.x and by calling
  disable_eager_execution() in TensorFlow 2.x), the following syntax
  can be used:

  ```python
  tf.compat.v1.disable_eager_execution()

  # Same as previous
  device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  # No need of .to_string() method.
  with tf.device(device_spec):
    my_var = tf.Variable(..., name="my_variable")
    squared_var = tf.square(my_var)
  ```

  If a `DeviceSpec` is partially specified, it will be merged with other
  `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
  components defined in inner scopes take precedence over those defined in
  outer scopes.

  ```python
  gpu0_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  with tf.device(DeviceSpec(job="train").to_string()):
    with tf.device(gpu0_spec.to_string()):
      # Nodes created here will be assigned to /job:ps/device:GPU:0.
    with tf.device(DeviceSpec(device_type="GPU", device_index=1).to_string()):
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

  __slots__ = ("_job", "_replica", "_task", "_device_type", "_device_index",
               "_as_string", "_hash")

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
    self._job = _as_str_or_none(job)
    self._replica = _as_int_or_none(replica)
    self._task = _as_int_or_none(task)
    self._device_type = _as_device_str_or_none(device_type)
    self._device_index = _as_int_or_none(device_index)
    self._as_string = self._components_to_string(
        job=self._job, replica=self._replica, task=self._task,
        device_type=self._device_type, device_index=self._device_index)
    self._hash = hash(self.to_string())

  def to_string(self):
    """Return a string representation of this `DeviceSpec`.

    Returns:
      a string of the form
      /job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>.
    """
    return self._as_string

  @classmethod
  def from_string(cls, spec):
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
    return cls(*cls._string_to_components(spec))

  def parse_from_string(self, spec):
    """Parse a `DeviceSpec` name into its components.

    **2.x behavior change**:

    In TensorFlow 1.x, this function mutates its own state and returns itself.
    In 2.x, DeviceSpecs are immutable, and this function will return a
      DeviceSpec which contains the spec.

    * Recommended:

      ```
      # my_spec and my_updated_spec are unrelated.
      my_spec = tf.DeviceSpec.from_string("/CPU:0")
      my_updated_spec = tf.DeviceSpec.from_string("/GPU:0")
      with tf.device(my_updated_spec):
        ...
      ```

    * Will work in 1.x and 2.x (though deprecated in 2.x):

      ```
      my_spec = tf.DeviceSpec.from_string("/CPU:0")
      my_updated_spec = my_spec.parse_from_string("/GPU:0")
      with tf.device(my_updated_spec):
        ...
      ```

    * Will NOT work in 2.x:

      ```
      my_spec = tf.DeviceSpec.from_string("/CPU:0")
      my_spec.parse_from_string("/GPU:0")  # <== Will not update my_spec
      with tf.device(my_spec):
        ...
      ```

    In general, `DeviceSpec.from_string` should completely replace
    `DeviceSpec.parse_from_string`, and `DeviceSpec.replace` should
    completely replace setting attributes directly.

    Args:
      spec: an optional string of the form
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
    return self.from_string(spec)

  def make_merged_spec(self, dev):
    """Returns a new DeviceSpec which incorporates `dev`.

    When combining specs, `dev` will take precedence over the current spec.
    So for instance:
    ```
    first_spec = tf.DeviceSpec(job=0, device_type="CPU")
    second_spec = tf.DeviceSpec(device_type="GPU")
    combined_spec = first_spec.make_merged_spec(second_spec)
    ```

    is equivalent to:
    ```
    combined_spec = tf.DeviceSpec(job=0, device_type="GPU")
    ```

    Args:
      dev: a `DeviceSpec`

    Returns:
      A new `DeviceSpec` which combines `self` and `dev`
    """
    return self.__class__(*self._get_combined_properties(dev))

  def replace(self, **kwargs):
    """Convenience method for making a new DeviceSpec by overriding fields.

    For instance:
    ```
    my_spec = DeviceSpec=(job="my_job", device="CPU")
    my_updated_spec = my_spec.replace(device="GPU")
    my_other_spec = my_spec.replace(device=None)
    ```

    Args:
      **kwargs: This method takes the same args as the DeviceSpec constructor

    Returns:
      A DeviceSpec with the fields specified in kwargs overridden.
    """
    init_kwargs = dict(
        job=self.job, replica=self.replica, task=self.task,
        device_type=self.device_type, device_index=self.device_index)

    # Explicitly provided kwargs take precedence.
    init_kwargs.update(kwargs)
    return self.__class__(**init_kwargs)

  @property
  def job(self):
    return self._job

  @property
  def replica(self):
    return self._replica

  @property
  def task(self):
    return self._task

  @property
  def device_type(self):
    return self._device_type

  @property
  def device_index(self):
    return self._device_index

  def _get_combined_properties(self, dev):
    """Combine the current DeviceSpec with another DeviceSpec.

    The combination of DeviceSpecs is will give priority to dev.

    Args:
      dev: a `DeviceSpec`

    Returns:
      A tuple of (job, replica, task, device_type, device_index) which
      represents the combination of self and dev.
    """
    return (
        dev.job if dev.job is not None else self.job,
        dev.replica if dev.replica is not None else self.replica,
        dev.task if dev.task is not None else self.task,
        dev.device_type if dev.device_type is not None else self.device_type,
        dev.device_index if dev.device_index is not None else self.device_index,
    )

  @staticmethod
  def _get_valid_device_types():
    valid_device_types = set({})
    physical_devices = pywrap_tfe.TF_ListPluggablePhysicalDevices()
    for device in physical_devices:
      valid_device_types.add(device.decode().split(":")[1])
    valid_device_types = valid_device_types | _VALID_DEVICE_TYPES
    return valid_device_types

  @staticmethod
  def _string_to_components(spec=None):
    """Stateless portion of device spec string parsing.

    Args:
      spec: An optional string specifying a device specification.

    Returns:
      The parsed components of `spec`. Note that the result of this function
      must go through attribute setters of DeviceSpec, and should therefore NOT
      be used directly.
    """
    cached_result = _STRING_TO_COMPONENTS_CACHE.get(spec)
    if cached_result is not None:
      return cached_result

    raw_spec = spec  # keep a copy of the original to update the cache
    job, replica, task, device_type, device_index = None, None, None, None, None

    spec = spec or ""
    splits = [x.split(":") for x in spec.split("/")]
    valid_device_types = DeviceSpecV2._get_valid_device_types()
    for y in splits:
      ly = len(y)
      if y:
        # NOTE(taylorrobie): these will go through setters later.
        if ly == 2 and y[0] == "job":
          job = y[1]
        elif ly == 2 and y[0] == "replica":
          replica = y[1]
        elif ly == 2 and y[0] == "task":
          task = y[1]
        elif ((ly == 1 or ly == 2) and (y[0].upper() in valid_device_types)):
          if device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          device_type = y[0].upper()
          if ly == 2 and y[1] != "*":
            device_index = int(y[1])
        elif ly == 3 and y[0] == "device":
          if device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          device_type = y[1]
          if y[2] != "*":
            device_index = int(y[2])
        elif ly and y[0] != "":  # pylint: disable=g-explicit-bool-comparison
          raise ValueError("Unknown attribute: '%s' in '%s'" % (y[0], spec))

    output = (job, replica, task, device_type, device_index)
    _STRING_TO_COMPONENTS_CACHE[raw_spec] = output
    return output

  @staticmethod
  def _components_to_string(job, replica, task, device_type, device_index):
    """Stateless portion of `to_string` (separated to allow caching)."""
    key = (job, replica, task, device_type, device_index)
    cached_result = _COMPONENTS_TO_STRING_CACHE.get(key)
    if cached_result is not None:
      return cached_result

    output = []
    if job is not None:
      output.append("/job:" + job)
    if replica is not None:
      output.append("/replica:" + str(replica))
    if task is not None:
      output.append("/task:" + str(task))
    if device_type is not None:
      device_index_string = "*"
      if device_index is not None:
        # Unlike the others, device_index is stored as an int.
        device_index_string = str(device_index)
      output.append("/device:%s:%s" % (device_type, device_index_string))

    output = "".join(output)
    _COMPONENTS_TO_STRING_CACHE[key] = output
    return output

  def __eq__(self, other):
    """Checks if the `other` DeviceSpec is same as the current instance, eg have

       same value for all the internal fields.

    Args:
      other: Another DeviceSpec

    Returns:
      Return `True` if `other` is also a DeviceSpec instance and has same value
      as the current instance.
      Return `False` otherwise.
    """
    return (isinstance(other, self.__class__) and
            self.to_string() == other.to_string())

  def __hash__(self):
    return self._hash


@tf_export(v1=["DeviceSpec"])  # pylint: disable=missing-docstring
class DeviceSpecV1(DeviceSpecV2):
  __doc__ = DeviceSpecV2.__doc__
  __slots__ = DeviceSpecV2.__slots__

  @DeviceSpecV2.job.setter
  def job(self, job):
    self._job = _as_str_or_none(job)
    self._as_string, self._hash = None, None

  @DeviceSpecV2.replica.setter
  def replica(self, replica):
    self._replica = _as_int_or_none(replica)
    self._as_string, self._hash = None, None

  @DeviceSpecV2.task.setter
  def task(self, task):
    self._task = _as_int_or_none(task)
    self._as_string, self._hash = None, None

  @DeviceSpecV2.device_type.setter
  def device_type(self, device_type):
    self._device_type = _as_device_str_or_none(device_type)
    self._as_string, self._hash = None, None

  @DeviceSpecV2.device_index.setter
  def device_index(self, device_index):
    self._device_index = _as_int_or_none(device_index)
    self._as_string, self._hash = None, None

  def __hash__(self):
    if self._hash is None:
      self._hash = hash(self.to_string())
    return self._hash

  def to_string(self):
    if self._as_string is None:
      self._as_string = self._components_to_string(
          job=self.job, replica=self.replica, task=self.task,
          device_type=self.device_type, device_index=self.device_index)
    return self._as_string

  def parse_from_string(self, spec):
    (self.job, self.replica, self.task, self.device_type, self.device_index
    ) = self._string_to_components(spec)

    return self

  def merge_from(self, dev):
    """Merge the properties of "dev" into this `DeviceSpec`.

    Note: Will be removed in TensorFlow 2.x since DeviceSpecs will become
          immutable.

    Args:
      dev: a `DeviceSpec`.
    """
    (self.job, self.replica, self.task, self.device_type, self.device_index
    ) = self._get_combined_properties(dev)

  # Use parent class docstrings for public methods.
  to_string.__doc__ = DeviceSpecV2.to_string.__doc__
  parse_from_string.__doc__ = DeviceSpecV2.parse_from_string.__doc__
