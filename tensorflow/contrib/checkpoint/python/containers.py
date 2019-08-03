"""Trackable data structures."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.tracking import base as trackable_lib
from tensorflow.python.training.tracking import data_structures


class UniqueNameTracker(data_structures.TrackableDataStructure):
  """Adds dependencies on trackable objects with name hints.

  Useful for creating dependencies with locally unique names.

  Example usage:
  ```python
  class SlotManager(tf.contrib.checkpoint.Checkpointable):

    def __init__(self):
      # Create a dependency named "slotdeps" on the container.
      self.slotdeps = tf.contrib.checkpoint.UniqueNameTracker()
      slotdeps = self.slotdeps
      slots = []
      slots.append(slotdeps.track(tf.Variable(3.), "x"))  # Named "x"
      slots.append(slotdeps.track(tf.Variable(4.), "y"))
      slots.append(slotdeps.track(tf.Variable(5.), "x"))  # Named "x_1"
  ```
  """

  def __init__(self):
    super(UniqueNameTracker, self).__init__()
    self._maybe_initialize_trackable()
    self._name_counts = {}

  @property
  def _values(self):
    return [dep.ref for dep in self._checkpoint_dependencies]

  def track(self, trackable, base_name):
    """Add a dependency on `trackable`.

    Args:
      trackable: An object to add a checkpoint dependency on.
      base_name: A name hint, which is uniquified to determine the dependency
        name.
    Returns:
      `trackable`, for chaining.
    Raises:
      ValueError: If `trackable` is not a trackable object.
    """

    if not isinstance(trackable, trackable_lib.Trackable):
      raise ValueError(
          ("Expected a trackable value, got %s which does not inherit "
           "from tf.track.Trackable.") % (trackable,))

    def _format_name(prefix, number):
      if number > 0:
        return "%s_%d" % (prefix, number)
      else:
        return prefix

    count = self._name_counts.get(base_name, 0)
    candidate = _format_name(base_name, count)
    while self._lookup_dependency(candidate) is not None:
      count += 1
      candidate = _format_name(base_name, count)
    self._name_counts[base_name] = count + 1
    self._track_value(trackable, name=candidate)
    return trackable
