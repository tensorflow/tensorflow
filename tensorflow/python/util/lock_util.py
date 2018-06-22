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
"""Locking related utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


class GroupLock(object):
  """A lock to allow many members of a group to access a resource exclusively.

  This lock provides a way to allow access to a resource by multiple threads
  belonging to a logical group at the same time, while restricting access to
  threads from all other groups. You can think of this as an extension of a
  reader-writer lock, where you allow multiple writers at the same time. We
  made it generic to support multiple groups instead of just two - readers and
  writers.

  Simple usage example with two groups accessing the same resource:

  ```python
  lock = GroupLock(num_groups=2)

  # In a member of group 0:
  with lock.group(0):
    # do stuff, access the resource
    # ...

  # In a member of group 1:
  with lock.group(1):
    # do stuff, access the resource
    # ...
  ```

  Using as a context manager with `.group(group_id)` is the easiest way. You
  can also use the `acquire` and `release` method directly.
  """

  def __init__(self, num_groups=2):
    """Initialize a group lock.

    Args:
      num_groups: The number of groups that will be accessing the resource under
        consideration. Should be a positive number.

    Returns:
      A group lock that can then be used to synchronize code.

    Raises:
      ValueError: If num_groups is less than 1.
    """
    if num_groups < 1:
      raise ValueError("num_groups must be a positive integer, got {}".format(
          num_groups))
    self._ready = threading.Condition(threading.Lock())
    self._num_groups = num_groups
    self._group_member_counts = [0] * self._num_groups

  def group(self, group_id):
    """Enter a context where the lock is with group `group_id`.

    Args:
      group_id: The group for which to acquire and release the lock.

    Returns:
      A context manager which will acquire the lock for `group_id`.
    """
    self._validate_group_id(group_id)
    return self._Context(self, group_id)

  def acquire(self, group_id):
    """Acquire the group lock for a specific group `group_id`."""
    self._validate_group_id(group_id)

    self._ready.acquire()
    while self._another_group_active(group_id):
      self._ready.wait()
    self._group_member_counts[group_id] += 1
    self._ready.release()

  def release(self, group_id):
    """Release the group lock for a specific group `group_id`."""
    self._validate_group_id(group_id)

    self._ready.acquire()
    self._group_member_counts[group_id] -= 1
    if self._group_member_counts[group_id] == 0:
      self._ready.notifyAll()
    self._ready.release()

  def _another_group_active(self, group_id):
    return any(
        c > 0 for g, c in enumerate(self._group_member_counts) if g != group_id)

  def _validate_group_id(self, group_id):
    if group_id < 0 or group_id >= self._num_groups:
      raise ValueError(
          "group_id={} should be between 0 and num_groups={}".format(
              group_id, self._num_groups))

  class _Context(object):
    """Context manager helper for `GroupLock`."""

    def __init__(self, lock, group_id):
      self._lock = lock
      self._group_id = group_id

    def __enter__(self):
      self._lock.acquire(self._group_id)

    def __exit__(self, type_arg, value_arg, traceback_arg):
      del type_arg, value_arg, traceback_arg
      self._lock.release(self._group_id)
