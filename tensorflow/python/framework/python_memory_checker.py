# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python memory leak detection utility.

Please don't use this class directly.  Instead, use `MemoryChecker` wrapper.
"""

import collections
import copy
import gc

from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace


def _get_typename(obj):
  """Return human readable pretty type name string."""
  objtype = type(obj)
  name = objtype.__name__
  module = getattr(objtype, '__module__', None)
  if module:
    return '{}.{}'.format(module, name)
  else:
    return name


def _create_python_object_snapshot():
  gc.collect()
  all_objects = gc.get_objects()
  result = collections.defaultdict(set)
  for obj in all_objects:
    result[_get_typename(obj)].add(id(obj))
  return result


def _snapshot_diff(old_snapshot, new_snapshot, exclude_ids):
  result = collections.Counter()
  for new_name, new_ids in new_snapshot.items():
    old_ids = old_snapshot[new_name]
    result[new_name] = len(new_ids - exclude_ids) - len(old_ids - exclude_ids)

  # This removes zero or negative value entries.
  result += collections.Counter()
  return result


class _PythonMemoryChecker(object):
  """Python memory leak detection class."""

  def __init__(self):
    self._snapshots = []
    # cache the function used by mark_stack_trace_and_call to avoid
    # contaminating the leak measurement.
    def _record_snapshot():
      self._snapshots.append(_create_python_object_snapshot())

    self._record_snapshot = _record_snapshot

  # We do not enable trace_wrapper on this function to avoid contaminating
  # the snapshot.
  def record_snapshot(self):
    # Function called using `mark_stack_trace_and_call` will have
    # "_python_memory_checker_helper" string in the C++ stack trace.  This will
    # be used to filter out C++ memory allocations caused by this function,
    # because we are not interested in detecting memory growth caused by memory
    # checker itself.
    _python_memory_checker_helper.mark_stack_trace_and_call(
        self._record_snapshot)

  @trace.trace_wrapper
  def report(self):
    # TODO(kkb): Implement.
    pass

  @trace.trace_wrapper
  def assert_no_leak_if_all_possibly_except_one(self):
    """Raises an exception if a leak is detected.

    This algorithm classifies a series of allocations as a leak if it's the same
    type at every snapshot, but possibly except one snapshot.
    """

    snapshot_diffs = []
    for i in range(0, len(self._snapshots) - 1):
      snapshot_diffs.append(self._snapshot_diff(i, i + 1))

    allocation_counter = collections.Counter()
    for diff in snapshot_diffs:
      for name, count in diff.items():
        if count > 0:
          allocation_counter[name] += 1

    leaking_object_names = {
        name for name, count in allocation_counter.items()
        if count >= len(snapshot_diffs) - 1
    }

    if leaking_object_names:
      object_list_to_print = '\n'.join(
          [' - ' + name for name in leaking_object_names])
      raise AssertionError(
          'These Python objects were allocated in every snapshot possibly '
          f'except one.\n\n{object_list_to_print}')

  @trace.trace_wrapper
  def assert_no_new_objects(self, threshold=None):
    """Assert no new Python objects."""

    if not threshold:
      threshold = {}

    count_diff = self._snapshot_diff(0, -1)
    original_count_diff = copy.deepcopy(count_diff)
    count_diff.subtract(collections.Counter(threshold))

    if max(count_diff.values() or [0]) > 0:
      raise AssertionError('New Python objects created exceeded the threshold.'
                           '\nPython object threshold:\n'
                           f'{threshold}\n\nNew Python objects:\n'
                           f'{original_count_diff.most_common()}')
    elif min(count_diff.values(), default=0) < 0:
      logging.warning('New Python objects created were less than the threshold.'
                      '\nPython object threshold:\n'
                      f'{threshold}\n\nNew Python objects:\n'
                      f'{original_count_diff.most_common()}')

  @trace.trace_wrapper
  def _snapshot_diff(self, old_index, new_index):
    return _snapshot_diff(self._snapshots[old_index],
                          self._snapshots[new_index],
                          self._get_internal_object_ids())

  @trace.trace_wrapper
  def _get_internal_object_ids(self):
    ids = set()
    for snapshot in self._snapshots:
      ids.add(id(snapshot))
      for v in snapshot.values():
        ids.add(id(v))
    return ids
