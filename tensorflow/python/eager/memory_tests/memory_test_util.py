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
"""Utils for memory tests."""

import collections
import gc
import time

from tensorflow.python.eager import context

# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


def _instance_count_by_class():
  counter = collections.Counter()

  for obj in gc.get_objects():
    try:
      counter[obj.__class__.__name__] += 1
    except Exception:  # pylint:disable=broad-except
      pass

  return counter


def assert_no_leak(f, num_iters=100000, increase_threshold_absolute_mb=10):
  """Assert memory usage doesn't increase beyond given threshold for f."""

  with context.eager_mode():
    # Warm up.
    f()

    # Wait for background threads to start up and take over memory.
    # FIXME: The nature of this test leaves few other options. Maybe there
    # is a better way to do this.
    time.sleep(4)

    gc.collect()
    initial = memory_profiler.memory_usage(-1)[0]
    instance_count_by_class_before = _instance_count_by_class()

    for _ in range(num_iters):
      f()

    gc.collect()
    increase = memory_profiler.memory_usage(-1)[0] - initial

    assert increase < increase_threshold_absolute_mb, (
        "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
        "Maximum allowed increase: %f MB. "
        "Instance count diff before/after: %s") % (
            initial, increase, increase_threshold_absolute_mb,
            _instance_count_by_class() - instance_count_by_class_before)


def memory_profiler_is_available():
  return memory_profiler is not None
