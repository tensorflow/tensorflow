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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six

from tensorflow.python.eager import context

# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


def assert_no_leak(f, num_iters=100000, increase_threshold_absolute_mb=10):
  """Assert memory usage doesn't increase beyond given threshold for f."""

  with context.eager_mode():
    # Warm up.
    f()

    # Wait for background threads to start up and take over memory.
    # FIXME: The nature of this test leaves few other options. Maybe there
    # is a better way to do this.
    time.sleep(4)

    initial = memory_profiler.memory_usage(-1)[0]

    for _ in six.moves.range(num_iters):
      f()

    increase = memory_profiler.memory_usage(-1)[0] - initial

    assert increase < increase_threshold_absolute_mb, (
        "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
        "Maximum allowed increase: %f") % (initial, increase,
                                           increase_threshold_absolute_mb)


def memory_profiler_is_available():
  return memory_profiler is not None
