# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""A wrapper for gen_sleep_op.py.

This defines a public API (and provides a docstring for it) for the C++ Op
defined by sleep_kernel.cc
"""

import tensorflow as tf
from tensorflow.python.platform import resource_loader

_sleep_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("sleep_kernel.so"))

examples_async_sleep = _sleep_module.examples_async_sleep
examples_sync_sleep = _sleep_module.examples_sync_sleep


# In this example, this Python function is a trivial wrapper for the C++ Op:
# it provides a public API and docstring that are equivalent to the API
# and documentation of the C++ op. The motivation for it is to be a placeholder
# that allows a wider variety of non-breaking future changes than are possible
# with the generated wrapper alone. Having this wrapper is optional.
def AsyncSleep(delay, name=None):
  """Pause for `delay` seconds (which need not be an integer).

  This is an asynchronous (non-blocking) version of a sleep op. It includes
  any time spent being blocked by another thread in `delay`. If it is blocked
  for a fraction of the time specified by `delay`, it only calls `sleep`
  (actually `usleep`) only for the remainder. If it is blocked for the full
  time specified by `delay` or more, it returns without explicitly calling
  `sleep`.

  Args:
    delay: tf.Tensor which is a scalar of type float.
    name: An optional name for the op.

  Returns:
    The `delay` value.
  """
  return examples_async_sleep(delay=delay, name=name)


def SyncSleep(delay, name=None):
  """Pause for `delay` seconds (which need not be an integer).

  This is a synchronous (blocking) version of a sleep op. It's purpose is
  to be contrasted with Examples>AsyncSleep.

  Args:
    delay: tf.Tensor which is a scalar of type float.
    name: An optional name for the op.

  Returns:
    The `delay` value.
  """
  return examples_sync_sleep(delay=delay, name=name)
