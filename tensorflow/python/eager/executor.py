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
"""Executor for eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tfe


class Executor(object):
  """A class for handling eager execution.

  The default behavior for asynchronous execution is to serialize all ops on
  a single thread. Having different `Executor` objects in different threads
  enables executing ops asynchronously in parallel:

  ```python
  def thread_function():
    executor = executor.Executor(enable_async=True):
    context.set_executor(executor)

  a = threading.Thread(target=thread_function)
  a.start()
  b = threading.Thread(target=thread_function)
  b.start()
  ```
  """

  __slots__ = ["_handle"]

  def __init__(self, handle):
    self._handle = handle

  def __del__(self):
    try:
      # pywrap_tfe.TFE_ExecutorWaitForAllPendingNodes(self._handle)
      pywrap_tfe.TFE_DeleteExecutor(self._handle)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the pywrap module
      # already being unloaded, self._handle. no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.

  def is_async(self):
    return pywrap_tfe.TFE_ExecutorIsAsync(self._handle)

  def handle(self):
    return self._handle

  def wait(self):
    """Waits for ops dispatched in this executor to finish."""
    pywrap_tfe.TFE_ExecutorWaitForAllPendingNodes(self._handle)

  def clear_error(self):
    """Clears errors raised in this executor during execution."""
    pywrap_tfe.TFE_ExecutorClearError(self._handle)


def new_executor(enable_async):
  handle = pywrap_tfe.TFE_NewExecutor(enable_async)
  return Executor(handle)
