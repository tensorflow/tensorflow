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
"""Cancellation support for eager execution."""

import functools
import threading
from typing import Any, Callable, Optional

from tensorflow.python import pywrap_tfe


class CancellationManager:
  """A mechanism for cancelling blocking computation."""

  __slots__ = ["_impl"]

  def __init__(self):
    self._impl = pywrap_tfe.TFE_NewCancellationManager()

  @property
  def is_cancelled(self):
    """Returns `True` if `CancellationManager.start_cancel` has been called."""
    return pywrap_tfe.TFE_CancellationManagerIsCancelled(self._impl)

  def start_cancel(self):
    """Cancels blocking operations that have been registered with this object."""
    pywrap_tfe.TFE_CancellationManagerStartCancel(self._impl)

  def get_cancelable_function(
      self, concrete_function: Callable[..., Any]
  ) -> Callable[..., Any]:
    """Wraps a ConcreteFunction to execute within this cancellation manager's context.

    Args:
      concrete_function: The ConcreteFunction to wrap.

    Returns:
      A callable that executes the concrete_function under this cancellation
      context.
    """

    @functools.wraps(concrete_function)
    def cancellable(*args, **kwargs):
      with CancellationManagerContext(self):
        return concrete_function(*args, **kwargs)

    return cancellable


_active_context = threading.local()


def context() -> Optional[CancellationManager]:
  """Returns the CancellationManager active in the current thread, or None."""
  stack = getattr(_active_context, "manager_stack", None)
  return stack[-1] if stack else None


class CancellationManagerContext:
  """A Python context for wrapping a cancellable ConcreteFunction."""

  def __init__(self, cancellation_manager):
    self._cancellation_manager = cancellation_manager

  def __enter__(self):
    if not hasattr(_active_context, "manager_stack"):
      _active_context.manager_stack = []
    _active_context.manager_stack.append(self._cancellation_manager)

  def __exit__(self, exc_type, exc_value, exc_tb):
    _active_context.manager_stack.pop()
