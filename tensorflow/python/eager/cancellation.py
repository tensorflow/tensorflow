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

from tensorflow.python import pywrap_tfe


class CancellationManager(object):
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

  def get_cancelable_function(self, concrete_function):
    # pylint: disable=protected-access
    return concrete_function._experimental_with_cancellation_manager(self)
