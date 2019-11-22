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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow


class CancellationManager(object):
  """A mechanism for cancelling blocking computation."""

  def __init__(self):
    self._impl = pywrap_tensorflow.TFE_NewCancellationManager()

  @property
  def is_cancelled(self):
    """Returns `True` if `CancellationManager.start_cancel` has been called."""
    return pywrap_tensorflow.TFE_CancellationManagerIsCancelled(self._impl)

  def start_cancel(self):
    """Cancels blocking operations that have been registered with this object."""
    pywrap_tensorflow.TFE_CancellationManagerStartCancel(self._impl)

  def get_cancelable_function(self, concrete_function):
    # pylint: disable=protected-access
    return concrete_function._experimental_with_cancellation_manager(self)

  def __del__(self):
    pywrap_tensorflow.TFE_DeleteCancellationManager(self._impl)
