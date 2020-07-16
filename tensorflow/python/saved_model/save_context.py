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
"""Context for building SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading


class SaveContext(threading.local):
  """A context for building a graph of SavedModel."""

  def __init__(self):
    super(SaveContext, self).__init__()
    self._in_save_context = False

  def enter_save_context(self):
    self._in_save_context = True

  def exit_save_context(self):
    self._in_save_context = False

  def in_save_context(self):
    return self._in_save_context

_save_context = SaveContext()


@contextlib.contextmanager
def save_context():
  _save_context.enter_save_context()
  try:
    yield
  finally:
    _save_context.exit_save_context()


def in_save_context():
  """Returns whether under a save context."""
  return _save_context.in_save_context()

