# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Context for storing options for loading a SavedModel."""

import contextlib
import threading

from tensorflow.python.util.tf_export import tf_export


class LoadContext(threading.local):
  """A context for loading a model."""

  def __init__(self):
    super().__init__()
    # Stack of load options, one entry per nested `load_context()` call. The
    # top of the stack is the currently active load options; popping on exit
    # restores whatever options (if any) were active before entering, rather
    # than always resetting to `None`.
    self._load_options_stack = []

  def set_load_options(self, load_options):
    self._load_options_stack.append(load_options)

  def clear_load_options(self):
    if not self._load_options_stack:
      raise RuntimeError(
          "clear_load_options() called when load context stack is empty."
      )
    self._load_options_stack.pop()

  def load_options(self):
    if not self._load_options_stack:
      return None
    return self._load_options_stack[-1]

  def in_load_context(self):
    return bool(self._load_options_stack)


_load_context = LoadContext()


@tf_export("__internal__.load_context", v1=[])
@contextlib.contextmanager
def load_context(load_options):
  _load_context.set_load_options(load_options)
  try:
    yield
  finally:
    _load_context.clear_load_options()


def get_load_options():
  """Returns the load options under a load context."""
  return _load_context.load_options()


def in_load_context():
  """Returns whether under a load context."""
  return _load_context.in_load_context()
