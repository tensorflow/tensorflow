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
    self._entered_load_context = []
    self._load_options = None

  def set_load_options(self, load_options):
    self._load_options = load_options
    self._entered_load_context.append(True)

  def clear_load_options(self):
    self._load_options = None
    self._entered_load_context.pop()

  def load_options(self):
    return self._load_options

  def in_load_context(self):
    return self._entered_load_context


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
