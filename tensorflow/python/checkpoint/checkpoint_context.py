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
"""Context for saving checkpoint."""

import contextlib
import threading


class PreemptionSaveContext(threading.local):
  """A context for monitoring saving checkpoint pre-preemption."""

  def __init__(self):
    super(PreemptionSaveContext, self).__init__()
    self._in_preemption_save_context = False

  def enter_preemption_save_context(self):
    self._in_preemption_save_context = True

  def exit_preemption_save_context(self):
    self._in_preemption_save_context = False

  def in_preemption_save_context(self):
    return self._in_preemption_save_context


_preemption_save_context = PreemptionSaveContext()


@contextlib.contextmanager
def preemption_save_context():
  _preemption_save_context.enter_preemption_save_context()
  try:
    yield
  finally:
    _preemption_save_context.exit_preemption_save_context()


def in_preemption_save_context():
  return _preemption_save_context.in_preemption_save_context()
