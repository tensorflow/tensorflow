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
"""Experimental impl for GradientTape using unified APIs, for testing only."""

from tensorflow.python.framework.experimental import _tape
from tensorflow.python.framework.experimental import context_stack
from tensorflow.python.framework.experimental import gradient_registry
from tensorflow.python.util import nest


class GradientTape(object):
  """GradientTape using the unified API."""

  def __init__(self, persistent=False):
    self._c_tape = _tape.Tape(persistent)
    ctx = context_stack.get_default()
    self._tape_context = _tape.TapeContext(
        ctx, self._c_tape, gradient_registry.get_global_registry())
    self._ctx_manager = None

  def watch(self, t):
    self._c_tape.Watch(t)

  # TODO(srbs): Add support for unconnected_gradients.
  def gradient(self, targets, sources, output_gradients=None):
    ctx = context_stack.get_default()
    flat_targets = nest.flatten(targets)
    flat_sources = nest.flatten(sources)
    out_grads = self._c_tape.ComputeGradient(ctx, flat_targets, flat_sources,
                                             output_gradients or [])
    return nest.pack_sequence_as(sources, out_grads)

  def __enter__(self):
    """Enters a context inside which operations are recorded on this tape."""
    self._ctx_manager = context_stack.set_default(self._tape_context)
    self._ctx_manager.__enter__()
    return self

  def __exit__(self, typ, value, traceback):
    self._ctx_manager.__exit__(typ, value, traceback)
    self._ctx_manager = None
