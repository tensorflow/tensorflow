# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Monitor is responsible for training, checkpointing and recovery."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.ops import variables


class Monitor(object):
  """Executes training steps, recovers and checkpoints.

  Note that this class is particularly preliminary, experimental, and
  expected to change.
  """
  # TODO(isaprykin): Support step functions that need multiple session calls.
  # TODO(isaprykin): Support extra arguments to the step function.
  # TODO(isaprykin): Support recovery, checkpointing and summaries.

  def __init__(self, step_callable, session=None):
    """Initialize the Monitor with components for executing training steps.

    Args:
      step_callable: a training `Step` that's capable of signaling when done.
      session: a `Session` instance that's needed for graph mode.

    Raises:
      ValueError: if `session` was provided for eager mode or not provided for
        graph mode.
    """
    if context.executing_eagerly():
      if session is not None:
        raise ValueError("Should not provide a `session` in Eager mode.")
      self._run_step = step_callable
    else:
      if session is None:
        raise ValueError("Should provide a `session` in Graph mode.")
      session.run(step_callable.initialize())
      self._run_step = session.make_callable(step_callable())
      session.run(variables.global_variables_initializer())

  def run_steps(self, num_steps=None):
    step = 0
    while num_steps is None or step < num_steps:
      try:
        self._run_step()
        step += 1
      except errors.OutOfRangeError:
        break
