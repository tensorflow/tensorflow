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
"""Experimental impl of tf.function using unified APIs, for testing only."""

from tensorflow.python.framework.experimental import _unified_api
from tensorflow.python.framework.experimental import context_stack as context_lib
from tensorflow.python.util import nest

NewTracingContext = _unified_api.NewTracingContext


class Function(object):
  """Helper for tf.function."""

  def __init__(self, func, name=None):
    self._python_func = func
    # TODO(srbs): Uniquify this name.
    self.name = name or func.__name__

  def __call__(self, *args, **kwargs):
    # Flatten arguments.
    flat_args = nest.flatten(args, expand_composites=True)
    flat_kwargs = nest.flatten(kwargs, expand_composites=True)
    all_args = flat_args + flat_kwargs

    # Trace
    outer_ctx = context_lib.get_default()
    ctx = NewTracingContext(self.name)
    with context_lib.set_default(ctx):
      # TODO(srbs): Iterating over list of inputs is a known performance
      # bottleneck. Add a pybind API for this.
      inputs = [ctx.AddParameter(arg.DataType()) for arg in all_args]
      structured_args = nest.pack_sequence_as(args, inputs[:len(flat_args)])
      structured_kwargs = nest.pack_sequence_as(kwargs, inputs[len(flat_args):])
      structured_outputs = self._python_func(*structured_args,
                                             **structured_kwargs)

      py_outputs = nest.flatten(structured_outputs, expand_composites=True)
      num_outputs = len(py_outputs)
      # TODO(srbs): Drop Nones before calling Finalize.
      finalized_f = ctx.Finalize(py_outputs)
      outer_ctx.RegisterFunction(finalized_f)

    # Build call op
    call_op = outer_ctx.CreateOperation(self.name, "")
    call_op.SetOpName(self.name)
    for arg in all_args:
      call_op.AddInput(arg)
    call_op_outputs = call_op.Execute(num_outputs)

    # Cleanup
    outer_ctx.RemoveFunction(self.name)

    return nest.pack_sequence_as(structured_outputs, call_op_outputs)


def function(func):
  return Function(func)
