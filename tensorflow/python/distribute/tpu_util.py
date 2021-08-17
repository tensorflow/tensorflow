# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for TPU."""

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.tpu import tpu


def enclosing_tpu_context():
  """Returns the TPUReplicateContext, which exists inside a tpu.rewrite()."""
  return enclosing_tpu_context_and_graph()[0]


def enclosing_tpu_context_and_graph():
  """Returns the TPUReplicateContext which exists inside a tpu.rewrite(), and its associated graph."""
  graph = ops.get_default_graph()
  while graph is not None:
    ctx = graph._get_control_flow_context()  # pylint: disable=protected-access
    while ctx is not None:
      if isinstance(ctx, tpu.TPUReplicateContext):
        return ctx, graph
      ctx = ctx.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None, None


@contextlib.contextmanager
def outside_or_skip_tpu_context():
  """Returns a context manager that skips current enclosing context if there is any."""
  ctx, graph = enclosing_tpu_context_and_graph()
  if ctx is None:
    yield
  else:
    saved_context = graph._get_control_flow_context()  # pylint: disable=protected-access
    graph._set_control_flow_context(ctx.outer_context)  # pylint: disable=protected-access
    yield
    graph._set_control_flow_context(saved_context)  # pylint: disable=protected-access
