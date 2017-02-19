# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Imperative mode for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.imperative import imperative_graph
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops


class ImperativeMode(object):
  """Imperative mode execution of TensorFlow graphs.

  This class is a container for an ImperativeGraph, a session, and other
  context managers that enable imperative mode execution. The following is
  the common usage pattern:

  ```python
  server = tf.train.Server.create_local_server()
  with ImperativeMode(server.target):
    a = tf.random_normal([])
    b = tf.random_normal([])
    c = a + b
    c_val = c.value
    d = c + 1.0
    d_val = d.value
    # Expect d_val == c_val + 1.0
  ```

  ImperativeMode provides the illusion of immediate execution. It still
  constructs a graph and defers op execution. But when an op executes for
  the first time, its results are cached and the cached value is returned for
  future executions. The __exit__ method clears this graph and cached values.
  To use ImperativeMode inside a loop, the `new_step` method can be used to
  create a temporary context around the loop body to clear the cache at loop
  exit as follows:

  ```python
  server = tf.train.Server.create_local_server()
  with ImperativeMode(server.target) as mode:
    w = tf.get_variable('w', [])
    for i in range(10):
      with mode.new_step():
        x = tf.random_uniform([])
        y = tf.random_uniform([])
        z = w.assign_add(x + y)
        print(z.value)
  ```

  ImperativeMode graph does not support all TensorFlow operations and features.
  Here are the current known limitations of ImperativeMode :
  * Stateful operations returned ref-typed tensors are limited to
  TensorFlow Variables and the associated operations. Data structures such as
  queues barriers, etc. are not supported in ImperativeMode.
  * Variables created and managed via `tf.variable_scope` and the associated
  `tf.get_variable` are not supported. (These use auxiliary data structures in
  addition to the graph, which are not aware of the imperative mode execution.)

  TODO(keveman): Remove the above restrictions on ImperativeMode.
  """

  def __init__(self, target, parent_graph=None):
    """Initializes an ImperativeMode.

    Args:
      target: The TensorFlow execution engine to connect to.
      parent_graph: (Optional) An ImperativeGraph.

    Raises:
      UnimplementedError: if non-None parent_graph is not an ImperativeGraph.
    """
    self._target = target
    self._parent_graph = parent_graph
    # Create a new graph
    self._graph = imperative_graph.ImperativeGraph(
        parent_graph=self._parent_graph)
    self._default_graph = self._graph.as_default()
    # Context manager to record variable inits
    self._record_variable_inits = self._graph.record_variable_inits()
    if self._parent_graph:
      if not isinstance(self._parent_graph, imperative_graph.ImperativeGraph):
        raise errors.UnimplementedError(None, None, 'ImperativeMode needs an '
                                        'ImperativeGraph')
      # Clone the `_parent_graph` in to the current graph. This is so that
      # operations used from the enclosing ImperativeMode context are
      # available in the current context.
      with self._graph.as_default(), self._graph.return_as_is():
        importer.import_graph_def(self._parent_graph.as_graph_def(), name='')
    self._session = session.Session(graph=self._graph, target=self._target)
    # Override the `_session`'s run, so that variable inits can be
    # called before the actual run.
    self._old_run = self._session.run
    self._session.run = self.run
    self._context_managers = [
        self._session.as_default(),
        self._default_graph,
        self._record_variable_inits,
        imperative_graph.add_session_attr(ops.Tensor, self._session)]

  def run(self, *args, **kwargs):
    """Runs the variable init ops before calling the original run method."""
    self._graph.run_pending_inits(self._session)
    ret = self._old_run(*args, **kwargs)
    return ret

  def __enter__(self):
    """Enters the runtime contexts of the `_context_managers`."""
    for c in self._context_managers:
      c.__enter__()
    return self

  def __exit__(self, exec_type, exec_value, exec_tb):
    """Cleans up resources, exits the runtime contexts in reverse order."""
    # pylint: disable=protected-access
    if self._graph._variable_cleanup_ops:
      self._session.run(self._graph._variable_cleanup_ops)
    # pylint: enable=protected-access
    self._session.close()

    for c in reversed(self._context_managers):
      c.__exit__(exec_type, exec_value, exec_tb)

  def new_step(self):
    """Returns a new 'child' ImperativeMode.

    `new_step` enables running the imperative mode inside a Python loop. The
    ImperativeGraph object and the tensors created and cached during the
    execution of that graph are destroyed when the context entered with the
    object returned from this function is 'exited'. However, the operations
    in `self._graph` and any of its ancestors can be freely used as
    operands to operations in the graph contained in the object returned
    by this function.

    Returns:
      A new ImperativeMode object.
    """
    self._graph.run_pending_inits(self._session)
    return ImperativeMode(self._target, parent_graph=self._graph)
