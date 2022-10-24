# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Library for controlling the Tensorflow/XLA JIT compiler."""

import contextlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export


_XLA_SCOPE_KEY = ("__xla_scope",)


class _XlaScope(object):
  """Keeps track of previous XLA scope calls, and depth of current call."""

  def __init__(self, count, depth):
    self.count = count
    self.depth = depth


@contextlib.contextmanager
@tf_export("xla.experimental.jit_scope")
def experimental_jit_scope(compile_ops=True, separate_compiled_gradients=False):
  """Enable or disable JIT compilation of operators within the scope.

  NOTE: This is an experimental feature.

  The compilation is a hint and only supported on a best-effort basis.

  Example usage:

    ```python
    with tf.xla.experimental.jit_scope():
      c = tf.matmul(a, b)  # compiled
    with tf.xla.experimental.jit_scope(compile_ops=False):
      d = tf.matmul(a, c)  # not compiled
    with tf.xla.experimental.jit_scope(
        compile_ops=lambda node_def: 'matmul' in node_def.op.lower()):
      e = tf.matmul(a, b) + d  # matmul is compiled, the addition is not.
    ```

  Example of `separate_compiled_gradients`:

    ```python
    # In the example below, the computations for f, g and h will all be compiled
    # in separate scopes.
    with tf.xla.experimental.jit_scope(
        separate_compiled_gradients=True):
      f = tf.matmul(a, b)
    g = tf.gradients([f], [a, b], name='mygrads1')
    h = tf.gradients([f], [a, b], name='mygrads2')
    ```

  Ops that are not in the scope may be clustered and compiled with ops in
  the scope with `compile_ops=True`, while the ops in the scope with
  `compile_ops=False` will never be compiled.

  For example:

    ```python
    # In the example below, x and loss may be clustered and compiled together,
    # while y will not be compiled.
    with tf.xla.experimental.jit_scope():
      x = tf.matmul(a, b)
    with tf.xla.experimental.jit_scope(compile_ops=False):
      y = tf.matmul(c, d)
    loss = x + y
    ```

  If you want to only compile the ops in the scope with `compile_ops=True`,
  consider adding an outer `jit_scope(compile_ops=False)`:

    ```python
    # In the example below, only x will be compiled.
    with tf.xla.experimental.jit_scope(compile_ops=False):
      with tf.xla.experimental.jit_scope():
        x = tf.matmul(a, b)
      y = tf.matmul(c, d)
      loss = x + y
    ```

  Args:
    compile_ops: Whether to enable or disable compilation in the scope.
      Either a Python bool, or a callable that accepts the parameter
      `node_def` and returns a python bool.
    separate_compiled_gradients: If true put each gradient subgraph into a
      separate compilation scope. This gives fine-grained control over which
      portions of the graph will be compiled as a single unit. Compiling
      gradients separately may yield better performance for some graphs.
      The scope is named based on the scope of the forward computation as well
      as the name of the gradients. As a result, the gradients will be compiled
      in a scope that is separate from both the forward computation, and from
      other gradients.
  Raises:
    RuntimeError: if called when eager execution is enabled.
  Yields:
    The current scope, enabling or disabling compilation.
  """
  if context.executing_eagerly():
    raise RuntimeError("xla.experimental.jit_scope is not supported when eager "
                       "execution is enabled. Try use it inside tf.function.")

  if callable(compile_ops):
    def xla_compile(node_def):
      return attr_value_pb2.AttrValue(b=compile_ops(node_def))
  else:
    xla_compile = attr_value_pb2.AttrValue(b=compile_ops)

  attrs = {
      "_XlaCompile":
          xla_compile,
      "_XlaSeparateCompiledGradients":
          attr_value_pb2.AttrValue(b=bool(separate_compiled_gradients))
  }

  # Find the singleton counter for the current scoped graph.  If it
  # doesn't exist, create one.
  xla_scope_counter = ops.get_collection(_XLA_SCOPE_KEY)
  if not xla_scope_counter:
    xla_scope_counter = _XlaScope(0, 0)
    ops.add_to_collection(_XLA_SCOPE_KEY, xla_scope_counter)
  else:
    xla_scope_counter = xla_scope_counter[0]

  if xla_scope_counter.depth == 0:
    # If we're at the root xla scope, we can increase the counter so
    # future calls to jit_scope use a different scope value.
    # If we're already within a scope, we'll be fusing using the scope
    # controlled by the parent.
    attrs["_XlaScope"] = attr_value_pb2.AttrValue(
        s=("jit_scope_%d" % xla_scope_counter.count).encode())
    xla_scope_counter.count += 1

  xla_scope_counter.depth += 1

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

  xla_scope_counter.depth -= 1
