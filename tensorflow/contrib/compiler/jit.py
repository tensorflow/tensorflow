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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops


@contextlib.contextmanager
def experimental_jit_scope(compile_ops=True):
  """Enable or disable JIT compilation of operators within the scope.

  NOTE: This is an experimental feature.

  The compilation is a hint and only supported on a best-effort basis.

  Example usage:
    with tf.contrib.compiler.experimental_jit_scope():
      c = tf.matmul(a, b)  # compiled
    with tf.contrib.compiler.experimental_jit_scope(compile_ops=False):
      d = tf.matmul(a, c)  # not compiled
    with tf.contrib.compiler.experimental_jit_scope(
        compile_ops=lambda node_def: 'matmul' in node_def.op.lower()):
      e = tf.matmul(a, b) + d  # matmul is compiled, the addition is not.

  Args:
    compile_ops: Whether to enable or disable compilation in the scope.
      Either a Python bool, or a callable that accepts the parameter
      `node_def` and returns a python bool.
  Yields:
    The current scope, enabling or disabling compilation.

  """
  if callable(compile_ops):
    def xla_compile(node_def):
      return attr_value_pb2.AttrValue(b=compile_ops(node_def))
  else:
    xla_compile = attr_value_pb2.AttrValue(b=compile_ops)
  attrs = {"_XlaCompile": xla_compile}

  # TODO(ebrevdo): Keep a global XlaScope counter and here create a
  # special scope that checks if already within a xla scope or creates
  # a new one with a new scope string.  Add a new attr _XlaScope
  # taking this string.  Modify the xla fusion to respect scope
  # boundaries.  Modify gradients_impl to either create a new gradient
  # scope with a suffix from the fw scope or to try to fuse with
  # the fw scope of the given op.  Should be backwards compatible to
  # avoid having to modify Defun compilation attributes.

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access
