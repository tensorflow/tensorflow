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

import tensorflow as tf


@contextlib.contextmanager
def experimental_jit_scope(compile_ops=True):
  """Enable or disable JIT compilation of operators within the scope.

  NOTE: This is an experimental feature.

  The compilation is a hint and only supported on a best-effort basis.

  Example usage:
    with tf.contrib.framework.experimental_jit_scope():
      c = tf.matmul(a, b)  # compiled
    with tf.contrib.framework.experimental_jit_scope(compile_ops=False):
        d = tf.matmul(a, c)  # not compiled

  Args:
    compile_ops: boolean, whether to enable or disable compilation in the scope.
  Yields:
    The current scope, enabling or disabling compilation.

  """
  attrs = {"_XlaCompile": tf.AttrValue(b=compile_ops)}
  # pylint: disable=protected-access
  with tf.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access
