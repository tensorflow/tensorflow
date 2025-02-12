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
"""A wrapper for gen_multiplex_4_op.py.

This defines a public API (and provides a docstring for it) for the C++ Op
defined by multiplex_4_kernel.cc
"""

import tensorflow as tf
from tensorflow.python.platform import resource_loader

_multiplex_4_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("multiplex_4_kernel.so"))

examples_multiplex_dense = _multiplex_4_module.examples_multiplex_dense


def multiplex(cond, a, b, name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.

  This is similar to `np.where` and `tf.where` if `cond` and `a` are tensors.
  This is similar to `np.select` if `cond` and `a` are lists of tensors.
  In either case, this is simplified to only handle the case of dense tensors,
  no optional parameters, no broadcasting, etc..

  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>

  >>> a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
  >>> a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
  >>> a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
  >>> b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
  >>> cond1 = tf.constant([False, False, True, False, False], dtype=bool)
  >>> cond2 = tf.constant([False, False, False, False, True], dtype=bool)
  >>> cond3 = tf.constant([True, False, True, False, True], dtype=bool)
  >>> multiplex_4_op.multiplex([cond1, cond2, cond3], [a1, a2, a3], b)
  <tf.Tensor: shape=(5,), ... numpy=array([ 11, 102,   3, 104,  10], ...)>

  Args:
    cond: tf.Tensor or list of tf.Tensor of type bool. Where True, yield `a`.
      When multiple corresponding `cond` elements are true, the first one yield
      based on the first one encountered.
    a: tf.Tensor or list of tf.Tensor, each with the same type and shape as `b`.
    b: tf.Tensor or list of tf.Tensor with the same type and shape as `a`. Yield
      `b` if all corresponding `cond` values is False.
    name: An optional name for the op.

  Returns:
    A tf.Tensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  if not isinstance(cond, (list, tuple)):
    # Support "old" use of multiplex where `cond` and `a` are tensors,
    # not lists of tensors.
    return examples_multiplex_dense(
        cond=[cond], a_values=[a], b_values=b, name=name)
  return examples_multiplex_dense(
      cond=cond, a_values=a, b_values=b, name=name)
