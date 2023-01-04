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
"""A wrapper for gen_multiplex_1_op.py.

This defines a public API and provides a docstring for the C++ Op defined by
multiplex_1_kernel.cc
"""

import tensorflow as tf
from tensorflow.python.platform import resource_loader

_multiplex_1_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("multiplex_1_kernel.so"))

examples_multiplex_dense = _multiplex_1_module.examples1_multiplex_dense

# In this example, this Python function is a trivial wrapper for the C++ Op:
# it provides a public API and docstring that are equivalent to the API
# and documentation of the C++ op. The motivation for it is to be a placeholder
# that allows a wider variety of non-breaking future changes than are possible
# with the generated wrapper alone. Having this wrapper is optional.
def multiplex(cond, a, b, name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.

  This is similar to `np.where` and `tf.where`, but simplified to only handle
  the case of dense tensors, no optional parameters, no broadcasting, etc..

  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>

  Args:
    cond: tf.Tensor of type bool. Where True, yield `a`, otherwise yield `b`.
    a: tf.Tensor with the same type and shape as `b`.
    b: tf.Tensor with the same type and shape as `a`.
    name: An optional name for the op.

  Returns:
    A tf.Tensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  return examples_multiplex_dense(
      cond=cond, a_values=a, b_values=b, name=name)
