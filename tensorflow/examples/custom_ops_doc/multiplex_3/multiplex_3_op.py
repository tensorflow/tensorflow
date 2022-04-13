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
"""A wrapper for gen_multiplex_3_op.py.

This defines a public API and provides a docstring for the C++ Op defined by
multiplex_3_kernel.cc
"""

import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_2.multiplex_2_op import examples_multiplex_dense
from tensorflow.python.platform import resource_loader


_multiplex_3_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("multiplex_3_kernel.so"))

examples_multiplex_sparse = _multiplex_3_module.examples_multiplex_sparse


# This Python function uses a decorator for "tensor API dispatch" so that it
# specializes the "gen_multiplex_2_op.examples_multiplex_dense" function that
# previously supported only dense tensors (using the C++ kernel from the
# previous multiplex_2 example) to now support sparse tensors (using the new
# C++ kernel from this example). Since multiplex_2_op.multiplex uses this,
# it now supports sparse tensors. Behavior for dense tensors is unchanged so
# this is a non-breaking change. See
# https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch
@tf.experimental.dispatch_for_api(examples_multiplex_dense)
def multiplex_sparse(cond: tf.SparseTensor,
                     a: tf.SparseTensor,
                     b: tf.SparseTensor,
                     name=None):
  """Return elements chosen from `a` or `b` depending on `cond`.


  This is similar to `np.where` and `tf.where`, but simplified to only handle
  the case of rank 1 sparse tensors, no optional parameters, no broadcasting,
  etc..

  >>> cond = tf.SparseTensor(
  ...     indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
  >>> a = tf.sparse.from_dense(['', 'a0', '', 'a1', '', 'a2', ''])
  >>> b = tf.sparse.from_dense(['b0', '', 'b1', 'b2', '', '', 'b3'])
  >>> multiplex_3_op.multiplex_sparse(cond, a, b)
  SparseTensorValue(indices=array([[0],
    [1],
    [2],
    [3]]), values=array([b'b0', b'a0', b'b1', b'b2'], dtype=object),
    dense_shape=array([7]))
  Args:
    cond: tf.SparseTensor of type bool. Where True, yield `a`, otherwise yield
      `b`.
    a: tf.SparseTensor with the same type and shape as `b`.
    b: tf.SparseTensor with the same type and shape as `a`.
    name: An optional name for the op.

  Returns:
    A tf.SparseTensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  (indices, values, shape) = examples_multiplex_sparse(
      cond_indices=cond.indices,
      cond_values=cond.values,
      cond_shape=cond.dense_shape,
      a_indices=a.indices,
      a_values=a.values,
      a_shape=a.dense_shape,
      b_indices=b.indices,
      b_values=b.values,
      b_shape=b.dense_shape,
      name=name)
  return tf.SparseTensor(indices, values, shape)
