# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.framework.python.ops import prettyprint_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class PrettyPrintOpsTest(test.TestCase):

  def testPrintTensorPassthrough(self):
    a = constant_op.constant([1])
    a = prettyprint_ops.print_op(a)
    with self.cached_session():
      self.assertEqual(a.eval(), constant_op.constant([1]).eval())

  def testPrintSparseTensorPassthrough(self):
    a = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    b = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    a = prettyprint_ops.print_op(a)
    with self.cached_session():
      self.assertAllEqual(
          sparse_ops.sparse_tensor_to_dense(a).eval(),
          sparse_ops.sparse_tensor_to_dense(b).eval())

  def testPrintTensorArrayPassthrough(self):
    a = tensor_array_ops.TensorArray(
        size=2, dtype=dtypes.int32, clear_after_read=False)
    a = a.write(1, 1)
    a = a.write(0, 0)
    a = prettyprint_ops.print_op(a)
    with self.cached_session():
      self.assertAllEqual(a.stack().eval(), constant_op.constant([0, 1]).eval())

  def testPrintVariable(self):
    a = variables.Variable(1.0)
    a = prettyprint_ops.print_op(a)
    with self.cached_session():
      variables.global_variables_initializer().run()
      a.eval()


if __name__ == "__main__":
  test.main()
