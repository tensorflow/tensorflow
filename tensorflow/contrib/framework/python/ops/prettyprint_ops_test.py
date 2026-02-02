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
import tensorflow as tf


class PrettyPrintOpsTest(tf.test.TestCase):

  def testPrintTensorPassthrough(self):
    a = tf.constant([1])
    a = tf.contrib.framework.print_op(a)
    with self.test_session():
      self.assertEqual(a.eval(), tf.constant([1]).eval())

  def testPrintSparseTensorPassthrough(self):
    a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], shape=[3, 4])
    b = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], shape=[3, 4])
    a = tf.contrib.framework.print_op(a)
    with self.test_session():
      self.assertAllEqual(tf.sparse_tensor_to_dense(a).eval(),
                          tf.sparse_tensor_to_dense(b).eval())

  def testPrintTensorArrayPassthrough(self):
    a = tf.TensorArray(size=2, dtype=tf.int32, clear_after_read=False)
    a = a.write(1, 1)
    a = a.write(0, 0)
    a = tf.contrib.framework.print_op(a)
    with self.test_session():
      self.assertAllEqual(a.pack().eval(), tf.constant([0, 1]).eval())

if __name__ == "__main__":
  tf.test.main()
