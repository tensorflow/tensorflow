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
"""Tests for misc module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.utils.misc import alias_tensors
from tensorflow.contrib.py2tf.utils.misc import dynamic_len
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops.variables import Variable
from tensorflow.python.platform import test


class ContextManagersTest(test.TestCase):

  def test_dynamic_len_tf_scalar(self):
    a = constant(1)

    with self.assertRaises(ValueError):
      with self.test_session() as sess:
        sess.run(dynamic_len(a))

  def test_dynamic_len_tf_array(self):
    a = constant([1, 2, 3])

    with self.test_session() as sess:
      self.assertEqual(3, sess.run(dynamic_len(a)))

  def test_dynamic_len_tf_matrix(self):
    a = constant([[1, 2], [3, 4]])

    with self.test_session() as sess:
      self.assertEqual(2, sess.run(dynamic_len(a)))

  def test_dynamic_len_py_list(self):
    a = [3] * 5

    self.assertEqual(5, dynamic_len(a))

  def test_alias_single_tensor(self):
    a = constant(1)

    new_a = alias_tensors(a)
    self.assertFalse(new_a is a)
    with self.test_session() as sess:
      self.assertEqual(1, sess.run(new_a))

  def test_alias_tensors(self):
    a = constant(1)
    v = Variable(2)
    s = 'a'
    l = [1, 2, 3]

    new_a, new_v, new_s, new_l = alias_tensors(a, v, s, l)

    self.assertFalse(new_a is a)
    self.assertTrue(new_v is v)
    self.assertTrue(new_s is s)
    self.assertTrue(new_l is l)
    with self.test_session() as sess:
      self.assertEqual(1, sess.run(new_a))


if __name__ == '__main__':
  test.main()
