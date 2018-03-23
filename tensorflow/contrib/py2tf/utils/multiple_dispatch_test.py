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
"""Tests for multiple_dispatch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.py2tf.utils import multiple_dispatch
from tensorflow.python.client.session import Session
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.platform import test


class MultipleDispatchTest(test.TestCase):

  def test_dynamic_is_python(self):
    a = np.eye(3)
    also_a = a
    not_actually_a = np.eye(3)
    should_be_true1 = multiple_dispatch.dynamic_is(a, also_a)
    should_be_false1 = multiple_dispatch.dynamic_is_not(a, also_a)
    should_be_true2 = multiple_dispatch.dynamic_is_not(a, not_actually_a)
    should_be_false2 = multiple_dispatch.dynamic_is(a, not_actually_a)
    self.assertTrue(should_be_true1)
    self.assertTrue(should_be_true2)
    self.assertFalse(should_be_false1)
    self.assertFalse(should_be_false2)

  def test_dynamic_is_tf(self):
    with Session().as_default():
      a = constant([2.0])
      also_a = a
      not_actually_a = constant([2.0])
      should_be_true1 = multiple_dispatch.dynamic_is(a, also_a)
      should_be_false1 = multiple_dispatch.dynamic_is_not(a, also_a)
      should_be_true2 = multiple_dispatch.dynamic_is_not(a, not_actually_a)
      should_be_false2 = multiple_dispatch.dynamic_is(a, not_actually_a)
      self.assertTrue(should_be_true1)
      self.assertTrue(should_be_true2)
      self.assertFalse(should_be_false1)
      self.assertFalse(should_be_false2)

  def test_run_cond_python(self):
    true_fn = lambda: 2.0
    false_fn = lambda: 3.0
    self.assertEqual(multiple_dispatch.run_cond(True, true_fn, false_fn), 2.0)
    self.assertEqual(multiple_dispatch.run_cond(False, true_fn, false_fn), 3.0)

  def test_run_cond_tf(self):

    true_fn = lambda: constant([2.0])
    false_fn = lambda: constant([3.0])
    with Session() as sess:
      out = multiple_dispatch.run_cond(constant(True), true_fn, false_fn)
      self.assertEqual(sess.run(out), 2.0)
      out = multiple_dispatch.run_cond(constant(False), true_fn, false_fn)
      self.assertEqual(sess.run(out), 3.0)

  def test_run_while_python(self):
    cond_fn = lambda x, t, s: x > t
    body_fn = lambda x, t, s: (x * s, t, s)

    x, _, _ = multiple_dispatch.run_while(cond_fn, body_fn, [3.0, 1.0, 0.5])
    self.assertEqual(x, 0.75)

    x, _, _ = multiple_dispatch.run_while(cond_fn, body_fn, [3.0, 4.0, 0.5])
    self.assertEqual(x, 3.0)

  def test_run_while_tf(self):
    cond_fn = lambda x, t, s: x > t
    body_fn = lambda x, t, s: (x * s, t, s)

    with Session() as sess:
      x, _, _ = multiple_dispatch.run_while(cond_fn, body_fn,
                                            [constant(3.0), 1.0, 0.5])
      self.assertEqual(sess.run(x), 0.75)

      x, _, _ = multiple_dispatch.run_while(cond_fn, body_fn,
                                            [constant(3.0), 4.0, 0.5])
      self.assertEqual(sess.run(x), 3.0)


if __name__ == '__main__':
  test.main()
