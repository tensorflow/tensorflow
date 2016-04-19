# Copyright 2016 Google Inc. All Rights Reserved.
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
"""tensor_util tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LocalVariableTest(tf.test.TestCase):

  def test_local_variable(self):
    with self.test_session() as sess:
      self.assertEquals([], tf.local_variables())
      value0 = 42
      tf.contrib.framework.local_variable(value0)
      value1 = 43
      tf.contrib.framework.local_variable(value1)
      variables = tf.local_variables()
      self.assertEquals(2, len(variables))
      self.assertRaises(tf.OpError, sess.run, variables)
      tf.initialize_variables(variables).run()
      self.assertAllEqual(set([value0, value1]), set(sess.run(variables)))


class GlobalStepTest(tf.test.TestCase):

  def _assert_global_step(self, global_step, expected_dtype=tf.int64):
    self.assertEquals("%s:0" % tf.GraphKeys.GLOBAL_STEP, global_step.name)
    self.assertEquals(expected_dtype, global_step.dtype.base_dtype)
    self.assertEquals([], global_step.get_shape().as_list())

  def test_invalid_dtype(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      tf.Variable(
          0.0, trainable=False, dtype=tf.float32, name=tf.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(
          TypeError, "does not have integer type",
          tf.contrib.framework.get_global_step)
    self.assertRaisesRegexp(
        TypeError, "does not have integer type",
        tf.contrib.framework.get_global_step, g)

  def test_invalid_shape(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      tf.Variable(
          [0], trainable=False, dtype=tf.int32, name=tf.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(
          TypeError, "not scalar",
          tf.contrib.framework.get_global_step)
    self.assertRaisesRegexp(
        TypeError, "not scalar",
        tf.contrib.framework.get_global_step, g)

  def test_create_global_step(self):
    self.assertEquals(None, tf.contrib.framework.get_global_step())
    with tf.Graph().as_default() as g:
      global_step = tf.contrib.framework.create_global_step()
      self._assert_global_step(global_step)
      self.assertRaisesRegexp(
          ValueError, "already exists", tf.contrib.framework.create_global_step)
      self.assertRaisesRegexp(
          ValueError, "already exists", tf.contrib.framework.create_global_step,
          g)
      self._assert_global_step(
          tf.contrib.framework.create_global_step(tf.Graph()))

  def test_get_global_step(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      tf.Variable(
          0, trainable=False, dtype=tf.int32, name=tf.GraphKeys.GLOBAL_STEP)
      self._assert_global_step(
          tf.contrib.framework.get_global_step(), expected_dtype=tf.int32)
    self._assert_global_step(
        tf.contrib.framework.get_global_step(g), expected_dtype=tf.int32)


if __name__ == "__main__":
  tf.test.main()
